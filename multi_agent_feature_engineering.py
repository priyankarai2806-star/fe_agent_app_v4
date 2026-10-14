"""
Multi-agent Feature Engineering system - v4 (LangGraph + Groq). Python 3.9 compatible.

New in v4
  * Use-case personas: every prompt is written for a specific modelling goal
    (demand forecasting, marketing mix modelling, price & promo elasticity,
    churn / propensity, general regression) and industry (CPG, Retail, General).
  * Data profiling step: the agent detects date, group (store / SKU) and column
    roles before planning, and keeps that profile in state.
  * Structured, explainable plan: the LLM returns observations + one row per
    feature with source columns, construction and reason.
  * Explicit test catalogue: statistical tests S1-S4 and business tests B1-B5,
    recorded PASS / FAIL per feature.

Graphs
  Orchestrator : __start__ -> orchestrator -> feature_engineering_agent -> orchestrator
                                           -> feature_selection_agent   -> orchestrator -> __end__
  FE agent     : profile_data -> recommend_feature_engineering_steps -> create_feature_engineering_code
                 -> execute_feature_engineering_code (-> fix_feature_engineering_code -> execute)*
                 -> report_feature_engineering
  FS agent     : prepare_candidates -> statistical_agent + business_agent (parallel)
                 -> select_features -> report_feature_selection
"""
import functools
import json
import operator
import os
import re
import time
import traceback
import uuid
import warnings
from datetime import date, datetime
from typing import Annotated, List, Optional, TypedDict

import numpy as np
import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# ===========================================================================
# SETTINGS (the Streamlit app changes these at run time)
# ===========================================================================
MAX_MISSING_PCT = 50.0      # S2: drop if more than this % missing
MIN_ABS_CORR = 0.05         # S3: drop if |rho| pooled AND within-group are both below this
COLLINEAR_CORR = 0.95       # S4: drop weaker feature of a pair with |rho| >= this
MIN_BUSINESS_SCORE = 3      # B1: relevance must be >= this (1-5)

# ===========================================================================
# INDUSTRY CONTEXT AND USE-CASE PERSONAS
# ===========================================================================
INDUSTRIES = {
    "CPG": (
        "Consumer packaged goods (CPG). Typical grain: SKU or brand x retailer/market x week. "
        "Typical drivers: base (regular) price and promoted price, temporary price reductions (TPR), "
        "feature and display support, distribution (ACV / weighted distribution), media spend or GRPs "
        "by channel (TV, digital, search, social), seasonality and holidays. Sales are often split "
        "into base and incremental volume."),
    "Retail": (
        "Retail. Typical grain: store (or store x category) x week or day. Typical drivers: shelf price "
        "and markdowns, promotions, store size and format, region, footfall, local events, holidays, "
        "weather and marketing spend. Differences between stores are large, so effects should be "
        "judged within each store as well as overall."),
    "General": "General tabular business data.",
}

USE_CASES = {
    "demand_forecasting": {
        "label": "Sales / demand forecasting",
        "persona": "a senior demand-forecasting data scientist with 10 years in CPG and retail",
        "model": ("a gradient-boosted tree or regularised regression model that predicts the target "
                  "H period(s) ahead at the grain of the data (e.g. store-week or SKU-week)"),
        "model_short": "a short-term demand forecasting model",
        "target_type": "regression",
        "allow_target_lags": True,
        "guidance": [
            "Lags and rolling means / std of the target within each group, shifted by at least the horizon",
            "Calendar: week of year, month, quarter, holiday and pre/post-holiday flags",
            "Price: price index vs the group's typical price, price change vs last period",
            "Promotion: promo flags, promo depth, promo x price and promo x media interactions",
            "Media carry-over (adstock) for TV / digital spend",
            "Group attributes: size, format, region (one-hot encoded)",
        ],
        "avoid": [
            "Any statistic of the target over the whole period (e.g. a store's mean sales over all weeks)",
            "Future values, centred rolling windows or lags shorter than the horizon",
            "Row identifiers encoded as numbers",
        ],
        "checks": ("Would a demand planner know this value when making the forecast, and does it help "
                   "explain week-to-week movement?"),
    },
    "mmm": {
        "label": "Marketing mix modelling (MMM)",
        "persona": "a marketing-mix-modelling econometrician who builds ROI models for CPG brands",
        "model": ("an interpretable linear (or Bayesian) regression of the target on media, price, "
                  "promotion and base drivers, used to estimate contribution and ROI per channel"),
        "model_short": "a marketing mix model",
        "target_type": "regression",
        "allow_target_lags": False,
        "guidance": [
            "Adstock (geometric carry-over) for every media / spend column, decay between 0.3 and 0.8",
            "Diminishing returns: log1p (or Hill) transform of the adstocked spend",
            "Base drivers: log price, distribution, seasonality (week of year or Fourier terms), holidays, trend",
            "Promotion flags and depth; promo x media interaction only if justified",
            "Keep the feature set small and every feature interpretable",
        ],
        "avoid": [
            "Lags or rolling statistics of the target (they absorb media effects and distort ROI)",
            "Several near-identical transforms of the same media channel",
            "Opaque encodings that a brand manager cannot interpret",
        ],
        "checks": ("Can a brand manager interpret this feature's coefficient, and does it represent a "
                   "real marketing lever or base driver?"),
    },
    "price_promo": {
        "label": "Price & promotion elasticity",
        "persona": "a pricing and revenue-management analyst for CPG and retail",
        "model": ("a log-log regression of the target on price and promotion variables that estimates "
                  "own-price elasticity and promotional lift, with controls for seasonality"),
        "model_short": "a price and promotion elasticity model",
        "target_type": "regression",
        "allow_target_lags": False,
        "guidance": [
            "Natural log of price; price index vs the group's regular (median) price",
            "Discount depth % vs regular price; price change vs last period",
            "Promotion type flags (TPR, feature, display) and promo x holiday",
            "Competitor or relative price ratios if such columns exist",
            "Seasonality and holiday controls; group fixed-effect encodings",
        ],
        "avoid": [
            "Lags or rolling statistics of the target",
            "Features that mix price with the target (e.g. revenue = price x sales)",
        ],
        "checks": "Does this feature help isolate the effect of price or promotion on the target?",
    },
    "churn": {
        "label": "Customer churn / propensity (classification)",
        "persona": "a CRM data scientist building propensity and churn models",
        "model": ("a binary classifier (logistic regression or gradient boosting) predicting the "
                  "probability of the target event"),
        "model_short": "a churn / propensity classifier",
        "target_type": "classification",
        "allow_target_lags": False,
        "guidance": [
            "Recency, frequency and monetary (RFM) style features",
            "Tenure and trend in activity or spend; ratios between recent and older activity",
            "One-hot encoding of low-cardinality categories; log of skewed amounts",
        ],
        "avoid": [
            "Anything recorded after the outcome",
            "Transformations of the target itself",
        ],
        "checks": "Would this be known before the customer churned, and does it plausibly drive the decision?",
    },
    "general": {
        "label": "General regression",
        "persona": "a senior data scientist",
        "model": "a supervised regression model of the target",
        "model_short": "a regression model",
        "target_type": "regression",
        "allow_target_lags": True,
        "guidance": [
            "Encodings for categories, log transforms for skewed columns, sensible interactions",
            "Date parts if a date exists; lags within groups if the data is a time series",
        ],
        "avoid": ["Using the current or future target value"],
        "checks": "Is this feature meaningful for the business question and known at prediction time?",
    },
}

DOMAIN_HINTS = [
    ("media / spend", ["spend", "grp", "tv", "media", "digital", "search", "social", "radio", "ooh", "impression"]),
    ("price", ["price", "cost", "markdown"]),
    ("promotion", ["promo", "tpr", "display", "feature", "discount", "deal"]),
    ("distribution", ["acv", "distribution", "tdp", "numeric_dist"]),
    ("calendar", ["holiday", "week", "month", "date", "season"]),
    ("store / product attribute", ["size", "region", "format", "store", "sku", "brand", "category"]),
    ("weather", ["temp", "rain", "weather"]),
]


def use_case_cfg(key: str) -> dict:
    return USE_CASES.get(key, USE_CASES["general"])


def stat_tests() -> List[dict]:
    """Statistical test catalogue with the current thresholds."""
    return [
        {"id": "S1", "test": "Variation", "rule": "More than one unique value",
         "why": "A constant column carries no information"},
        {"id": "S2", "test": "Completeness", "rule": f"Missing values <= {MAX_MISSING_PCT:.0f}%",
         "why": "Too many gaps make the feature unreliable"},
        {"id": "S3", "test": "Signal with target",
         "rule": f"|Spearman rho| >= {MIN_ABS_CORR} overall OR on average within each group",
         "why": "The feature should move with the target, across all rows or inside each store / SKU"},
        {"id": "S4", "test": "Redundancy",
         "rule": f"Not a near-duplicate (|rho| >= {COLLINEAR_CORR}) of a stronger feature",
         "why": "Duplicates add noise and unstable coefficients"},
    ]


def business_tests(use_case: str) -> List[dict]:
    cfg = use_case_cfg(use_case)
    return [
        {"id": "B1", "test": "Business relevance", "rule": f"LLM relevance score >= {MIN_BUSINESS_SCORE} (1-5)",
         "why": cfg["checks"]},
        {"id": "B2", "test": "No leakage", "rule": "LLM leakage risk is not 'high'",
         "why": "Uses no current / future target values or whole-period target statistics"},
        {"id": "B3", "test": "Known at prediction time", "rule": "LLM says the value is available in advance",
         "why": "Planned levers (price, promo, media, holidays) count as known"},
        {"id": "B4", "test": "Fits the model", "rule": f"LLM says it suits {cfg['model_short']}",
         "why": "Feature type must match the modelling approach"},
        {"id": "B5", "test": "Use-case rule", "rule": ("Target-derived features allowed" if cfg["allow_target_lags"]
                                                       else "No feature built from the target"),
         "why": "Deterministic rule from the use case (e.g. MMM excludes sales lags)"},
    ]


# ===========================================================================
# CHECKPOINT-SAFE STATE HELPERS
# ===========================================================================
def make_safe(obj):
    """Convert numpy / pandas values to plain Python so LangGraph can save them."""
    if isinstance(obj, dict):
        return {make_safe(k): make_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_safe(v) for v in obj]
    if obj is pd.NaT:
        return None
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, float) and np.isinf(obj):
        return None
    return obj


def df_to_state(df: pd.DataFrame) -> dict:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    out = out.reset_index(drop=True)
    return make_safe(out.to_dict(orient="list"))


def state_to_df(data: Optional[dict]) -> pd.DataFrame:
    return pd.DataFrame(data) if data else pd.DataFrame()


def safe_node(func):
    @functools.wraps(func)
    def wrapper(state):
        return make_safe(func(state))
    return wrapper


class SafeStateGraph(StateGraph):
    """StateGraph that automatically wraps every node with safe_node()."""
    def add_node(self, node, action=None, **kwargs):
        return super().add_node(node, safe_node(action), **kwargs)


# ===========================================================================
# GENERAL HELPERS
# ===========================================================================
def fill(template: str, **kw) -> str:
    """Replace <<key>> placeholders (avoids brace escaping in JSON examples)."""
    for k, v in kw.items():
        template = template.replace(f"<<{k}>>", str(v))
    return template


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def extract_json(text: str):
    """Return the first JSON object or list found in an LLM answer."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    body = fenced.group(1) if fenced else text
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = body.find(open_c), body.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(body[start:end + 1])
            except Exception:
                continue
    raise ValueError("No JSON found in LLM answer")


def run_feature_code(code: str, df: pd.DataFrame, target: str) -> pd.DataFrame:
    namespace = {"pd": pd, "np": np}
    exec(code, namespace)
    result = namespace["feature_engineer"](df.copy())
    if not isinstance(result, pd.DataFrame):
        raise TypeError("feature_engineer() must return a pandas DataFrame")
    if target and target not in result.columns:
        raise ValueError(f"Target column '{target}' was removed")
    if len(result) != len(df):
        raise ValueError(f"Row count changed from {len(df)} to {len(result)}")
    return result


def describe_value(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, dict) and value and all(isinstance(v, list) for v in value.values()):
        rows = len(next(iter(value.values())))
        return f"DataFrame: {rows} rows x {len(value)} cols"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{len(value)} keys]"
    text = str(value).replace("\n", " ")
    return text if len(text) <= 80 else text[:77] + "..."


def log_row(agent: str, node: str, detail: str, start: float, status: str = "ok") -> List[dict]:
    return [{"agent": agent, "node": node,
             "time": datetime.now().strftime("%H:%M:%S"),
             "duration_s": round(time.time() - start, 2),
             "status": status, "detail": detail}]


def llm_text(model, prompt: str) -> str:
    answer = model.invoke(prompt)
    return answer.content if hasattr(answer, "content") else str(answer)


# ===========================================================================
# DATA PROFILING (deterministic, no LLM)
# ===========================================================================
def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(50)
            if sample.empty or not sample.str.contains(r"\d").all():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.9:
                return col
    return None


def detect_group_columns(df: pd.DataFrame, target: str, date_col: Optional[str]) -> List[str]:
    """Columns that, with the date, identify one row (e.g. store_id, or sku + retailer for CPG)."""
    n = len(df)
    cands = [c for c in df.columns
             if c not in (target, date_col) and not pd.api.types.is_numeric_dtype(df[c])
             and 2 <= df[c].nunique() <= max(2, n // 2)]
    if not cands:
        return []
    if date_col:
        single = [c for c in cands if not df.duplicated([c, date_col]).any()]
        if single:
            return [max(single, key=lambda c: df[c].nunique())]
        pairs = [(a, b) for i, a in enumerate(cands) for b in cands[i + 1:]
                 if not df.duplicated([a, b, date_col]).any()]
        if pairs:
            return list(max(pairs, key=lambda ab: df[list(ab)].drop_duplicates().shape[0]))
    return [max(cands, key=lambda c: df[c].nunique())]


def build_data_context(df: pd.DataFrame, target: str, date_col=None, group_col=None) -> dict:
    date_col = date_col if date_col in df.columns else detect_date_column(df)
    if group_col == "__none__":
        groups = []
    elif group_col in df.columns:
        groups = [group_col]
    else:
        groups = detect_group_columns(df, target, date_col)
    group_col = " + ".join(groups) if groups else None
    ctx = {"rows": len(df), "columns": df.shape[1], "target": target,
           "target_type": "binary" if df[target].nunique() == 2 else "numeric",
           "date_column": date_col, "group_column": group_col, "group_columns": groups,
           "n_groups": int(df[groups].drop_duplicates().shape[0]) if groups else None,
           "frequency": None, "periods": None, "date_min": None, "date_max": None,
           "grain_unique": None}
    if date_col:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = pd.to_datetime(df[date_col], errors="coerce")
        uniq = d.dropna().drop_duplicates().sort_values()
        step = uniq.diff().dt.days.median() if len(uniq) > 1 else None
        ctx["frequency"] = ("daily" if step == 1 else "weekly" if step and 6 <= step <= 8
                            else "monthly" if step and 28 <= step <= 31 else
                            (f"every {step:.0f} days" if step else None))
        ctx["periods"] = int(len(uniq))
        ctx["date_min"] = str(uniq.min().date()) if len(uniq) else None
        ctx["date_max"] = str(uniq.max().date()) if len(uniq) else None
        if groups:
            ctx["grain_unique"] = not df.duplicated(groups + [date_col]).any()
    return ctx


def profile_columns(df: pd.DataFrame, ctx: dict) -> List[dict]:
    rows = []
    for col in df.columns:
        s = df[col]
        num = pd.api.types.is_numeric_dtype(s)
        row = {"column": col, "dtype": str(s.dtype),
               "missing_pct": round(float(s.isna().mean() * 100), 2),
               "unique": int(s.nunique()), "zero_pct": None, "skew": None, "role": "", "hint": ""}
        if num:
            row["zero_pct"] = round(float((s == 0).mean() * 100), 1)
            row["skew"] = round(float(s.skew()), 2) if s.nunique() > 2 else None
        if col == ctx["target"]:
            row["role"] = "target"
        elif col == ctx["date_column"]:
            row["role"] = "date"
        elif col in ctx.get("group_columns", []):
            row["role"] = "group id"
        elif not num:
            row["role"] = "categorical"
        elif set(s.dropna().unique()) <= {0, 1}:
            row["role"] = "binary flag"
        elif row["skew"] is not None and row["skew"] > 1:
            row["role"] = "numeric (right-skewed)"
        else:
            row["role"] = "numeric"
        low = col.lower()
        for hint, keys in DOMAIN_HINTS:
            if any(k in low for k in keys):
                row["hint"] = hint
                break
        if "noise" in low or "random" in low:
            row["hint"] = "suspicious: name suggests noise"
        rows.append(row)
    return rows


def summary_for_llm(df: pd.DataFrame, ctx: dict, profile: List[dict]) -> str:
    grain = (f"{ctx['group_column']} x {ctx['date_column']}" if ctx["group_column"] and ctx["date_column"]
             else ctx["date_column"] or "one row per record")
    lines = [f"Rows: {ctx['rows']}, columns: {ctx['columns']}",
             f"Target: {ctx['target']} ({ctx['target_type']})",
             f"Grain: {grain} (unique per row: {ctx['grain_unique']})",
             f"Date column: {ctx['date_column']}, frequency: {ctx['frequency']}, "
             f"periods: {ctx['periods']}, range: {ctx['date_min']} to {ctx['date_max']}",
             f"Group column: {ctx['group_column']}, groups: {ctx['n_groups']}", "",
             "Column profile (column | role | dtype | missing% | unique | zero% | skew | hint):"]
    for r in profile:
        lines.append(f"{r['column']} | {r['role']} | {r['dtype']} | {r['missing_pct']} | {r['unique']} | "
                     f"{r['zero_pct']} | {r['skew']} | {r['hint']}")
    lines += ["", "First rows:", df.head(5).to_string()]
    num = df.select_dtypes(include=np.number)
    if not num.empty:
        lines += ["", "Numeric summary:", num.describe().T.round(3).to_string()]
    return "\n".join(lines)


# ===========================================================================
# BASE AGENT
# ===========================================================================
class BaseAgent:
    """Common parts: own checkpointer, own thread id per run, state history."""
    name = "agent"

    def __init__(self, model, log: bool = False, log_path: Optional[str] = None):
        self.model = model
        self.log = log
        self.log_path = log_path or os.path.join(os.getcwd(), "logs")
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        self.config = None
        self.response = None

    def _build_graph(self):
        raise NotImplementedError

    def run(self, inputs: dict, thread_id: str) -> dict:
        self.config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
        self.response = self.graph.invoke(make_safe(inputs), self.config)
        return self.response

    def _save_code(self, code: str, name: str):
        if self.log:
            os.makedirs(self.log_path, exist_ok=True)
            with open(os.path.join(self.log_path, name), "w") as f:
                f.write(code)

    def get_state_history(self, show_values: bool = False) -> pd.DataFrame:
        rows = []
        for snap in self.graph.get_state_history(self.config):
            v = snap.values or {}
            row = {"agent": self.name,
                   "thread_id": self.config["configurable"]["thread_id"],
                   "checkpoint_id": snap.config["configurable"]["checkpoint_id"],
                   "step": (snap.metadata or {}).get("step"),
                   "next_node": ", ".join(snap.next) if snap.next else "END",
                   "keys_in_state": len(v)}
            if show_values:
                row.update({k: describe_value(x) for k, x in v.items() if k != "session_log"})
            rows.append(row)
        return pd.DataFrame(rows[::-1]).reset_index(drop=True)

    def get_state_at(self, checkpoint_id: str) -> dict:
        cfg = {"configurable": {"thread_id": self.config["configurable"]["thread_id"],
                                "checkpoint_id": checkpoint_id}}
        return self.graph.get_state(cfg).values


# ===========================================================================
# PROMPTS
# ===========================================================================
RECOMMEND_PROMPT = """You are <<persona>>.

INDUSTRY CONTEXT
<<industry>>

MODELLING GOAL
Use case: <<use_case>>
The features are for: <<model>>
Target: <<target>> (<<target_type>>)
Forecast horizon: <<horizon>> period(s) ahead
Grain: <<grain>>

GOOD FEATURES FOR THIS MODEL
<<guidance>>

DO NOT CREATE
<<avoid>>

LEAKAGE RULES
- Never use the current or future value of the target.
- <<lag_rule>>
- Build lags, rolling windows and adstock WITHIN each <<group>> after sorting by <<date>>.
- Only pandas and numpy are available.

USER INSTRUCTIONS AND ASSUMPTIONS
<<instructions>>

DATA PROFILE (this is everything you know about the data)
<<summary>>

TASK
1. Write 3 to 6 short observations about the data that matter for this model
   (skew, zero-heavy columns, flags, grain, seasonality, suspicious columns, missing values).
2. Propose 8 to 25 features. Each must be justified by the data profile, the model or a user assumption.

Return ONLY JSON in exactly this format:
{"observations": ["..."],
 "features": [{"name": "snake_case_name",
               "category": "lag | rolling | calendar | price | promotion | media | interaction | encoding | transform | other",
               "source_columns": ["column"],
               "how": "exact construction in one line",
               "reason": "why it helps this model",
               "supports": "the assumption or model need it serves"}]}"""

CODE_PROMPT = """You are a senior Python engineer implementing a feature plan for <<use_case>>.

Write ONE function feature_engineer(data_raw) that creates EXACTLY the features in the plan,
using the exact names given.

FEATURE PLAN
<<plan>>

DATA FACTS
- Date column: <<date>> (convert with pd.to_datetime before using it)
- Group columns (use as a list in groupby): <<group>>
- Target: <<target>> (never modify it)
- <<lag_rule>>
- Available columns: <<columns>>

RULES
- Use ONLY pandas and numpy; put the imports inside the function.
- df = data_raw.copy(); keep all original columns and the same number of rows.
- For time-based features sort by [group, date], compute with groupby(group) + shift / rolling / transform,
  then restore the original row order with df.sort_index() before returning.
- Leave NaN where no history exists. Do not fill or change the target.
- If a planned feature cannot be built from the available columns, skip it.
- Return the DataFrame.

Return only python code in a ```python block."""

FIX_PROMPT = """The feature_engineer function below raised an error. Fix it.
Keep the name feature_engineer(data_raw), keep imports inside it, use ONLY pandas and numpy,
keep the planned feature names: <<names>>
Date column: <<date>>. Group columns: <<group>>. Target: <<target>> (never modify it).

Code:
<<code>>

Error:
<<error>>

Available columns: <<columns>>

Return only the corrected python code in a ```python block."""

BUSINESS_PROMPT = """You are <<persona>> reviewing candidate features before they go into
<<model>>.
Industry: <<industry>>. Target: <<target>>. Horizon: <<horizon>> period(s) ahead.
User assumptions: <<instructions>>
Key business question for every feature: <<checks>>
Treat planned business levers (price, promotions, media spend, holidays, calendar, store or product
attributes) as known in advance.

CANDIDATE FEATURES (name: how it was built)
<<features>>

For EACH feature answer:
- relevance: 1-5 (5 = core driver for this model and these assumptions; 1 = no business meaning,
  for example random noise or an ID)
- leakage_risk: low | medium | high (high = uses current / future target or whole-period target statistics)
- available_at_prediction: true or false
- fits_model: true or false (is it appropriate for <<model_short>>?)
- reason: one sentence

Return ONLY a JSON list:
[{"feature": "...", "relevance": 4, "leakage_risk": "low", "available_at_prediction": true,
  "fits_model": true, "reason": "..."}]"""


def _bullets(items):
    return "\n".join(f"- {x}" for x in items)


def lag_rule(cfg: dict, horizon: int) -> str:
    if cfg["allow_target_lags"]:
        return (f"Target lags and rolling statistics are allowed ONLY with shift({horizon}) or more, so "
                f"they are known {horizon} period(s) before the target.")
    return ("Do NOT create any lag, rolling or other feature from the target column; this model must "
            "explain the target with drivers only.")


# ===========================================================================
# AGENT 1: FEATURE ENGINEERING AGENT
# ===========================================================================
class FEState(TypedDict, total=False):
    data_raw: dict
    user_instructions: str
    target_variable: str
    use_case: str
    industry: str
    horizon: int
    date_column: Optional[str]
    group_column: Optional[str]
    max_retries: int
    retry_count: int
    data_context: dict
    data_profile: List[dict]
    data_summary: str
    observations: List[str]
    feature_plan: List[dict]
    plan_parsed: bool
    recommended_steps: str
    feature_engineer_function: str
    feature_engineer_error: Optional[str]
    data_engineered: dict
    new_features: List[str]
    fe_status: str
    fe_report: str
    session_log: Annotated[List[dict], operator.add]


class FeatureEngineeringAgent(BaseAgent):
    name = "feature_engineering_agent"

    def _profile(self, state: FEState):
        print("    [FE] profile_data")
        start = time.time()
        df = state_to_df(state["data_raw"])
        ctx = build_data_context(df, state["target_variable"],
                                 state.get("date_column"), state.get("group_column"))
        prof = profile_columns(df, ctx)
        summary = summary_for_llm(df, ctx, prof)
        grain = (f"{ctx['group_column']} x {ctx['date_column']}" if ctx["group_column"] and ctx["date_column"]
                 else "no time grain")
        return {"data_context": ctx, "data_profile": prof, "data_summary": summary,
                "session_log": log_row(self.name, "profile_data",
                                       f"grain {grain}, {ctx['frequency'] or 'no date'}, "
                                       f"{ctx['n_groups'] or 0} groups", start)}

    def _recommend(self, state: FEState):
        print("    [FE] recommend_feature_engineering_steps")
        start = time.time()
        cfg = use_case_cfg(state.get("use_case"))
        ctx = state["data_context"]
        h = int(state.get("horizon") or 1)
        grain = (f"{ctx['group_column']} x {ctx['date_column']}" if ctx["group_column"] and ctx["date_column"]
                 else "one row per record")
        prompt = fill(RECOMMEND_PROMPT,
                      persona=cfg["persona"], industry=INDUSTRIES.get(state.get("industry"), INDUSTRIES["General"]),
                      use_case=cfg["label"], model=cfg["model"].replace("H period", f"{h} period"),
                      target=ctx["target"], target_type=cfg["target_type"], horizon=h, grain=grain,
                      guidance=_bullets(cfg["guidance"]), avoid=_bullets(cfg["avoid"]),
                      lag_rule=lag_rule(cfg, h), group=ctx["group_column"] or "group (none detected)",
                      date=ctx["date_column"] or "date (none detected)",
                      instructions=state.get("user_instructions") or "None given",
                      summary=state["data_summary"])
        answer = llm_text(self.model, prompt)
        observations, plan, parsed = [], [], True
        try:
            obj = extract_json(answer)
            observations = [str(o) for o in obj.get("observations", [])]
            for f in obj.get("features", []):
                if isinstance(f, dict) and f.get("name"):
                    plan.append({"name": str(f.get("name")), "category": str(f.get("category", "")),
                                 "source_columns": [str(c) for c in (f.get("source_columns") or [])],
                                 "how": str(f.get("how", "")), "reason": str(f.get("reason", "")),
                                 "supports": str(f.get("supports", ""))})
        except Exception:
            parsed = False
        if plan:
            steps = "\n".join(f"{i}. {p['name']}: {p['how']} (reason: {p['reason']})"
                              for i, p in enumerate(plan, 1))
        else:
            steps = answer
        return {"observations": observations, "feature_plan": plan, "plan_parsed": parsed,
                "recommended_steps": steps,
                "session_log": log_row(self.name, "recommend_feature_engineering_steps",
                                       f"{len(plan)} features planned, {len(observations)} observations"
                                       + ("" if parsed else " (plan not in JSON, used as text)"), start)}

    def _create_code(self, state: FEState):
        print("    [FE] create_feature_engineering_code")
        start = time.time()
        cfg = use_case_cfg(state.get("use_case"))
        ctx = state["data_context"]
        plan_text = (json.dumps([{k: p[k] for k in ("name", "source_columns", "how")}
                                 for p in state["feature_plan"]], indent=1)
                     if state.get("feature_plan") else state["recommended_steps"])
        code = extract_python_code(llm_text(self.model, fill(
            CODE_PROMPT, use_case=cfg["label"], plan=plan_text,
            date=ctx["date_column"] or "none", group=ctx.get("group_columns") or "none",
            target=ctx["target"], lag_rule=lag_rule(cfg, int(state.get("horizon") or 1)),
            columns=", ".join(state["data_raw"].keys()))))
        self._save_code(code, "feature_engineer.py")
        return {"feature_engineer_function": code,
                "session_log": log_row(self.name, "create_feature_engineering_code",
                                       f"{len(code.splitlines())} lines of code", start)}

    def _execute(self, state: FEState):
        print("    [FE] execute_feature_engineering_code")
        start = time.time()
        df = state_to_df(state["data_raw"])
        try:
            result = run_feature_code(state["feature_engineer_function"], df,
                                      state.get("target_variable", ""))
        except Exception:
            err = traceback.format_exc(limit=3)
            print("       ! Error:", err.strip().splitlines()[-1])
            return {"feature_engineer_error": err,
                    "session_log": log_row(self.name, "execute_feature_engineering_code",
                                           "error: " + err.strip().splitlines()[-1], start, "error")}
        result = result.replace([np.inf, -np.inf], np.nan)
        new = [str(c) for c in result.columns if c not in df.columns]
        planned = {p["name"] for p in state.get("feature_plan", [])}
        built = len(planned & set(new)) if planned else len(new)
        return {"data_engineered": df_to_state(result), "new_features": new,
                "feature_engineer_error": None,
                "session_log": log_row(self.name, "execute_feature_engineering_code",
                                       f"success: {len(new)} new features"
                                       + (f" ({built}/{len(planned)} of plan)" if planned else ""), start)}

    @staticmethod
    def _route_after_execute(state: FEState):
        if state.get("feature_engineer_error") and \
                state.get("retry_count", 0) < state.get("max_retries", 3):
            return "fix_code"
        return "explain_code"

    def _fix(self, state: FEState):
        print("    [FE] fix_feature_engineering_code")
        start = time.time()
        ctx = state["data_context"]
        code = extract_python_code(llm_text(self.model, fill(
            FIX_PROMPT, names=", ".join(p["name"] for p in state.get("feature_plan", [])) or "as in the code",
            date=ctx["date_column"] or "none", group=ctx.get("group_columns") or "none", target=ctx["target"],
            code=state["feature_engineer_function"], error=state["feature_engineer_error"],
            columns=", ".join(state["data_raw"].keys()))))
        self._save_code(code, "feature_engineer.py")
        attempt = state.get("retry_count", 0) + 1
        return {"feature_engineer_function": code, "retry_count": attempt,
                "session_log": log_row(self.name, "fix_feature_engineering_code",
                                       f"code fixed (attempt {attempt})", start)}

    def _report(self, state: FEState):
        print("    [FE] report_feature_engineering")
        start = time.time()
        if state.get("feature_engineer_error"):
            status = "FAILED"
            report = "Feature engineering FAILED after retries.\n" + state["feature_engineer_error"]
        else:
            status = "SUCCESS"
            new = state.get("new_features", [])
            report = f"Created {len(new)} new features: {new}"
        return {"fe_status": status, "fe_report": report,
                "session_log": log_row(self.name, "report_feature_engineering",
                                       f"{status} -> back to orchestrator", start)}

    def _build_graph(self):
        g = SafeStateGraph(FEState)
        g.add_node("profile_data", self._profile)
        g.add_node("recommend_feature_engineering_steps", self._recommend)
        g.add_node("create_feature_engineering_code", self._create_code)
        g.add_node("execute_feature_engineering_code", self._execute)
        g.add_node("fix_feature_engineering_code", self._fix)
        g.add_node("report_feature_engineering", self._report)

        g.add_edge(START, "profile_data")
        g.add_edge("profile_data", "recommend_feature_engineering_steps")
        g.add_edge("recommend_feature_engineering_steps", "create_feature_engineering_code")
        g.add_edge("create_feature_engineering_code", "execute_feature_engineering_code")
        g.add_conditional_edges("execute_feature_engineering_code", self._route_after_execute,
                                {"fix_code": "fix_feature_engineering_code",
                                 "explain_code": "report_feature_engineering"})
        g.add_edge("fix_feature_engineering_code", "execute_feature_engineering_code")
        g.add_edge("report_feature_engineering", END)
        return g.compile(checkpointer=self.checkpointer)


# ===========================================================================
# AGENT 2: FEATURE SELECTION AGENT
# ===========================================================================
class FSState(TypedDict, total=False):
    data_engineered: dict
    new_features: List[str]
    feature_plan: List[dict]
    data_context: dict
    user_instructions: str
    target_variable: str
    use_case: str
    industry: str
    horizon: int
    candidate_features: List[str]
    key_columns: List[str]
    statistical_review: List[dict]
    business_review: List[dict]
    feature_decisions: List[dict]
    selected_features: List[str]
    data_selected: dict
    fs_status: str
    fs_report: str
    session_log: Annotated[List[dict], operator.add]


def within_group_rho(df: pd.DataFrame, feat: str, target: str, groups: List[str]) -> Optional[float]:
    """Average Spearman rho computed separately inside each group (store / SKU)."""
    groups = [g for g in (groups or []) if g in df.columns and g != feat]
    if not groups:
        return None
    vals = []
    for _, g in df[[feat, target] + groups].dropna().groupby(groups):
        if len(g) >= 5 and g[feat].nunique() > 1 and g[target].nunique() > 1:
            r = g[feat].astype(float).corr(g[target].astype(float), method="spearman")
            if pd.notna(r):
                vals.append(float(r))
    return round(float(np.mean(vals)), 4) if vals else None


class FeatureSelectionAgent(BaseAgent):
    name = "feature_selection_agent"

    def _prepare(self, state: FSState):
        print("    [FS] prepare_candidates")
        start = time.time()
        df = state_to_df(state["data_engineered"])
        target = state.get("target_variable", "")
        new = [f for f in state.get("new_features", []) if f in df.columns]
        originals = [c for c in df.columns if c not in new and c != target]
        key_cols = [c for c in originals if not pd.api.types.is_numeric_dtype(df[c])]
        original_predictors = [c for c in originals if c not in key_cols]
        candidates = original_predictors + new
        return {"candidate_features": candidates, "key_columns": key_cols,
                "session_log": log_row(self.name, "prepare_candidates",
                                       f"{len(original_predictors)} original + {len(new)} new candidates; "
                                       f"keys kept as-is: {key_cols}", start)}

    def _statistical(self, state: FSState):
        print("    [FS] statistical_agent")
        start = time.time()
        df = state_to_df(state["data_engineered"])
        target = state.get("target_variable", "")
        group = (state.get("data_context") or {}).get("group_columns") or []
        feats = state.get("candidate_features", [])
        numeric = [f for f in feats if pd.api.types.is_numeric_dtype(df[f])]
        y = pd.to_numeric(df[target], errors="coerce")

        review = []
        for feat in feats:
            row = {"feature": feat,
                   "missing_pct": round(float(df[feat].isna().mean() * 100), 2),
                   "unique_values": int(df[feat].nunique()),
                   "rho_pooled": None, "rho_within": None, "collinear_with": None,
                   "S1": "PASS", "S2": "PASS", "S3": "n/a", "S4": "n/a",
                   "stats_keep": True, "stats_reason": "All statistical tests passed"}
            if row["unique_values"] <= 1:
                row["S1"] = "FAIL"
            if row["missing_pct"] > MAX_MISSING_PCT:
                row["S2"] = "FAIL"
            if feat in numeric and row["S1"] == "PASS":
                rho = df[feat].astype(float).corr(y, method="spearman")
                row["rho_pooled"] = None if pd.isna(rho) else round(float(rho), 4)
                row["rho_within"] = within_group_rho(df, feat, target, group)
                strength = max(abs(row["rho_pooled"] or 0), abs(row["rho_within"] or 0))
                row["S3"] = "PASS" if strength >= MIN_ABS_CORR else "FAIL"
                row["S4"] = "PASS"
            review.append(row)

        lookup = {r["feature"]: r for r in review}
        strength = {f: max(abs(lookup[f]["rho_pooled"] or 0), abs(lookup[f]["rho_within"] or 0))
                    for f in lookup}
        cands = [f for f in numeric
                 if all(lookup[f][t] in ("PASS", "n/a") for t in ("S1", "S2", "S3"))]
        if len(cands) > 1:
            corr = df[cands].astype(float).corr(method="spearman").abs()
            for i, a in enumerate(cands):
                for b in cands[i + 1:]:
                    if lookup[a]["S4"] == "FAIL" or lookup[b]["S4"] == "FAIL":
                        continue
                    if corr.loc[a, b] >= COLLINEAR_CORR:
                        weaker, stronger = (a, b) if strength[a] < strength[b] else (b, a)
                        lookup[weaker]["S4"] = "FAIL"
                        lookup[weaker]["collinear_with"] = f"{stronger} (|rho|={float(corr.loc[a, b]):.2f})"

        names = {t["id"]: t["test"] for t in stat_tests()}
        for r in review:
            fails = [f"{t} {names[t]}" for t in ("S1", "S2", "S3", "S4") if r[t] == "FAIL"]
            r["stats_keep"] = not fails
            if fails:
                r["stats_reason"] = "; ".join(fails)
                if r["S3"] == "FAIL":
                    r["stats_reason"] += f" (rho {r['rho_pooled']} overall, {r['rho_within']} within group)"
                if r["S4"] == "FAIL":
                    r["stats_reason"] += f" (duplicate of {r['collinear_with']})"
        passed = sum(r["stats_keep"] for r in review)
        return {"statistical_review": review,
                "session_log": log_row(self.name, "statistical_agent",
                                       f"4 tests on {len(review)} features, {passed} passed all", start)}

    def _business(self, state: FSState):
        print("    [FS] business_agent")
        start = time.time()
        feats = state.get("candidate_features", [])
        cfg = use_case_cfg(state.get("use_case"))
        if not feats:
            return {"business_review": [],
                    "session_log": log_row(self.name, "business_agent", "no features to review", start)}
        how = {p["name"]: p["how"] for p in state.get("feature_plan", [])}
        lines = [f"- {f}: {how.get(f, 'original column in the data' if f not in state.get('new_features', []) else 'engineered feature')}"
                 for f in feats]
        h = int(state.get("horizon") or 1)
        answer = llm_text(self.model, fill(
            BUSINESS_PROMPT, persona=cfg["persona"], model=cfg["model"].replace("H period", f"{h} period"),
            industry=state.get("industry", "General"), target=state.get("target_variable", ""), horizon=h,
            instructions=state.get("user_instructions") or "None given", checks=cfg["checks"],
            features="\n".join(lines), model_short=cfg["model_short"]))
        try:
            review = extract_json(answer)
            review = review if isinstance(review, list) else []
        except Exception:
            review = []
        reviewed = {str(r.get("feature")): r for r in review if isinstance(r, dict)}
        target = (state.get("target_variable") or "").lower()
        clean = []
        for feat in feats:
            r = reviewed.get(feat, {})
            try:
                relevance = int(r.get("relevance", MIN_BUSINESS_SCORE))
            except (TypeError, ValueError):
                relevance = MIN_BUSINESS_SCORE
            leak = str(r.get("leakage_risk", "low")).lower()
            avail = r.get("available_at_prediction", True)
            fits = r.get("fits_model", True)
            avail = str(avail).lower() not in ("false", "no", "0")
            fits = str(fits).lower() not in ("false", "no", "0")
            from_target = bool(target) and target in feat.lower()
            clean.append({
                "feature": feat, "relevance": relevance, "leakage_risk": leak,
                "available_at_prediction": avail, "fits_model": fits,
                "B1": "PASS" if relevance >= MIN_BUSINESS_SCORE else "FAIL",
                "B2": "FAIL" if leak == "high" else "PASS",
                "B3": "PASS" if avail else "FAIL",
                "B4": "PASS" if fits else "FAIL",
                "B5": "FAIL" if (from_target and not cfg["allow_target_lags"]) else "PASS",
                "business_reason": r.get("reason", "Not reviewed by the LLM - kept by default"),
            })
        ok = sum(all(c[t] == "PASS" for t in ("B1", "B2", "B3", "B4", "B5")) for c in clean)
        return {"business_review": clean,
                "session_log": log_row(self.name, "business_agent",
                                       f"5 tests on {len(clean)} features, {ok} passed all", start)}

    def _select(self, state: FSState):
        print("    [FS] select_features")
        start = time.time()
        stats = {r["feature"]: r for r in state.get("statistical_review", [])}
        business = {r["feature"]: r for r in state.get("business_review", [])}
        new = set(state.get("new_features", []))
        bnames = {t["id"]: t["test"] for t in business_tests(state.get("use_case", "general"))}
        decisions = []
        for feat in state.get("candidate_features", []):
            s, b = stats.get(feat, {}), business.get(feat, {})
            fails = []
            if s and not s.get("stats_keep", True):
                fails.append(s["stats_reason"])
            bf = [f"{t} {bnames[t]}" for t in ("B1", "B2", "B3", "B4", "B5") if b.get(t) == "FAIL"]
            if bf:
                fails.append("; ".join(bf))
            row = {"feature": feat, "source": "new" if feat in new else "original",
                   "decision": "DROP" if fails else "KEEP"}
            for t in ("S1", "S2", "S3", "S4"):
                row[t] = s.get(t, "n/a")
            for t in ("B1", "B2", "B3", "B4", "B5"):
                row[t] = b.get(t, "n/a")
            row.update({"rho_pooled": s.get("rho_pooled"), "rho_within": s.get("rho_within"),
                        "relevance": b.get("relevance"), "leakage_risk": b.get("leakage_risk"),
                        "reason": " | ".join(fails) if fails else "Passed all tests",
                        "business_reason": b.get("business_reason", "")})
            decisions.append(row)
        selected = [d["feature"] for d in decisions if d["decision"] == "KEEP"]
        dropped = [d["feature"] for d in decisions if d["decision"] == "DROP"]
        target = state.get("target_variable", "")
        keep_cols = list(dict.fromkeys(state.get("key_columns", []) + selected + [target]))
        df = state_to_df(state["data_engineered"])[keep_cols]
        return {"feature_decisions": decisions, "selected_features": selected,
                "data_selected": df_to_state(df),
                "session_log": log_row(self.name, "select_features",
                                       f"kept {len(selected)}, dropped {len(dropped)}", start)}

    def _report(self, state: FSState):
        print("    [FS] report_feature_selection")
        start = time.time()
        sel = state.get("selected_features", [])
        dropped = [d["feature"] for d in state.get("feature_decisions", []) if d["decision"] == "DROP"]
        report = f"Selected {len(sel)} features: {sel}\nDropped {len(dropped)}: {dropped}"
        return {"fs_status": "SUCCESS", "fs_report": report,
                "session_log": log_row(self.name, "report_feature_selection",
                                       "SUCCESS -> back to orchestrator", start)}

    def _build_graph(self):
        g = SafeStateGraph(FSState)
        g.add_node("prepare_candidates", self._prepare)
        g.add_node("statistical_agent", self._statistical)
        g.add_node("business_agent", self._business)
        g.add_node("select_features", self._select)
        g.add_node("report_feature_selection", self._report)

        g.add_edge(START, "prepare_candidates")
        g.add_edge("prepare_candidates", "statistical_agent")
        g.add_edge("prepare_candidates", "business_agent")
        g.add_edge(["statistical_agent", "business_agent"], "select_features")
        g.add_edge("select_features", "report_feature_selection")
        g.add_edge("report_feature_selection", END)
        return g.compile(checkpointer=self.checkpointer)


# ===========================================================================
# AGENT 0: ORCHESTRATOR (manager)
# ===========================================================================
class OrchestratorState(TypedDict, total=False):
    data_raw: dict
    user_instructions: str
    target_variable: str
    use_case: str
    industry: str
    horizon: int
    date_column: Optional[str]
    group_column: Optional[str]
    max_retries: int
    session_id: str
    session_started: str
    session_finished: str
    session_status: str
    next_agent: str
    agent_threads: dict
    fe_result: dict
    fs_result: dict
    final_output: dict
    session_log: Annotated[List[dict], operator.add]


TEAM = {
    "feature_engineering_agent": "Profiles the data, plans features for the use case, writes and runs the code",
    "feature_selection_agent": "Runs 4 statistical and 5 business tests on every feature, keeps those that pass",
}


class OrchestratorAgent(BaseAgent):
    """The manager. Knows the team, decides who works next, returns final output."""
    name = "orchestrator"

    def __init__(self, model, log: bool = False, log_path: Optional[str] = None):
        self.fe_agent = FeatureEngineeringAgent(model, log, log_path)
        self.fs_agent = FeatureSelectionAgent(model, log, log_path)
        super().__init__(model, log, log_path)

    @property
    def agents(self):
        return [self, self.fe_agent, self.fs_agent]

    def _orchestrator(self, state: OrchestratorState):
        start = time.time()
        fe, fs = state.get("fe_result"), state.get("fs_result")
        if fe is None:
            print(f"ORCHESTRATOR | session {state['session_id']} started")
            target = state.get("target_variable", "")
            if target and target not in (state.get("data_raw") or {}):
                raise ValueError(f"Target '{target}' not found in data columns")
            cfg = use_case_cfg(state.get("use_case"))
            nxt = "feature_engineering_agent"
            detail = f"brief: {cfg['label']} | {state.get('industry')} -> feature engineering"
            extra = {"session_status": "RUNNING"}
        elif fe.get("fe_status") != "SUCCESS":
            nxt, detail = "finish", "feature engineering failed -> stop"
            extra = self._finalise(state, "FAILED")
        elif fs is None:
            nxt = "feature_selection_agent"
            detail = f"{len(fe.get('new_features') or [])} features created -> feature selection"
            extra = {}
        else:
            nxt, detail = "finish", "selection done -> final output returned"
            extra = self._finalise(state, "SUCCESS")
        print(f"[ORCHESTRATOR] next -> {nxt}  ({detail})")
        out = {"next_agent": nxt,
               "session_log": log_row(self.name, "orchestrator", detail, start, "decision")}
        out.update(extra)
        return out

    def _finalise(self, state, status):
        fe, fs = state.get("fe_result") or {}, state.get("fs_result") or {}
        final = {"session_id": state.get("session_id"), "status": status,
                 "use_case": use_case_cfg(state.get("use_case"))["label"], "industry": state.get("industry"),
                 "fe_report": fe.get("fe_report"), "fs_report": fs.get("fs_report"),
                 "created_features": fe.get("new_features") or [],
                 "selected_features": fs.get("selected_features") or [],
                 "retries_used": fe.get("retry_count") or 0,
                 "feature_engineer_function": fe.get("feature_engineer_function"),
                 "data_selected": fs.get("data_selected")}
        print(f"ORCHESTRATOR | session {state['session_id']} {status}")
        return {"final_output": final, "session_status": status,
                "session_finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    @staticmethod
    def _route(state: OrchestratorState):
        return state["next_agent"]

    def _call_fe(self, state: OrchestratorState):
        print("  -> FEATURE ENGINEERING AGENT")
        thread = f"{state['session_id']}-fe"
        res = self.fe_agent.run({
            "data_raw": state["data_raw"], "user_instructions": state.get("user_instructions", ""),
            "target_variable": state.get("target_variable", ""), "use_case": state.get("use_case"),
            "industry": state.get("industry"), "horizon": state.get("horizon", 1),
            "date_column": state.get("date_column"), "group_column": state.get("group_column"),
            "max_retries": state.get("max_retries", 3), "retry_count": 0, "session_log": [],
        }, thread)
        threads = dict(state.get("agent_threads") or {})
        threads["feature_engineering_agent"] = thread
        fe_result = {k: res.get(k) for k in (
            "fe_status", "fe_report", "new_features", "recommended_steps", "feature_plan",
            "observations", "data_context", "feature_engineer_function", "retry_count", "data_engineered")}
        return {"fe_result": fe_result, "agent_threads": threads,
                "session_log": res.get("session_log", [])}

    def _call_fs(self, state: OrchestratorState):
        print("  -> FEATURE SELECTION AGENT")
        thread = f"{state['session_id']}-fs"
        fe = state["fe_result"]
        res = self.fs_agent.run({
            "data_engineered": fe["data_engineered"], "new_features": fe.get("new_features") or [],
            "feature_plan": fe.get("feature_plan") or [], "data_context": fe.get("data_context") or {},
            "user_instructions": state.get("user_instructions", ""),
            "target_variable": state.get("target_variable", ""), "use_case": state.get("use_case"),
            "industry": state.get("industry"), "horizon": state.get("horizon", 1), "session_log": [],
        }, thread)
        threads = dict(state.get("agent_threads") or {})
        threads["feature_selection_agent"] = thread
        fs_result = {k: res.get(k) for k in (
            "fs_status", "fs_report", "selected_features", "feature_decisions",
            "statistical_review", "business_review", "data_selected")}
        return {"fs_result": fs_result, "agent_threads": threads,
                "session_log": res.get("session_log", [])}

    def _build_graph(self):
        g = SafeStateGraph(OrchestratorState)
        g.add_node("orchestrator", self._orchestrator)
        g.add_node("feature_engineering_agent", self._call_fe)
        g.add_node("feature_selection_agent", self._call_fs)
        g.add_edge(START, "orchestrator")
        g.add_conditional_edges("orchestrator", self._route,
                                {"feature_engineering_agent": "feature_engineering_agent",
                                 "feature_selection_agent": "feature_selection_agent",
                                 "finish": END})
        g.add_edge("feature_engineering_agent", "orchestrator")
        g.add_edge("feature_selection_agent", "orchestrator")
        return g.compile(checkpointer=self.checkpointer)

    # =====================================================================
    # PUBLIC API
    # =====================================================================
    def invoke_agent(self, data_raw: pd.DataFrame, user_instructions: str = "",
                     target_variable: str = "", max_retries: int = 3,
                     use_case: str = "demand_forecasting", industry: str = "Retail",
                     horizon: int = 1, date_column: Optional[str] = None,
                     group_column: Optional[str] = None) -> dict:
        session_id = uuid.uuid4().hex[:8]
        self.run({
            "data_raw": df_to_state(data_raw), "user_instructions": user_instructions,
            "target_variable": target_variable, "use_case": use_case if use_case in USE_CASES else "general",
            "industry": industry if industry in INDUSTRIES else "General", "horizon": int(horizon),
            "date_column": date_column, "group_column": group_column, "max_retries": max_retries,
            "session_id": session_id, "session_started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "agent_threads": {"orchestrator": session_id}, "session_log": [],
        }, session_id)
        final = self.response.get("final_output", {})
        print(str(final.get("fe_report")) + "\n" + str(final.get("fs_report")))
        return final

    # ---- session / history ----
    def get_session_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.response.get("session_log", []))

    def get_all_state_history(self, show_values: bool = False) -> pd.DataFrame:
        frames = [a.get_state_history(show_values) for a in self.agents if a.config]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_final_output(self):
        return self.response.get("final_output")

    def _fe(self, key, default=None):
        return (self.fe_agent.response or {}).get(key, default)

    def _fs(self, key, default=None):
        return (self.fs_agent.response or {}).get(key, default)

    # ---- brief and reasoning ----
    def get_brief(self) -> dict:
        r = self.response or {}
        cfg = use_case_cfg(r.get("use_case"))
        return {"industry": r.get("industry"), "use_case": cfg["label"], "persona": cfg["persona"],
                "model": cfg["model"].replace("H period", f"{r.get('horizon', 1)} period"),
                "horizon": r.get("horizon", 1), "allow_target_lags": cfg["allow_target_lags"],
                "guidance": cfg["guidance"], "avoid": cfg["avoid"],
                "instructions": r.get("user_instructions", "")}

    def get_data_context(self) -> dict:
        return self._fe("data_context", {}) or {}

    def get_data_profile(self) -> pd.DataFrame:
        return pd.DataFrame(self._fe("data_profile", []))

    def get_data_summary(self) -> str:
        return self._fe("data_summary", "") or ""

    def get_observations(self) -> List[str]:
        return self._fe("observations", []) or []

    def get_feature_plan(self) -> pd.DataFrame:
        plan = pd.DataFrame(self._fe("feature_plan", []))
        if plan.empty:
            return plan
        created = set(self._fe("new_features", []) or [])
        plan.insert(1, "created", plan["name"].isin(created).map({True: "yes", False: "no"}))
        plan["source_columns"] = plan["source_columns"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        return plan

    def get_recommended_feature_engineering_steps(self):
        return self._fe("recommended_steps")

    def get_feature_engineer_function(self):
        return self._fe("feature_engineer_function")

    def get_data_engineered(self) -> pd.DataFrame:
        return state_to_df(self._fe("data_engineered"))

    # ---- selection ----
    def get_stat_tests(self) -> pd.DataFrame:
        return pd.DataFrame(stat_tests())

    def get_business_tests(self) -> pd.DataFrame:
        return pd.DataFrame(business_tests((self.response or {}).get("use_case", "general")))

    def get_statistical_review(self) -> pd.DataFrame:
        return pd.DataFrame(self._fs("statistical_review", []))

    def get_business_review(self) -> pd.DataFrame:
        return pd.DataFrame(self._fs("business_review", []))

    def get_feature_decisions(self) -> pd.DataFrame:
        return pd.DataFrame(self._fs("feature_decisions", []))

    def get_data_selected(self) -> pd.DataFrame:
        return state_to_df(self._fs("data_selected"))
