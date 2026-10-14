"""
Streamlit app for the multi-agent feature engineering system.
Version 4 - use-case personas (CPG / Retail), agent reasoning and test catalogue in the Flow tab.

Run from this folder:   bash run_app.sh      (or: streamlit run app.py)
"""
import contextlib
import html
import io
import os
import sys
import threading
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import find_dotenv, load_dotenv

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import multi_agent_feature_engineering as mafe  # noqa: E402
from multi_agent_feature_engineering import (INDUSTRIES, USE_CASES, OrchestratorAgent,  # noqa: E402
                                             describe_value, state_to_df)

APP_VERSION = "4.0"
load_dotenv(os.path.join(APP_DIR, ".env"), override=True)
load_dotenv(find_dotenv(usecwd=True), override=False)
st.set_page_config(page_title="Feature Engineering Agents", page_icon="🟣", layout="wide")

# ===========================================================================
# ACCENTURE PURPLE THEME
# ===========================================================================
PURPLE, DARK_PURPLE, DEEP_PURPLE = "#A100FF", "#7500C0", "#460073"
LIGHT_PURPLE, PALE_PURPLE, BLACK, GREY = "#E6D7FF", "#F7F1FF", "#000000", "#8A8A8A"
AGENT_FILL, AGENT_FONT = DEEP_PURPLE, "white"
TOOL_FILL, TOOL_FONT = LIGHT_PURPLE, DEEP_PURPLE

st.markdown(f"""
<style>
.acn-banner {{background: linear-gradient(90deg,{DEEP_PURPLE} 0%,{DARK_PURPLE} 55%,{PURPLE} 100%);
  color:#fff; padding:18px 26px; border-radius:10px; margin-bottom:14px;}}
.acn-banner h1 {{color:#fff !important; font-size:1.7rem; margin:0; font-weight:700;}}
.acn-banner p {{color:#EBDDFF; margin:4px 0 0 0; font-size:.95rem;}}
.acn-ver {{float:right; background:rgba(255,255,255,.18); padding:2px 10px; border-radius:10px; font-size:.8rem;}}
.stButton > button, .stDownloadButton > button {{border-color:{PURPLE} !important; color:{DEEP_PURPLE} !important;}}
.stButton > button:hover, .stDownloadButton > button:hover {{background:{PALE_PURPLE} !important;}}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"],
button[data-testid="baseButton-primary"] {{background:{PURPLE} !important; border-color:{PURPLE} !important; color:#fff !important;}}
button[data-baseweb="tab"][aria-selected="true"] p {{color:{PURPLE} !important; font-weight:700;}}
div[data-baseweb="tab-highlight"] {{background-color:{PURPLE} !important;}}
div[data-baseweb="slider"] div[role="slider"] {{background-color:{PURPLE} !important; box-shadow:none !important;}}
div[data-testid="stMetric"] {{background:{PALE_PURPLE}; border-left:5px solid {PURPLE}; padding:10px 14px; border-radius:6px;}}
div[data-testid="stMetricValue"] {{color:{DEEP_PURPLE};}}
section[data-testid="stSidebar"] h2 {{color:{DEEP_PURPLE};}}
.chip {{display:inline-block; padding:3px 10px; border-radius:12px; font-size:.8rem; margin:0 8px 6px 0; font-weight:600;}}
.step-row {{display:flex; align-items:center; gap:10px; padding:6px 8px; border-bottom:1px solid #EEE6F7; font-size:.9rem;}}
.step-no {{color:{GREY}; width:28px; text-align:right;}}
.step-node {{font-family:monospace; color:{DEEP_PURPLE}; min-width:300px;}}
.step-detail {{flex:1; color:#333;}}
.step-time {{color:{GREY}; width:75px; text-align:right;}}
.crumb {{font-size:1rem; color:{GREY}; margin:4px 0 8px 0;}}
.crumb b {{color:{DEEP_PURPLE};}}
.card {{border:1px solid #E3D8F2; border-left:5px solid {PURPLE}; border-radius:6px; padding:10px 14px;
  background:{PALE_PURPLE}; margin-bottom:10px;}}
.card h4 {{color:{DEEP_PURPLE}; margin:0 0 6px 0; font-size:1rem;}}
.obs {{background:#fff; border:1px solid #E3D8F2; border-radius:6px; padding:8px 12px; margin-bottom:6px;}}
</style>
""", unsafe_allow_html=True)

MODELS = ["openai/gpt-oss-120b"]
UC_KEYS = list(USE_CASES)
UC_LABELS = {k: v["label"] for k, v in USE_CASES.items()}
AGENT_TITLES = {"orchestrator": "Orchestrator", "feature_engineering_agent": "Feature Engineering Agent",
                "feature_selection_agent": "Feature Selection Agent"}
NODE_ROLES = {
    "orchestrator": "routes work between agents",
    "feature_engineering_agent": "creates features",
    "feature_selection_agent": "selects features",
    "profile_data": "Python - detects grain & column roles",
    "recommend_feature_engineering_steps": "LLM - persona plans features",
    "create_feature_engineering_code": "LLM - writes pandas code",
    "execute_feature_engineering_code": "Python - runs the code",
    "fix_feature_engineering_code": "LLM - fixes failed code",
    "report_feature_engineering": "Python - summarises",
    "prepare_candidates": "Python - lists candidates",
    "statistical_agent": "pandas - tests S1 to S4",
    "business_agent": "LLM persona - tests B1 to B5",
    "select_features": "Python - KEEP if all pass",
    "report_feature_selection": "Python - summarises",
}


def is_agent(node: str) -> bool:
    return node == "orchestrator" or node.endswith("_agent")


# ===========================================================================
# HELPERS
# ===========================================================================
def fmt_secs(s) -> str:
    if s is None:
        return "-"
    s = float(s)
    return f"{s:.1f} s" if s < 60 else f"{int(s // 60)} min {int(s % 60)} s"


def read_table(source, name: str) -> pd.DataFrame:
    name = name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(source)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(source)
    if name.endswith(".parquet"):
        return pd.read_parquet(source)
    raise ValueError("Supported files: .csv, .xlsx, .xls, .parquet")


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, api_key: str):
    from langchain_groq import ChatGroq
    return ChatGroq(model=model_name, temperature=0, api_key=api_key)


class LiveLog(io.TextIOBase):
    """Captures print() from the agents and shows it live while they run."""

    def __init__(self, placeholder):
        self.placeholder, self.lines = placeholder, []
        self.owner = threading.current_thread()

    def write(self, text):
        for line in text.splitlines():
            if line.strip():
                self.lines.append(line)
        if threading.current_thread() is self.owner:
            self.placeholder.code("\n".join(self.lines), language="text")
        return len(text)


def node_stats(log: pd.DataFrame):
    runs, secs = defaultdict(Counter), defaultdict(lambda: defaultdict(float))
    for _, r in log.iterrows():
        runs[r["agent"]][r["node"]] += 1
        secs[r["agent"]][r["node"]] += float(r["duration_s"])
        if r["agent"] != "orchestrator":
            runs["orchestrator"][r["agent"]] = 1
            secs["orchestrator"][r["agent"]] += float(r["duration_s"])
    return runs, secs


def graph_to_dot(agent, runs: Counter, secs: dict) -> str:
    g = agent.graph.get_graph()
    out = ['digraph G {', 'rankdir=TB; bgcolor="transparent"; nodesep=0.45; ranksep=0.5;',
           'node [fontname="Helvetica", fontsize=11, margin="0.2,0.08"];',
           f'edge [fontname="Helvetica", fontsize=9, color="{GREY}", fontcolor="{DARK_PURPLE}"];']
    for node in g.nodes:
        if node in ("__start__", "__end__"):
            out.append(f'"{node}" [label="{node.strip("_")}", shape=oval, style=filled, '
                       f'fillcolor="{BLACK}", fontcolor="white", color="{BLACK}"];')
            continue
        a = is_agent(node)
        fill, font = (AGENT_FILL, AGENT_FONT) if a else (TOOL_FILL, TOOL_FONT)
        n = runs.get(node, 0)
        drill = node in ("feature_engineering_agent", "feature_selection_agent")
        info = f"ran {n}x · {secs.get(node, 0):.1f}s" if n else "not run"
        label = f"{'AGENT' if a else 'TOOL'}\\n{node}{'  ▸' if drill else ''}\\n{NODE_ROLES.get(node, '')}\\n{info}"
        style = ("rounded," if a else "") + "filled" + (",bold" if n else ",dashed")
        out.append(f'"{node}" [label="{label}", shape=box, style="{style}", fillcolor="{fill}", '
                   f'fontcolor="{font}", color="{PURPLE if n else GREY}", penwidth={3 if n else 1}'
                   f'{", peripheries=2" if drill else ""}];')
    for e in g.edges:
        style = "dashed" if getattr(e, "conditional", False) else "solid"
        label = f', label=" {e.data}"' if getattr(e, "data", None) else ""
        out.append(f'"{e.source}" -> "{e.target}" [style={style}{label}];')
    out.append("}")
    return "\n".join(out)


def legend():
    st.markdown(
        f'<span class="chip" style="background:{AGENT_FILL};color:{AGENT_FONT}">AGENT · orchestrator / *_agent</span>'
        f'<span class="chip" style="background:{TOOL_FILL};color:{TOOL_FONT}">TOOL · all other steps</span>'
        f'<span class="chip" style="background:{BLACK};color:white">start / end</span>'
        f'<span class="chip" style="border:2px solid {PURPLE};color:{DEEP_PURPLE}">thick border = ran</span>'
        f'<span class="chip" style="border:2px dashed {GREY};color:{GREY}">dashed = not run</span>',
        unsafe_allow_html=True)


def steps_timeline(log: pd.DataFrame):
    if log.empty:
        st.info("No steps recorded.")
        return
    icons = {"ok": "✅", "error": "⚠️", "decision": "🔀"}
    rows = []
    for i, r in enumerate(log.itertuples(), 1):
        a = is_agent(r.node)
        rows.append(
            f'<div class="step-row"><span class="step-no">{i}</span><span>{icons.get(r.status, "•")}</span>'
            f'<span class="chip" style="background:{AGENT_FILL if a else TOOL_FILL};'
            f'color:{AGENT_FONT if a else TOOL_FONT}">{"AGENT" if a else "TOOL"}</span>'
            f'<span class="step-node">{html.escape(AGENT_TITLES.get(r.agent, r.agent))} › {html.escape(r.node)}</span>'
            f'<span class="step-detail">{html.escape(str(r.detail))}</span>'
            f'<span class="step-time">{fmt_secs(r.duration_s)}</span></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)


def pass_fail_style(df: pd.DataFrame, cols):
    def colour(v):
        if v == "PASS":
            return "background-color:#E3F1EA; color:#2E7D5B; font-weight:600"
        if v == "FAIL":
            return "background-color:#F6E3DE; color:#B5452B; font-weight:600"
        if v == "KEEP":
            return f"background-color:{PALE_PURPLE}; color:{DEEP_PURPLE}; font-weight:700"
        if v == "DROP":
            return "background-color:#F2F2F2; color:#777777; font-weight:700"
        return ""
    cols = [c for c in cols if c in df.columns]
    styler = df.style
    return styler.map(colour, subset=cols) if hasattr(styler, "map") else styler.applymap(colour, subset=cols)


def show_state_value(key, value):
    if isinstance(value, dict) and value and all(isinstance(v, list) for v in value.values()):
        df = state_to_df(value)
        st.caption(f"{key}: DataFrame {df.shape[0]} rows x {df.shape[1]} cols (first 20 rows)")
        st.dataframe(df.head(20), use_container_width=True)
    elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        st.caption(f"{key}: {len(value)} rows")
        st.dataframe(pd.DataFrame(value), use_container_width=True)
    elif isinstance(value, str) and ("def " in value or "\n" in value):
        st.caption(key)
        st.code(value, language="python" if "def " in value else "text")
    else:
        st.write(f"**{key}:**", value)


def data_insights(df: pd.DataFrame, target: str):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", f"{df.isna().mean().mean() * 100:.1f}%")
    c4.metric("Duplicate rows", int(df.duplicated().sum()))
    profile = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing_%": (df.isna().mean() * 100).round(2),
                            "unique": df.nunique()})
    num = df.select_dtypes(include=np.number)
    if not num.empty:
        profile["mean"] = num.mean().round(3)
        profile["skew"] = num.skew().round(3)
    left, right = st.columns(2)
    with left:
        st.markdown("**Column profile**")
        st.dataframe(profile, use_container_width=True)
    if not target or target not in num.columns:
        return
    y = df[target]
    with right:
        st.markdown(f"**Spearman correlation with `{target}`**")
        corr = num.drop(columns=[target]).corrwith(y, method="spearman").dropna().sort_values()
        if not corr.empty:
            st.bar_chart(corr.rename("rho"), color=PURPLE)
    binary = [c for c in num.columns if c != target and set(num[c].dropna().unique()) <= {0, 1}]
    if binary:
        st.markdown(f"**Average `{target}` lift for 0/1 columns** (avg when 1 minus avg when 0)")
        st.dataframe(pd.DataFrame([{
            "column": c, f"{target} when 1": round(y[df[c] == 1].mean(), 2),
            f"{target} when 0": round(y[df[c] == 0].mean(), 2),
            "lift": round(y[df[c] == 1].mean() - y[df[c] == 0].mean(), 2)} for c in binary]
        ).sort_values("lift", ascending=False), use_container_width=True, hide_index=True)


INSIGHT_PROMPT = """You are {persona}.
Write a short, plain-English insight summary (max 8 bullet points) for a business audience.

Use case: {use_case}. Industry: {industry}. Target: {target}.
Assumptions: {instructions}

Feature decisions (KEEP/DROP with reasons):
{decisions}

Cover: which drivers matter most for this model, what was removed and why, any leakage risk,
and the next step before modelling. Do not invent numbers that are not given."""

# ===========================================================================
# SESSION STATE + SIDEBAR
# ===========================================================================
st.session_state.setdefault("runs", [])
st.session_state.setdefault("flow_view", "orchestrator")

with st.sidebar:
    st.caption(f"App version {APP_VERSION}")
    st.header("LLM")
    env_key = os.getenv("GROQ_API_KEY", "")
    if not env_key:
        try:
            env_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            env_key = ""
    if env_key:
        st.success("Groq key loaded")
        api_key = env_key
    else:
        api_key = st.text_input("Groq API key", type="password")
    model_name = st.selectbox("Model", MODELS)

    st.header("Data")
    source = st.radio("Load data from", ["Upload a file", "Browse a local folder"])
    df, file_name = None, None
    if source == "Upload a file":
        up = st.file_uploader("CSV, Excel or Parquet", type=["csv", "xlsx", "xls", "parquet"])
        if up is not None:
            file_name, df = up.name, read_table(up, up.name)
    else:
        folder = st.text_input("Folder path", value=os.path.join(APP_DIR, "data"))
        if os.path.isdir(folder):
            files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".csv", ".xlsx", ".xls", ".parquet")))
            if files:
                file_name = st.selectbox("File", files)
                df = read_table(os.path.join(folder, file_name), file_name)
            else:
                st.warning("No CSV / Excel / Parquet files in this folder.")
        else:
            st.error("Folder not found.")

    st.header("Selection thresholds")
    mafe.MAX_MISSING_PCT = st.slider("S2 max missing %", 0, 100, 50)
    mafe.MIN_ABS_CORR = st.slider("S3 min |Spearman rho|", 0.0, 0.5, 0.05, 0.01)
    mafe.COLLINEAR_CORR = st.slider("S4 duplicate cut-off |rho|", 0.5, 1.0, 0.95, 0.01)
    mafe.MIN_BUSINESS_SCORE = st.slider("B1 min business relevance (1-5)", 1, 5, 3)
    max_retries = st.number_input("Max code-fix retries", 0, 5, 3)

    current = None
    if st.session_state.runs:
        st.header("Runs")
        labels = [f"{r['id']} | {r['file']} | {r['status']}" for r in st.session_state.runs]
        chosen = st.selectbox("Show run", labels, index=len(labels) - 1)
        current = st.session_state.runs[labels.index(chosen)]

st.markdown(f"""
<div class="acn-banner"><span class="acn-ver">v{APP_VERSION}</span>
  <h1>&gt; Multi-Agent Feature Engineering</h1>
  <p>Use-case aware agents for CPG and retail data: Orchestrator → Feature Engineering → Feature Selection</p>
</div>""", unsafe_allow_html=True)

tab_data, tab_run, tab_flow, tab_hist, tab_res = st.tabs(
    ["📊 Data & insights", "▶️ Run agent", "🔀 Flow & reasoning", "🕓 State history", "✅ Results"])

# ---------------- Data & insights ----------------
with tab_data:
    if df is None:
        st.info("Load a file from the sidebar to start.")
    else:
        st.subheader(file_name)
        st.dataframe(df.head(50), use_container_width=True)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            data_insights(df, st.selectbox("Target for insights", num_cols, index=len(num_cols) - 1,
                                           key="insight_target"))

# ---------------- Run agent ----------------
with tab_run:
    if df is None:
        st.info("Load a file from the sidebar first.")
    else:
        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        st.markdown("#### 1. Brief the agents")
        b1, b2, b3 = st.columns(3)
        industry = b1.selectbox("Industry", list(INDUSTRIES), index=1)
        use_case = b2.selectbox("Modelling use case", UC_KEYS, format_func=lambda k: UC_LABELS[k])
        horizon = b3.number_input("Forecast horizon (periods ahead)", 1, 52, 1,
                                  help="Target lags must be shifted by at least this many periods")
        cfg = USE_CASES[use_case]
        with st.expander(f"Persona and rules the agents will use: {cfg['label']}", expanded=False):
            st.markdown(f"**Persona:** {cfg['persona']}  \n**Model:** "
                        f"{cfg['model'].replace('H period', f'{horizon} period')}  \n"
                        f"**Target lags allowed:** {'yes, shifted by the horizon' if cfg['allow_target_lags'] else 'no'}")
            g1, g2 = st.columns(2)
            g1.markdown("**Good features**\n" + "\n".join(f"- {x}" for x in cfg["guidance"]))
            g2.markdown("**Avoid**\n" + "\n".join(f"- {x}" for x in cfg["avoid"]))
            st.caption(f"Industry context: {INDUSTRIES[industry]}")

        st.markdown("#### 2. Data roles")
        r1, r2, r3 = st.columns(3)
        default_target = num_cols[-1] if num_cols else cols[-1]
        target = r1.selectbox("Target variable", cols, index=cols.index(default_target))
        date_pick = r2.selectbox("Date column", ["(auto-detect)"] + cols)
        group_pick = r3.selectbox("Group column (store / SKU)", ["(auto-detect)", "(none)"] + cols)
        instructions = st.text_area(
            "3. Business assumptions", height=90,
            value="TV has carry-over effects. Promotions lift sales." if industry != "General" else "")
        if not api_key:
            st.warning("Add your Groq API key in the sidebar to run the agents.")

        if st.button("🚀 Run agents", type="primary", disabled=not api_key):
            orch = OrchestratorAgent(model=get_llm(model_name, api_key), log=True)
            run_status, error = "ERROR", None
            with st.status("Agents working...", expanded=True) as status:
                placeholder = st.empty()
                live = LiveLog(placeholder)
                t0 = time.perf_counter()
                try:
                    with contextlib.redirect_stdout(live):
                        final = orch.invoke_agent(
                            df, instructions, target, int(max_retries), use_case=use_case, industry=industry,
                            horizon=int(horizon),
                            date_column=None if date_pick == "(auto-detect)" else date_pick,
                            group_column={"(auto-detect)": None, "(none)": "__none__"}.get(group_pick, group_pick))
                    run_status = final.get("status", "UNKNOWN")
                except Exception as exc:
                    error = repr(exc)
                elapsed = time.perf_counter() - t0
                status.update(label=f"Agents finished: {run_status} in {fmt_secs(elapsed)}",
                              state="complete" if run_status == "SUCCESS" else "error", expanded=True)
            if orch.response:
                st.session_state.runs.append({
                    "id": orch.response.get("session_id"), "file": file_name, "target": target,
                    "instructions": instructions, "model": model_name, "use_case": use_case,
                    "industry": industry, "status": run_status, "elapsed_s": elapsed,
                    "console": list(live.lines), "error": error, "orch": orch})
                st.session_state.flow_view = "orchestrator"
                st.rerun()
            else:
                st.error(f"Run failed before any step finished: {error}")

    if current is not None:
        st.markdown("---")
        with st.status(f"{'✅' if current['status'] == 'SUCCESS' else '⚠️'} Agents finished: {current['status']} · "
                       f"session {current['id']} · {UC_LABELS[current['use_case']]} · {current['industry']} · "
                       f"total execution time {fmt_secs(current['elapsed_s'])}",
                       state="complete" if current["status"] == "SUCCESS" else "error", expanded=True):
            st.code("\n".join(current["console"]), language="text")
        if current.get("error"):
            st.error(current["error"])
        st.markdown("**Agent working steps**")
        steps_timeline(current["orch"].get_session_log())

if current is None:
    for t in (tab_flow, tab_hist, tab_res):
        with t:
            st.info("Run the agents to see this view.")
    st.stop()

orch: OrchestratorAgent = current["orch"]
log = orch.get_session_log()
runs, secs = node_stats(log)
brief = orch.get_brief()
ctx = orch.get_data_context()

# ---------------- Flow & reasoning ----------------
with tab_flow:
    view = st.session_state.flow_view
    if view == "orchestrator":
        st.markdown('<div class="crumb"><b>Orchestrator</b> (open an agent to see its flow, reasoning and tests)</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", current["status"])
        c2.metric("Use case", brief["use_case"].split(" (")[0])
        c3.metric("Total execution time", fmt_secs(current["elapsed_s"]))
        c4.metric("Steps", len(log))
        st.markdown(f'<div class="card"><h4>Brief given to the team</h4>'
                    f'<b>Industry:</b> {html.escape(str(brief["industry"]))} · <b>Persona:</b> {html.escape(brief["persona"])}<br>'
                    f'<b>Model:</b> {html.escape(brief["model"])}<br>'
                    f'<b>Assumptions:</b> {html.escape(brief["instructions"] or "none")}</div>', unsafe_allow_html=True)
        legend()
        left, right = st.columns([1.4, 1])
        with left:
            st.graphviz_chart(graph_to_dot(orch, runs["orchestrator"], secs["orchestrator"]), use_container_width=True)
        with right:
            for key, desc in mafe.TEAM.items():
                ran = key in runs["orchestrator"]
                st.markdown(f"**{AGENT_TITLES[key]}**  \n{desc}")
                st.caption(f"{sum(runs[key].values())} steps · {fmt_secs(sum(secs[key].values()))}" if ran else "Not run")
                if st.button(f"Open {AGENT_TITLES[key]} ▸", key=f"open_{key}", type="primary",
                             disabled=not ran, use_container_width=True):
                    st.session_state.flow_view = key
                    st.rerun()
                st.markdown("---")
    else:
        agent = orch.fe_agent if view == "feature_engineering_agent" else orch.fs_agent
        st.markdown(f'<div class="crumb">Orchestrator › <b>{AGENT_TITLES[view]}</b></div>', unsafe_allow_html=True)
        if st.button("◂ Back to Orchestrator"):
            st.session_state.flow_view = "orchestrator"
            st.rerun()
        agent_log = log[log["agent"] == view]
        c1, c2, c3 = st.columns(3)
        c1.metric("Agent execution time", fmt_secs(agent_log["duration_s"].sum()))
        c2.metric("Steps run", len(agent_log))
        c3.metric("Errors", int((agent_log["status"] == "error").sum()))

        if view == "feature_engineering_agent":
            t_flow, t_reason, t_saw, t_brief, t_code = st.tabs(
                ["Flow", "💡 Reasoning & feature plan", "🔎 What the agent saw", "🧭 Brief & persona", "Code"])
        else:
            t_flow, t_tests, t_matrix, t_fail = st.tabs(
                ["Flow", "🧪 Tests performed", "📋 Test results per feature", "📉 Failures by test"])

        with t_flow:
            legend()
            left, right = st.columns([1.4, 1])
            with left:
                st.graphviz_chart(graph_to_dot(agent, runs[view], secs[view]), use_container_width=True)
            with right:
                nodes = [n for n in agent.graph.get_graph().nodes if not n.startswith("__")]
                st.dataframe(pd.DataFrame([{"node": n, "type": "agent" if is_agent(n) else "tool",
                                            "role": NODE_ROLES.get(n, ""), "runs": runs[view].get(n, 0),
                                            "seconds": round(secs[view].get(n, 0.0), 2)} for n in nodes]),
                             use_container_width=True, hide_index=True)
            st.markdown("**Steps taken by this agent**")
            steps_timeline(agent_log)

        if view == "feature_engineering_agent":
            with t_reason:
                obs = orch.get_observations()
                st.markdown("##### What the agent noticed in the data")
                if obs:
                    st.markdown("".join(f'<div class="obs">🔍 {html.escape(o)}</div>' for o in obs),
                                unsafe_allow_html=True)
                else:
                    st.info("The LLM did not return observations in JSON for this run.")
                st.markdown("##### Feature plan: what it decided to build and why")
                plan = orch.get_feature_plan()
                if plan.empty:
                    st.warning("Plan was returned as free text:")
                    st.markdown(orch.get_recommended_feature_engineering_steps() or "-")
                else:
                    cats = plan["category"].value_counts()
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Features planned", len(plan))
                    p2.metric("Built by the code", int((plan["created"] == "yes").sum()))
                    p3.metric("Categories", len(cats))
                    st.dataframe(plan[["name", "created", "category", "source_columns", "how", "reason", "supports"]],
                                 use_container_width=True, hide_index=True)
                    st.bar_chart(cats, color=PURPLE)

            with t_saw:
                st.markdown("##### Data context detected by profile_data")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Grain", f"{ctx.get('group_column') or '-'} x {ctx.get('date_column') or '-'}")
                d2.metric("Frequency", ctx.get("frequency") or "-")
                d3.metric("Groups", ctx.get("n_groups") or "-")
                d4.metric("Periods", ctx.get("periods") or "-")
                st.caption(f"Date range {ctx.get('date_min')} to {ctx.get('date_max')} · target "
                           f"{ctx.get('target')} ({ctx.get('target_type')}) · one row per grain: {ctx.get('grain_unique')}")
                st.markdown("##### Column roles and hints")
                st.dataframe(orch.get_data_profile(), use_container_width=True, hide_index=True)
                with st.expander("Exact data summary sent to the LLM"):
                    st.code(orch.get_data_summary(), language="text")

            with t_brief:
                st.markdown(f'<div class="card"><h4>Persona</h4>{html.escape(brief["persona"])}</div>'
                            f'<div class="card"><h4>Model the data is prepared for</h4>{html.escape(brief["model"])}<br>'
                            f'Horizon: {brief["horizon"]} period(s) · target lags allowed: '
                            f'{"yes, shifted by the horizon" if brief["allow_target_lags"] else "no"}</div>',
                            unsafe_allow_html=True)
                g1, g2 = st.columns(2)
                g1.markdown("**Good features for this model**\n" + "\n".join(f"- {x}" for x in brief["guidance"]))
                g2.markdown("**Told to avoid**\n" + "\n".join(f"- {x}" for x in brief["avoid"]))
                st.caption(f"Industry context: {INDUSTRIES.get(brief['industry'], '')}")

            with t_code:
                st.code(orch.get_feature_engineer_function() or "", language="python")

        else:
            dec = orch.get_feature_decisions()
            with t_tests:
                st.markdown("##### Statistical agent (pandas, no LLM)")
                st.dataframe(orch.get_stat_tests(), use_container_width=True, hide_index=True)
                st.caption("Within-group rho is the average Spearman correlation computed separately in each "
                           f"{ctx.get('group_column') or 'group'}, so store or SKU size differences do not hide weekly effects.")
                st.markdown(f"##### Business agent (LLM persona: {brief['persona']})")
                st.dataframe(orch.get_business_tests(), use_container_width=True, hide_index=True)
                st.markdown("**Decision rule:** KEEP only if every S and B test passes.")
            with t_matrix:
                if dec.empty:
                    st.info("No decisions recorded.")
                else:
                    f1, f2 = st.columns(2)
                    dsel = f1.multiselect("Decision", ["KEEP", "DROP"], default=["KEEP", "DROP"], key="m_dec")
                    ssel = f2.multiselect("Source", ["new", "original"], default=["new", "original"], key="m_src")
                    show = dec[dec["decision"].isin(dsel) & dec["source"].isin(ssel)]
                    tcols = ["S1", "S2", "S3", "S4", "B1", "B2", "B3", "B4", "B5"]
                    st.dataframe(pass_fail_style(
                        show[["feature", "source", "decision"] + tcols + ["rho_pooled", "rho_within", "relevance",
                                                                         "leakage_risk", "reason", "business_reason"]],
                        tcols + ["decision"]), use_container_width=True, hide_index=True)
            with t_fail:
                if not dec.empty:
                    tcols = ["S1", "S2", "S3", "S4", "B1", "B2", "B3", "B4", "B5"]
                    fails = pd.Series({t: int((dec[t] == "FAIL").sum()) for t in tcols}, name="features failing")
                    st.bar_chart(fails, color=PURPLE)
                    names = {**{r["id"]: r["test"] for r in mafe.stat_tests()},
                             **{r["id"]: r["test"] for r in mafe.business_tests(orch.response.get("use_case"))}}
                    st.dataframe(pd.DataFrame([{"test": t, "name": names[t], "features failing": int(fails[t]),
                                                "examples": ", ".join(dec.loc[dec[t] == "FAIL", "feature"].head(5))}
                                               for t in tcols]), use_container_width=True, hide_index=True)

# ---------------- State history ----------------
with tab_hist:
    st.markdown("Every agent saves a **checkpoint** after each step (`graph.get_state_history(config)`).")
    st.dataframe(orch.get_all_state_history(), use_container_width=True, hide_index=True)
    st.subheader("Inspect one checkpoint")
    names = {"Orchestrator": orch, "Feature Engineering Agent": orch.fe_agent, "Feature Selection Agent": orch.fs_agent}
    agent = names[st.selectbox("Agent", list(names))]
    if agent.config is None:
        st.info("This agent did not run in this session.")
    else:
        hist = agent.get_state_history(show_values=True)
        options = [f"step {row.step} | next: {row.next_node} | {row.checkpoint_id[:8]}" for row in hist.itertuples()]
        idx = (st.select_slider("Checkpoint (oldest → newest)", options=list(range(len(options))),
                                value=len(options) - 1, format_func=lambda i: options[i]) if len(options) > 1 else 0)
        values = agent.get_state_at(hist.loc[idx, "checkpoint_id"])
        prev = agent.get_state_at(hist.loc[idx - 1, "checkpoint_id"]) if idx > 0 else {}
        changed = [k for k in values if values.get(k) != prev.get(k)]
        st.markdown(f"**Keys changed at this checkpoint:** {', '.join(changed) or 'none'}")
        st.dataframe(pd.DataFrame([{"key": k, "value": describe_value(v), "changed": k in changed}
                                   for k, v in values.items() if k != "session_log"]),
                     use_container_width=True, hide_index=True)
        key = st.selectbox("Show full value of", [k for k in values if k != "session_log"])
        if key:
            show_state_value(key, values[key])

# ---------------- Results ----------------
with tab_res:
    final = orch.get_final_output() or {}
    decisions = orch.get_feature_decisions()
    kept = int((decisions["decision"] == "KEEP").sum()) if not decisions.empty else 0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("New features created", len(final.get("created_features", [])))
    c2.metric("Features reviewed", len(decisions))
    c3.metric("Kept", kept)
    c4.metric("Dropped", len(decisions) - kept)
    c5.metric("Total execution time", fmt_secs(current["elapsed_s"]))
    if final.get("status") != "SUCCESS":
        st.error(final.get("fe_report") or "Run did not finish successfully.")
    if not decisions.empty:
        st.subheader("Feature decisions")
        st.dataframe(pass_fail_style(decisions[["feature", "source", "decision", "reason", "rho_pooled",
                                                "rho_within", "relevance", "business_reason"]], ["decision"]),
                     use_container_width=True, hide_index=True)
        rho = decisions.dropna(subset=["rho_pooled"]).set_index("feature")
        if not rho.empty:
            st.markdown(f"**Spearman rho with `{current['target']}`: overall vs within group**")
            st.bar_chart(rho[["rho_pooled", "rho_within"]].fillna(0), color=[PURPLE, DEEP_PURPLE])
    selected = orch.get_data_selected()
    if not selected.empty:
        st.subheader("Final model-ready data")
        st.dataframe(selected.head(50), use_container_width=True)
    st.subheader("Downloads")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Selected data (CSV)", selected.to_csv(index=False), f"selected_{current['id']}.csv",
                       disabled=selected.empty)
    eng = orch.get_data_engineered()
    d2.download_button("Engineered data (CSV)", eng.to_csv(index=False), f"engineered_{current['id']}.csv",
                       disabled=eng.empty)
    d3.download_button("Decisions (CSV)", decisions.to_csv(index=False), f"decisions_{current['id']}.csv",
                       disabled=decisions.empty)
    d4.download_button("Feature code (.py)", orch.get_feature_engineer_function() or "",
                       f"feature_engineer_{current['id']}.py")
    st.subheader("AI insight summary")
    if st.button("Generate insight summary", type="primary", disabled=decisions.empty or not api_key):
        with st.spinner("Writing summary..."):
            st.session_state[f"insight_{current['id']}"] = mafe.llm_text(
                get_llm(current["model"], api_key), INSIGHT_PROMPT.format(
                    persona=brief["persona"], use_case=brief["use_case"], industry=brief["industry"],
                    target=current["target"], instructions=current["instructions"],
                    decisions=decisions[["feature", "decision", "reason"]].to_string(index=False)))
    if st.session_state.get(f"insight_{current['id']}"):
        st.markdown(st.session_state[f"insight_{current['id']}"])
