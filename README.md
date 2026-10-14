# Feature Engineering Agents - app v4

The banner and sidebar show "v4.0".

## What is new in v4
- Brief the agents: Industry (CPG / Retail / General), Modelling use case, Forecast horizon,
  Date and Group columns (auto-detected; CPG grain such as sku + retailer is detected automatically)
- Use-case personas: Sales/demand forecasting, Marketing mix modelling, Price & promotion elasticity,
  Churn/propensity, General regression. Each sets the persona, target model, good features,
  features to avoid and whether target lags are allowed.
- New profile_data step: detects grain, frequency, column roles and domain hints before planning.
- Flow & reasoning tab
  - Feature Engineering Agent: observations, feature plan (name, source columns, how, reason,
    assumption served, built yes/no), what the agent saw, brief and persona, code
  - Feature Selection Agent: test catalogue (S1-S4 statistical, B1-B5 business), PASS/FAIL per
    feature, failures by test
- Statistical signal test now uses overall AND within-group Spearman rho.

## Run
    cd ~/Downloads/fe_agent_app_v4
    source ~/Downloads/multi_agent_feature_engineering/.venv/bin/activate
    python -m pip install -r requirements.txt
    cp ~/Downloads/fe_agent_app_v3/.env .
    bash run_app.sh

Demo data in ./data: sample_retail_weekly.csv and sample_cpg_weekly.csv (synthetic).

## Deploy (Streamlit Community Cloud)
Push this folder to GitHub, set main file app.py, add GROQ_API_KEY in Secrets,
and invite viewers by email under Share.
