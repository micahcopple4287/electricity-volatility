
# dashboard.py
# Streamlit dashboard for electricity price analysis
#
# Run with: streamlit run app/dashboard.py

 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import json
import os
 

# DARK THEME FOR ALL CHARTS

plt.style.use('dark_background')
 

# PAGE CONFIG

st.set_page_config(
    page_title="US Electricity Price Explorer",
    page_icon="⚡",
    layout="wide"
)
 
st.title("⚡ US Electricity Price Volatility Explorer")
st.markdown("Analyzing retail electricity prices across CA, TX, CO, NY, and IL (2001–2026)")
 

# LOAD DATA

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prices     = pd.read_csv(os.path.join(base, "data/raw/electricity_prices.csv"))
    generation = pd.read_csv(os.path.join(base, "data/raw/generation_by_source.csv"))
    features   = pd.read_csv(os.path.join(base, "data/processed/features.csv"),
                              parse_dates=["date"])
    results    = pd.read_csv(os.path.join(base, "data/processed/test_results.csv"),
                              parse_dates=["date"])
    return prices, generation, features, results
 
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reg_model = joblib.load(os.path.join(base, "data/processed/reg_model.pkl"))
    clf_model = joblib.load(os.path.join(base, "data/processed/clf_model.pkl"))
    le        = joblib.load(os.path.join(base, "data/processed/label_encoder.pkl"))
    with open(os.path.join(base, "data/processed/features.json")) as f:
        feature_list = json.load(f)
    return reg_model, clf_model, le, feature_list
 
prices, generation, features, test_results = load_data()
reg_model, clf_model, le, FEATURES = load_models()
 
# Build residential price series
res = prices[prices["sectorid"] == "RES"].copy()
res["date"] = pd.to_datetime(res[["year", "month"]].assign(day=1))
res = res.sort_values(["stateid", "date"])
 

# SIDEBAR CONTROLS

st.sidebar.header("Filters")
 
all_states = sorted(res["stateid"].unique().tolist())
selected_states = st.sidebar.multiselect(
    "Select states",
    options=all_states,
    default=all_states
)
 
year_min = int(res["year"].min())
year_max = int(res["year"].max())
year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(2010, year_max)
)
 
sector = st.sidebar.selectbox(
    "Sector",
    options=["RES", "COM", "IND"],
    format_func=lambda x: {"RES": "Residential",
                            "COM": "Commercial",
                            "IND": "Industrial"}[x]
)
 
# Filter data
filtered = prices[
    (prices["sectorid"] == sector) &
    (prices["stateid"].isin(selected_states)) &
    (prices["year"] >= year_range[0]) &
    (prices["year"] <= year_range[1])
].copy()
filtered["date"] = pd.to_datetime(filtered[["year", "month"]].assign(day=1))
filtered = filtered.sort_values(["stateid", "date"])
 

# TAB LAYOUT

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price Trends",
    "🌡️ Seasonality",
    "⚡ Generation Mix",
    "📊 Volatility",
    "🤖 Model Predictions"
])
 

# TAB 1: PRICE TRENDS

with tab1:
    st.subheader("Retail Electricity Prices Over Time")
 
    fig, ax = plt.subplots(figsize=(12, 5))
    for state in selected_states:
        subset = filtered[filtered["stateid"] == state]
        ax.plot(subset["date"], subset["price"], label=state, linewidth=1.5)
 
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (cents/kWh)")
    ax.set_title(f"{'Residential' if sector == 'RES' else sector} Electricity Prices")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    st.subheader("Summary Statistics (filtered period)")
    summary = (
        filtered.groupby("stateid")["price"]
        .agg(["mean", "std", "min", "max"])
        .round(2)
        .rename(columns={"mean": "Avg Price", "std": "Std Dev",
                          "min": "Min", "max": "Max"})
    )
    summary["Volatility (CV)"] = (summary["Std Dev"] / summary["Avg Price"]).round(3)
    st.dataframe(summary, use_container_width=True)
 

# TAB 2: SEASONALITY

with tab2:
    st.subheader("Seasonal Price Patterns by State")
 
    seasonal = (
        filtered.groupby(["stateid", "month"])["price"]
        .mean()
        .reset_index()
    )
 
    fig, ax = plt.subplots(figsize=(12, 5))
    for state in selected_states:
        subset = seasonal[seasonal["stateid"] == state]
        ax.plot(subset["month"], subset["price"], marker="o",
                label=state, linewidth=1.5)
 
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Price (cents/kWh)")
    ax.set_title("Average Price by Month (Seasonality)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    st.markdown("""
    **What to look for:**
    - Summer peaks (Jun–Aug) signal air conditioning demand driving prices up
    - Winter peaks (Dec–Jan) signal heating demand
    - Flat lines indicate fossil-fuel grids that absorb demand swings more easily
    """)
 

# TAB 3: GENERATION MIX

with tab3:
    st.subheader("Generation Mix by State (Last 12 Months)")
 
    gen = generation.copy()
    gen["date"] = pd.to_datetime(gen[["year", "month"]].assign(day=1))
    recent_gen = gen[gen["date"] >= "2025-01-01"]
 
    mix = (
        recent_gen.groupby(["location", "fueltypedescription"])["generation"]
        .sum()
        .reset_index()
    )
 
    selected_gen_state = st.selectbox("Select state", options=selected_states)
 
    state_mix = (
        mix[mix["location"] == selected_gen_state]
        .sort_values("generation", ascending=False)
    )
    state_mix = state_mix[state_mix["generation"] > 0].head(8)
 
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(state_mix)), state_mix["generation"],
                   color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(state_mix)))
    ax.set_xticklabels(state_mix["fueltypedescription"],
                        rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Generation (thousand MWh)")
    ax.set_title(f"{selected_gen_state} — Generation Mix")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 

# TAB 4: VOLATILITY

with tab4:
    st.subheader("Price Volatility Over Time")
    st.markdown("Rolling 12-month standard deviation — higher = more volatile")
 
    feat_filtered = features[
        (features["stateid"].isin(selected_states)) &
        (features["year"] >= year_range[0]) &
        (features["year"] <= year_range[1])
    ]
 
    fig, ax = plt.subplots(figsize=(12, 5))
    for state in selected_states:
        subset = feat_filtered[feat_filtered["stateid"] == state]
        ax.plot(subset["date"], subset["rolling_std_12m"],
                label=state, linewidth=1.5)
 
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Std Dev (cents/kWh)")
    ax.set_title("12-Month Rolling Price Volatility")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    for state in selected_states:
        subset = feat_filtered[feat_filtered["stateid"] == state]
        ax2.plot(subset["date"], subset["yoy_change"] * 100,
                  label=state, linewidth=1.5)
 
    ax2.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("YoY Change (%)")
    ax2.set_title("Year-over-Year Price Change by State")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
 

# TAB 5: MODEL PREDICTIONS

with tab5:
    st.subheader("XGBoost Model Predictions (Test Set: 2021–2026)")
 
    st.markdown("""
    Model trained on **2001–2021**, tested on **2021–2026**.
    The post-2021 price surge in CA and NY represents a structural regime change
    that lag-based features struggle to anticipate.
    """)
 
    # --- Actual vs Predicted ---
    st.markdown("### Actual vs Predicted Price")
 
    pred_state = st.selectbox("Select state to inspect",
                               options=sorted(test_results["state"].unique()))
 
    subset = test_results[test_results["state"] == pred_state].copy()
 
    # Run classifier on test features to get spike flags
    feat_test = features[features["stateid"] == pred_state].copy()
    feat_test = feat_test[feat_test["date"] >= "2021-05-01"]
 
    if not feat_test.empty:
        feat_test["state_encoded"] = le.transform(feat_test["stateid"])
        X_pred = feat_test[FEATURES]
        feat_test["spike_pred"] = clf_model.predict(X_pred)
        spike_dates = feat_test[feat_test["spike_pred"] == 1]["date"]
    else:
        spike_dates = pd.Series(dtype="datetime64[ns]")
 
    fig, ax = plt.subplots(figsize=(12, 5))
 
    # Plot actual and predicted lines
    ax.plot(subset["date"], subset["actual"],
            label="Actual", linewidth=2, color="#4fc3f7")
    ax.plot(subset["date"], subset["predicted"],
            label="Predicted", linewidth=2, linestyle="--", color="#ffb74d")
 
    # Shade error
    ax.fill_between(subset["date"], subset["actual"], subset["predicted"],
                     alpha=0.15, color="red", label="Error")
 
    # Highlight predicted spike months
    for spike_date in spike_dates:
        ax.axvline(spike_date, color="yellow", alpha=0.15, linewidth=4)
 
    # Add spike legend entry
    from matplotlib.patches import Patch
    spike_legend = Patch(facecolor="yellow", alpha=0.4, label="Predicted spike month")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [spike_legend],
          labels=labels + ["Predicted spike month"],
          fontsize=9)

 
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (cents/kWh)")
    ax.set_title(f"{pred_state} — Actual vs Predicted (yellow = predicted spike)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    # --- Model metrics ---
    st.markdown("### Model Performance")
    col1, col2, col3, col4 = st.columns(4)
 
    state_sub = test_results[test_results["state"] == pred_state]
    mae  = (state_sub["actual"] - state_sub["predicted"]).abs().mean()
    rmse = ((state_sub["actual"] - state_sub["predicted"]) ** 2).mean() ** 0.5
    avg  = state_sub["actual"].mean()
    pct_err = mae / avg * 100
 
    col1.metric("MAE", f"{mae:.2f} ¢/kWh")
    col2.metric("RMSE", f"{rmse:.2f} ¢/kWh")
    col3.metric("Avg Actual Price", f"{avg:.2f} ¢/kWh")
    col4.metric("Avg % Error", f"{pct_err:.1f}%")
 
    # --- Feature Importance ---
    st.markdown("### Feature Importance")
 
    col_a, col_b = st.columns(2)
 
    with col_a:
        st.markdown("**Regression Model (predicts price)**")
        imp_reg = pd.Series(reg_model.feature_importances_,
                             index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(6, 5))
        imp_reg.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white",
                      linewidth=0.3)
        ax.set_title("What predicts price?")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    with col_b:
        st.markdown("**Classifier Model (predicts spikes)**")
        imp_clf = pd.Series(clf_model.feature_importances_,
                             index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(6, 5))
        imp_clf.plot(kind="barh", ax=ax, color="#ff7043", edgecolor="white",
                      linewidth=0.3)
        ax.set_title("What predicts spikes?")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()