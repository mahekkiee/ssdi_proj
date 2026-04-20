"""
SSDI Stock Sector Analysis — Streamlit Dashboard
Reads the 9 NSE CSVs from the `data/` folder and runs the same analysis
as the Jupyter notebook.
"""

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.multivariate.manova import MANOVA

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="NSE Stock Sector Analysis",
    page_icon="📈",
    layout="wide",
)
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# Ticker / sector configuration  (same as notebook)
# ---------------------------------------------------------------
DATA_FOLDER = "data"

TICKERS = ["TCS", "INFY", "WIPRO",
           "HDFCBANK", "ICICIBANK", "SBIN",
           "SUNPHARMA", "CIPLA", "DRREDDY"]

SECTOR_MAP = {
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT",
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "SUNPHARMA": "Pharma", "CIPLA": "Pharma", "DRREDDY": "Pharma",
}

# ---------------------------------------------------------------
# Data loader — reads 9 NSE CSVs, cleans, adds derived columns
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.isdir(DATA_FOLDER):
        raise FileNotFoundError(
            f"Folder `{DATA_FOLDER}/` not found. "
            "Create it in the repo root and upload the 9 NSE CSV files there."
        )

    all_csvs = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    if not all_csvs:
        raise FileNotFoundError(
            f"No CSV files found in `{DATA_FOLDER}/`. "
            "Upload the 9 NSE CSV files there."
        )

    all_dfs, missing = [], []
    for ticker in TICKERS:
        # case-insensitive substring match on filename
        matches = [f for f in all_csvs
                   if ticker.upper() in os.path.basename(f).upper()]
        if not matches:
            missing.append(ticker)
            continue
        temp = pd.read_csv(matches[0], encoding="utf-8-sig", thousands=",")
        temp.columns = temp.columns.str.strip()
        temp["Ticker"] = ticker
        temp["Sector"] = SECTOR_MAP[ticker]
        all_dfs.append(temp)

    if missing:
        raise FileNotFoundError(
            f"Could not find CSVs for these tickers in `{DATA_FOLDER}/`: "
            f"{', '.join(missing)}. "
            f"Make sure each ticker name appears in its filename."
        )

    df = pd.concat(all_dfs, ignore_index=True)

    # Clean — same steps as the notebook
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d-%b-%Y")
    df = df.rename(columns={
        "OPEN": "Open", "HIGH": "High", "LOW": "Low",
        "CLOSE": "Close", "VOLUME": "Volume",
    })
    df = df.sort_values(["Ticker", "DATE"]).reset_index(drop=True)
    df = df[["DATE", "Ticker", "Sector",
             "Open", "High", "Low", "Close", "Volume"]]

    # Derived columns
    df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"] * 100
    df["Daily_Range_Pct"] = (df["High"] - df["Low"]) / df["Open"] * 100
    df["DayOfWeek"] = df["DATE"].dt.day_name()
    return df


try:
    df = load_data()
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()

# ---------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "🔍 Exploratory Analysis",
        "📉 Linear Regression",
        "🧪 One-way ANOVA",
        "🧩 Two-way ANOVA",
        "🎯 MANOVA",
        "📝 Conclusions",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Dataset:** {len(df):,} rows · "
    f"{df['Ticker'].nunique()} tickers · "
    f"{df['Sector'].nunique()} sectors"
)

# ---------------------------------------------------------------
# 1. OVERVIEW
# ---------------------------------------------------------------
if page == "🏠 Overview":
    st.title("📈 NSE Stock Sector Analysis")
    st.caption("Statistical study of 9 NSE stocks across IT, Banking & Pharma")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows", f"{len(df):,}")
    c2.metric("Tickers", df["Ticker"].nunique())
    c3.metric("Sectors", df["Sector"].nunique())
    c4.metric("Date range", f"{df['DATE'].min().date()} → {df['DATE'].max().date()}")

    st.markdown("### Project summary")
    st.markdown(
        """
        This dashboard presents the findings of a statistical analysis on 1 year of
        NSE (National Stock Exchange of India) data for **9 stocks** grouped into
        **3 sectors**:

        - **IT** — TCS, INFY, WIPRO
        - **Banking** — HDFCBANK, ICICIBANK, SBIN
        - **Pharma** — SUNPHARMA, CIPLA, DRREDDY

        We apply the same techniques taught in the SSDI module: linear regression
        with VIF diagnostics, one-way & two-way ANOVA with Tukey HSD post-hoc,
        and MANOVA on the OHLC price vector.
        """
    )

    st.markdown("### Sample of the data")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Rows per ticker")
    st.bar_chart(df["Ticker"].value_counts())

# ---------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------
elif page == "🔍 Exploratory Analysis":
    st.title("🔍 Exploratory Data Analysis")

    st.markdown("### Summary statistics per sector")
    num_cols = ["Open", "High", "Low", "Close", "Volume",
                "Daily_Return", "Daily_Range_Pct"]
    st.dataframe(df.groupby("Sector")[num_cols].mean().round(3),
                 use_container_width=True)

    st.markdown("### Correlation heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0, ax=ax)
    ax.set_title("Correlation of numeric variables")
    st.pyplot(fig)
    st.info("**Takeaway:** Open/High/Low/Close are all ≈1.0 correlated — "
            "a first hint of multicollinearity, confirmed later by VIF.")

    st.markdown("### Close price over time")
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    for ticker in df["Ticker"].unique():
        sub = df[df["Ticker"] == ticker]
        ax2.plot(sub["DATE"], sub["Close"], label=ticker, linewidth=1.2)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Close Price (INR)")
    ax2.set_title("Close Price over Time — All 9 Stocks")
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig2)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Daily Return by Sector")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="Sector", y="Daily_Return", palette="Set2", ax=ax3)
        ax3.axhline(0, color="red", linestyle="--", alpha=0.5)
        st.pyplot(fig3)

    with col2:
        st.markdown("#### Intraday Volatility by Sector")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="Sector", y="Daily_Range_Pct", palette="Set2", ax=ax4)
        st.pyplot(fig4)

    st.success("**Observation:** Return boxes overlap heavily, but volatility "
               "boxes clearly differ — Banking is the calmest sector.")

# ---------------------------------------------------------------
# 3. LINEAR REGRESSION
# ---------------------------------------------------------------
elif page == "📉 Linear Regression":
    st.title("📉 Linear Regression — Predicting Close Price")

    st.markdown("### Model comparison (lower AIC = better)")
    models = {
        "M1: Close ~ Open": "Close ~ Open",
        "M2: Close ~ High": "Close ~ High",
        "M3: Close ~ Low": "Close ~ Low",
        "M4: Close ~ Volume": "Close ~ Volume",
        "M5: Close ~ Open + Volume": "Close ~ Open + Volume",
        "M6: Close ~ Open + High": "Close ~ Open + High",
        "M7: Close ~ Open * Volume": "Close ~ Open * Volume",
        "M8: Close ~ Open + Volume + Open²": "Close ~ Open + Volume + I(Open**2)",
        "M9: Close ~ Open + Volume + Daily_Range_Pct": "Close ~ Open + Volume + Daily_Range_Pct",
        "M10: Close ~ Open + High + Low + Volume": "Close ~ Open + High + Low + Volume",
    }
    rows = []
    for name, formula in models.items():
        fit = smf.ols(formula, data=df).fit()
        rows.append({
            "Model": name,
            "R²": round(fit.rsquared, 6),
            "Adj R²": round(fit.rsquared_adj, 6),
            "AIC": round(fit.aic, 2),
            "BIC": round(fit.bic, 2),
            "# Predictors": int(fit.df_model),
        })
    comp = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    st.dataframe(comp, use_container_width=True)
    st.success(f"🏆 Best model by AIC: **{comp.iloc[0]['Model']}**")

    st.markdown("### VIF — multicollinearity check on full model")
    X = add_constant(df[["Open", "High", "Low", "Volume"]])
    vif_full = pd.DataFrame({
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    st.dataframe(vif_full, use_container_width=True)
    st.warning("Open, High, Low all have VIF > 1000 — extremely collinear. "
               "Drop High & Low, keep Open + Volume.")

    st.markdown("### Refined model — Close ~ Open + Volume")
    lm_refined = smf.ols("Close ~ Open + Volume", data=df).fit()
    st.code(lm_refined.summary().as_text(), language="text")

    st.markdown("### VIF after refinement")
    X2 = add_constant(df[["Open", "Volume"]])
    vif_ref = pd.DataFrame({
        "Feature": X2.columns,
        "VIF": [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])],
    })
    st.dataframe(vif_ref, use_container_width=True)
    st.success("✓ All VIF values well below 10 — multicollinearity solved.")

    st.markdown("### Diagnostic plots")
    y_pred = lm_refined.predict(df[["Open", "Volume"]])
    residuals = df["Close"] - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(df["Close"], y_pred, alpha=0.3, s=12)
    axes[0].plot([df["Close"].min(), df["Close"].max()],
                 [df["Close"].min(), df["Close"].max()], "r--")
    axes[0].set_xlabel("Actual Close")
    axes[0].set_ylabel("Predicted Close")
    axes[0].set_title("Predicted vs Actual")
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=12)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted Close")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs Predicted")
    st.pyplot(fig)

# ---------------------------------------------------------------
# 4. ONE-WAY ANOVA
# ---------------------------------------------------------------
elif page == "🧪 One-way ANOVA":
    st.title("🧪 One-way ANOVA + Tukey HSD")

    st.markdown("### Q1. Do daily returns differ across sectors?")
    fit_ret = smf.ols("Daily_Return ~ Sector", data=df).fit()
    anova_ret = sm.stats.anova_lm(fit_ret, typ=1)
    st.dataframe(anova_ret, use_container_width=True)
    p_ret = anova_ret["PR(>F)"].iloc[0]
    if p_ret > 0.05:
        st.info(f"p = {p_ret:.4f} > 0.05 → **fail to reject H₀**. "
                "Sectors do NOT differ in average daily returns "
                "(consistent with efficient-market theory).")
    else:
        st.warning(f"p = {p_ret:.4f} < 0.05 → reject H₀.")

    tukey_ret = pairwise_tukeyhsd(df["Daily_Return"], df["Sector"])
    st.code(tukey_ret.summary().as_text(), language="text")

    st.markdown("---")
    st.markdown("### Q2. Does intraday volatility differ across sectors?")
    fit_vol = smf.ols("Daily_Range_Pct ~ Sector", data=df).fit()
    anova_vol = sm.stats.anova_lm(fit_vol, typ=1)
    st.dataframe(anova_vol, use_container_width=True)
    p_vol = anova_vol["PR(>F)"].iloc[0]
    if p_vol < 0.05:
        st.success(f"p = {p_vol:.2e} < 0.05 → **reject H₀**. "
                   "Volatility DOES differ significantly across sectors.")

    tukey_vol = pairwise_tukeyhsd(df["Daily_Range_Pct"], df["Sector"])
    st.code(tukey_vol.summary().as_text(), language="text")

    # Tukey visualization
    tukey_df = pd.DataFrame(tukey_vol._results_table.data[1:],
                            columns=tukey_vol._results_table.data[0])
    tukey_df["meandiff"] = tukey_df["meandiff"].astype(float)
    tukey_df["lower"] = tukey_df["lower"].astype(float)
    tukey_df["upper"] = tukey_df["upper"].astype(float)
    tukey_df["pair"] = tukey_df["group1"] + " vs " + tukey_df["group2"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for _, row in tukey_df.iterrows():
        color = "red" if row["reject"] else "gray"
        ax.errorbar(row["meandiff"], row["pair"],
                    xerr=[[row["meandiff"] - row["lower"]],
                          [row["upper"] - row["meandiff"]]],
                    fmt="o", color=color, ecolor=color, capsize=5)
    ax.axvline(0, color="blue", linestyle="--", alpha=0.5)
    ax.set_title("Tukey HSD — Volatility between sectors "
                 "(red = significant)")
    ax.set_xlabel("Mean difference in Daily_Range_Pct")
    st.pyplot(fig)

# ---------------------------------------------------------------
# 5. TWO-WAY ANOVA
# ---------------------------------------------------------------
elif page == "🧩 Two-way ANOVA":
    st.title("🧩 Two-way ANOVA — Sector × Day-of-week")

    st.markdown("### Model: `Daily_Range_Pct ~ Sector * DayOfWeek`")
    fit_2way = smf.ols("Daily_Range_Pct ~ Sector * DayOfWeek", data=df).fit()
    anova_2way = sm.stats.anova_lm(fit_2way, typ=2)
    st.dataframe(anova_2way, use_container_width=True)

    st.markdown("### Interaction plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.pointplot(
        data=df, x="DayOfWeek", y="Daily_Range_Pct", hue="Sector",
        order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        palette="Set2", ax=ax,
    )
    ax.set_title("Volatility by Day-of-week and Sector")
    ax.set_ylabel("Mean Daily_Range_Pct")
    st.pyplot(fig)

    st.info("**Reading the plot:** parallel lines → no interaction; "
            "crossing/diverging lines → interaction exists. "
            "In this dataset, lines are roughly parallel → sector ranking "
            "stays consistent across weekdays.")

# ---------------------------------------------------------------
# 6. MANOVA
# ---------------------------------------------------------------
elif page == "🎯 MANOVA":
    st.title("🎯 MANOVA — Do OHLC prices jointly differ by sector?")

    maov = MANOVA.from_formula("Open + High + Low + Close ~ Sector", data=df)
    st.code(str(maov.mv_test()), language="text")
    st.success("Wilks' Lambda p-value < 0.001 → **reject H₀**. "
               "The four OHLC price variables jointly differ across sectors.")

    st.markdown("### Per-variable Tukey (follow-up)")
    for col in ["Open", "High", "Low", "Close"]:
        with st.expander(f"Tukey HSD on {col}"):
            tuk = pairwise_tukeyhsd(df[col], df["Sector"])
            st.code(tuk.summary().as_text(), language="text")

# ---------------------------------------------------------------
# 7. CONCLUSIONS
# ---------------------------------------------------------------
elif page == "📝 Conclusions":
    st.title("📝 Conclusions")

    summary = pd.DataFrame([
        ["Linear Regression", "Close ~ Open + Volume", "R² ≈ 0.99",
         "Open price is the strongest predictor. VIF flagged High/Low as redundant."],
        ["One-way ANOVA", "Daily_Return ~ Sector", "p > 0.05",
         "Sectors do NOT differ in average daily returns (efficient-market consistent)."],
        ["One-way ANOVA", "Daily_Range_Pct ~ Sector", "p ≈ 0.000",
         "Sectors DO differ in volatility. Banking is calmest."],
        ["Two-way ANOVA", "Volatility ~ Sector * DayOfWeek", "see output",
         "Tests sector, weekday, and interaction effects."],
        ["MANOVA", "OHLC ~ Sector", "Wilks' λ p < 0.001",
         "OHLC price levels jointly differ across sectors."],
    ], columns=["Analysis", "Test", "Result", "Interpretation"])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("### Real-world use cases")
    st.markdown(
        """
        - **Portfolio construction** — allocate more to Banking for low-volatility exposure
        - **Risk management** — size positions based on sector-specific volatility
        - **Day-trading strategy** — target IT / Pharma for intraday moves
        - **Market efficiency research** — non-significant return ANOVA supports EMH at sector level
        """
    )

    st.markdown("### Assumptions & limitations")
    st.markdown(
        """
        - Only 1 year of data (≈ 247 trading days per stock)
        - 3 sectors, 3 stocks each — a wider universe would increase statistical power
        - ANOVA assumes normality & equal variances — could be formally tested
          (Shapiro-Wilk, Levene's test)
        - Financial returns have fat tails; non-parametric alternatives (Kruskal-Wallis)
          are worth exploring
        """
    )

    st.caption("Project built for the SSDI module · Data sourced from NSE India.")
