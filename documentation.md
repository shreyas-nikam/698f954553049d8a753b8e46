id: 698f954553049d8a753b8e46_documentation
summary: Lab 7: Fundamental Data Analysis (Structured Data) Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Building a Robust Financial Data Pipeline: A Codelab for Structured Data Analysis

## Introduction to QuLab: Fundamental Data Analysis (Structured Data)
Duration: 0:05

Welcome to QuLab's Lab 7, focusing on **Fundamental Data Analysis for Structured Data**. In this codelab, we'll guide you through a comprehensive Streamlit application designed to tackle common challenges in financial data processing and prepare a clean, reliable dataset for in-depth analysis.

<aside class="positive">
This codelab is designed to provide a <b>comprehensive guide</b> for developers to understand the application's functionalities, the underlying financial concepts, and best practices in data engineering for finance.
</aside>

### The Problem: Ava's Frustrations at Alpha Investments

Imagine Ava, a Senior Equity Analyst at Alpha Investments, whose primary goal is to identify undervalued technology stocks. She relies heavily on financial ratios and relative valuation. However, raw financial data is notoriously messy:
*   **Missing Values:** Incomplete disclosures lead to gaps.
*   **Outliers:** Extreme values distort statistical measures and peer comparisons.
*   **Temporal Misalignment (Look-Ahead Bias):** Using financial data "as of" its fiscal year-end date, rather than its public release date, can lead to unrealistic historical analysis.

Manually cleaning this data is time-consuming, error-prone, and diverts Ava from higher-value analytical work.

### The Solution: A Systematic Data Pipeline

This Streamlit application demonstrates a systematic and reproducible workflow to address Ava's challenges. We will build a data pipeline that automatically:
1.  **Acquires** raw financial data from public sources.
2.  Performs **Initial Data Quality Assessment** to diagnose issues.
3.  **Computes** meaningful financial ratios and performs **DuPont Decomposition**.
4.  **Imputes** missing values using domain-appropriate methods.
5.  **Treats Outliers** to ensure robust comparisons.
6.  Applies **Point-in-Time (PIT) Alignment** to prevent look-ahead bias.
7.  Generates a **Final Report and Visualizations** for actionable insights.

This pipeline ensures high-quality, comparable valuation metrics, enabling Ava to make confident, data-driven investment recommendations.

### Key Concepts You Will Learn

*   **Data Engineering for Finance:** Best practices in handling financial time-series and cross-sectional data.
*   **DuPont Analysis:** Decomposing Return on Equity (ROE) into its fundamental drivers.
*   **Sector-Median Imputation:** A robust method for filling missing values while preserving sector characteristics.
*   **Winsorization:** Treating extreme outliers without discarding valuable data.
*   **Point-in-Time (PIT) Alignment:** A critical technique to prevent look-ahead bias in financial backtesting and analysis.

### Application Architecture Overview

The application follows a modular, multi-step pipeline approach. Each step builds upon the output of the previous one, ensuring a progressive data transformation. Streamlit's `st.session_state` is leveraged to maintain data and user configurations across page navigations, creating a seamless user experience.

Here's a high-level flowchart of the data pipeline implemented in this codelab:

```mermaid
graph TD
    A[Start: User Interaction] --> B{Data Acquisition};
    B --> C[Raw Financial Data];
    C --> D{Initial Data Quality Assessment};
    D --> E[Diagnosed Data Quality];
    E --> F{Ratio Computation & DuPont Decomposition};
    F --> G[Computed Financial Ratios];
    G --> H{Missing Value Imputation};
    H --> I[Imputed Ratios with Flags];
    I --> J{Outlier Treatment (Winsorization)};
    J --> K[Winsorized Ratios];
    K --> L{Point-in-Time Alignment};
    L --> M[Clean, PIT-Aligned Data];
    M --> N{Final Report & Visualizations};
    N --> O[End: Actionable Insights];
```

<aside class="positive">
This codelab is interactive! You'll be using Streamlit components to trigger each step of the pipeline and observe the data transformations in real-time.
</aside>

## 1. Setting up the Streamlit Application
Duration: 0:05

This section guides you through setting up and understanding the basic structure of the Streamlit application. The entire application is orchestrated using `streamlit` for the UI and `pandas`, `numpy`, `matplotlib`, `seaborn`, `missingno` for data processing and visualization. The core data transformation logic is encapsulated in a `source.py` file (not directly provided in this document, but its functions are called by `app.py`).

### Understanding `st.session_state`

Streamlit applications re-run from top to bottom every time a widget is interacted with. To persist data and user selections across these reruns and different pages, `st.session_state` is crucial. In this application, `st.session_state` stores:
*   `current_page`: Tracks the active page in the sidebar navigation.
*   `raw_df`: The initial data fetched from the API.
*   `financial_ratios_df`: Data after ratio computation.
*   `financial_ratios_imputed_df`: Data after missing value imputation.
*   `financial_ratios_winsorized_df`: Data after outlier treatment.
*   `final_analysis_data`: The fully cleaned and PIT-aligned data.
*   User-defined parameters like `num_companies`, `filing_lag_days`, `winsorization_limits`.

The `initialize_session_state()` function at the beginning of `app.py` ensures all these variables have a default value when the application starts.

### Running the Application

To run the Streamlit application, ensure you have Streamlit installed (`pip install streamlit`). Save the provided Python code as `app.py` and the `source.py` (which contains helper functions like `fetch_sp500_tech_data`, `compute_financial_ratios`, `sector_median_impute`, `winsorize_ratios`, `add_pit_date`, `generate_quality_report`).

```bash
streamlit run app.py
```

This command will open the Streamlit application in your default web browser.

### Application Structure in `app.py`

The `app.py` file sets up the Streamlit page configuration, initializes `st.session_state`, defines a helper for final visualizations, and then uses a `st.sidebar.selectbox` for navigation. Each `if st.session_state.current_page == "..."` block corresponds to a different step of our data pipeline, rendering its specific UI and logic.

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
from source import * # Imports all helper functions from source.py

# Set Streamlit page configuration
st.set_page_config(page_title="QuLab: Lab 7: Fundamental Data Analysis (Structured Data)", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 7: Fundamental Data Analysis (Structured Data)")
st.divider()

warnings.filterwarnings('ignore') # Suppress warnings

#  st.session_state Design 
def initialize_session_state():
    # ... (initialization code) ...
    pass # Placeholder for brevity

initialize_session_state()

#  Helper function for Final Visualizations 
# (Contains logic for DuPont, correlation, P/E vs ROE plots, and peer comparison)
def generate_final_visualizations(df: pd.DataFrame, ratio_cols: list):
    # ... (visualization code) ...
    pass # Placeholder for brevity

#  Sidebar Navigation 
st.sidebar.title("Navigation")
pages = [
    "Introduction",
    "1. Data Acquisition",
    "2. Initial Data Quality Assessment",
    "3. Ratio Computation & DuPont Decomposition",
    "4. Missing Value Imputation",
    "5. Outlier Treatment (Winsorization)",
    "6. Point-in-Time Alignment & Export",
    "7. Final Report & Visualizations"
]

st.session_state.current_page = st.sidebar.selectbox(
    "Go to",
    pages,
    index=pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0
)

#  Application Pages (Conditional Rendering) 
# Each page logic is rendered based on st.session_state.current_page
if st.session_state.current_page == "Introduction":
    # ... (Introduction page content) ...
    pass
elif st.session_state.current_page == "1. Data Acquisition":
    # ... (Data Acquisition page content) ...
    pass
# ... and so on for other pages ...
```

<aside class="positive">
The use of <code>st.session_state</code> is a best practice in Streamlit for building multi-page applications or preserving complex data objects across user interactions.
</aside>

## 2. Data Acquisition: Retrieving Raw Financial Statement Data
Duration: 0:10

Ava's journey begins with gathering raw financial data. Instead of manual data entry, which is slow and prone to human error, she opts for programmatic data retrieval. This step uses the `yfinance` library (abstracted within `source.py`) to fetch key financial statement items for a user-specified number of S&P 500 technology companies.

### Why Automated Data Acquisition?

*   **Consistency:** Ensures data from the same source and format every time.
*   **Speed:** Quickly builds a foundational dataset for analysis.
*   **Scalability:** Easily adjust the number of companies or data points.
*   **Reproducibility:** The entire process can be rerun with identical results.

### How it Works

On the "1. Data Acquisition" page:
1.  A `st.slider` allows you to select the `num_companies` (between 10 and 100) for which to fetch data. This value is stored in `st.session_state.num_companies`.
2.  A `st.button` labeled "Fetch Raw Financial Data" triggers the data retrieval.
3.  Upon clicking, the `fetch_sp500_tech_data()` function from `source.py` is called with the selected `num_companies`.
4.  The retrieved `pandas.DataFrame` is then stored in `st.session_state.raw_df` and a copy in `st.session_state.initial_raw_data_copy` for later comparison in the final report.
5.  The first few rows of the fetched data are displayed using `st.dataframe()`.

<aside class="negative">
Fetching live financial data can sometimes be slow or encounter API rate limits. The application handles this with a spinner and error messages, but it's a consideration for production environments.
</aside>

### Code Snippet (`app.py` for this page)

```python
elif st.session_state.current_page == "1. Data Acquisition":
    st.header("1. Data Acquisition: Retrieving Raw Financial Statement Data")
    st.markdown("""
    Ava's first step is to gather financial data...
    """)

    num_companies_input = st.slider("Select number of S&P 500 Tech Companies to fetch data for:", min_value=10, max_value=100, value=st.session_state.num_companies, step=10)
    st.session_state.num_companies = num_companies_input

    if st.button("Fetch Raw Financial Data"):
        with st.spinner(f"Fetching data for {st.session_state.num_companies} companies... This may take a moment."):
            # Invocation: fetch_sp500_tech_data from source.py
            raw_data = fetch_sp500_tech_data(num_companies=st.session_state.num_companies)
            if not raw_data.empty:
                st.session_state.raw_df = raw_data
                st.session_state.initial_raw_data_copy = raw_data.copy()
                st.success(f"Successfully retrieved raw data for {len(st.session_state.raw_df)} companies.")
                st.markdown(f"Initial shape: {st.session_state.raw_df.shape}")
                st.markdown(f"First 5 rows of raw data:")
                st.dataframe(st.session_state.raw_df.head())
            else:
                st.error("Failed to fetch any data. Please try again or reduce the number of companies.")
```

**Exercise:**
1. Navigate to "1. Data Acquisition".
2. Select a number of companies (e.g., 50).
3. Click "Fetch Raw Financial Data" and observe the initial data structure. Notice potential `NaN` values.

## 3. Initial Data Quality Assessment
Duration: 0:15

Before any serious analysis, Ava must understand the quality of her raw data. This crucial step identifies missing values, distributional characteristics, and common financial anomalies that can severely impact the reliability of her models and investment decisions. "Garbage in, garbage out" is a fundamental truth in data analysis.

### Why Data Quality Assessment is Crucial in Finance

*   **Reliability:** Flawed data leads to incorrect valuation metrics.
*   **Bias Detection:** Missing data patterns might indicate systematic issues (e.g., smaller companies having less disclosure).
*   **Outlier Identification:** Extreme values can distort averages and skew relative comparisons.
*   **Domain-Specific Anomalies:** Negative revenue or equity require careful consideration, as they could indicate distress or data errors.

### Assessment Techniques

This page performs several diagnostics:
1.  **Missing Value Analysis:**
    *   Calculates the percentage of `NaN` values per feature.
    *   Uses `missingno.matrix` to visualize missing data patterns, helping to identify if missingness is random or structured.
2.  **Distributional Summary:**
    *   `df.describe(percentiles=...)` provides key statistics, including the 1st and 99th percentiles, which are excellent indicators of extreme values and heavy-tailed distributions.
3.  **Common Financial Data Anomalies:**
    *   Counts occurrences of `negative_revenue`, `negative_assets`, `zero_equity`, `negative_equity`, `zero_interest_expense`, and `zero_total_debt`. These values can break ratio calculations or signal distress.
4.  **Sector Distribution:**
    *   Shows the count of companies per sector, useful for understanding the composition of the dataset.

The **Jarque-Bera test** is conceptually used to check for normality in distributions, although financial ratios are almost always heavy-tailed. The formula for the Jarque-Bera statistic is:
$$ \text{{JB}} = \frac{{n}}{{6}} \left( S^2 + \frac{{(K-3)^2}}{{4}} \right) $$
where $n$ is the number of observations, $S$ is the skewness, and $K$ is the kurtosis. High JB values suggest non-normality, which is typical for financial ratios and often requires transformation or robust methods like winsorization.

### Code Snippet (`app.py` for this page)

```python
elif st.session_state.current_page == "2. Initial Data Quality Assessment":
    st.header("2. Initial Data Quality Assessment")
    st.markdown("""
    Before Ava can trust any valuation models...
    """)
    st.markdown(r"The **Jarque-Bera test** is conceptually used to check for normality...")
    st.markdown(r"$$ \text{{JB}} = \frac{{n}}{{6}} \left( S^2 + \frac{{(K-3)^2}}{{4}} \right) $$")

    if st.session_state.raw_df is None:
        st.warning("Please fetch raw financial data first on the '1. Data Acquisition' page.")
    else:
        if st.button("Assess Data Quality"):
            st.subheader(" Data Quality Assessment Results ")
            # Missing Value Analysis
            missing_pct = st.session_state.raw_df.isnull().mean().sort_values(ascending=False)
            st.write("Percentage of missing values per feature (only > 0%):")
            st.dataframe(missing_pct[missing_pct > 0].round(3))

            # Missing data matrix
            fig, ax = plt.subplots(figsize=(12, 6))
            msno.matrix(st.session_state.raw_df, ax=ax, fontsize=8)
            ax.set_title('Missing Data Pattern Across Features and Companies', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Distributional Summary
            desc = st.session_state.raw_df.describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]).T[['count', 'mean', 'std', '1%', '99%']]
            st.dataframe(desc.round(2))

            # Anomalies
            anomalies = pd.DataFrame({
                'negative_revenue': (st.session_state.raw_df['revenue'] < 0).sum(),
                # ... other anomaly counts ...
            }, index=['count'])
            st.dataframe(anomalies.T)

            # Sector Distribution
            if 'sector' in st.session_state.raw_df.columns:
                st.dataframe(st.session_state.raw_df['sector'].value_counts())
            else:
                st.markdown("Sector column not found for distribution analysis.")
```

**Exercise:**
1. Navigate to "2. Initial Data Quality Assessment".
2. Click "Assess Data Quality".
3. Examine the outputs:
    *   Which features have the most missing values?
    *   What patterns do you see in the `missingno` matrix? Are certain columns often missing together?
    *   Look at the `1%` and `99%` percentiles in the distributional summary. What does a large difference suggest?
    *   Are there any companies with negative equity or zero interest expense?

## 4. Financial Ratio Computation and DuPont Decomposition
Duration: 0:15

After assessing the raw data's quality, Ava's next critical step is to transform raw financial statement items into meaningful financial ratios. These ratios are the cornerstone of fundamental analysis and relative valuation. Beyond single ratios, the **DuPont Analysis** provides deeper insights into a company's profitability drivers.

### Why Financial Ratios and DuPont Analysis?

*   **Standardization:** Ratios normalize financial data, allowing for comparisons across companies of different sizes.
*   **Insight Generation:** They distill complex financial statements into digestible metrics (profitability, liquidity, leverage, efficiency, valuation).
*   **DuPont Decomposition:** Breaks down Return on Equity (ROE) into three components:
    1.  **Net Profit Margin:** Operating efficiency (Net Income / Revenue).
    2.  **Asset Turnover:** Asset utilization efficiency (Revenue / Total Assets).
    3.  **Equity Multiplier:** Financial leverage (Total Assets / Total Equity).
    This helps Ava understand *why* a company's ROE is high or low, revealing the underlying drivers (e.g., strong margins vs. high debt).

The **DuPont Decomposition of ROE** is given by:
$$ \text{{ROE}} = \frac{{\text{{Net Income}}}}{{\text{{Revenue}}}} \times \frac{{\text{{Revenue}}}}{{\text{{Total Assets}}}} \times \frac{{\text{{Total Assets}}}}{{\text{{Total Equity}}}} $$
$$ \text{{ROE}} = \text{{Net Profit Margin}} \times \text{{Asset Turnover}} \times \text{{Equity Multiplier}} $$
where $\text{{Net Profit Margin}}$ measures operating efficiency, $\text{{Asset Turnover}}$ measures asset use efficiency, and $\text{{Equity Multiplier}}$ measures financial leverage.

### Handling Financial Data Traps

The `compute_financial_ratios()` function (from `source.py`) also handles common pitfalls:
*   **Negative Equity:** If `Total Equity` is negative, ratios like ROE become undefined or misleading. The function treats these cases to prevent errors.
*   **Zero Denominators:** For ratios like `Interest Coverage` (EBIT / Interest Expense), a zero `Interest Expense` would lead to infinity. Such values are often capped (e.g., at 100x) to avoid extreme distortions.

### Code Snippet (`app.py` for this page)

```python
elif st.session_state.current_page == "3. Ratio Computation & DuPont Decomposition":
    st.header("3. Financial Ratio Computation and DuPont Decomposition")
    st.markdown("""
    With an understanding of the raw data's quality...
    """)
    st.markdown(r"The **DuPont Decomposition of ROE** is given by:")
    st.markdown(r"$$ \text{{ROE}} = \frac{{\text{{Net Income}}}}{{\text{{Revenue}}}} \times \frac{{\text{{Revenue}}}}{{\text{{Total Assets}}}} \times \frac{{\text{{Total Assets}}}}{{\text{{Total Equity}}}} $$")
    st.markdown(r"$$ \text{{ROE}} = \text{{Net Profit Margin}} \times \text{{Asset Turnover}} \times \text{{Equity Multiplier}} $$")

    if st.session_state.raw_df is None:
        st.warning("Please fetch raw financial data first on the '1. Data Acquisition' page.")
    else:
        if st.button("Compute Financial Ratios"):
            with st.spinner("Computing financial ratios and DuPont components..."):
                # Invocation: compute_financial_ratios from source.py
                financial_ratios = compute_financial_ratios(st.session_state.raw_df)
                if not financial_ratios.empty:
                    st.session_state.financial_ratios_df = financial_ratios
                    st.session_state.ratio_columns = [col for col in financial_ratios.columns
                                                    if col not in ['ticker', 'sector', 'fiscal_year_end']]
                    st.success(f"Computed {len(st.session_state.ratio_columns)} ratios for {financial_ratios.shape[0]} companies.")
                    st.markdown(f"First 5 rows of computed ratios:")
                    st.dataframe(st.session_state.financial_ratios_df.head())

                    st.markdown(f"**DuPont ROE consistency check (ROE vs. DuPont components product):**")
                    sample_dupont_check = financial_ratios[['ticker', 'roe', 'dupont_roe_check']].dropna().head()
                    st.dataframe(sample_dupont_check)
                    if not sample_dupont_check.empty:
                        st.markdown(f"Average absolute difference between ROE and DuPont check: {(sample_dupont_check['roe'] - sample_dupont_check['dupont_roe_check']).abs().mean():.4f}")
                    else:
                        st.markdown(f"No sufficient data to perform DuPont consistency check.")
                else:
                    st.error("Failed to compute financial ratios.")
```

**Exercise:**
1. Navigate to "3. Ratio Computation & DuPont Decomposition".
2. Click "Compute Financial Ratios".
3. Observe the new `financial_ratios_df` with various ratios and DuPont components. Check the "DuPont ROE consistency check" to confirm accurate calculation.

## 5. Missing Value Imputation: Sector-Median Approach
Duration: 0:10

Even after computing ratios, some will inevitably have missing values. Simply dropping rows with `NaN`s can lead to significant data loss and sample bias. Ava needs a robust imputation strategy that makes financial sense. The **sector-median imputation** method is ideal as it fills missing values using the median of a company's direct peers within the same sector, preserving critical industry characteristics.

### Why Sector-Median Imputation?

*   **Preserves Data:** Avoids discarding valuable company data due to a few missing ratios.
*   **Domain Relevance:** Imputing with sector medians ensures the filled values are characteristic of the company's industry, which is crucial for relative valuation. Using a global median would dilute sector-specific nuances.
*   **Transparency:** Crucially, this method adds imputation flag columns (e.g., `roe_imputed`), indicating which values were filled. This metadata is vital for downstream models to understand the "certainty" of each data point.

The formula for sector-median imputation for a missing feature $x_{{ij}}$ (feature $j$ of company $i$ in sector $s_i$) is:
$$ \hat{{x}}_{{ij}} = \text{{median}}\{x_{{kj}} : k \in \text{{sector }} s_i, x_{{kj}} \text{{ not missing}}\} $$
where $\hat{x}_{ij}$ is the imputed value for feature $j$ of company $i$, and the median is taken over all companies $k$ in the same sector $s_i$ for which $x_{kj}$ is not missing.

### How it Works

On the "4. Missing Value Imputation" page:
1.  It checks if `st.session_state.financial_ratios_df` (from the previous step) is available.
2.  Upon clicking "Impute Missing Values", the `sector_median_impute()` function from `source.py` is called. It takes the ratio DataFrame, the list of ratio columns, and the grouping column (`'sector'`).
3.  The function computes the median for each ratio within each sector and uses these medians to fill `NaN` values.
4.  It also creates new binary flag columns (e.g., `pe_ratio_imputed`), which are `1` if the value was imputed and `0` otherwise.
5.  The resulting DataFrame is stored in `st.session_state.financial_ratios_imputed_df`.
6.  The imputation rates and the head of the imputed data are displayed.

### Code Snippet (`app.py` for this page)

```python
elif st.session_state.current_page == "4. Missing Value Imputation":
    st.header("4. Missing Value Imputation: Sector-Median Approach")
    st.markdown("""
    Despite the robust data acquisition...
    """)
    st.markdown(r"The formula for sector-median imputation for a missing feature $x_{{ij}}$...")
    st.markdown(r"$$ \hat{{x}}_{{ij}} = \text{{median}}\{x_{{kj}} : k \in \text{{sector }} s_i, x_{{kj}} \text{{ not missing}}\} $$")

    if st.session_state.financial_ratios_df is None:
        st.warning("Please compute financial ratios first on the '3. Ratio Computation & DuPont Decomposition' page.")
    else:
        if st.button("Impute Missing Values"):
            with st.spinner("Applying sector-median imputation..."):
                # Invocation: sector_median_impute from source.py
                financial_ratios_imputed = sector_median_impute(st.session_state.financial_ratios_df, st.session_state.ratio_columns, 'sector')
                if not financial_ratios_imputed.empty:
                    st.session_state.financial_ratios_imputed_df = financial_ratios_imputed

                    imputation_flags_cols = [f'{col}_imputed' for col in st.session_state.ratio_columns if f'{col}_imputed' in st.session_state.financial_ratios_imputed_df.columns]
                    if imputation_flags_cols:
                        imputation_rate = st.session_state.financial_ratios_imputed_df[imputation_flags_cols].mean().sort_values(ascending=False)
                        st.markdown(f"**Imputation rate per feature (only > 0%):**")
                        st.dataframe(imputation_rate[imputation_rate > 0].round(3))
                    else:
                        st.markdown(f"No imputation flags found, or no values were imputed.")

                    st.markdown(f"First 5 rows of imputed data with flags:")
                    st.dataframe(st.session_state.financial_ratios_imputed_df.head())
                else:
                    st.error("Failed to impute missing values.")
```

**Exercise:**
1. Navigate to "4. Missing Value Imputation".
2. Click "Impute Missing Values".
3. Review the "Imputation rate per feature" to see which ratios had the most missing values.
4. Examine the `st.dataframe` output, noting the new `_imputed` flag columns.

## 6. Outlier Treatment: Winsorization
Duration: 0:15

Financial ratios often exhibit heavy-tailed distributions with extreme outliers. These outliers, though sometimes genuine, can disproportionately influence statistical analyses and bias comparative valuation models. Instead of simply removing these extreme values (which can discard valuable information), Ava opts for **winsorization**. This method caps extreme values at a specified percentile, effectively "pulling in" the tails of the distribution without deleting observations.

### Why Winsorization over Trimming?

*   **Mitigates Extreme Influence:** Reduces the impact of outliers on statistical measures (mean, standard deviation, regression coefficients).
*   **Retains Data:** Unlike trimming (which removes outliers), winsorization preserves the number of observations, which is beneficial for sample size and avoiding bias if outliers are concentrated in certain groups.
*   **Robustness:** Leads to more robust estimates and more reliable relative valuation.

**Winsorization** at the $\alpha$-level replaces extreme values with the $\alpha$-th percentile boundary:
$$ x_i^{{(w)}} = \begin{{cases}} q_\alpha & \text{{if }} x_i < q_\alpha \\ q_{{1-\alpha}} & \text{{if }} x_i > q_{{1-\alpha}} \\ x_i & \text{{otherwise}} \end{{cases}} $$
where $q_\alpha$ is the $\alpha$-th sample quantile, $q_{{1-\alpha}}$ is the $(1-\alpha)$-th sample quantile, and $x_i^{{(w)}}$ is the winsorized value of $x_i$.

### How it Works

On the "5. Outlier Treatment (Winsorization)" page:
1.  It requires `st.session_state.financial_ratios_imputed_df` from the previous step.
2.  Two `st.number_input` widgets allow you to specify the `lower_limit` and `upper_limit` for winsorization (e.g., 0.01 for 1st percentile and 0.01 for 99th percentile, implying a 1% winsorization on each tail). These values are stored in `st.session_state.winsorization_limits`.
3.  Clicking "Apply Winsorization" calls the `winsorize_ratios()` function from `source.py`.
4.  This function applies winsorization to each ratio column and returns the winsorized DataFrame, stored in `st.session_state.financial_ratios_winsorized_df`, and `detected_outlier_counts`.
5.  A summary of the number of winsorized values per feature is displayed.
6.  Crucially, **distribution plots (histograms)** are generated for key ratios, showing their shape *before* and *after* winsorization, visually demonstrating the effect of capping the tails.

### Code Snippet (`app.py` for this page)

```python
elif st.session_state.current_page == "5. Outlier Treatment (Winsorization)":
    st.header("5. Outlier Detection and Treatment: Winsorization")
    st.markdown("""
    Even after imputing missing values...
    """)
    st.markdown(r"**Winsorization** at the $\alpha$-level replaces extreme values...")
    st.markdown(r"$$ x_i^{{(w)}} = \begin{{cases}} q_\alpha & \text{{if }} x_i < q_\alpha \\ q_{{1-\alpha}} & \text{{if }} x_i > q_{{1-\alpha}} \\ x_i & \text{{otherwise}} \end{{cases}} $$")

    if st.session_state.financial_ratios_imputed_df is None:
        st.warning("Please impute missing values first on the '4. Missing Value Imputation' page.")
    else:
        lower_limit_input = st.number_input("Lower Winsorization Percentile (e.g., 0.01 for 1st percentile):", min_value=0.0, max_value=0.49, value=st.session_state.winsorization_limits[0], step=0.005, format="%.3f")
        upper_limit_input = st.number_input("Upper Winsorization Percentile (e.g., 0.01 for 99th percentile):", min_value=0.0, max_value=0.49, value=st.session_state.winsorization_limits[1], step=0.005, format="%.3f")

        st.session_state.winsorization_limits = (lower_limit_input, upper_limit_input)

        if st.button("Apply Winsorization"):
            with st.spinner("Applying winsorization to financial ratios..."):
                # Invocation: winsorize_ratios from source.py
                financial_ratios_winsorized, detected_outlier_counts = winsorize_ratios(
                    st.session_state.financial_ratios_imputed_df,
                    st.session_state.ratio_columns,
                    limits=st.session_state.winsorization_limits
                )
                if not financial_ratios_winsorized.empty:
                    st.session_state.financial_ratios_winsorized_df = financial_ratios_winsorized
                    st.session_state.detected_outlier_counts = detected_outlier_counts
                    st.success(f"Ratios winsorized. Data shape: {st.session_state.financial_ratios_winsorized_df.shape}")

                    outlier_summary = pd.Series(st.session_state.detected_outlier_counts)
                    st.markdown(f"**Number of values detected as outliers (and winsorized) per feature (initial counts):**")
                    st.dataframe(outlier_summary[outlier_summary > 0].sort_values(ascending=False))

                    st.markdown(f"First 5 rows of winsorized data:")
                    st.dataframe(st.session_state.financial_ratios_winsorized_df.head())

                    st.markdown(f"### Ratio Distributions Before vs. After Winsorization ({lower_limit_input*100:.0f}th/{ (1-upper_limit_input)*100:.0f}th Percentile)")
                    key_ratios_for_viz = ['pe_ratio', 'roe', 'debt_to_equity', 'net_margin', 'interest_coverage', 'asset_turnover']

                    valid_key_ratios = [
                        col for col in key_ratios_for_viz
                        if col in st.session_state.financial_ratios_imputed_df.columns and
                           col in st.session_state.financial_ratios_winsorized_df.columns
                    ]

                    if valid_key_ratios:
                        fig, axes = plt.subplots(len(valid_key_ratios), 2, figsize=(15, 4 * len(valid_key_ratios)))
                        fig.suptitle('Ratio Distributions Before vs. After Winsorization', fontsize=16, y=1.02)

                        for i, col in enumerate(valid_key_ratios):
                            # Before Winsorization
                            sns.histplot(st.session_state.financial_ratios_imputed_df[col].dropna(), bins=50, kde=True, ax=axes[i, 0], color='skyblue', alpha=0.7)
                            axes[i, 0].axvline(st.session_state.financial_ratios_imputed_df[col].median(), color='red', linestyle='--', label='Median')
                            axes[i, 0].set_title(f'Original: {col}')
                            axes[i, 0].legend()

                            # After Winsorization
                            sns.histplot(st.session_state.financial_ratios_winsorized_df[col].dropna(), bins=50, kde=True, ax=axes[i, 1], color='lightgreen', alpha=0.7)
                            axes[i, 1].axvline(st.session_state.financial_ratios_winsorized_df[col].median(), color='red', linestyle='--', label='Median')
                            axes[i, 1].set_title(f'Winsorized: {col}')
                            axes[i, 1].legend()

                        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.markdown(f"Not enough valid key ratios to generate before/after winsorization plots.")
                else:
                    st.error("Failed to apply winsorization.")
```

**Exercise:**
1. Navigate to "5. Outlier Treatment (Winsorization)".
2. Adjust the "Lower" and "Upper Winsorization Percentile" if desired (default 0.01 for both is common).
3. Click "Apply Winsorization".
4. Review the "Number of values detected as outliers" summary.
5. **Crucially, examine the "Ratio Distributions Before vs. After Winsorization" plots.** Observe how the tails of the distributions are compressed after winsorization, particularly for ratios like P/E or Debt-to-Equity.

## 7. Point-in-Time Alignment and Export
Duration: 0:10

A critical and often overlooked aspect of financial data analysis is **Point-in-Time (PIT) alignment**, which directly addresses **look-ahead bias**. Financial statements are released with a delay after a company's fiscal year-end. For example, a 10-K report for a December 31st fiscal year might not be publicly available until March of the following year. Using the December 31st data for a decision made in January would be using future information, leading to unrealistic backtest results or flawed historical analysis.

### Why Point-in-Time Alignment is Critical

*   **Prevents Look-Ahead Bias:** Ensures that any analysis or model training uses only information that was genuinely available to the market at the time of a hypothetical investment decision.
*   **Realistic Backtesting:** Essential for building and validating robust quantitative investment strategies.
*   **Data Integrity:** Creates a more accurate representation of historical information flow.

The `pit_date` is calculated as:
$$ \text{{pit\_date}} = \text{{fiscal\_year\_end}} + \text{{filing\_lag\_days}} $$
where $\text{{fiscal\_year\_end}}$ is the company's fiscal year-end date, and $\text{{filing\_lag\_days}}$ is the assumed number of days for the financial report to become publicly available.

### How it Works

On the "6. Point-in-Time Alignment & Export" page:
1.  It requires `st.session_state.financial_ratios_winsorized_df` from the previous step.
2.  An `st.number_input` allows you to specify the `filing_lag_days` (e.g., 90 days is common for annual reports). This value is stored in `st.session_state.filing_lag_days`.
3.  Clicking "Apply Point-in-Time Alignment" calls the `add_pit_date()` function from `source.py`.
4.  This function takes the winsorized DataFrame and the `filing_lag_days`, calculating a new `pit_date` for each record.
5.  The final, clean DataFrame is stored in `st.session_state.final_analysis_data`.
6.  The first few rows, showing `fiscal_year_end` and the new `pit_date`, are displayed.
7.  A **Download button** is provided to export the `sp500_tech_ratios_clean.csv`, making the cleaned data ready for external use.

### Code Snippet (`app.py` for this page)

```python
elif st.session_state.current_page == "6. Point-in-Time Alignment & Export":
    st.header("6. Point-in-Time Alignment and Export")
    st.markdown("""
    Ava understands a critical pitfall in financial data analysis: **look-ahead bias**...
    """)
    st.markdown(r"The `pit_date` is calculated as:")
    st.markdown(r"$$ \text{{pit\_date}} = \text{{fiscal\_year\_end}} + \text{{filing\_lag\_days}} $$")

    if st.session_state.financial_ratios_winsorized_df is None:
        st.warning("Please apply winsorization first on the '5. Outlier Treatment (Winsorization)' page.")
    else:
        filing_lag_input = st.number_input("Enter filing lag in days (e.g., 90 for annual reports):", min_value=0, max_value=365, value=st.session_state.filing_lag_days, step=1)
        st.session_state.filing_lag_days = filing_lag_input

        if st.button("Apply Point-in-Time Alignment"):
            with st.spinner("Applying Point-in-Time alignment..."):
                # Invocation: add_pit_date from source.py
                final_data = add_pit_date(st.session_state.financial_ratios_winsorized_df, filing_lag_days=st.session_state.filing_lag_days)
                if not final_data.empty:
                    st.session_state.final_analysis_data = final_data
                    st.success(f"Point-in-Time Alignment applied. Data shape: {st.session_state.final_analysis_data.shape}")
                    st.markdown(f"Assumed filing lag: {st.session_state.filing_lag_days} days.")
                    st.markdown(f"First 5 rows with 'fiscal_year_end' and 'pit_date':")
                    st.dataframe(st.session_state.final_analysis_data[['ticker', 'fiscal_year_end', 'pit_date', 'pe_ratio']].head())
                else:
                    st.error("Failed to apply Point-in-Time alignment.")

    if st.session_state.final_analysis_data is not None:
        st.markdown(f"### Export Cleaned Data")
        csv_data = st.session_state.final_analysis_data.to_csv(index=False).encode('utf-8')
        <button>
          [Download Cleaned Financial Ratios as CSV](data:text/csv;base64,{csv_data.decode('utf-8')})
        </button>
        st.markdown(f"The cleaned, analysis-ready dataset is now available for download. This dataset contains {st.session_state.final_analysis_data.shape[0]} companies and {st.session_state.final_analysis_data.shape[1]} features.")
```
<aside class="negative">
The download button above uses an illustrative `data:text/csv` URI for demonstration within the codelab markdown. In the actual Streamlit app, `st.download_button` handles the file download directly.
</aside>

**Exercise:**
1. Navigate to "6. Point-in-Time Alignment & Export".
2. You can adjust the "filing lag in days" if needed (default 90 days).
3. Click "Apply Point-in-Time Alignment".
4. Observe the new `pit_date` column in the displayed DataFrame and how it relates to `fiscal_year_end`.
5. Click the "Download Cleaned Financial Ratios as CSV" button to get your final dataset.

## 8. Final Report & Visualizations for Relative Valuation
Duration: 0:20

Having meticulously cleaned, imputed, winsorized, and point-in-time aligned her financial data, Ava is now ready for the final, most crucial step: generating actionable insights for relative valuation. This page provides a comprehensive data quality report summarizing the cleaning process and a suite of visualizations vital for an equity analyst.

### Why a Final Report and Visualizations?

*   **Validation and Transparency:** The data quality report quantifies the impact of each cleaning step, providing an audit trail and building confidence in the data.
*   **Actionable Insights:** Visualizations allow Ava to quickly identify patterns, compare companies against peers, and spot potential investment opportunities (undervalued/overvalued).
*   **Communication:** Clear charts and tables are essential tools for communicating findings to portfolio managers and other stakeholders.

### Components of the Final Report

1.  **Comprehensive Data Quality Report:** Generated by `generate_quality_report()` from `source.py`. This report summarizes:
    *   Initial and final data completeness.
    *   Features with significant missingness.
    *   Total number of outliers treated.
    *   Confirmed PIT alignment status.
    This offers a quantitative overview of the data engineering effort.

2.  **DuPont Decomposition Stacked Bar Chart:**
    *   **Purpose:** Visualizes the components of ROE (Net Profit Margin, Asset Turnover, Equity Multiplier) for selected high-ROE companies.
    *   **Insight:** Helps Ava understand if high ROE is driven by operational efficiency, asset utilization, or financial leverage, revealing underlying risk and sustainability.

3.  **Correlation Heatmap of Financial Ratios:**
    *   **Purpose:** Displays the pairwise correlations between all computed financial ratios.
    *   **Insight:** Helps identify highly correlated ratios, which might suggest multicollinearity for future predictive model building or indicate redundant features.

4.  **P/E vs. ROE Relative Value Scatter Plot:**
    *   **Purpose:** A classic relative valuation tool plotting Price-to-Earnings (P/E) Ratio against Return on Equity (ROE).
    *   **Insight:**
        *   **Undervalued candidates:** Companies with low P/E and high ROE (bottom-right quadrant).
        *   **Overvalued candidates:** Companies with high P/E and low ROE (top-left quadrant).
        *   The size of the points is scaled by `market_cap`, adding another dimension to the analysis.

5.  **Peer Comparison Table (Ranked by P/E Ratio):**
    *   **Purpose:** A ranked list of companies by key valuation and profitability metrics.
    *   **Insight:** Provides a quick, digestible way to compare a company's fundamental performance and valuation multiples directly against its peers.

These visualizations are generated by the `generate_final_visualizations()` helper function defined in `app.py`.

### Code Snippet (`app.py` for this page)

```python
elif st.session_state.current_page == "7. Final Report & Visualizations":
    st.header("7. Comprehensive Data Quality Report & Relative Valuation Insights")
    st.markdown("""
    Now that Ava has meticulously cleaned, imputed, winsorized...
    """)

    if st.session_state.final_analysis_data is None:
        st.warning("Please complete previous data cleaning steps up to '6. Point-in-Time Alignment & Export' to generate the final report.")
    elif st.session_state.initial_raw_data_copy is None:
        st.warning("Initial raw data was not saved. Please re-run '1. Data Acquisition' and subsequent steps.")
    else:
        if st.button("Generate Final Report and Visualizations"):
            with st.spinner("Generating data quality report and visualizations..."):
                # Invocation: generate_quality_report from source.py
                quality_report = generate_quality_report(
                    st.session_state.initial_raw_data_copy,
                    st.session_state.final_analysis_data,
                    st.session_state.ratio_columns,
                    st.session_state.detected_outlier_counts,
                    st.session_state.filing_lag_days
                )

                st.subheader(" Comprehensive Data Quality Report ")
                for k, v in quality_report.items():
                    st.markdown(f"**{k}:** {v}")

                # Invocation: Helper function generate_final_visualizations
                generate_final_visualizations(st.session_state.final_analysis_data, st.session_state.ratio_columns)
```

**Exercise:**
1. Navigate to "7. Final Report & Visualizations".
2. Click "Generate Final Report and Visualizations".
3. Carefully review the "Comprehensive Data Quality Report" to understand the impact of your cleaning steps.
4. Analyze each visualization:
    *   **DuPont Chart:** What drives ROE for the top performers?
    *   **Correlation Heatmap:** Are there highly correlated ratios? What does this imply for future modeling?
    *   **P/E vs. ROE Plot:** Can you identify any potentially undervalued (low P/E, high ROE) or overvalued (high P/E, low ROE) companies?
    *   **Peer Comparison Table:** How do companies rank against each other on key metrics?

This final step completes the entire data engineering pipeline, transforming raw, messy financial data into a clean, insightful, and actionable dataset for an equity analyst like Ava.
