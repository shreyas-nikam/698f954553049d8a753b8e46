
# Streamlit Application Specification: Fundamental Data Analysis Pipeline

## Application Overview

This Streamlit application, "Fundamental Data Analysis: Building a Clean Financial Data Pipeline," provides a hands-on, interactive experience for financial professionals to understand and apply rigorous data engineering techniques to raw financial statement data. It simulates the real-world workflow of Ava, a Senior Equity Analyst, who aims to identify undervalued technology stocks but struggles with inconsistent and messy financial data.

The application guides the user through a systematic process, from data acquisition to generating a final, analysis-ready dataset and comprehensive data quality report, addressing common pitfalls like missing values, outliers, and look-ahead bias. By demonstrating each step with interactive widgets and clear explanations, the app empowers learners to transform raw financial data into a reliable foundation for robust investment analysis and machine learning models.

**High-Level Story Flow:**

1.  **Introduction**: Meet Ava, understand her challenges with raw financial data, and grasp the application's purpose in automating and enhancing her data cleaning workflow.
2.  **Data Acquisition**: Ava programmatically fetches raw financial statement data for a universe of S&P 500 technology companies using a robust API.
3.  **Initial Data Quality Assessment**: She then diagnoses the fetched data for missing values, distributional anomalies, and common financial data traps using descriptive statistics and visualizations (e.g., missing data matrix).
4.  **Financial Ratio Computation & DuPont Decomposition**: The raw data is transformed into a comprehensive set of financial ratios, including a detailed DuPont decomposition of ROE, a critical feature engineering step.
5.  **Missing Value Imputation**: Missing ratio values are addressed using a domain-appropriate sector-median imputation method, with flags to track imputed data for transparency.
6.  **Outlier Treatment (Winsorization)**: Extreme outliers, common in financial ratios, are treated using winsorization to prevent distortion of analysis, with visual comparison of distributions before and after.
7.  **Point-in-Time Alignment & Export**: To prevent look-ahead bias, a Point-in-Time (PIT) date is added, accounting for reporting lags, making the data truly analysis-ready. The cleaned data is then ready for export.
8.  **Final Report & Visualizations**: A comprehensive data quality report summarizes the cleaning process, and interactive visualizations (DuPont charts, correlation heatmaps, relative value scatter plots, peer comparison tables) provide actionable insights for investment decisions.

## Code Requirements

```python
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno # Required for missingno.matrix visualization
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Import all functions from source.py ---
from source import (
    fetch_sp500_tech_data,
    compute_financial_ratios,
    sector_median_impute,
    winsorize_ratios,
    add_pit_date,
    generate_quality_report
    # Note: assess_data_quality is not imported as a function, but its logic is
    # adapted and performed inline using st.session_state.raw_df and matplotlib/pandas methods,
    # as the original function prints and shows plots which are directly handled by Streamlit.
)

# --- st.session_state Design ---
# Keys are initialized, updated by UI interactions/function calls, and read across pages.

# Initialization:
# All session state variables are initialized with default values or None.
# This block runs only once when the app starts or a script rerun is triggered.
def initialize_session_state():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Introduction'
    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = None # Stores raw data from fetch_sp500_tech_data
    if 'initial_raw_data_copy' not in st.session_state: # Copy of raw_df to preserve original state for final report's completeness metric
        st.session_state.initial_raw_data_copy = None
    if 'financial_ratios_df' not in st.session_state:
        st.session_state.financial_ratios_df = None # Stores ratios from compute_financial_ratios
    if 'financial_ratios_imputed_df' not in st.session_state:
        st.session_state.financial_ratios_imputed_df = None # Stores imputed ratios from sector_median_impute
    if 'financial_ratios_winsorized_df' not in st.session_state:
        st.session_state.financial_ratios_winsorized_df = None # Stores winsorized ratios from winsorize_ratios
    if 'detected_outlier_counts' not in st.session_state:
        st.session_state.detected_outlier_counts = {} # Stores outlier counts from winsorize_ratios
    if 'final_analysis_data' not in st.session_state:
        st.session_state.final_analysis_data = None # Stores final data from add_pit_date
    if 'ratio_columns' not in st.session_state:
        st.session_state.ratio_columns = [] # List of columns considered as ratios for imputation/winsorization
    if 'num_companies' not in st.session_state:
        st.session_state.num_companies = 50 # User selection for data acquisition
    if 'filing_lag_days' not in st.session_state:
        st.session_state.filing_lag_days = 90 # User selection for PIT alignment
    if 'winsorization_limits' not in st.session_state:
        st.session_state.winsorization_limits = (0.01, 0.01) # User selection for winsorization percentiles

initialize_session_state()

# --- Helper function for Final Visualizations (not in source.py, but required by spec) ---
# This function encapsulates the visualization requirements for the final report page.
def generate_final_visualizations(df: pd.DataFrame, ratio_cols: list):
    st.markdown(f"### DuPont Decomposition Stacked Bar Chart")
    st.markdown(f"Visualize the components of ROE for selected companies, revealing profitability drivers.")
    
    if all(col in df.columns for col in ['roe', 'dupont_margin', 'dupont_turnover', 'dupont_leverage']):
        # Select top 5 companies by ROE for visualization
        selected_companies = df.nlargest(5, 'roe', default_value=0.0).copy()
        selected_companies = selected_companies.set_index('ticker')
        dupont_components = selected_companies[['dupont_margin', 'dupont_turnover', 'dupont_leverage']]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        dupont_components.plot(kind='bar', stacked=True, ax=ax, cmap='viridis')
        ax.set_title('DuPont Decomposition of ROE', fontsize=16)
        ax.set_ylabel('Component Value', fontsize=12)
        ax.set_xlabel('Company Ticker', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(['Net Profit Margin', 'Asset Turnover', 'Equity Multiplier'], bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.markdown(f"Required DuPont components (roe, dupont_margin, dupont_turnover, dupont_leverage) not found or insufficient data for visualization.")

    st.markdown(f"### Correlation Heatmap of Financial Ratios")
    st.markdown(f"Explore the relationships between various computed ratios to understand potential multicollinearity for future model building.")
    
    # Filter for numeric ratio columns, excluding imputation flags and non-numeric identifiers
    numeric_ratio_cols = [col for col in ratio_cols if df[col].dtype in [np.float64, np.int64] and col in df.columns]
    
    if numeric_ratio_cols:
        corr_matrix = df[numeric_ratio_cols].corr()
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", ax=ax, linewidths=.5)
        ax.set_title('Correlation Heatmap of Financial Ratios', fontsize=16)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.markdown(f"No numeric ratio columns available for correlation heatmap.")

    st.markdown(f"### P/E vs. ROE Relative Value Scatter Plot")
    st.markdown(f"Identify potentially undervalued (low P/E, high ROE) or overvalued (high P/E, low ROE) companies within the sector.")
    
    if all(col in df.columns for col in ['pe_ratio', 'roe', 'market_cap', 'sector']):
        plot_df = df.dropna(subset=['pe_ratio', 'roe', 'market_cap'])
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(
                data=plot_df,
                x='roe',
                y='pe_ratio',
                size='market_cap',
                sizes=(50, 1000), # Size range for points
                hue='sector',
                alpha=0.7,
                ax=ax,
                palette='tab10'
            )
            ax.set_title('P/E Ratio vs. ROE (Size by Market Cap)', fontsize=16)
            ax.set_xlabel('Return on Equity (ROE)', fontsize=12)
            ax.set_ylabel('Price-to-Earnings (P/E) Ratio', fontsize=12)
            ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Sector')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.markdown(f"Not enough data to generate P/E vs. ROE scatter plot after dropping NaNs.")
    else:
        st.markdown(f"Required columns (pe_ratio, roe, market_cap, sector) not found for scatter plot.")

    st.markdown(f"### Peer Comparison Table (Ranked by P/E Ratio)")
    st.markdown(f"A ranked list of companies by key valuation and profitability metrics for quick peer comparison.")
    
    # Define columns to display in the peer comparison table
    peer_display_cols = ['ticker', 'sector', 'pe_ratio', 'roe', 'net_margin', 'debt_to_equity', 'current_ratio', 'market_cap']
    # Filter for columns that actually exist in the DataFrame
    valid_peer_display_cols = [col for col in peer_display_cols if col in df.columns]

    if valid_peer_display_cols:
        display_df = df[valid_peer_display_cols].sort_values(by='pe_ratio', ascending=True).reset_index(drop=True)
        st.dataframe(display_df.round(2))
    else:
        st.markdown(f"Required columns for peer comparison table not found.")


# --- Sidebar Navigation ---
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

# --- Application Pages (Conditional Rendering) ---

# Introduction Page
if st.session_state.current_page == "Introduction":
    st.title("Fundamental Data Analysis: Building a Clean Financial Data Pipeline")
    st.markdown(f"**Persona:** Ava, a Senior Equity Analyst at Alpha Investments.")
    st.markdown(f"**Organization:** Alpha Investments, focused on identifying undervalued technology stocks for their actively managed portfolios.")
    st.markdown(f"""
    Ava's role involves meticulously evaluating companies to provide actionable investment recommendations. A core part of her workflow is performing relative valuation, comparing target companies against their peers using various financial multiples. However, she constantly faces a significant challenge: the raw financial data she retrieves is often plagued with inconsistencies. Missing values, extreme outliers, and misaligned reporting dates frequently skew her analysis, leading to unreliable valuation metrics and, potentially, suboptimal investment decisions. Manually cleaning this data for dozens of companies is time-consuming and prone to human error, diverting her attention from deeper analytical work.

    This application outlines a systematic and reproducible workflow to address Ava's frustrations. We will build a data pipeline that automatically retrieves, cleans, and standardizes financial statement data for a universe of technology companies. By applying robust data quality measures—including missing value imputation, outlier treatment, and point-in-time alignment—Ava will be able to generate high-quality, comparable valuation metrics, leading to more confident and data-driven investment recommendations. This automation will free up her time for strategic insights, enhancing Alpha Investments' research capabilities and competitive edge.
    """)

# 1. Data Acquisition
elif st.session_state.current_page == "1. Data Acquisition":
    st.header("1. Data Acquisition: Retrieving Raw Financial Statement Data")
    st.markdown(f"""
    Ava's first step is to gather financial data for her target universe of companies. Historically, this involved manually pulling data from various sources like Bloomberg terminals or company reports, a process notorious for being slow and error-prone. To streamline this, Ava will use `yfinance` to programmatically retrieve key financial statement items for a basket of S&P 500 technology companies. This automation ensures consistency and speed, allowing her to quickly establish a foundational dataset for her valuation models.

    We will focus on the technology sector, a high-growth but often volatile segment requiring precise data for accurate valuation.
    """)

    num_companies_input = st.slider("Select number of S&P 500 Tech Companies to fetch data for:", min_value=10, max_value=100, value=st.session_state.num_companies, step=10)
    # Update: st.session_state.num_companies is automatically updated by slider value, but explicit assignment is harmless.
    st.session_state.num_companies = num_companies_input

    if st.button("Fetch Raw Financial Data"):
        with st.spinner(f"Fetching data for {st.session_state.num_companies} companies... This may take a moment."):
            # Invocation: fetch_sp500_tech_data from source.py
            raw_data = fetch_sp500_tech_data(num_companies=st.session_state.num_companies)
            if not raw_data.empty:
                # Update: st.session_state.raw_df
                st.session_state.raw_df = raw_data
                # Update: st.session_state.initial_raw_data_copy for final report
                st.session_state.initial_raw_data_copy = raw_data.copy()
                st.success(f"Successfully retrieved raw data for {len(st.session_state.raw_df)} companies.")
                st.markdown(f"Initial shape: {st.session_state.raw_df.shape}")
                st.markdown(f"First 5 rows of raw data:")
                st.dataframe(st.session_state.raw_df.head())
            else:
                st.error("Failed to fetch any data. Please try again or reduce the number of companies.")
    
    # Read: st.session_state.raw_df
    if st.session_state.raw_df is not None:
        st.markdown(f"**Explanation of Execution:**")
        st.markdown(f"""
        The code successfully retrieved raw financial data for a subset of S&P 500 technology companies. Ava can now see the initial structure of her data, with key financial metrics and company information. The output of `raw_financial_data.head()` provides a quick snapshot, immediately revealing potential issues like missing values (represented by `NaN`) in certain columns, which is a common challenge with raw financial APIs. This early visibility confirms the need for comprehensive data quality assessment and cleaning, validating Ava's decision to automate this part of her workflow.
        """)

# 2. Initial Data Quality Assessment
elif st.session_state.current_page == "2. Initial Data Quality Assessment":
    st.header("2. Initial Data Quality Assessment")
    st.markdown(f"""
    Before Ava can trust any valuation models built on this data, she needs to understand its quality. She's aware that raw financial data often contains missing values, outliers, and data entry anomalies that can severely distort her analysis. This section focuses on systematically diagnosing these issues using descriptive statistics and visualizations. For an equity analyst, identifying these problems early is crucial to prevent "garbage in, garbage out" scenarios, where flawed data leads to incorrect investment conclusions.

    A key concern for Ava is identifying *why* data might be missing or anomalous. For instance, negative equity might indicate a distressed company, not just a data error. Similarly, zero interest expense might signify an unlevered company, not simply missing data.
    """)

    st.markdown(r"The **Jarque-Bera test** is conceptually used to check for normality in distributions, although financial ratios are almost always heavy-tailed. The formula for the Jarque-Bera statistic is:")
    st.markdown(r"$$ \text{{JB}} = \frac{{n}}{{6}} \left( S^2 + \frac{{(K-3)^2}}{{4}} \right) $$")
    st.markdown(r"where $n$ is the number of observations, $S$ is the skewness, and $K$ is the kurtosis. High JB values suggest non-normality, which is typical for financial ratios and often requires transformation or robust methods like winsorization.")

    # Read: st.session_state.raw_df
    if st.session_state.raw_df is None:
        st.warning("Please fetch raw financial data first on the '1. Data Acquisition' page.")
    else:
        if st.button("Assess Data Quality"):
            st.subheader("--- Data Quality Assessment Results ---")
            st.markdown("### 3.1 Missing Value Analysis:")
            missing_pct = st.session_state.raw_df.isnull().mean().sort_values(ascending=False)
            st.write("Percentage of missing values per feature (only > 0%):")
            st.dataframe(missing_pct[missing_pct > 0].round(3))

            st.markdown("### Missing Data Pattern Across Features and Companies:")
            fig, ax = plt.subplots(figsize=(12, 6))
            msno.matrix(st.session_state.raw_df, ax=ax, fontsize=8)
            ax.set_title('Missing Data Pattern Across Features and Companies', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### 3.2 Distributional Summary (Key Percentiles for Outlier Detection):")
            desc = st.session_state.raw_df.describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]).T[['count', 'mean', 'std', '1%', '99%']]
            st.dataframe(desc.round(2))

            st.markdown("### 3.3 Common Financial Data Anomalies:")
            anomalies = pd.DataFrame({
                'negative_revenue': (st.session_state.raw_df['revenue'] < 0).sum(),
                'negative_assets': (st.session_state.raw_df['total_assets'] < 0).sum(),
                'zero_equity': (st.session_state.raw_df['total_equity'] == 0).sum(),
                'negative_equity': (st.session_state.raw_df['total_equity'] < 0).sum(),
                'zero_interest_expense': (st.session_state.raw_df['interest_expense'] == 0).sum(),
                'zero_total_debt': (st.session_state.raw_df['total_debt'] == 0).sum()
            }, index=['count'])
            st.dataframe(anomalies.T)

            st.markdown("### 3.4 Sector Distribution:")
            if 'sector' in st.session_state.raw_df.columns:
                st.dataframe(st.session_state.raw_df['sector'].value_counts())
            else:
                st.markdown("Sector column not found for distribution analysis.")
            
            st.markdown(f"**Explanation of Execution:**")
            st.markdown(f"""
            Ava's data quality assessment reveals several critical insights:
            1.  **Missing Value Percentages:** The output clearly shows which financial line items have the highest percentage of missing data. For example, `interest_expense`, `ebit`, or `gross_profit` might be frequently absent. This informs Ava that she'll need a robust imputation strategy.
            2.  **Missing Data Pattern (missingno matrix):** The visualization graphically illustrates the patterns of missingness. Ava can observe if certain features are often missing together, or if entire companies have sparse data. This visual aid helps confirm that simple dropping of rows/columns might lead to significant data loss and bias.
            3.  **Distributional Summary:** The `describe` output, especially the `1%` and `99%` percentiles, highlights the presence of extreme values (potential outliers) in many financial metrics. For instance, a vast difference between the 95th and 99th percentile suggests a heavy-tailed distribution, which is common in financial data and signals the need for outlier treatment.
            4.  **Financial Anomalies:** The count of negative revenues, negative/zero assets, or negative/zero equity values immediately flags potentially problematic data points. As a CFA Charterholder, Ava understands that while some of these (e.g., negative equity for distressed companies) are real, they can cause issues when computing ratios (like ROE becoming undefined or misleading) and require special handling. Zero interest expense means an unlevered firm, which impacts interest coverage ratio calculations.

            This systematic diagnosis confirms Ava's initial concerns about data quality and provides a clear roadmap for the subsequent cleaning steps.
            """)

# 3. Financial Ratio Computation and DuPont Decomposition
elif st.session_state.current_page == "3. Ratio Computation & DuPont Decomposition":
    st.header("3. Financial Ratio Computation and DuPont Decomposition")
    st.markdown(f"""
    With an understanding of the raw data's quality, Ava's next crucial step is to transform these raw financial statement items into meaningful financial ratios. These ratios are the bedrock of fundamental analysis and relative valuation. Instead of just computing a single Return on Equity (ROE), Ava knows that decomposing ROE using the **DuPont Analysis** provides deeper insights into a company's profitability drivers (profit margin, asset turnover, financial leverage). This decomposition helps her understand *why* a company's ROE is high or low, allowing for more nuanced peer comparisons. For instance, a high ROE driven by excessive leverage might signal higher risk compared to a high ROE driven by superior operating margins. This is core CFA-level feature engineering.
    """)

    st.markdown(r"The **DuPont Decomposition of ROE** is given by:")
    st.markdown(r"$$ \text{{ROE}} = \frac{{\text{{Net Income}}}}{{\text{{Revenue}}}} \times \frac{{\text{{Revenue}}}}{{\text{{Total Assets}}}} \times \frac{{\text{{Total Assets}}}}{{\text{{Total Equity}}}} $$")
    st.markdown(r"$$ \text{{ROE}} = \text{{Net Profit Margin}} \times \text{{Asset Turnover}} \times \text{{Equity Multiplier}} $$")
    st.markdown(r"where $\text{{Net Profit Margin}}$ measures operating efficiency, $\text{{Asset Turnover}}$ measures asset use efficiency, and $\text{{Equity Multiplier}}$ measures financial leverage.")

    st.markdown(f"""
    We also need to handle common financial data traps, such as:
    -   **Negative Equity:** If Total Equity is negative, ROE and D/E become undefined or misleading.
    -   **Zero Denominators:** For ratios like Interest Coverage (EBIT / Interest Expense), if Interest Expense is zero, the ratio approaches infinity. We cap such values (e.g., at 100x) or treat them as "unlevered" to prevent extreme distortions.
    """)

    # Read: st.session_state.raw_df
    if st.session_state.raw_df is None:
        st.warning("Please fetch raw financial data first on the '1. Data Acquisition' page.")
    else:
        if st.button("Compute Financial Ratios"):
            with st.spinner("Computing financial ratios and DuPont components..."):
                # Invocation: compute_financial_ratios from source.py
                financial_ratios = compute_financial_ratios(st.session_state.raw_df)
                if not financial_ratios.empty:
                    # Update: st.session_state.financial_ratios_df
                    st.session_state.financial_ratios_df = financial_ratios
                    # Update: st.session_state.ratio_columns
                    st.session_state.ratio_columns = [col for col in financial_ratios.columns 
                                                    if col not in ['ticker', 'sector', 'fiscal_year_end']]
                    st.success(f"Computed {len(st.session_state.ratio_columns)} ratios for {financial_ratios.shape[0]} companies.")
                    st.markdown(f"First 5 rows of computed ratios:")
                    st.dataframe(st.session_state.financial_ratios_df.head())

                    st.markdown(f"**DuPont ROE consistency check (ROE vs. DuPont components product):**")
                    # Read: st.session_state.financial_ratios_df
                    sample_dupont_check = financial_ratios[['ticker', 'roe', 'dupont_roe_check']].dropna().head()
                    st.dataframe(sample_dupont_check)
                    if not sample_dupont_check.empty:
                        st.markdown(f"Average absolute difference between ROE and DuPont check: {(sample_dupont_check['roe'] - sample_dupont_check['dupont_roe_check']).abs().mean():.4f}")
                    else:
                        st.markdown(f"No sufficient data to perform DuPont consistency check.")
                else:
                    st.error("Failed to compute financial ratios.")
            
            st.markdown(f"**Explanation of Execution:**")
            st.markdown(f"""
            Ava now has a comprehensive set of financial ratios for her selected technology companies. The output shows the first few rows of the `financial_ratios` DataFrame, indicating the successful computation of metrics like ROE, P/E, and the DuPont components. The consistency check for DuPont ROE confirms that the decomposition holds true, assuring Ava that these foundational metrics are correctly calculated.

            For Ava, a CFA Charterholder, this is a critical step because:
            -   **Structured Analysis:** Ratios organize complex financial data into digestible metrics (profitability, leverage, liquidity, efficiency, valuation), facilitating systematic comparison.
            -   **Enhanced Insights:** DuPont decomposition allows her to dissect the drivers of ROE, helping her differentiate between a company with strong operational efficiency (high asset turnover) versus one relying heavily on debt (high equity multiplier) for similar headline profitability. This nuance is vital for assessing risk and sustainability of earnings.
            -   **Domain-Specific Handling:** The explicit handling of negative equity and zero denominators (e.g., capping interest coverage) prevents misleading results that could arise from naive ratio calculations, ensuring the integrity of her analysis.

            This cleaned and enriched dataset of ratios is now ready for further refinement through imputation and outlier treatment, moving Ava closer to building robust valuation models.
            """)

# 4. Missing Value Imputation
elif st.session_state.current_page == "4. Missing Value Imputation":
    st.header("4. Missing Value Imputation: Sector-Median Approach")
    st.markdown(f"""
    Despite the robust data acquisition, Ava knows that some financial ratios will inevitably have missing values, especially for smaller or less transparent companies. Simply dropping rows with missing data could lead to significant data loss and sample bias, as companies with incomplete disclosures might be systematically different from those with full data. As an equity analyst, Ava prefers domain-appropriate imputation methods. A missing ROE for a tech company, for instance, is best estimated by the median ROE of its tech sector peers, rather than the overall market median. This **sector-median imputation** preserves the sector-specific characteristics of the data, which is crucial for relative valuation. Critically, Ava needs to track *which* values were imputed, as imputed data inherently carries less certainty than observed data. This information is vital for downstream machine learning models to potentially down-weight these observations.
    """)

    st.markdown(r"The formula for sector-median imputation for a missing feature $x_{{ij}}$ (feature $j$ of company $i$ in sector $s_i$) is:")
    st.markdown(r"$$ \hat{{x}}_{{ij}} = \text{{median}}\{x_{{kj}} : k \in \text{{sector }} s_i, x_{{kj}} \text{{ not missing}}\} $$")
    st.markdown(r"where $\hat{x}_{ij}$ is the imputed value for feature $j$ of company $i$, and the median is taken over all companies $k$ in the same sector $s_i$ for which $x_{kj}$ is not missing.")

    # Read: st.session_state.financial_ratios_df
    if st.session_state.financial_ratios_df is None:
        st.warning("Please compute financial ratios first on the '3. Ratio Computation & DuPont Decomposition' page.")
    else:
        if st.button("Impute Missing Values"):
            with st.spinner("Applying sector-median imputation..."):
                # Invocation: sector_median_impute from source.py
                # Read: st.session_state.ratio_columns
                financial_ratios_imputed = sector_median_impute(st.session_state.financial_ratios_df, st.session_state.ratio_columns, 'sector')
                if not financial_ratios_imputed.empty:
                    # Update: st.session_state.financial_ratios_imputed_df
                    st.session_state.financial_ratios_imputed_df = financial_ratios_imputed
                    st.success(f"Missing values imputed. Data shape: {st.session_state.financial_ratios_imputed_df.shape}")

                    imputation_flags_cols = [f'{col}_imputed' for col in st.session_state.ratio_columns if f'{col}_imputed' in st.session_state.financial_ratios_imputed_df.columns]
                    if imputation_flags_cols:
                        # Read: st.session_state.financial_ratios_imputed_df
                        imputation_rate = st.session_state.financial_ratios_imputed_df[imputation_flags_cols].mean().sort_values(ascending=False)
                        st.markdown(f"**Imputation rate per feature (only > 0%):**")
                        st.dataframe(imputation_rate[imputation_rate > 0].round(3))
                    else:
                        st.markdown(f"No imputation flags found, or no values were imputed.")
                    
                    st.markdown(f"First 5 rows of imputed data with flags:")
                    st.dataframe(st.session_state.financial_ratios_imputed_df.head())
                else:
                    st.error("Failed to impute missing values.")
            
            st.markdown(f"**Explanation of Execution:**")
            st.markdown(f"""
            Ava's financial ratios have now been imputed using sector-specific medians, and critically, a set of binary flags indicates exactly which values were filled. The output shows the imputation rates for features that had missing values, confirming the extent of imputation. The `head()` of the `financial_ratios_imputed` DataFrame now includes new columns like `roe_imputed`, `pe_ratio_imputed`, etc., which are `1` if the corresponding ratio was imputed and `0` otherwise.

            For Ava, this step offers several benefits:
            -   **Completeness for Analysis:** She now has a more complete dataset, allowing her to include companies that previously had partial data in her relative valuation screens without discarding valuable observations.
            -   **Preservation of Sector Context:** Using sector medians ensures that imputed values are financially sensible within a company's peer group, maintaining the integrity of her relative valuation analysis.
            -   **Transparency and Robustness:** The imputation flags provide crucial metadata. For any quantitative model Ava might use later, these flags allow the model to learn that imputed values might be "less real" or carry higher uncertainty, preventing overconfidence in predictions based on filled data. This is a best practice from production credit scoring that enhances model robustness and interpretability.

            This moves Ava closer to a clean, analysis-ready dataset, with a clear understanding of data quality at each step.
            """)

# 5. Outlier Treatment (Winsorization)
elif st.session_state.current_page == "5. Outlier Treatment (Winsorization)":
    st.header("5. Outlier Detection and Treatment: Winsorization")
    st.markdown(f"""
    Even after imputing missing values, Ava knows that financial ratios are often characterized by heavy-tailed distributions. Extreme values, while sometimes genuine (e.g., a highly leveraged company), can disproportionately influence statistical measures (like mean) and bias her comparative analysis. For example, a single company with an astronomically high P/E ratio due to transient low earnings can skew the entire sector's average P/E. Simply removing these "outliers" might discard valuable information. Instead, Ava opts for **winsorization**, a method that caps extreme values at a specified percentile (e.g., the 1st and 99th percentiles) while retaining the bulk of the distribution. This approach mitigates the impact of extreme values without outright deleting legitimate, albeit unusual, observations. This ensures that her relative valuation comparisons are robust and not unduly influenced by a few anomalous data points.
    """)

    st.markdown(r"**Winsorization** at the $\alpha$-level replaces extreme values with the $\alpha$-th percentile boundary:")
    st.markdown(r"$$ x_i^{{(w)}} = \begin{{cases}} q_\alpha & \text{{if }} x_i < q_\alpha \\ q_{{1-\alpha}} & \text{{if }} x_i > q_{{1-\alpha}} \\ x_i & \text{{otherwise}} \end{{cases}} $$")
    st.markdown(r"where $q_\alpha$ is the $\alpha$-th sample quantile, $q_{{1-\alpha}}$ is the $(1-\alpha)$-th sample quantile, and $x_i^{{(w)}}$ is the winsorized value of $x_i$.")

    # Read: st.session_state.financial_ratios_imputed_df
    if st.session_state.financial_ratios_imputed_df is None:
        st.warning("Please impute missing values first on the '4. Missing Value Imputation' page.")
    else:
        # Widgets for winsorization limits
        lower_limit_input = st.number_input("Lower Winsorization Percentile (e.g., 0.01 for 1st percentile):", min_value=0.0, max_value=0.49, value=st.session_state.winsorization_limits[0], step=0.005, format="%.3f")
        upper_limit_input = st.number_input("Upper Winsorization Percentile (e.g., 0.01 for 99th percentile):", min_value=0.0, max_value=0.49, value=st.session_state.winsorization_limits[1], step=0.005, format="%.3f")
        
        # Update: st.session_state.winsorization_limits
        st.session_state.winsorization_limits = (lower_limit_input, upper_limit_input)

        if st.button("Apply Winsorization"):
            with st.spinner("Applying winsorization to financial ratios..."):
                # Invocation: winsorize_ratios from source.py
                # Read: st.session_state.ratio_columns, st.session_state.winsorization_limits
                financial_ratios_winsorized, detected_outlier_counts = winsorize_ratios(
                    st.session_state.financial_ratios_imputed_df, 
                    st.session_state.ratio_columns, 
                    limits=st.session_state.winsorization_limits
                )
                if not financial_ratios_winsorized.empty:
                    # Update: st.session_state.financial_ratios_winsorized_df
                    st.session_state.financial_ratios_winsorized_df = financial_ratios_winsorized
                    # Update: st.session_state.detected_outlier_counts
                    st.session_state.detected_outlier_counts = detected_outlier_counts
                    st.success(f"Ratios winsorized. Data shape: {st.session_state.financial_ratios_winsorized_df.shape}")

                    # Read: st.session_state.detected_outlier_counts
                    outlier_summary = pd.Series(st.session_state.detected_outlier_counts)
                    st.markdown(f"**Number of values detected as outliers (and winsorized) per feature (initial counts):**")
                    st.dataframe(outlier_summary[outlier_summary > 0].sort_values(ascending=False))

                    st.markdown(f"First 5 rows of winsorized data:")
                    st.dataframe(st.session_state.financial_ratios_winsorized_df.head())

                    st.markdown(f"### Ratio Distributions Before vs. After Winsorization ({lower_limit_input*100:.0f}th/{ (1-upper_limit_input)*100:.0f}th Percentile)")
                    key_ratios_for_viz = ['pe_ratio', 'roe', 'debt_to_equity', 'net_margin', 'interest_coverage', 'asset_turnover']
                    
                    # Ensure key_ratios_for_viz are present in both dataframes before plotting
                    valid_key_ratios = [
                        col for col in key_ratios_for_viz 
                        if col in st.session_state.financial_ratios_imputed_df.columns and 
                           col in st.session_state.financial_ratios_winsorized_df.columns
                    ]
                    
                    if valid_key_ratios:
                        fig, axes = plt.subplots(len(valid_key_ratios), 2, figsize=(15, 4 * len(valid_key_ratios)))
                        fig.suptitle('Ratio Distributions Before vs. After Winsorization', fontsize=16, y=1.02)

                        for i, col in enumerate(valid_key_ratios):
                            # Before Winsorization (Read: st.session_state.financial_ratios_imputed_df)
                            sns.histplot(st.session_state.financial_ratios_imputed_df[col].dropna(), bins=50, kde=True, ax=axes[i, 0], color='skyblue', alpha=0.7)
                            axes[i, 0].axvline(st.session_state.financial_ratios_imputed_df[col].median(), color='red', linestyle='--', label='Median')
                            axes[i, 0].set_title(f'Original: {col}')
                            axes[i, 0].legend()

                            # After Winsorization (Read: st.session_state.financial_ratios_winsorized_df)
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
            
            st.markdown(f"**Explanation of Execution:**")
            st.markdown(f"""
            Ava's data has now undergone outlier treatment through winsorization. The output shows the number of values that were detected and capped for each ratio, indicating where extreme values were most prevalent. The `head()` of the `financial_ratios_winsorized` DataFrame confirms the process.

            Most importantly, the **Before/After Distribution plots** visually demonstrate the effect of winsorization. Ava can observe:
            -   **Tail Compression:** The histograms for ratios like P/E or Debt/Equity clearly show that the extreme tails of the distribution are pulled in towards the 1st and 99th percentiles. This reduces the undue influence of highly unusual observations.
            -   **Preservation of Shape:** The core shape and central tendency of the distributions remain largely intact, confirming that the method retains the overall signal while mitigating noise from extremes.

            For Ava, an equity analyst, this step is crucial because:
            -   **Robust Peer Comparisons:** Winsorization prevents a few outlier companies from distorting sector averages or medians, allowing for more reliable relative valuation. Her scatter plots and comparison tables will now reflect the typical range of multiples, making it easier to identify truly undervalued or overvalued companies.
            -   **Prevention of Model Bias:** If this data were fed into a quantitative model, winsorization would prevent extreme values from dominating the model's learning process, leading to more stable and generalizable predictions.
            -   **Domain-Appropriate Handling:** Unlike simply dropping outliers (which can remove genuine information about highly leveraged or highly profitable firms), winsorization acknowledges that these firms exist but caps their extreme influence, a nuanced approach fitting financial analysis.

            With imputed and winsorized data, Ava is now one step closer to producing a truly robust and analysis-ready dataset for her investment decisions.
            """)

# 6. Point-in-Time Alignment & Export
elif st.session_state.current_page == "6. Point-in-Time Alignment & Export":
    st.header("6. Point-in-Time Alignment and Export")
    st.markdown(f"""
    Ava understands a critical pitfall in financial data analysis: **look-ahead bias**. When building historical models or backtesting strategies, it's easy to inadvertently use information that wasn't actually available to investors at the time of a decision. Financial statements are typically filed with a lag after the fiscal year-end (e.g., a 10-K report for December 31st fiscal year might not be publicly available until March of the following year). Using the "as-of" fiscal year-end date for a prediction made in January introduces look-ahead bias, inflating simulated returns. To prevent this, Ava must implement **Point-in-Time (PIT) alignment**. She will add a `pit_date` column, representing the earliest date a financial record would have been publicly available, accounting for a typical filing lag (e.g., 90 days for annual reports). This ensures that any subsequent analysis or model training uses only information that was truly available at each decision point, creating a realistic and robust dataset for investment research.
    """)

    st.markdown(r"The `pit_date` is calculated as:")
    st.markdown(r"$$ \text{{pit\_date}} = \text{{fiscal\_year\_end}} + \text{{filing\_lag\_days}} $$")
    st.markdown(r"where $\text{{fiscal\_year\_end}}$ is the company's fiscal year-end date, and $\text{{filing\_lag\_days}}$ is the assumed number of days for the financial report to become publicly available.")

    # Read: st.session_state.financial_ratios_winsorized_df
    if st.session_state.financial_ratios_winsorized_df is None:
        st.warning("Please apply winsorization first on the '5. Outlier Treatment (Winsorization)' page.")
    else:
        # Widget for filing lag
        filing_lag_input = st.number_input("Enter filing lag in days (e.g., 90 for annual reports):", min_value=0, max_value=365, value=st.session_state.filing_lag_days, step=1)
        # Update: st.session_state.filing_lag_days
        st.session_state.filing_lag_days = filing_lag_input

        if st.button("Apply Point-in-Time Alignment"):
            with st.spinner("Applying Point-in-Time alignment..."):
                # Invocation: add_pit_date from source.py
                # Read: st.session_state.filing_lag_days
                final_data = add_pit_date(st.session_state.financial_ratios_winsorized_df, filing_lag_days=st.session_state.filing_lag_days)
                if not final_data.empty:
                    # Update: st.session_state.final_analysis_data
                    st.session_state.final_analysis_data = final_data
                    st.success(f"Point-in-Time Alignment applied. Data shape: {st.session_state.final_analysis_data.shape}")
                    st.markdown(f"Assumed filing lag: {st.session_state.filing_lag_days} days.")
                    st.markdown(f"First 5 rows with 'fiscal_year_end' and 'pit_date':")
                    st.dataframe(st.session_state.final_analysis_data[['ticker', 'fiscal_year_end', 'pit_date', 'pe_ratio']].head())
                else:
                    st.error("Failed to apply Point-in-Time alignment.")
            
            st.markdown(f"**Explanation of Execution:**")
            st.markdown(f"""
            Ava has successfully applied Point-in-Time (PIT) alignment to her dataset. The output shows the newly added `pit_date` column, which, for each company, is `filing_lag_days` after its `fiscal_year_end`. For example, a company with a fiscal year-end of December 31st will have its financial data `pit_date` set to approximately March 30th of the following year (assuming a 90-day lag). This crucial step prevents look-ahead bias, ensuring that her valuation models and backtests use only information that was genuinely available at the time.

            For Ava, a CFA Charterholder focused on robust investment research, this is non-negotiable because:
            -   **Valid Backtesting:** Without PIT alignment, any historical simulation of investment strategies would produce artificially inflated returns, leading to overconfidence in a strategy's efficacy. This is a common and costly mistake in quantitative finance.
            -   **Realistic Decision-Making:** She can now confidently use this dataset for building predictive models or performing current relative valuation, knowing that her inputs accurately reflect the information available to the market.
            -   **Foundation for Downstream Models:** The exported `sp500_tech_ratios_clean.csv` is now a clean, analysis-ready feature set, suitable for direct input into machine learning models for tasks like stock screening or factor investing, without needing further data preprocessing.

            This concludes the data engineering phase, providing Ava with a high-quality foundation for her investment analysis.
            """)
    
    # Read: st.session_state.final_analysis_data
    if st.session_state.final_analysis_data is not None:
        st.markdown("### Export Cleaned Data")
        csv_data = st.session_state.final_analysis_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Financial Ratios as CSV",
            data=csv_data,
            file_name="sp500_tech_ratios_clean.csv",
            mime="text/csv"
        )
        st.markdown(f"The cleaned, analysis-ready dataset is now available for download. This dataset contains {st.session_state.final_analysis_data.shape[0]} companies and {st.session_state.final_analysis_data.shape[1]} features.")


# 7. Final Report & Visualizations
elif st.session_state.current_page == "7. Final Report & Visualizations":
    st.header("7. Comprehensive Data Quality Report & Relative Valuation Insights")
    st.markdown(f"""
    Now that Ava has meticulously cleaned, imputed, winsorized, and point-in-time aligned her financial data, she needs to summarize the results of her efforts and derive actionable insights for relative valuation. A comprehensive data quality report not only validates her process but also provides transparency for her team. Furthermore, she can now generate interactive visualizations and peer comparison tables, which are vital tools for an equity analyst to identify undervalued or overvalued companies within the technology sector. This final step connects all her data engineering work back to her primary goal: making informed investment decisions.
    """)

    st.markdown(f"""
    The **DuPont Stacked Bar Chart** will visually demonstrate the components of ROE for selected companies, allowing Ava to quickly see if a company's high ROE is driven by margins, asset utilization, or leverage. This provides immediate context for relative valuation.
    **Sector Box Plots** will illustrate the distribution of key ratios across sectors (if multiple sectors were included, here we have only tech sector, but can show within tech how different sub-sectors or groups compare if available) and help identify outliers still present, allowing Ava to gauge the typical range for peers.
    A **Correlation Heatmap** will show relationships between different ratios, helping Ava understand potential multicollinearity for future model building.
    """)

    # Read: st.session_state.final_analysis_data, st.session_state.initial_raw_data_copy
    if st.session_state.final_analysis_data is None:
        st.warning("Please complete previous data cleaning steps up to '6. Point-in-Time Alignment & Export' to generate the final report.")
    elif st.session_state.initial_raw_data_copy is None:
        st.warning("Initial raw data was not saved. Please re-run '1. Data Acquisition' and subsequent steps.")
    else:
        if st.button("Generate Final Report and Visualizations"):
            with st.spinner("Generating data quality report and visualizations..."):
                # Invocation: generate_quality_report from source.py
                # Read: st.session_state.initial_raw_data_copy, st.session_state.final_analysis_data,
                # st.session_state.ratio_columns, st.session_state.detected_outlier_counts,
                # st.session_state.filing_lag_days
                quality_report = generate_quality_report(
                    st.session_state.initial_raw_data_copy,
                    st.session_state.final_analysis_data,
                    st.session_state.ratio_columns,
                    st.session_state.detected_outlier_counts,
                    st.session_state.filing_lag_days
                )

                st.subheader("--- Comprehensive Data Quality Report ---")
                for k, v in quality_report.items():
                    st.markdown(f"**{k}:** {v}")
                
                # Invocation: Helper function generate_final_visualizations
                # Read: st.session_state.final_analysis_data, st.session_state.ratio_columns
                generate_final_visualizations(st.session_state.final_analysis_data, st.session_state.ratio_columns)
            
            st.markdown(f"**Explanation of Execution:**")
            st.markdown(f"""
            Ava now has a comprehensive overview of her cleaned data and actionable insights for relative valuation:

            1.  **Data Quality Report:** The summary report confirms the successful transformation of the raw data. It quantifies the completeness before and after imputation, highlights features that initially had significant missingness, reports the total number of outliers treated, and confirms the PIT alignment status. This report serves as a critical audit trail for Ava and her firm, demonstrating the rigor applied to data preparation.

            2.  **DuPont Stacked Bar Chart:** This visualization provides an immediate, intuitive understanding of what drives ROE for the selected companies. For example, Ava can quickly discern if a high ROE is a result of strong operating margins (e.g., luxury tech brands) or aggressive financial leverage, allowing for a more informed assessment of risk and sustainability.

            3.  **Correlation Heatmap:** The heatmap shows the pairwise relationships between various financial ratios. Ava uses this to understand potential multicollinearity among features, which is important for building future predictive models (e.g., if P/B and P/S are highly correlated, using both might not add much new information).

            4.  **P/E vs. ROE Scatter Plot:** This is a classic relative valuation tool.
                *   Companies in the **bottom-right quadrant** (low P/E, high ROE) might be **undervalued**, potentially overlooked by the market given their strong profitability.
                *   Companies in the **top-left quadrant** (high P/E, low ROE) might be **overvalued**, trading at rich multiples despite weaker profitability.
                *   The `market_cap` as size helps Ava visualize the significance of each company in the sector.
                This plot directly helps Ava identify potential investment candidates for deeper dives.

            5.  **Peer Comparison Table:** The ranked table, here shown by P/E ratio, allows Ava to quickly compare key valuation and profitability metrics across her target universe. She can easily see how a company stacks up against its peers on fundamental measures, providing the final input for her relative valuation calls.

            By following this robust workflow, Ava has transformed messy, raw financial data into a clean, insightful dataset ready for sophisticated analysis. She can now confidently identify investment opportunities, articulate her rationale with data-backed evidence, and streamline a previously tedious and error-prone process, ultimately enhancing Alpha Investments' research capabilities.
            """)
```
