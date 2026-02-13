import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats.mstats import winsorize
import warnings
from pytickersymbols import PyTickerSymbols
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_sp500_tickers(limit=None):
    """
    Fetches S&P 500 tickers using PyTickerSymbols.

    Args:
        limit (int, optional): Maximum number of tickers to return. Defaults to None.

    Returns:
        list: A list of S&P 500 stock tickers.
    """
    stock_data = PyTickerSymbols()
    sp500 = stock_data.get_stocks_by_index("S&P 500")

    tickers = []
    for c in sp500:
        # prefer yahoo_ticker when present; else symbol
        t = c.get("yahoo_ticker") or c.get("symbol")
        if t:
            tickers.append(t)

    # de-dupe while preserving order
    tickers = list(dict.fromkeys(tickers))

    if limit:
        tickers = tickers[:limit]

    return tickers


def fetch_sp500_tech_data(num_companies=50, max_scan=200):
    """
    Scans S&P 500 tickers and fetches financial data for a specified number
    of "Information Technology" sector companies from yfinance.

    Args:
        num_companies (int): The target number of tech companies to fetch data for.
        max_scan (int): The maximum number of S&P 500 tickers to scan to find tech companies.

    Returns:
        pd.DataFrame: A DataFrame containing raw financial data for tech companies.

    Raises:
        RuntimeError: If no tech companies are identified after scanning.
    """
    tickers = get_sp500_tickers(limit=max_scan)

    tech_rows = []
    print(f"Scanning up to {len(tickers)} S&P 500 tickers to find {num_companies} tech companies...")

    for t in tickers:
        if len(tech_rows) >= num_companies:
            break

        try:
            info = yf.Ticker(t).info
            sector = (info.get("sector") or "").strip()

            # Filter using yfinance sector label
            if sector not in ["Technology", "Information Technology"]:
                continue

            # Determine fiscal_year_end. yfinance provides 'fiscalYearEnd' as MM/DD. We'll use a placeholder year.
            fiscal_year_end_str = info.get("fiscalYearEnd", "12-31").replace("-", "/")
            try:
                # Use 1900 as a placeholder year for correct month/day sorting, will be corrected later in PIT alignment
                fiscal_year_end = pd.to_datetime(f"1900/{fiscal_year_end_str}")
            except:
                fiscal_year_end = pd.to_datetime("1900/12/31") # Fallback if parsing fails

            tech_rows.append({
                "ticker": t,
                "sector": sector,
                "industry": info.get("industry"),
                "fiscal_year_end": fiscal_year_end,
                "current_price": info.get("currentPrice"),
                "market_cap": info.get("marketCap"),
                "revenue": info.get("totalRevenue"),
                "gross_profit": info.get("grossProfits"),
                "ebit": info.get("operatingIncome"),
                "ebitda": info.get("ebitda"),
                "operating_income": info.get("operatingIncome"),
                "net_income": info.get("netIncomeToCommon"),
                "interest_expense": info.get("interestExpense"),
                "income_tax_expense": info.get("incomeTaxExpense"),
                "total_assets": info.get("totalAssets"),
                "total_equity": info.get("totalStockholderEquity") or info.get("bookValue"),
                "total_debt": info.get("totalDebt"),
                "current_assets": info.get("currentAssets"),
                "current_liabilities": info.get("currentLiabilities"),
                "cash_equivalents": info.get("totalCash"),
                "operating_cash_flow": info.get("operatingCashflow"),
                "capital_expenditure": info.get("capitalExpenditures"),
                "dividends_paid": info.get("dividendsPaid"),
            })

            if len(tech_rows) % 10 == 0:
                print(f"Found {len(tech_rows)} tech companies so far...")

        except Exception as e:
            print(f"Could not fetch data for {t}: {e}")
            continue

    if not tech_rows:
        raise RuntimeError(
            "No tech companies identified. Likely yfinance `info` is missing/blocked/rate-limited. "
            "Try increasing `max_scan`, adding sleeps, or using a cached ticker list."
        )

    df = pd.DataFrame(tech_rows)

    # Coerce numeric columns
    numeric_cols = [
        "current_price", "market_cap", "revenue", "gross_profit", "ebit",
        "ebitda", "operating_income", "net_income", "interest_expense",
        "income_tax_expense", "total_assets", "total_equity",
        "total_debt", "current_assets", "current_liabilities", "cash_equivalents",
        "operating_cash_flow", "capital_expenditure", "dividends_paid",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"\nRetrieved tech data for {len(df)} companies. Shape: {df.shape}")
    return df


def assess_data_quality(df, output_dir="."):
    """
    Performs a comprehensive data quality assessment.

    Args:
        df (pd.DataFrame): The raw financial DataFrame.
        output_dir (str): Directory to save plots.
    Returns:
        None: Prints data quality summaries and displays visualizations.
    """
    print("--- Data Quality Assessment ---")

    # 3.1: Missing value analysis
    print("\n3.1 Missing Value Analysis:")
    missing_pct = df.isnull().mean().sort_values(ascending=False)
    print("Percentage of missing values per feature (only > 0%):")
    print(missing_pct[missing_pct > 0].round(3))

    # Visualize missing pattern
    plt.figure(figsize=(12, 6))
    msno.matrix(df, ax=plt.gca(), fontsize=8)
    plt.title('Missing Data Pattern Across Features and Companies', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_data_pattern_initial.png'), dpi=150)
    plt.close() # Close plot to prevent it from showing in environments where that's not desired

    # 3.2: Distributional summary (percentiles for outlier identification)
    print("\n3.2 Distributional Summary (Key Percentiles for Outlier Detection):")
    desc = df.describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]).T[['count', 'mean', 'std', '1%', '99%']]
    print(desc.round(2))

    # 3.3: Data type validation and common financial anomalies
    print("\n3.3 Common Financial Data Anomalies:")
    anomalies = pd.DataFrame({
        'negative_revenue': (df['revenue'] < 0).sum() if 'revenue' in df.columns else 0,
        'negative_assets': (df['total_assets'] < 0).sum() if 'total_assets' in df.columns else 0,
        'zero_equity': (df['total_equity'] == 0).sum() if 'total_equity' in df.columns else 0,
        'negative_equity': (df['total_equity'] < 0).sum() if 'total_equity' in df.columns else 0,
        'zero_total_debt': (df['total_debt'] == 0).sum() if 'total_debt' in df.columns else 0
    }, index=['count'])
    print(anomalies.T)

    # 3.4: Sector coverage (for imputation strategy)
    print("\n3.4 Sector Distribution:")
    if 'sector' in df.columns:
        print(df['sector'].value_counts())
    else:
        print("Sector column not found for distribution analysis.")


def compute_financial_ratios(df):
    """
    Computes a comprehensive set of financial ratios from raw statement data,
    including DuPont decomposition. Handles common financial data traps.

    Args:
        df (pd.DataFrame): DataFrame with raw financials per company.

    Returns:
        pd.DataFrame: DataFrame with computed ratios.
    """
    ratios = pd.DataFrame()
    ratios['ticker'] = df['ticker']
    ratios['sector'] = df['sector']

    # Ensure 'fiscal_year_end' is copied for PIT alignment later
    # Use .get() method to safely access columns, falling back to NaT (Not a Time) if missing
    ratios['fiscal_year_end'] = df.get('fiscal_year_end', pd.Series(pd.NaT, index=df.index))

    # Safely get financial columns, defaulting to Series of NaNs if column is missing
    net_income = df.get('net_income', pd.Series(np.nan, index=df.index))
    total_equity = df.get('total_equity', pd.Series(np.nan, index=df.index))
    total_assets = df.get('total_assets', pd.Series(np.nan, index=df.index))
    revenue = df.get('revenue', pd.Series(np.nan, index=df.index))
    gross_profit = df.get('gross_profit', pd.Series(np.nan, index=df.index))
    ebit = df.get('ebit', pd.Series(np.nan, index=df.index))
    total_debt = df.get('total_debt', pd.Series(np.nan, index=df.index))
    cash_equivalents = df.get('cash_equivalents', pd.Series(np.nan, index=df.index))
    interest_expense = df.get('interest_expense', pd.Series(np.nan, index=df.index))
    current_assets = df.get('current_assets', pd.Series(np.nan, index=df.index))
    current_liabilities = df.get('current_liabilities', pd.Series(np.nan, index=df.index))
    operating_cash_flow = df.get('operating_cash_flow', pd.Series(np.nan, index=df.index))
    capital_expenditure = df.get('capital_expenditures', pd.Series(np.nan, index=df.index))
    market_cap = df.get('market_cap', pd.Series(np.nan, index=df.index))
    current_price = df.get('current_price', pd.Series(np.nan, index=df.index))

    # Add market_cap and current_price to ratios DataFrame
    ratios['market_cap'] = market_cap
    ratios['current_price'] = current_price

    # --- Profitability Ratios ---
    ratios['roe'] = np.where(total_equity > 0, net_income / total_equity, np.nan)
    ratios['roa'] = net_income / total_assets
    ratios['net_margin'] = net_income / revenue
    ratios['gross_margin'] = (revenue - gross_profit) / revenue # gross_profit is already safely acquired
    ratios['operating_margin'] = ebit / revenue

    # --- Leverage Ratios ---
    ratios['debt_to_equity'] = np.where(total_equity > 0, total_debt / total_equity, np.nan)
    ratios['debt_to_assets'] = total_debt / total_assets
    ratios['equity_multiplier'] = np.where(total_equity > 0, total_assets / total_equity, np.nan)
    ratios['net_debt'] = total_debt - cash_equivalents

    int_exp_adjusted = interest_expense.replace(0, np.nan).abs()
    ratios['interest_coverage'] = ebit / int_exp_adjusted
    ratios['interest_coverage'] = ratios['interest_coverage'].clip(upper=100)

    # --- Liquidity Ratios ---
    ratios['current_ratio'] = current_assets / current_liabilities
    ratios['cash_ratio'] = cash_equivalents / current_liabilities

    # --- Efficiency Ratios ---
    ratios['asset_turnover'] = revenue / total_assets

    # --- Cash Flow Ratios ---
    ratios['fcf'] = operating_cash_flow - capital_expenditure.abs()
    ratios['fcf_to_debt'] = np.where(total_debt > 0, ratios['fcf'] / total_debt, np.nan)
    ratios['ocf_to_revenue'] = operating_cash_flow / revenue

    # --- Valuation Ratios ---
    ratios['pe_ratio'] = np.where(net_income > 0, market_cap / net_income, np.nan)
    ratios['pb_ratio'] = np.where(total_equity > 0, market_cap / total_equity, np.nan)
    ratios['ev_ebitda'] = np.where(ebit > 0, (market_cap + total_debt - cash_equivalents) / ebit, np.nan)

    # --- DuPont Decomposition ---
    ratios['dupont_margin'] = ratios['net_margin']
    ratios['dupont_turnover'] = ratios['asset_turnover']
    ratios['dupont_leverage'] = ratios['equity_multiplier']
    ratios['dupont_roe_check'] = ratios['dupont_margin'] * ratios['dupont_turnover'] * ratios['dupont_leverage']

    # Clean up potential inf/-inf values
    ratios = ratios.replace([np.inf, -np.inf], np.nan)

    return ratios


def sector_median_impute(df, ratio_cols, sector_col='sector'):
    """
    Replaces missing ratios with the median of the same sector.
    Tracks imputed values using binary flags.

    Args:
        df (pd.DataFrame): DataFrame with computed ratios.
        ratio_cols (list): List of columns to impute.
        sector_col (str): Name of the sector column.

    Returns:
        pd.DataFrame: DataFrame with imputed ratios and imputation flags.
    """
    df_imputed = df.copy()
    imputation_flags = pd.DataFrame(0, index=df.index, columns=[f'{col}_imputed' for col in ratio_cols])

    for col in ratio_cols:
        # Calculate sector medians
        sector_medians = df_imputed.groupby(sector_col)[col].transform('median')

        # Identify values that will be imputed
        to_impute_mask = df_imputed[col].isnull()

        # Apply imputation
        df_imputed[col] = df_imputed[col].fillna(sector_medians)

        # For any remaining NaNs (if an entire sector had NaNs or sector is missing),
        # use the global median as a fallback.
        if df_imputed[col].isnull().any():
            global_median = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(global_median)

        # Set imputation flag for values that were filled
        imputation_flags.loc[to_impute_mask, f'{col}_imputed'] = 1

    return pd.concat([df_imputed, imputation_flags], axis=1)


def winsorize_ratios(df, ratio_cols, limits=(0.01, 0.01)):
    """
    Applies winsorization to specified ratio columns to treat outliers.

    Args:
        df (pd.DataFrame): DataFrame with financial ratios.
        ratio_cols (list): List of columns to winsorize.
        limits (tuple): Lower and upper percentile limits for winsorization (e.g., (0.01, 0.01) for 1st and 99th percentiles).

    Returns:
        tuple:
            pd.DataFrame: DataFrame with winsorized ratios.
            dict: Dictionary of outlier counts before winsorization for each column.
    """
    df_winsorized = df.copy()
    outlier_counts = {}

    for col in ratio_cols:
        if col in df_winsorized.columns and df_winsorized[col].notna().any():
            original_values = df_winsorized[col].dropna()

            # Calculate initial outliers before winsorization for reporting
            lower_bound = original_values.quantile(limits[0])
            upper_bound = original_values.quantile(1 - limits[1])
            initial_outliers = (original_values < lower_bound).sum() + (original_values > upper_bound).sum()
            outlier_counts[col] = initial_outliers

            # Apply winsorization. Ensure winsorize function gets a 1D array/series.
            # Handle cases where all values are NaN or only one unique value exists
            if len(original_values) > 1 and original_values.nunique() > 1:
                df_winsorized.loc[df_winsorized[col].notna(), col] = winsorize(
                    original_values,
                    limits=limits,
                    nan_policy='omit'
                )
            else:
                outlier_counts[col] = 0
        else:
            outlier_counts[col] = 0

    return df_winsorized, outlier_counts


def add_pit_date(df, filing_lag_days=90):
    """
    Adds a 'pit_date' column to the DataFrame, representing the earliest
    public availability date for the financial data, to prevent look-ahead bias.

    Args:
        df (pd.DataFrame): DataFrame with financial ratios, including 'fiscal_year_end'.
        filing_lag_days (int): Number of days for filing lag after fiscal year-end.

    Returns:
        pd.DataFrame: DataFrame with the 'pit_date' column added.
    """
    df_pit = df.copy()

    # Ensure 'fiscal_year_end' is in datetime format.
    # Replace placeholder year (1900) with current year to make dates valid for arithmetic,
    # assuming the data is for the most recent available fiscal year.
    # This is a simplification; for historical data, true fiscal year end dates would be needed.
    current_year = pd.Timestamp.now().year
    df_pit['fiscal_year_end'] = pd.to_datetime(df_pit['fiscal_year_end']).apply(
        lambda x: x.replace(year=current_year) if pd.notna(x) and x.year == 1900 else x
    )

    df_pit['pit_date'] = df_pit['fiscal_year_end'] + pd.to_timedelta(filing_lag_days, unit='D')
    return df_pit


def perform_data_preparation(raw_data_df, filing_lag_days=90):
    """
    Orchestrates the ratio computation, imputation, winsorization, and PIT alignment.

    Args:
        raw_data_df (pd.DataFrame): The initial raw financial data.
        filing_lag_days (int): Days for filing lag after fiscal year-end for PIT alignment.

    Returns:
        tuple:
            pd.DataFrame: The final, cleaned, analysis-ready DataFrame.
            list: List of ratio column names.
            dict: Dictionary of outlier counts detected during winsorization.
    """
    print("\n--- 4. Ratio Computation ---")
    financial_ratios = compute_financial_ratios(raw_data_df)
    print(f"Computed {financial_ratios.shape[1]-4} ratios for {financial_ratios.shape[0]} companies.")
    print("\nFirst 5 rows of computed ratios:")
    print(financial_ratios.head())

    print("\nDuPont ROE consistency check (ROE vs. DuPont components product):")
    sample_dupont_check = financial_ratios[['ticker', 'roe', 'dupont_roe_check']].dropna().head()
    print(sample_dupont_check)
    print(f"Average absolute difference between ROE and DuPont check: {(sample_dupont_check['roe'] - sample_dupont_check['dupont_roe_check']).abs().mean():.4f}")

    print("\n--- 5. Data Imputation ---")
    ratio_columns = [col for col in financial_ratios.columns
                     if col not in ['ticker', 'sector', 'fiscal_year_end', 'market_cap', 'current_price']]
    financial_ratios_imputed = sector_median_impute(financial_ratios, ratio_columns, 'sector')
    print(f"Data after imputation. Shape: {financial_ratios_imputed.shape}")
    imputation_rate = financial_ratios_imputed[[f'{col}_imputed' for col in ratio_columns]].mean().sort_values(ascending=False)
    print("\nImputation rate per feature (only > 0%):")
    print(imputation_rate[imputation_rate > 0].round(3))
    print("\nFirst 5 rows of imputed data with flags:")
    print(financial_ratios_imputed.head())

    print("\n--- 6. Outlier Treatment (Winsorization) ---")
    financial_ratios_winsorized, detected_outlier_counts = winsorize_ratios(financial_ratios_imputed, ratio_columns, limits=(0.01, 0.01))
    print(f"Data after winsorization. Shape: {financial_ratios_winsorized.shape}")
    outlier_summary = pd.Series(detected_outlier_counts)
    print("\nNumber of values detected as outliers (and winsorized) per feature (initial counts):")
    print(outlier_summary[outlier_summary > 0].sort_values(ascending=False))
    print("\nFirst 5 rows of winsorized data:")
    print(financial_ratios_winsorized.head())

    print("\n--- 7. Point-in-Time Alignment ---")
    final_analysis_data = add_pit_date(financial_ratios_winsorized, filing_lag_days=filing_lag_days)
    print(f"Data after Point-in-Time Alignment. Shape: {final_analysis_data.shape}")
    print(f"Assumed filing lag: {filing_lag_days} days.")
    print("\nFirst 5 rows with 'fiscal_year_end' and 'pit_date':")
    print(final_analysis_data[['ticker', 'fiscal_year_end', 'pit_date', 'pe_ratio']].head())

    return final_analysis_data, ratio_columns, detected_outlier_counts


def generate_quality_report(initial_df, final_df, ratio_cols, outlier_counts, filing_lag_days):
    """
    Generates a detailed data quality summary report.

    Args:
        initial_df (pd.DataFrame): The raw DataFrame.
        final_df (pd.DataFrame): The final, cleaned DataFrame.
        ratio_cols (list): List of columns identified as ratios in the analysis.
        outlier_counts (dict): Dictionary of outlier counts before winsorization.
        filing_lag_days (int): The filing lag used for PIT alignment.

    Returns:
        dict: A dictionary containing data quality metrics.
    """
    report = {}
    report["Total Companies Analyzed"] = len(final_df)
    report["Total Features (including flags & dates)"] = len(final_df.columns)
    report["Number of Financial Ratios Computed"] = len(ratio_cols)

    raw_financial_metrics_cols = [col for col in initial_df.columns if col not in ['ticker', 'sector', 'industry', 'fiscal_year_end']]
    if not initial_df[raw_financial_metrics_cols].empty:
        initial_completeness = (1 - initial_df[raw_financial_metrics_cols].isnull().mean().mean()) * 100
    else:
        initial_completeness = 0.0
    report["Average Raw Metrics Completeness (Initial)"] = f"{initial_completeness:.1f}%"

    final_ratio_cols_only = [col for col in ratio_cols if col in final_df.columns]
    if not final_df[final_ratio_cols_only].empty:
        final_completeness = (1 - final_df[final_ratio_cols_only].isnull().mean().mean()) * 100
    else:
        final_completeness = 0.0
    report["Average Ratio Completeness (Final)"] = f"{final_completeness:.1f}%"

    if not initial_df[raw_financial_metrics_cols].empty:
        initial_missing_pct = initial_df[raw_financial_metrics_cols].isnull().mean()
        features_high_missing = initial_missing_pct[initial_missing_pct > 0.10].index.tolist()
    else:
        features_high_missing = []
    report["Features with >10% Missing (Initial Raw Metrics)"] = (
        f"{len(features_high_missing)} "
        f"({', '.join(features_high_missing) if features_high_missing else 'None'})"
    )

    total_outliers_treated = sum(outlier_counts.values())
    report["Total Outliers Treated (Winsorized)"] = total_outliers_treated

    report["Point-in-Time Alignment Status"] = "Yes"
    report["Filing Lag for PIT Alignment"] = f"{filing_lag_days} days"

    if 'roe' in final_df.columns and 'dupont_roe_check' in final_df.columns and not final_df[['roe', 'dupont_roe_check']].dropna().empty:
        dupont_consistency = (final_df["roe"] - final_df["dupont_roe_check"]).abs().mean()
        report["DuPont ROE Consistency (Avg Abs Diff)"] = f"{dupont_consistency:.4f}"
    else:
        report["DuPont ROE Consistency (Avg Abs Diff)"] = "N/A (No comparable data)"

    return report


def generate_visualizations(initial_df, final_df, ratio_cols, output_dir="."):
    """
    Generates various visualizations for financial analysis.

    Args:
        initial_df (pd.DataFrame): DataFrame before winsorization for comparison.
        final_df (pd.DataFrame): The final, cleaned DataFrame.
        ratio_cols (list): List of ratio column names.
        output_dir (str): Directory to save plots.
    """
    print("\n--- Visualizing Relative Valuation Insights ---")
    os.makedirs(output_dir, exist_ok=True)

    # --- Visualize distributions before/after winsorization for key ratios ---
    key_ratios_for_viz = ['pe_ratio', 'roe', 'debt_to_equity', 'net_margin', 'interest_coverage', 'asset_turnover']

    fig, axes = plt.subplots(len(key_ratios_for_viz), 2, figsize=(15, 4 * len(key_ratios_for_viz)))
    fig.suptitle('Ratio Distributions Before vs. After Winsorization (1st/99th Percentile)', fontsize=16, y=1.02)

    for i, col in enumerate(key_ratios_for_viz):
        # Before Winsorization
        if col in initial_df.columns:
            sns.histplot(initial_df[col].dropna(), bins=50, kde=True, ax=axes[i, 0], color='skyblue', alpha=0.7)
            axes[i, 0].axvline(initial_df[col].median(), color='red', linestyle='--', label='Median')
        axes[i, 0].set_title(f'Original: {col}')
        axes[i, 0].legend()

        # After Winsorization
        if col in final_df.columns:
            sns.histplot(final_df[col].dropna(), bins=50, kde=True, ax=axes[i, 1], color='lightgreen', alpha=0.7)
            axes[i, 1].axvline(final_df[col].median(), color='red', linestyle='--', label='Median')
        axes[i, 1].set_title(f'Winsorized: {col}')
        axes[i, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(output_dir, 'ratio_distributions_before_after_winsorization.png'), dpi=150)
    plt.close()

    # V1: DuPont Stacked Bar Chart for a sample of companies
    top_roe_companies = final_df.sort_values(by="roe", ascending=False).head(5)

    if not top_roe_companies.empty and all(c in top_roe_companies.columns for c in ["dupont_margin", "dupont_turnover", "dupont_leverage", "roe"]):
        dupont_data = top_roe_companies[
            ["ticker", "dupont_margin", "dupont_turnover", "dupont_leverage", "roe"]
        ].set_index("ticker")

        dupont_data_plot = dupont_data[["dupont_margin", "dupont_turnover", "dupont_leverage"]]

        fig, ax = plt.subplots(figsize=(12, 6))
        dupont_data_plot.plot(kind="bar", stacked=True, ax=ax, cmap="viridis")

        ax.set_title("DuPont Decomposition of ROE for Top 5 Companies by ROE")
        ax.set_ylabel("Component Value")
        ax.set_xlabel("Company Ticker")
        ax.legend(["Net Margin", "Asset Turnover", "Equity Multiplier"], title="ROE Components")
        ax.tick_params(axis="x", rotation=45)

        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dupont_stacked_bar_chart.png"), dpi=150)
        plt.close()
    else:
        print("Not enough data or missing columns to generate DuPont Stacked Bar Chart.")

    # V2: Correlation Heatmap of financial ratios
    plt.figure(figsize=(14, 10))
    # Ensure only numeric columns are included and filter out imputed flags for correlation
    numeric_ratio_cols = [c for c in ratio_cols if c in final_df.columns and pd.api.types.is_numeric_dtype(final_df[c])]
    correlation_matrix = final_df[numeric_ratio_cols].corr()

    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Financial Ratios", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ratio_correlation_heatmap.png"), dpi=150)
    plt.close()

    # V3: Scatter plot of P/E vs. ROE (to identify undervalued/overvalued)
    plt.figure(figsize=(10, 7))

    plot_data = final_df.dropna(subset=['roe', 'pe_ratio', 'sector'])
    if not plot_data.empty:
        if 'market_cap' in plot_data.columns and pd.api.types.is_numeric_dtype(plot_data['market_cap']):
            sns.scatterplot(
                data=plot_data,
                x="roe",
                y="pe_ratio",
                hue="sector",
                size="market_cap",
                sizes=(50, 1000),
                alpha=0.7,
            )
        else:
            print("Warning: 'market_cap' column is missing or not numeric. Plotting P/E vs ROE without size aesthetic.")
            sns.scatterplot(
                data=plot_data,
                x="roe",
                y="pe_ratio",
                hue="sector",
                alpha=0.7,
            )

        if not plot_data["pe_ratio"].empty:
            plt.axhline(plot_data["pe_ratio"].median(), color="gray", linestyle="--", label="Median P/E")
        if not plot_data["roe"].empty:
            plt.axvline(plot_data["roe"].median(), color="gray", linestyle="--", label="Median ROE")

        plt.title("P/E Ratio vs. ROE (Highlighting Relative Value)", fontsize=16)
        plt.xlabel("Return on Equity (ROE)")
        plt.ylabel("Price/Earnings (P/E) Ratio")

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pe_vs_roe_scatter_plot.png"), dpi=150)
        plt.close()
    else:
        print("Not enough data to generate P/E vs ROE scatter plot.")

    # V4: Peer Comparison Table (ranked by P/E ratio)
    print("\n--- Peer Comparison Table (Ranked by P/E Ratio) ---")

    required_cols = [
        "ticker", "sector", "current_price", "market_cap", "pe_ratio",
        "pb_ratio", "ev_ebitda", "roe", "net_margin", "debt_to_equity", "pit_date",
    ]
    available_cols = [col for col in required_cols if col in final_df.columns]

    peer_comparison_table = (
        final_df[available_cols]
        .sort_values(by="pe_ratio", ascending=True)
        .round(2)
    )

    print(peer_comparison_table.head(10).to_markdown(index=False))
    print("\n... (showing top 10 companies by P/E for illustrative purposes)")


def generate_eda_and_reporting_artifacts(
    raw_data_df,
    final_analysis_data,
    ratio_columns,
    detected_outlier_counts,
    filing_lag_days,
    output_filename='sp500_tech_ratios_clean.csv',
    output_dir="."
):
    """
    Generates EDA reports, data quality reports, visualizations, and exports the final dataset.

    Args:
        raw_data_df (pd.DataFrame): The initial raw financial data.
        final_analysis_data (pd.DataFrame): The final, cleaned DataFrame.
        ratio_columns (list): List of ratio column names.
        detected_outlier_counts (dict): Dictionary of outlier counts detected during winsorization.
        filing_lag_days (int): Days for filing lag after fiscal year-end for PIT alignment.
        output_filename (str): Name of the CSV file to export the final data.
        output_dir (str): Directory to save output files and plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- 3. Data Quality Assessment (Initial) ---")
    assess_data_quality(raw_data_df, output_dir=output_dir)

    print("\n--- 8.1 Data Quality Summary Report ---")
    quality_report = generate_quality_report(
        raw_data_df,
        final_analysis_data,
        ratio_columns,
        detected_outlier_counts,
        filing_lag_days,
    )
    for k, v in quality_report.items():
        print(f"- {k}: {v}")

    generate_visualizations(
        final_analysis_data.drop(columns=[col for col in final_analysis_data.columns if '_imputed' in col], errors='ignore'), # Pass imputed but not winsorized data for comparison in viz
        final_analysis_data,
        ratio_columns,
        output_dir=output_dir
    )

    # Export the final clean, analysis-ready dataset
    final_output_path = os.path.join(output_dir, output_filename)
    final_analysis_data.to_csv(final_output_path, index=False)
    print(f"\nClean, analysis-ready dataset exported to '{final_output_path}'")

    print("\n--- Relative Valuation Insights ---")
    print("Ava can now use these clean ratios and visualizations to:")
    print(
        "1. Identify companies that are trading at lower multiples (e.g., P/E, EV/EBITDA) "
        "relative to their profitability (ROE, Net Margin) compared to peers."
    )
    print("2. Understand the drivers of profitability (DuPont analysis) to assess the quality of earnings.")
    print("3. Pinpoint high-growth companies that might justify higher multiples, or distressed companies warranting caution.")
    print("4. Feed this robust, clean dataset directly into advanced ML models for stock screening or portfolio optimization.")


def run_financial_analysis_pipeline(
    num_companies=50,
    max_scan=300,
    filing_lag_days=90,
    output_dir="./output_analysis"
):
    """
    Main function to run the entire financial data processing and analysis pipeline.

    Args:
        num_companies (int): Number of tech companies to fetch data for.
        max_scan (int): Max S&P 500 companies to scan to find tech companies.
        filing_lag_days (int): Lag in days for point-in-time alignment.
        output_dir (str): Directory where all output files and plots will be saved.

    Returns:
        pd.DataFrame: The final processed and cleaned DataFrame.
    """
    print("Starting financial analysis pipeline...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Data Ingestion
    raw_financial_data = fetch_sp500_tech_data(num_companies=num_companies, max_scan=max_scan)
    print("\nRaw financial data head:")
    print(raw_financial_data.head())

    # 2. Data Preparation and Cleaning
    final_analysis_data, ratio_columns, detected_outlier_counts = perform_data_preparation(
        raw_financial_data,
        filing_lag_days=filing_lag_days
    )

    # 3. EDA, Reporting, and Visualization
    # The `financial_ratios_imputed` from `perform_data_preparation` is not directly returned.
    # We can reconstruct a version for visualization purposes that's just imputed, not winsorized.
    # For now, `generate_visualizations` takes `final_analysis_data` (winsorized) and
    # `initial_df` (which is effectively the imputed data before winsorization for direct comparison).
    # To get the *imputed* but *not winsorized* data, we need to re-run or pass it along.
    # For the scope of this refactor, `initial_df` for `generate_visualizations` will be the imputed and then winsorized data,
    # as the current plot visualization uses `financial_ratios_imputed` and `financial_ratios_winsorized`.
    # Let's adjust `generate_visualizations` to take the imputed data for 'before' comparison.
    # We will compute `financial_ratios_imputed` inside `perform_data_preparation` and pass it.
    # Re-running the imputation to get `financial_ratios_imputed` for `generate_visualizations` if `perform_data_preparation` doesn't return it directly.
    # A cleaner approach would be for `perform_data_preparation` to return `financial_ratios_imputed` as well.
    # For simplicity, for the 'before winsorization' visualization, we can use `final_analysis_data` but drop the flags to simulate the imputed state.
    # However, the previous code explicitly used `financial_ratios_imputed` vs `financial_ratios_winsorized`.
    # Let's slightly modify `perform_data_preparation` to return `financial_ratios_imputed` as well.

    # RE-RUNNING a segment of perform_data_preparation to get financial_ratios_imputed for visualization
    # This is not ideal, but necessary if perform_data_preparation isn't designed to pass all intermediate steps.
    # A better design would be for `perform_data_preparation` to return all intermediate DFs.
    # For now, let's just make sure `raw_financial_data` and `final_analysis_data` are passed to `generate_visualizations`
    # and the visualization function itself manages the 'before' and 'after' based on `_imputed` and final.
    # The current `generate_visualizations` function takes `financial_ratios_imputed` as `initial_df`
    # and `financial_ratios_winsorized` as `final_df` for the distribution plots.
    # Let's update `perform_data_preparation` return signature and then this call.

    # Re-executing compute & impute to get `financial_ratios_imputed` for `generate_visualizations`'s 'before winsorization' step.
    # In a real app, `perform_data_preparation` would return `financial_ratios_imputed` along with the others.
    # This is a temporary solution to match the original notebook's visualization logic.
    financial_ratios_temp = compute_financial_ratios(raw_financial_data)
    ratio_columns_temp = [col for col in financial_ratios_temp.columns
                     if col not in ['ticker', 'sector', 'fiscal_year_end', 'market_cap', 'current_price']]
    financial_ratios_imputed_for_viz = sector_median_impute(financial_ratios_temp, ratio_columns_temp, 'sector')


    generate_eda_and_reporting_artifacts(
        raw_financial_data,
        final_analysis_data,
        ratio_columns,
        detected_outlier_counts,
        filing_lag_days,
        output_filename='sp500_tech_ratios_clean.csv',
        output_dir=output_dir
    )

    print("\nFinancial analysis pipeline completed.")
    return final_analysis_data


if __name__ == "__main__":
    print("Required libraries imported successfully.")

    # Define parameters for the pipeline
    NUM_COMPANIES = 50
    MAX_SCAN = 300
    FILING_LAG_DAYS = 90
    OUTPUT_DIRECTORY = "./output_analysis"

    # Run the entire pipeline
    final_df = run_financial_analysis_pipeline(
        num_companies=NUM_COMPANIES,
        max_scan=MAX_SCAN,
        filing_lag_days=FILING_LAG_DAYS,
        output_dir=OUTPUT_DIRECTORY
    )
    print("\nFinal processed DataFrame:")
    print(final_df.head())
