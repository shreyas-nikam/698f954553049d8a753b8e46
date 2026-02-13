
import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest
import sys
from unittest.mock import patch, MagicMock

# --- Dummy Data for Mocking Streamlit App State ---
# These dataframes simulate the output of functions in source.py.
# In a real testing environment, you would use actual mocks for source.py functions.
# This setup ensures that AppTest can run without a fully functional source.py.

dummy_raw_df = pd.DataFrame({
    'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    'sector': ['Technology', 'Technology', 'Technology', 'Technology', 'Technology'],
    'fiscal_year_end': pd.to_datetime(['2022-09-24', '2022-06-30', '2022-12-31', '2022-12-31', '2023-01-29']),
    'revenue': [394328, 198270, 282836, 513983, 26974],
    'net_income': [99803, 72738, 59972, 30425, 4368],
    'total_assets': [352755, 364843, 364741, 461875, 54593],
    'total_equity': [60105, 166549, 256860, 160408, 38161],
    'interest_expense': [0, 2038, 200, 1500, 100],
    'ebit': [122346, 83610, 83520, 68000, 9000],
    'gross_profit': [170782, 135620, 157200, 220000, 15000],
    'current_assets': [135405, 184224, 178000, 160000, 30000],
    'current_liabilities': [160290, 165842, 120000, 150000, 15000],
    'total_debt': [153350, 60473, 10000, 100000, 5000],
    'market_cap': [2.8e12, 2.5e12, 1.8e12, 1.5e12, 1.2e12],
    'shares_outstanding': [15.7e9, 7.4e9, 13.1e9, 10.2e9, 2.5e9],
    'price_to_earnings': [25.0, 30.0, 22.0, 45.0, 60.0]
})

dummy_financial_ratios_df = dummy_raw_df.copy()
dummy_financial_ratios_df['roe'] = dummy_financial_ratios_df['net_income'] / dummy_financial_ratios_df['total_equity']
dummy_financial_ratios_df['pe_ratio'] = dummy_financial_ratios_df['price_to_earnings']
dummy_financial_ratios_df['net_margin'] = dummy_financial_ratios_df['net_income'] / dummy_financial_ratios_df['revenue']
dummy_financial_ratios_df['asset_turnover'] = dummy_financial_ratios_df['revenue'] / dummy_financial_ratios_df['total_assets']
dummy_financial_ratios_df['debt_to_equity'] = dummy_financial_ratios_df['total_debt'] / dummy_financial_ratios_df['total_equity']
dummy_financial_ratios_df['current_ratio'] = dummy_financial_ratios_df['current_assets'] / dummy_financial_ratios_df['current_liabilities']
dummy_financial_ratios_df['interest_coverage'] = dummy_financial_ratios_df['ebit'] / dummy_financial_ratios_df['interest_expense'].replace(0, np.nan)
dummy_financial_ratios_df['dupont_margin'] = dummy_financial_ratios_df['net_income'] / dummy_financial_ratios_df['revenue']
dummy_financial_ratios_df['dupont_turnover'] = dummy_financial_ratios_df['revenue'] / dummy_financial_ratios_df['total_assets']
dummy_financial_ratios_df['dupont_leverage'] = dummy_financial_ratios_df['total_assets'] / dummy_financial_ratios_df['total_equity']
dummy_financial_ratios_df['dupont_roe_check'] = dummy_financial_ratios_df['dupont_margin'] * dummy_financial_ratios_df['dupont_turnover'] * dummy_financial_ratios_df['dupont_leverage']

dummy_financial_ratios_df.loc[0, 'pe_ratio'] = np.nan
dummy_financial_ratios_df.loc[2, 'debt_to_equity'] = np.nan

dummy_ratio_columns = [col for col in dummy_financial_ratios_df.columns if col not in ['ticker', 'sector', 'fiscal_year_end', 'market_cap', 'shares_outstanding', 'price_to_earnings', 'net_income', 'revenue', 'total_assets', 'total_equity', 'interest_expense', 'ebit', 'gross_profit', 'current_assets', 'current_liabilities', 'total_debt']]


dummy_financial_ratios_imputed_df = dummy_financial_ratios_df.copy()
median_pe_tech = dummy_financial_ratios_imputed_df[dummy_financial_ratios_imputed_df['sector'] == 'Technology']['pe_ratio'].median()
median_debt_to_equity_tech = dummy_financial_ratios_imputed_df[dummy_financial_ratios_imputed_df['sector'] == 'Technology']['debt_to_equity'].median()

imputed_pe_ratio_flag = dummy_financial_ratios_imputed_df['pe_ratio'].isna()
imputed_debt_to_equity_flag = dummy_financial_ratios_imputed_df['debt_to_equity'].isna()

dummy_financial_ratios_imputed_df['pe_ratio'] = dummy_financial_ratios_imputed_df['pe_ratio'].fillna(median_pe_tech)
dummy_financial_ratios_imputed_df['debt_to_equity'] = dummy_financial_ratios_imputed_df['debt_to_equity'].fillna(median_debt_to_equity_tech)

dummy_financial_ratios_imputed_df['pe_ratio_imputed'] = imputed_pe_ratio_flag.astype(int)
dummy_financial_ratios_imputed_df['debt_to_equity_imputed'] = imputed_debt_to_equity_flag.astype(int)

for col in dummy_ratio_columns:
    if f'{col}_imputed' not in dummy_financial_ratios_imputed_df.columns:
        dummy_financial_ratios_imputed_df[f'{col}_imputed'] = 0

dummy_financial_ratios_winsorized_df = dummy_financial_ratios_imputed_df.copy()
dummy_detected_outlier_counts = {
    'pe_ratio': 1,
    'roe': 0,
    'debt_to_equity': 1,
    'net_margin': 0, 'asset_turnover': 0, 'current_ratio': 0, 'interest_coverage': 0,
    'dupont_margin': 0, 'dupont_turnover': 0, 'dupont_leverage': 0, 'dupont_roe_check': 0
}

dummy_final_analysis_data = dummy_financial_ratios_winsorized_df.copy()
dummy_final_analysis_data['pit_date'] = dummy_final_analysis_data['fiscal_year_end'] + pd.to_timedelta(90, unit='D')


# Mocking the 'source' module to prevent actual external calls during tests
mock_source_module = MagicMock()
mock_source_module.fetch_sp500_tech_data.return_value = dummy_raw_df
mock_source_module.compute_financial_ratios.return_value = dummy_financial_ratios_df
mock_source_module.sector_median_impute.return_value = dummy_financial_ratios_imputed_df
mock_source_module.winsorize_ratios.return_value = (dummy_financial_ratios_winsorized_df, dummy_detected_outlier_counts)
mock_source_module.add_pit_date.return_value = dummy_final_analysis_data
mock_source_module.generate_quality_report.return_value = {
    "Initial Data Completeness (Rows, Cols)": f"({dummy_raw_df.shape[0]}, {dummy_raw_df.shape[1]})",
    "Final Data Completeness (Rows, Cols)": f"({dummy_final_analysis_data.shape[0]}, {dummy_final_analysis_data.shape[1]})",
    "Total Ratios Imputed": sum(dummy_financial_ratios_imputed_df[[col for col in dummy_financial_ratios_imputed_df.columns if '_imputed' in col]].sum()),
    "Total Outliers Winsorized": sum(dummy_detected_outlier_counts.values()),
    "Point-in-Time Alignment Lag": "90 days applied"
}

sys.modules['source'] = mock_source_module

_APP_FILE = "app.py"


def test_initial_page_load_and_introduction():
    """Tests the initial loading of the app and the content of the Introduction page."""
    at = AppTest.from_file(_APP_FILE).run()
    assert at.session_state.current_page == 'Introduction'
    assert "QuLab: Lab 7: Fundamental Data Analysis (Structured Data)" in at.title[0].value
    assert "Ava, a Senior Equity Analyst at Alpha Investments." in at.markdown[0].value
    assert "Alpha Investments, focused on identifying undervalued technology stocks" in at.markdown[1].value


def test_data_acquisition():
    """Tests the data acquisition page, including slider interaction and data fetching."""
    at = AppTest.from_file(_APP_FILE)
    at.sidebar.selectbox[0].set_value("1. Data Acquisition").run()

    at.slider[0].set_value(50).run()
    assert at.session_state.num_companies == 50

    at.button[0].click().run()

    assert at.session_state.raw_df is not None
    assert at.session_state.initial_raw_data_copy is not None
    assert not at.session_state.raw_df.empty
    assert "Successfully retrieved raw data" in at.success[0].value
    assert at.dataframe[0].value.equals(dummy_raw_df.head())


def test_initial_data_quality_assessment():
    """Tests the initial data quality assessment page, including displaying various data quality metrics."""
    at = AppTest.from_file(_APP_FILE)
    at.session_state.raw_df = dummy_raw_df
    at.sidebar.selectbox[0].set_value("2. Initial Data Quality Assessment").run()

    at.button[0].click().run()

    assert "--- Data Quality Assessment Results ---" in at.subheader[0].value
    assert "Percentage of missing values per feature" in at.write[0].value
    assert at.dataframe[0].value.shape[0] > 0
    assert at.pyplot[0].exists
    assert "Distributional Summary" in at.markdown[2].value
    assert at.dataframe[1].value.shape[0] > 0
    assert "Common Financial Data Anomalies" in at.markdown[3].value
    assert at.dataframe[2].value.shape[0] > 0
    assert "Sector Distribution" in at.markdown[4].value
    assert at.dataframe[3].value.shape[0] > 0


def test_ratio_computation():
    """Tests the financial ratio computation and DuPont decomposition page."""
    at = AppTest.from_file(_APP_FILE)
    at.session_state.raw_df = dummy_raw_df
    at.sidebar.selectbox[0].set_value("3. Ratio Computation & DuPont Decomposition").run()

    at.button[0].click().run()

    assert at.session_state.financial_ratios_df is not None
    assert not at.session_state.financial_ratios_df.empty
    assert "Computed 13 ratios" in at.success[0].value
    assert at.dataframe[0].value.equals(dummy_financial_ratios_df.head())
    assert "DuPont ROE consistency check" in at.markdown[2].value
    assert at.dataframe[1].value.shape[0] > 0


def test_missing_value_imputation():
    """Tests the missing value imputation page, verifying imputation and flags."""
    at = AppTest.from_file(_APP_FILE)
    at.session_state.financial_ratios_df = dummy_financial_ratios_df
    at.session_state.ratio_columns = dummy_ratio_columns
    at.sidebar.selectbox[0].set_value("4. Missing Value Imputation").run()

    at.button[0].click().run()

    assert at.session_state.financial_ratios_imputed_df is not None
    assert not at.session_state.financial_ratios_imputed_df.empty
    assert "Missing values imputed. Data shape" in at.success[0].value
    assert at.dataframe[0].value.shape[0] > 0
    assert at.dataframe[1].value.equals(dummy_financial_ratios_imputed_df.head())


def test_outlier_treatment():
    """Tests the outlier treatment (winsorization) page, including limit settings and visualization."""
    at = AppTest.from_file(_APP_FILE)
    at.session_state.financial_ratios_imputed_df = dummy_financial_ratios_imputed_df
    at.session_state.ratio_columns = dummy_ratio_columns
    at.session_state.winsorization_limits = (0.01, 0.01)
    at.sidebar.selectbox[0].set_value("5. Outlier Treatment (Winsorization)").run()

    at.number_input[0].set_value(0.01).run()
    at.number_input[1].set_value(0.01).run()
    assert at.session_state.winsorization_limits == (0.01, 0.01)

    at.button[0].click().run()

    assert at.session_state.financial_ratios_winsorized_df is not None
    assert not at.session_state.financial_ratios_winsorized_df.empty
    assert at.session_state.detected_outlier_counts is not None
    assert "Ratios winsorized. Data shape" in at.success[0].value
    assert at.dataframe[0].value.shape[0] > 0
    assert at.dataframe[1].value.equals(dummy_financial_ratios_winsorized_df.head())
    assert len(at.pyplot) >= 12 # At least 2 plots per key ratio (before/after winsorization)


def test_point_in_time_alignment():
    """Tests the point-in-time alignment and data export page."""
    at = AppTest.from_file(_APP_FILE)
    at.session_state.financial_ratios_winsorized_df = dummy_financial_ratios_winsorized_df
    at.sidebar.selectbox[0].set_value("6. Point-in-Time Alignment & Export").run()

    at.number_input[0].set_value(90).run()
    assert at.session_state.filing_lag_days == 90

    at.button[0].click().run()

    assert at.session_state.final_analysis_data is not None
    assert not at.session_state.final_analysis_data.empty
    assert "Point-in-Time Alignment applied. Data shape" in at.success[0].value
    assert "pit_date" in at.session_state.final_analysis_data.columns
    assert at.dataframe[0].value.equals(dummy_final_analysis_data[['ticker', 'fiscal_year_end', 'pit_date', 'pe_ratio']].head())
    assert at.download_button[0].exists


def test_final_report_and_visualizations():
    """Tests the final report page, including quality report and various visualizations."""
    at = AppTest.from_file(_APP_FILE)
    at.session_state.initial_raw_data_copy = dummy_raw_df
    at.session_state.final_analysis_data = dummy_final_analysis_data
    at.session_state.ratio_columns = dummy_ratio_columns
    at.session_state.detected_outlier_counts = dummy_detected_outlier_counts
    at.session_state.filing_lag_days = 90
    at.sidebar.selectbox[0].set_value("7. Final Report & Visualizations").run()

    at.button[0].click().run()

    assert "--- Comprehensive Data Quality Report ---" in at.subheader[0].value
    assert "Initial Data Completeness" in at.markdown[0].value
    assert "Final Data Completeness" in at.markdown[1].value
    assert "Total Ratios Imputed" in at.markdown[2].value
    assert "Total Outliers Winsorized" in at.markdown[3].value
    assert "Point-in-Time Alignment Lag" in at.markdown[4].value

    assert at.pyplot[0].exists
    assert at.pyplot[1].exists
    assert at.pyplot[2].exists
    assert at.dataframe[0].exists

# Clean up: Restore the original 'source' module after tests are defined
del sys.modules['source']
