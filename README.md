# QuLab: Lab 7: Fundamental Data Analysis (Structured Data)

## Project Title

**QuLab: Lab 7: Fundamental Data Analysis (Structured Data) - Building a Clean Financial Data Pipeline**

## Project Description

This Streamlit application, developed as part of QuLab's Lab 7, demonstrates a systematic and reproducible workflow for cleaning, transforming, and analyzing structured financial data. It addresses the common challenges faced by financial analysts, such as missing values, extreme outliers, and look-ahead bias, which can lead to unreliable valuation metrics and flawed investment decisions.

The application follows the persona of Ava, a Senior Equity Analyst at Alpha Investments, who needs high-quality, comparable financial data for relative valuation of technology stocks. By automating the data retrieval, cleaning, and standardization processes, this tool enables Ava to generate robust valuation metrics, freeing her to focus on deeper strategic insights.

The pipeline covers several critical data engineering steps:
1.  **Data Acquisition**: Programmatically fetching raw financial statement data.
2.  **Initial Data Quality Assessment**: Diagnosing missing values, anomalies, and distributional properties.
3.  **Ratio Computation & DuPont Decomposition**: Calculating key financial ratios and breaking down Return on Equity (ROE) into its drivers.
4.  **Missing Value Imputation**: Applying domain-appropriate techniques like sector-median imputation.
5.  **Outlier Treatment (Winsorization)**: Mitigating the impact of extreme values without deleting valuable observations.
6.  **Point-in-Time (PIT) Alignment & Export**: Ensuring data accurately reflects information available at a given time to prevent look-ahead bias, and exporting the clean dataset.
7.  **Final Report & Visualizations**: Summarizing the cleaning process and deriving actionable insights through comparative visualizations.

This project emphasizes best practices in financial data preparation, crucial for robust quantitative analysis and informed investment decision-making.

## Features

The application is structured as a multi-page interactive dashboard, guiding users through a comprehensive data cleaning and analysis pipeline:

*   **1. Data Acquisition**:
    *   Fetch raw financial data for a user-specified number of S&P 500 Technology companies using `yfinance`.
    *   Display initial raw data structure.

*   **2. Initial Data Quality Assessment**:
    *   Analyze missing value percentages and patterns using `missingno`.
    *   Provide descriptive statistics with key percentiles for outlier detection.
    *   Identify common financial data anomalies (e.g., negative equity, zero interest expense).
    *   Show sector distribution.

*   **3. Ratio Computation & DuPont Decomposition**:
    *   Compute a comprehensive set of financial ratios (e.g., P/E, ROE, Debt/Equity, Current Ratio, Net Margin).
    *   Perform DuPont Decomposition of ROE into Net Profit Margin, Asset Turnover, and Equity Multiplier.
    *   Handle financial data traps like negative equity and zero denominators.
    *   Display a consistency check for DuPont ROE.

*   **4. Missing Value Imputation**:
    *   Apply sector-median imputation for missing financial ratios.
    *   Generate imputation flags (`_imputed` columns) to track which values were filled.
    *   Report imputation rates per feature.

*   **5. Outlier Treatment (Winsorization)**:
    *   Allow user-defined lower and upper percentile limits for winsorization.
    *   Apply winsorization to cap extreme ratio values.
    *   Report the count of values treated as outliers for each ratio.
    *   Visualize ratio distributions *before* and *after* winsorization for key ratios.

*   **6. Point-in-Time Alignment & Export**:
    *   Calculate `pit_date` (point-in-time date) for each financial record based on fiscal year-end and a user-defined filing lag.
    *   Prevent look-ahead bias for robust historical analysis.
    *   Provide an option to download the fully cleaned and aligned dataset as a CSV.

*   **7. Final Report & Visualizations**:
    *   Generate a comprehensive data quality report summarizing data completeness, imputation efforts, and outlier treatment.
    *   **DuPont Decomposition Stacked Bar Chart**: Visualize ROE components for top companies.
    *   **Correlation Heatmap**: Explore relationships between financial ratios.
    *   **P/E vs. ROE Relative Value Scatter Plot**: Identify potentially undervalued/overvalued companies (sized by market cap, colored by sector).
    *   **Peer Comparison Table**: Display a ranked list of companies by key valuation and profitability metrics.

## Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

Ensure you have the following installed:

*   Python (version 3.8 or higher is recommended)
*   pip (Python package installer)

### Installation

1.  **Clone the repository** (if this project were hosted on GitHub):

    ```bash
    git clone https://github.com/your-username/qualt-lab7-data-analysis.git
    cd qualt-lab7-data-analysis
    ```
    *(If you only have the provided files, create a directory and place `app.py` and `source.py` inside it.)*

2.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required Python packages**:
    Create a `requirements.txt` file in your project directory with the following content:

    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    missingno
    yfinance
    scipy
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Ensure `app.py` and `source.py` are in the same directory.**
    The `source.py` file should contain the implementations for `fetch_sp500_tech_data`, `compute_financial_ratios`, `sector_median_impute`, `winsorize_ratios`, and `add_pit_date`, and `generate_quality_report`.

2.  **Run the Streamlit application**:

    ```bash
    streamlit run app.py
    ```

3.  Your default web browser should automatically open a new tab with the Streamlit application. If not, open your browser and navigate to the URL displayed in your terminal (usually `http://localhost:8501`).

4.  **Navigate through the application**: Use the sidebar navigation to progress through the data analysis pipeline step-by-step. Follow the instructions and click the buttons on each page to execute the data processing tasks.

## Project Structure

```
.
├── app.py                  # Main Streamlit application file
├── source.py               # Contains core data processing functions
├── README.md               # This README file
└── requirements.txt        # List of Python dependencies
```
*(Note: The `source.py` file is critical and assumed to contain the backend logic for data fetching, computation, imputation, winsorization, and PIT alignment.)*

## Technology Stack

*   **Frontend**: Streamlit
*   **Data Manipulation**: Pandas, NumPy
*   **Data Acquisition**: `yfinance` (used within `source.py`)
*   **Data Visualization**: Matplotlib, Seaborn, Missingno
*   **Statistical Operations**: SciPy (for winsorization, potentially within `source.py`)
*   **Programming Language**: Python

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes and ensure the code adheres to a consistent style.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file (if applicable) for details.

```
MIT License

Copyright (c) [Year] [Your Name or Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact

For any questions or inquiries, please reach out via:

*   **GitHub Issues**: [Link to Issues Page](https://github.com/your-username/qualt-lab7-data-analysis/issues) (if applicable)
*   **Email**: [your.email@example.com](mailto:your.email@example.com)
*   **QuantUniversity**: [www.quantuniversity.com](https://www.quantuniversity.com/) (For QuLab-specific queries)

---
**QuantUniversity - QuLab Project**
