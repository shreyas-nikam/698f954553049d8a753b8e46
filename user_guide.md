id: 698f954553049d8a753b8e46_user_guide
summary: Lab 7: Fundamental Data Analysis (Structured Data) User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Lab 7: Fundamental Data Analysis (Structured Data)

## Introduction to Fundamental Data Analysis with QuLab
Duration: 02:00

<aside class="positive">
This codelab is designed to guide you through a complete data engineering pipeline for fundamental financial analysis. By the end, you will understand how to transform raw, messy financial data into a clean, analysis-ready dataset suitable for robust investment decisions.
</aside>

**Persona:** Ava, a Senior Equity Analyst at Alpha Investments.
**Organization:** Alpha Investments, focused on identifying undervalued technology stocks for their actively managed portfolios.

Ava's role involves meticulously evaluating companies to provide actionable investment recommendations. A core part of her workflow is performing relative valuation, comparing target companies against their peers using various financial multiples. However, she constantly faces a significant challenge: the raw financial data she retrieves is often plagued with inconsistencies. Missing values, extreme outliers, and misaligned reporting dates frequently skew her analysis, leading to unreliable valuation metrics and, potentially, suboptimal investment decisions. Manually cleaning this data for dozens of companies is time-consuming and prone to human error, diverting her attention from deeper analytical work.

This application outlines a systematic and reproducible workflow to address Ava's frustrations. We will build a data pipeline that automatically retrieves, cleans, and standardizes financial statement data for a universe of technology companies. By applying robust data quality measures—including missing value imputation, outlier treatment, and point-in-time alignment—Ava will be able to generate high-quality, comparable valuation metrics, leading to more confident and data-driven investment recommendations. This automation will free up her time for strategic insights, enhancing Alpha Investments' research capabilities and competitive edge.

## 1. Data Acquisition: Retrieving Raw Financial Statement Data
Duration: 03:00

Ava's first step is to gather financial data for her target universe of companies. Historically, this involved manually pulling data from various sources like Bloomberg terminals or company reports, a process notorious for being slow and error-prone. To streamline this, Ava will use `yfinance` to programmatically retrieve key financial statement items for a basket of S&P 500 technology companies. This automation ensures consistency and speed, allowing her to quickly establish a foundational dataset for her valuation models.

We will focus on the technology sector, a high-growth but often volatile segment requiring precise data for accurate valuation.

**Instructions:**
1.  Use the slider to "Select number of S&P 500 Tech Companies to fetch data for:".
2.  Click the "Fetch Raw Financial Data" button.

**Expected Outcome:**
The application will display the initial shape of the raw data and the first 5 rows of the retrieved data. You will observe columns like `ticker`, `sector`, `revenue`, `net_income`, `total_assets`, `total_equity`, and potentially many `NaN` values, indicating missing data.

**Explanation of Execution:**
The application successfully retrieved raw financial data for a subset of S&P 500 technology companies. Ava can now see the initial structure of her data, with key financial metrics and company information. The output of `raw_financial_data.head()` provides a quick snapshot, immediately revealing potential issues like missing values (represented by `NaN`) in certain columns, which is a common challenge with raw financial APIs. This early visibility confirms the need for comprehensive data quality assessment and cleaning, validating Ava's decision to automate this part of her workflow.

## 2. Initial Data Quality Assessment
Duration: 05:00

Before Ava can trust any valuation models built on this data, she needs to understand its quality. She's aware that raw financial data often contains missing values, outliers, and data entry anomalies that can severely distort her analysis. This section focuses on systematically diagnosing these issues using descriptive statistics and visualizations. For an equity analyst, identifying these problems early is crucial to prevent "garbage in, garbage out" scenarios, where flawed data leads to incorrect investment conclusions.

A key concern for Ava is identifying *why* data might be missing or anomalous. For instance, negative equity might indicate a distressed company, not just a data error. Similarly, zero interest expense might signify an unlevered company, not simply missing data.

The **Jarque-Bera test** is conceptually used to check for normality in distributions, although financial ratios are almost always heavy-tailed. The formula for the Jarque-Bera statistic is:
$$ \text{{JB}} = \frac{{n}}{{6}} \left( S^2 + \frac{{(K-3)^2}}{{4}} \right) $$
where $n$ is the number of observations, $S$ is the skewness, and $K$ is the kurtosis. High JB values suggest non-normality, which is typical for financial ratios and often requires transformation or robust methods like winsorization.

**Instructions:**
1.  Ensure you have fetched data in the previous step.
2.  Click the "Assess Data Quality" button.

**Expected Outcome:**
The application will display:
*   Percentage of missing values per feature.
*   A `missingno` matrix visualizing the pattern of missing data.
*   A descriptive summary of key percentiles (1st, 99th) for outlier detection.
*   Counts of common financial data anomalies (e.g., negative revenue, zero equity).
*   Sector distribution of the companies.

**Explanation of Execution:**
Ava's data quality assessment reveals several critical insights:
1.  <b>Missing Value Percentages:</b> The output clearly shows which financial line items have the highest percentage of missing data. For example, `interest_expense`, `ebit`, or `gross_profit` might be frequently absent. This informs Ava that she'll need a robust imputation strategy.
2.  <b>Missing Data Pattern (missingno matrix):</b> The visualization graphically illustrates the patterns of missingness. Ava can observe if certain features are often missing together, or if entire companies have sparse data. This visual aid helps confirm that simple dropping of rows/columns might lead to significant data loss and bias.
3.  <b>Distributional Summary:</b> The `describe` output, especially the `1%` and `99%` percentiles, highlights the presence of extreme values (potential outliers) in many financial metrics. For instance, a vast difference between the 95th and 99th percentile suggests a heavy-tailed distribution, which is common in financial data and signals the need for outlier treatment.
4.  <b>Financial Anomalies:</b> The count of negative revenues, negative/zero assets, or negative/zero equity values immediately flags potentially problematic data points. As a CFA Charterholder, Ava understands that while some of these (e.g., negative equity for distressed companies) are real, they can cause issues when computing ratios (like ROE becoming undefined or misleading) and require special handling. Zero interest expense means an unlevered firm, which impacts interest coverage ratio calculations.

This systematic diagnosis confirms Ava's initial concerns about data quality and provides a clear roadmap for the subsequent cleaning steps.

## 3. Financial Ratio Computation and DuPont Decomposition
Duration: 04:00

With an understanding of the raw data's quality, Ava's next crucial step is to transform these raw financial statement items into meaningful financial ratios. These ratios are the bedrock of fundamental analysis and relative valuation. Instead of just computing a single Return on Equity (ROE), Ava knows that decomposing ROE using the **DuPont Analysis** provides deeper insights into a company's profitability drivers (profit margin, asset turnover, financial leverage). This decomposition helps her understand *why* a company's ROE is high or low, allowing for more nuanced peer comparisons. For instance, a high ROE driven by excessive leverage might signal higher risk compared to a high ROE driven by superior operating margins. This is core CFA-level feature engineering.

The **DuPont Decomposition of ROE** is given by:
$$ \text{{ROE}} = \frac{{\text{{Net Income}}}}{{\text{{Revenue}}}} \times \frac{{\text{{Revenue}}}}{{\text{{Total Assets}}}} \times \frac{{\text{{Total Assets}}}}{{\text{{Total Equity}}}} $$
$$ \text{{ROE}} = \text{{Net Profit Margin}} \times \text{{Asset Turnover}} \times \text{{Equity Multiplier}} $$
where $\text{{Net Profit Margin}}$ measures operating efficiency, $\text{{Asset Turnover}}$ measures asset use efficiency, and $\text{{Equity Multiplier}}$ measures financial leverage.

We also need to handle common financial data traps, such as:
*   **Negative Equity:** If Total Equity is negative, ROE and D/E become undefined or misleading.
*   **Zero Denominators:** For ratios like Interest Coverage (EBIT / Interest Expense), if Interest Expense is zero, the ratio approaches infinity. We cap such values (e.g., at 100x) or treat them as "unlevered" to prevent extreme distortions.

**Instructions:**
1.  Ensure you have completed the previous steps.
2.  Click the "Compute Financial Ratios" button.

**Expected Outcome:**
The application will display the first 5 rows of the DataFrame with newly computed financial ratios, including the DuPont components. A consistency check for DuPont ROE will also be presented.

**Explanation of Execution:**
Ava now has a comprehensive set of financial ratios for her selected technology companies. The output shows the first few rows of the `financial_ratios` DataFrame, indicating the successful computation of metrics like ROE, P/E, and the DuPont components. The consistency check for DuPont ROE confirms that the decomposition holds true, assuring Ava that these foundational metrics are correctly calculated.

For Ava, a CFA Charterholder, this is a critical step because:
*   <b>Structured Analysis:</b> Ratios organize complex financial data into digestible metrics (profitability, leverage, liquidity, efficiency, valuation), facilitating systematic comparison.
*   <b>Enhanced Insights:</b> DuPont decomposition allows her to dissect the drivers of ROE, helping her differentiate between a company with strong operational efficiency (high asset turnover) versus one relying heavily on debt (high equity multiplier) for similar headline profitability. This nuance is vital for assessing risk and sustainability of earnings.
*   <b>Domain-Specific Handling:</b> The explicit handling of negative equity and zero denominators (e.g., capping interest coverage) prevents misleading results that could arise from naive ratio calculations, ensuring the integrity of her analysis.

This cleaned and enriched dataset of ratios is now ready for further refinement through imputation and outlier treatment, moving Ava closer to building robust valuation models.

## 4. Missing Value Imputation: Sector-Median Approach
Duration: 04:00

Despite the robust data acquisition, Ava knows that some financial ratios will inevitably have missing values, especially for smaller or less transparent companies. Simply dropping rows with missing data could lead to significant data loss and sample bias, as companies with incomplete disclosures might be systematically different from those with full data. As an equity analyst, Ava prefers domain-appropriate imputation methods. A missing ROE for a tech company, for instance, is best estimated by the median ROE of its tech sector peers, rather than the overall market median. This **sector-median imputation** preserves the sector-specific characteristics of the data, which is crucial for relative valuation. Critically, Ava needs to track *which* values were imputed, as imputed data inherently carries less certainty than observed data. This information is vital for downstream machine learning models to potentially down-weight these observations.

The formula for sector-median imputation for a missing feature $x_{{ij}}$ (feature $j$ of company $i$ in sector $s_i$) is:
$$ \hat{{x}}_{{ij}} = \text{{median}}\{x_{{kj}} : k \in \text{{sector }} s_i, x_{{kj}} \text{{ not missing}}\} $$
where $\hat{x}_{ij}$ is the imputed value for feature $j$ of company $i$, and the median is taken over all companies $k$ in the same sector $s_i$ for which $x_{kj}$ is not missing.

**Instructions:**
1.  Ensure you have computed financial ratios in the previous step.
2.  Click the "Impute Missing Values" button.

**Expected Outcome:**
The application will display the imputation rate per feature (showing which features had missing values imputed) and the first 5 rows of the imputed data, including new imputation flag columns (e.g., `roe_imputed`).

**Explanation of Execution:**
Ava's financial ratios have now been imputed using sector-specific medians, and critically, a set of binary flags indicates exactly which values were filled. The output shows the imputation rates for features that had missing values, confirming the extent of imputation. The `head()` of the `financial_ratios_imputed` DataFrame now includes new columns like `roe_imputed`, `pe_ratio_imputed`, etc., which are `1` if the corresponding ratio was imputed and `0` otherwise.

For Ava, this step offers several benefits:
*   <b>Completeness for Analysis:</b> She now has a more complete dataset, allowing her to include companies that previously had partial data in her relative valuation screens without discarding valuable observations.
*   <b>Preservation of Sector Context:</b> Using sector medians ensures that imputed values are financially sensible within a company's peer group, maintaining the integrity of her relative valuation analysis.
*   <b>Transparency and Robustness:</b> The imputation flags provide crucial metadata. For any quantitative model Ava might use later, these flags allow the model to learn that imputed values might be "less real" or carry higher uncertainty, preventing overconfidence in predictions based on filled data. This is a best practice from production credit scoring that enhances model robustness and interpretability.

This moves Ava closer to a clean, analysis-ready dataset, with a clear understanding of data quality at each step.

## 5. Outlier Detection and Treatment: Winsorization
Duration: 05:00

Even after imputing missing values, Ava knows that financial ratios are often characterized by heavy-tailed distributions. Extreme values, while sometimes genuine (e.g., a highly leveraged company), can disproportionately influence statistical measures (like mean) and bias her comparative analysis. For example, a single company with an astronomically high P/E ratio due to transient low earnings can skew the entire sector's average P/E. Simply removing these "outliers" might discard valuable information. Instead, Ava opts for **winsorization**, a method that caps extreme values at a specified percentile (e.g., the 1st and 99th percentiles) while retaining the bulk of the distribution. This approach mitigates the impact of extreme values without outright deleting legitimate, albeit unusual, observations. This ensures that her relative valuation comparisons are robust and not unduly influenced by a few anomalous data points.

**Winsorization** at the $\alpha$-level replaces extreme values with the $\alpha$-th percentile boundary:
$$ x_i^{{(w)}} = \begin{{cases}} q_\alpha & \text{{if }} x_i < q_\alpha \\ q_{{1-\alpha}} & \text{{if }} x_i > q_{{1-\alpha}} \\ x_i & \text{{otherwise}} \end{{cases}} $$
where $q_\alpha$ is the $\alpha$-th sample quantile, $q_{{1-\alpha}}$ is the $(1-\alpha)$-th sample quantile, and $x_i^{{(w)}}$ is the winsorized value of $x_i$.

**Instructions:**
1.  Ensure you have imputed missing values in the previous step.
2.  Adjust the "Lower Winsorization Percentile" and "Upper Winsorization Percentile" sliders as desired (default 0.01 for both 1st and 99th percentiles).
3.  Click the "Apply Winsorization" button.

**Expected Outcome:**
The application will display the number of values winsorized for each feature, the first 5 rows of the winsorized data, and distribution plots showing "Before vs. After Winsorization" for key financial ratios.

**Explanation of Execution:**
Ava's data has now undergone outlier treatment through winsorization. The output shows the number of values that were detected and capped for each ratio, indicating where extreme values were most prevalent. The `head()` of the `financial_ratios_winsorized` DataFrame confirms the process.

Most importantly, the <b>Before/After Distribution plots</b> visually demonstrate the effect of winsorization. Ava can observe:
*   <b>Tail Compression:</b> The histograms for ratios like P/E or Debt/Equity clearly show that the extreme tails of the distribution are pulled in towards the 1st and 99th percentiles. This reduces the undue influence of highly unusual observations.
*   <b>Preservation of Shape:</b> The core shape and central tendency of the distributions remain largely intact, confirming that the method retains the overall signal while mitigating noise from extremes.

For Ava, an equity analyst, this step is crucial because:
*   <b>Robust Peer Comparisons:</b> Winsorization prevents a few outlier companies from distorting sector averages or medians, allowing for more reliable relative valuation. Her scatter plots and comparison tables will now reflect the typical range of multiples, making it easier to identify truly undervalued or overvalued companies.
*   <b>Prevention of Model Bias:</b> If this data were fed into a quantitative model, winsorization would prevent extreme values from dominating the model's learning process, leading to more stable and generalizable predictions.
*   <b>Domain-Appropriate Handling:</b> Unlike simply dropping outliers (which can remove genuine information about highly leveraged or highly profitable firms), winsorization acknowledges that these firms exist but caps their extreme influence, a nuanced approach fitting financial analysis.

With imputed and winsorized data, Ava is now one step closer to producing a truly robust and analysis-ready dataset for her investment decisions.

## 6. Point-in-Time Alignment and Export
Duration: 03:00

Ava understands a critical pitfall in financial data analysis: **look-ahead bias**. When building historical models or backtesting strategies, it's easy to inadvertently use information that wasn't actually available to investors at the time of a decision. Financial statements are typically filed with a lag after the fiscal year-end (e.g., a 10-K report for December 31st fiscal year might not be publicly available until March of the following year). Using the "as-of" fiscal year-end date for a prediction made in January introduces look-ahead bias, inflating simulated returns. To prevent this, Ava must implement **Point-in-Time (PIT) alignment**. She will add a `pit_date` column, representing the earliest date a financial record would have been publicly available, accounting for a typical filing lag (e.g., 90 days for annual reports). This ensures that any subsequent analysis or model training uses only information that was truly available at each decision point, creating a realistic and robust dataset for investment research.

The `pit_date` is calculated as:
$$ \text{{pit\_date}} = \text{{fiscal\_year\_end}} + \text{{filing\_lag\_days}} $$
where $\text{{fiscal\_year\_end}}$ is the company's fiscal year-end date, and $\text{{filing\_lag\_days}}$ is the assumed number of days for the financial report to become publicly available.

**Instructions:**
1.  Ensure you have applied winsorization in the previous step.
2.  Adjust the "Enter filing lag in days" as appropriate (default is 90 days).
3.  Click the "Apply Point-in-Time Alignment" button.
4.  Optionally, click "Download Cleaned Financial Ratios as CSV" to save the final dataset.

**Expected Outcome:**
The application will display the first 5 rows of the data including the new `pit_date` column, showing how it relates to the `fiscal_year_end`. The download button will become active, allowing you to save the processed data.

**Explanation of Execution:**
Ava has successfully applied Point-in-Time (PIT) alignment to her dataset. The output shows the newly added `pit_date` column, which, for each company, is `filing_lag_days` after its `fiscal_year_end`. For example, a company with a fiscal year-end of December 31st will have its financial data `pit_date` set to approximately March 30th of the following year (assuming a 90-day lag). This crucial step prevents look-ahead bias, ensuring that her valuation models and backtests use only information that was genuinely available at the time.

For Ava, a CFA Charterholder focused on robust investment research, this is non-negotiable because:
*   <b>Valid Backtesting:</b> Without PIT alignment, any historical simulation of investment strategies would produce artificially inflated returns, leading to overconfidence in a strategy's efficacy. This is a common and costly mistake in quantitative finance.
*   <b>Realistic Decision-Making:</b> She can now confidently use this dataset for building predictive models or performing current relative valuation, knowing that her inputs accurately reflect the information available to the market.
*   <b>Foundation for Downstream Models:</b> The exported `sp500_tech_ratios_clean.csv` is now a clean, analysis-ready feature set, suitable for direct input into machine learning models for tasks like stock screening or factor investing, without needing further data preprocessing.

This concludes the data engineering phase, providing Ava with a high-quality foundation for her investment analysis.

## 7. Comprehensive Data Quality Report & Relative Valuation Insights
Duration: 06:00

Now that Ava has meticulously cleaned, imputed, winsorized, and point-in-time aligned her financial data, she needs to summarize the results of her efforts and derive actionable insights for relative valuation. A comprehensive data quality report not only validates her process but also provides transparency for her team. Furthermore, she can now generate interactive visualizations and peer comparison tables, which are vital tools for an equity analyst to identify undervalued or overvalued companies within the technology sector. This final step connects all her data engineering work back to her primary goal: making informed investment decisions.

The **DuPont Stacked Bar Chart** will visually demonstrate the components of ROE for selected companies, allowing Ava to quickly see if a company's high ROE is driven by margins, asset utilization, or leverage. This provides immediate context for relative valuation.
A **Correlation Heatmap** will show relationships between various computed ratios to understand potential multicollinearity for future model building.
A **P/E vs. ROE Relative Value Scatter Plot** will help identify potentially undervalued (low P/E, high ROE) or overvalued (high P/E, low ROE) companies within the sector.
A **Peer Comparison Table** will provide a ranked list of companies by key valuation and profitability metrics for quick peer comparison.

**Instructions:**
1.  Ensure you have completed all previous data cleaning steps up to '6. Point-in-Time Alignment & Export'.
2.  Click the "Generate Final Report and Visualizations" button.

**Expected Outcome:**
The application will display a summary data quality report detailing the impact of the cleaning steps (completeness, outliers treated). Following this, it will present several key visualizations:
*   DuPont Decomposition Stacked Bar Chart.
*   Correlation Heatmap of Financial Ratios.
*   P/E vs. ROE Relative Value Scatter Plot (sized by Market Cap, colored by Sector).
*   Peer Comparison Table (ranked by P/E Ratio).

**Explanation of Execution:**
Ava now has a comprehensive overview of her cleaned data and actionable insights for relative valuation:

1.  <b>Data Quality Report:</b> The summary report confirms the successful transformation of the raw data. It quantifies the completeness before and after imputation, highlights features that initially had significant missingness, reports the total number of outliers treated, and confirms the PIT alignment status. This report serves as a critical audit trail for Ava and her firm, demonstrating the rigor applied to data preparation.

2.  <b>DuPont Stacked Bar Chart:</b> This visualization provides an immediate, intuitive understanding of what drives ROE for the selected companies. For example, Ava can quickly discern if a high ROE is a result of strong operating margins (e.g., luxury tech brands) or aggressive financial leverage, allowing for a more informed assessment of risk and sustainability.

3.  <b>Correlation Heatmap:</b> The heatmap shows the pairwise relationships between various financial ratios. Ava uses this to understand potential multicollinearity among features, which is important for building future predictive models (e.g., if P/B and P/S are highly correlated, using both might not add much new information).

4.  <b>P/E vs. ROE Scatter Plot:</b> This is a classic relative valuation tool.
    *   Companies in the **bottom-right quadrant** (low P/E, high ROE) might be **undervalued**, potentially overlooked by the market given their strong profitability.
    *   Companies in the **top-left quadrant** (high P/E, low ROE) might be **overvalued**, trading at rich multiples despite weaker profitability.
    *   The `market_cap` as size helps Ava visualize the significance of each company in the sector.
    This plot directly helps Ava identify potential investment candidates for deeper dives.

5.  <b>Peer Comparison Table:</b> The ranked table, here shown by P/E ratio, allows Ava to quickly compare key valuation and profitability metrics across her target universe. She can easily see how a company stacks up against its peers on fundamental measures, providing the final input for her relative valuation calls.

By following this robust workflow, Ava has transformed messy, raw financial data into a clean, insightful dataset ready for sophisticated analysis. She can now confidently identify investment opportunities, articulate her rationale with data-backed evidence, and streamline a previously tedious and error-prone process, ultimately enhancing Alpha Investments' research capabilities.
