# Data Analysis of happiness.csv
This report presents a comprehensive analysis of a dataset comprising 2,363 observations across 14 variables, including a range of socio-economic indicators such as Life Ladder, Log GDP per capita, and measures of social support and happiness across various countries. The dataset spans multiple years, with most data captured around 2014-2023. While the majority of columns are numeric, measures related to social and psychological well-being highlight significant variability and patterns. The analysis also addresses notable missing values in key metrics, such as Generosity and Perceptions of Corruption, which could impact the interpretation of well-being across countries. By exploring these dimensions, the report aims to elucidate the factors influencing life satisfaction and quality of life on a global scale.
- **Dataset Name**: happiness.csv
- **Total Rows**: 2363
- **Total Columns**: 14
- **Data Types**: 
  - Numerical (float64, int64)
  - Categorical (object)
- **Key Columns**: 
  - Country name (object)
  - year (int64)
  - Life Ladder (float64)
  - Log GDP per capita (float64)
  - Social support (float64)
  - Healthy life expectancy at birth (float64)
  - Freedom to make life choices (float64)
  - Generosity (float64)
  - Perceptions of corruption (float64)
  - Positive affect (float64)
  - Negative affect (float64)
  - Anomaly (int64)
  - DBSCAN_Anomaly (int64)
  - SVM_Anomaly (int64)
- **Statistics**: 
  - Mean Life Ladder: 5.48
  - Mean Log GDP per capita: 9.40
  - Mean Healthy life expectancy: (missing data)
- **Missing Data**: 
  - Log GDP per capita: 28 missing values
  - Social support: 13 missing values
  - Healthy life expectancy at birth: 63 missing values
  - Freedom to make life choices: 36 missing values
  - Generosity: 81 missing values
  - Perceptions of corruption: 125 missing values
  - Positive affect: 24 missing values
  - Negative affect: 16 missing values
- **Unique Country Names**: 165

## Contents
- [Missing Values Summary](#missing-values-summary)
- [Anomalies Detected](#anomalies-detected)
- [Graphs](#graphs)
- [Analysis Results](#analysis-results)
- [Recommnedations](#Recommnedations)

## Missing Values Summary
The table below shows the count of missing values for each column in the dataset.
| Column Name                      |   Missing Values |
|:---------------------------------|-----------------:|
| Country name                     |                0 |
| year                             |                0 |
| Life Ladder                      |                0 |
| Log GDP per capita               |               28 |
| Social support                   |               13 |
| Healthy life expectancy at birth |               63 |
| Freedom to make life choices     |               36 |
| Generosity                       |               81 |
| Perceptions of corruption        |              125 |
| Positive affect                  |               24 |
| Negative affect                  |               16 |
| Anomaly                          |                0 |
| DBSCAN_Anomaly                   |                0 |
| SVM_Anomaly                      |                0 |

## Anomalies Detected
Anomalies were detected using three methods. The results are summarized below:

### Isolation Forest
- Number of anomalies detected: **119**
- Method: Identifies anomalies by isolating data points through recursive partitioning.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Number of anomalies detected: **2287**
- Method: Identifies anomalies as points in low-density regions using density-based clustering.

### One-Class SVM (Support Vector Machine)
- Number of anomalies detected: **109**
- Method: Learns a decision boundary to separate normal data points from anomalies.

## Graphs
Here are some key visualizations:
![Histogram](happiness\histogram.png)  

## Analysis Results
### Comprehensive Analysis of the Dataset

1. **Correlations between Numeric and Categorical Variables:**
   - The correlation between **Log GDP per capita** and **Life Ladder** is notably strong (0.78), indicating that higher GDP per capita is associated with higher life satisfaction or happiness.
   - A positive correlation exists between **Healthy life expectancy at birth** and **Life Ladder** (0.71), suggesting that longevity may influence perceived life satisfaction.
   - **Social support** has a strong correlation with **Life Ladder** (0.72), indicating that a feeling of social support can significantly impact life satisfaction.
   - **Freedom to make life choices** also shows a positive correlation with **Life Ladder** (0.54), implying autonomy plays a vital role in well-being.
   - **Generosity** shows weak to negligible correlations with **Life Ladder** (-0.00) and other variables, indicating it may not be a significant predictor of life satisfaction within this dataset.
   - **Perceptions of corruption** exhibit a negative correlation with **Life Ladder** (-0.43), signaling that higher corruption perceptions correlate with lower life satisfaction.
   - The anomaly detection columns (**Anomaly**, **DBSCAN_Anomaly**, and **SVM_Anomaly**) have modest correlations with the Life Ladder and other happiness indicators, which may indicate the effectiveness of these models in capturing deviations from standard patterns.

2. **Feature Distributions for Numerical Data:**
   - **Life Ladder**: Mean = 5.48, Std = 1.12, Range = 1.28 (min) to 8.02 (max).
   - **Log GDP per capita**: Mean = 9.40, Std = 1.15, Range = 5.53 (min) to 11.68 (max). 
   - **Social Support**: Mean = 0.82, Std = 0.16, Range = 0.33 (min) to 1.16 (max).
   - **Healthy life expectancy at birth**: Mean = 73.04, Std = 7.60, Range = 57.10 (min) to 81.80 (max).
   - **Freedom to make life choices**: Mean = 0.63, Std = 0.15, Range = 0.33 (min) to 0.92 (max).
   - **Generosity**: Mean = 0.11, Std = 0.26, Range = -0.50 (min) to 0.85 (max).
   - Potential skewness can be observed in **Generosity**, which has a broad range with many values concentrated towards zero.

3. **Identification of Outliers or Extreme Values:**
   - The maximum value of **SVM_Anomaly** (13.00) indicates a significant potential outlier compared to other anomaly detection features, suggesting unusual deviations.
   - The **Generosity** feature might also present a few outliers due to the observed range of values from negative to positive.
   - Consideration of outlier handling may enhance model accuracy for any further analysis conducted on this dataset.

4. **Trends in Missing Data or Categorical Distributions:**
   - The following features have missing data:
     - **Log GDP per capita**: 28 missing entries
     - **Social support**: 13 missing entries
     - **Healthy life expectancy at birth**: 63 missing entries
     - **Freedom to make life choices**: 36 missing entries
     - **Generosity**: 81 missing entries
     - **Perceptions of corruption**: 125 missing entries
     - **Positive affect**: 24 missing entries
     - **Negative affect**: 16 missing entries
   - Handling these missing values may be necessary to prevent bias in analyses, especially if these features are integral to understanding happiness and life satisfaction.
   - Notably, the **Country name** field does not contain any missing values, contributing to demographic consistency.

5. **General Observations:**
   - The dataset contains a diverse set of countries but is likely skewed toward several countries with repeated entries over multiple years, hence the high frequency for **Argentina**.
   - The average Life Ladder score is relatively low at 5.48, suggesting that the overall perceived life satisfaction may be moderate.
   - Considering **Perceptions of corruption** correlated negatively with several factors emphasizes the need for governance and institutional integrity in fostering a happy society.
   - Features related to health, economic performance, and social support persist as the strongest predictors of life satisfaction, reinforcing existing theories in the field of happiness economics.

### Conclusion:
This dataset presents a rich source of information for exploring the factors influencing happiness and life satisfaction across nations. The identified correlations and trends provide a groundwork for further analysis, including possible machine learning applications or causal analysis to inform policy-making for enhancing societal well-being.

### Correlation
![Correlation Heatmap](happiness\correlation_matrix.png)

### Outliers
Outlier detection results:
![Box Plot of Outliers](happiness\boxplot.png)

## Recommnedations
Based on the provided dataset summary and details about the columns, here are some recommendations to improve the dataset and prepare it for further analysis:

### 1. **Handling Missing Data:**
   - **Imputation**: For the columns with missing values, consider using imputation techniques:
     - For numerical columns like **Log GDP per capita**, **Freedom to make life choices**, **Generosity**, etc., you can use the mean or median values for imputation.
     - You can also explore more sophisticated imputation techniques, such as K-nearest neighbors (KNN) or regression-based imputation.
   - **Deletion**: If the percentage of missing data is very high (e.g., over 10%), you might consider dropping those columns entirely if they are not critical for your analysis.

### 2. **Standardization/Normalization:**
   - Since this dataset could be used for clustering or other machine learning tasks, standardizing or normalizing the numerical features may be necessary to ensure that all features contribute equally to the distance calculations.

### 3. **Feature Engineering:**
   - Consider creating new features based on existing ones. For example, you could combine **Positive affect** and **Negative affect** into a new feature called **Net affect** (Positive - Negative).
   - Creating categorical variables from continuous ones (e.g., categorizing years into 'Pre-2010', '2010-2015', and 'Post-2015') may help in some analyses.

### 4. **Examine Anomalies:**
   - The columns for anomalies indicate potential outliers or special cases. Analyze the data in these columns to determine if they represent meaningful insights or if they should be further investigated or removed.

### 5. **Visualize Data:**
   - Create visualizations (e.g., histograms, box plots, scatter plots) to better understand distributions, relationships between features, and the impact of missing data imputation.
   - Use correlation matrices to identify which features are correlated and may impact each other. 

### 6. **Year-Based Analysis:**
   - Investigate trends over time. It may be useful to create visualizations that show how the metrics like **Life Ladder**, **Log GDP per capita**, and others change over the years for different countries.
   - Consider aggregating the data by year or by country to analyze the broad trends versus individual country trends.

### 7. **Categorical Analysis:**
   - For the **Country name** column, analyze the data to understand regional differences or trends. You might want to encode the country names into numerical categories for machine learning models.

### 8. **Statistical Tests:**
   - If you aim to compare groups (such as countries with higher vs. lower GDP), consider conducting appropriate statistical tests (like t-tests or ANOVA) to validate any differences observed in means.

### 9. **Documentation:**
   - Ensure that you document your findings, transformations, and any assumptions made during data cleaning and preprocessing. This documentation is crucial for reproducibility and understanding the data's context.

### 10. **Consider Outlier Treatment:**
   - After exploring anomalies, consider whether they should be removed from the dataset, capped, or transformed (e.g., via log transformations) to reduce their influence on analyses.

By systematically addressing these issues, you can enhance the quality of your dataset and ensure robust analytical results.

