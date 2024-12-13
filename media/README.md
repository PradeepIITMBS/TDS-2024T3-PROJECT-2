# Data Analysis of media.csv
The dataset under analysis contains 2,652 entries and 11 attributes, encompassing both numerical and categorical data. It features columns such as 'date,' 'language,' 'type,' 'title,' and several anomaly detection metrics, including DBSCAN and SVM anomalies. With a focus on characteristics such as 'overall,' 'quality,' and 'repeatability,' the dataset offers valuable insights into the performance of various entities, predominantly in the film industry, indicated by the prevalence of English-language movies. There are some missing values, particularly in the 'date' column and the 'by' attribute, suggesting areas for further investigation. This analysis aims to uncover underlying patterns and anomalies within the data to enhance understanding and inform future research or operational strategies.
- **Dataset Name**: media.csv
- **Dimensions**: 2652 rows and 11 columns
- **Column Types**:
  - **Object**: date, language, type, title, by
  - **Integer**: overall, quality, repeatability, Anomaly, DBSCAN_Anomaly, SVM_Anomaly
- **Key Statistics**:
  - **Numerical Columns**:
    - **Overall**: Integer values
    - **Quality**: Integer values
    - **Repeatability**: Mean of ~1.49 (min: 1, max: 3)
    - **Anomaly**: Mean of ~0.91 (min: -1, max: 1)
    - **DBSCAN Anomaly**: Mean of ~5.22 (min: -1, max: 20)
    - **SVM Anomaly**: Mean of ~0.48 (min: -1, max: 1)
- **Missing Values**:
  - **Date**: 99 missing entries
  - **By**: 262 missing entries
  - No missing entries in other columns
- **Unique Values**:
  - **Date**: 2055 unique dates
  - **Language**: 11 unique languages
  - **Type**: 8 unique types
  - **Title**: 2312 unique titles

This summary provides a comprehensive overview of the dataset's structure and characteristics.

## Contents
- [Missing Values Summary](#missing-values-summary)
- [Anomalies Detected](#anomalies-detected)
- [Graphs](#graphs)
- [Analysis Results](#analysis-results)
- [Recommnedations](#Recommnedations)

## Missing Values Summary
The table below shows the count of missing values for each column in the dataset.
| Column Name    |   Missing Values |
|:---------------|-----------------:|
| date           |               99 |
| language       |                0 |
| type           |                0 |
| title          |                0 |
| by             |              262 |
| overall        |                0 |
| quality        |                0 |
| repeatability  |                0 |
| Anomaly        |                0 |
| DBSCAN_Anomaly |                0 |
| SVM_Anomaly    |                0 |

## Anomalies Detected
Anomalies were detected using three methods. The results are summarized below:

### Isolation Forest
- Number of anomalies detected: **120**
- Method: Identifies anomalies by isolating data points through recursive partitioning.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Number of anomalies detected: **18**
- Method: Identifies anomalies as points in low-density regions using density-based clustering.

### One-Class SVM (Support Vector Machine)
- Number of anomalies detected: **326**
- Method: Learns a decision boundary to separate normal data points from anomalies.

## Graphs
Here are some key visualizations:
![Histogram](media\histogram.png)  

## Analysis Results
### Comprehensive Analysis of the Dataset

#### 1. Correlations Between Numeric Variables:
- **Overall vs. Quality**: Strong positive correlation (0.826), suggesting that higher overall ratings are associated with better quality assessments.
- **Overall vs. Repeatability**: Moderate positive correlation (0.513), indicating a connection between overall ratings and the repeatability of ratings.
- **Quality vs. Repeatability**: Moderate positive correlation (0.312), suggesting that higher quality ratings may be associated with more consistent assessments.
- **Anomaly Metrics**:
  - **Anomaly vs. DBSCAN_Anomaly**: Moderate negative correlation (-0.483), implying that high anomaly scores are often associated with lower DBSCAN anomalies.
  - **Anomaly vs. SVM_Anomaly**: Weak positive correlation (0.054), indicating limited association between traditional anomaly scores and SVM detected anomalies.
  - **DBSCAN_Anomaly vs. SVM_Anomaly**: Weak negative correlation (-0.116), suggesting that DBSCAN and SVM anomalies do not have a strong relationship.

#### 2. Feature Distributions for Numerical Data:
- **Overall Ratings**: 
  - Mean: 1.49, Std: 0.60, Min: 1, Max: 3.
- **Quality Ratings**:
  - Mean: Not explicitly stated, Std: Not stated, indicating the need for clear summary statistics.
- **Repeatability**:
  - Mean: 1.49, Std: 0.60, indicating that most repeatability measures are low, likely showing a predominance of a single score.
- **Anomaly Scores**:
  - Mean: 0.91, Min: -1, Max: 1. Since the minimum is -1, it indicates the presence of some unusual evaluations.
- **DBSCAN Anomaly**:
  - Mean: 5.22, Std: 4.03, Min: -1, Max: 20. Presence of outliers above the maximum reasonable value.
- **SVM Anomaly**:
  - Mean: 0.48, Min: -1, Max: 1. The maximum of 1 indicates lower overall anomaly detection.

#### 3. Identification of Outliers or Extreme Values:
- The maximum of **DBSCAN_Anomaly** (20) compared with its mean (5.22) and standard deviation (4.03) indicates potential outliers influencing this metric.
- Anomalies represented by -1 in several columns could signify missing or unclassifiable data points, indicating areas for further analysis.

#### 4. Trends in Missing Data:
- The **date** column has 99 missing entries, which could affect temporal analyses.
- The **by** column is highly missing (262 out of 2652), indicating a significant data gap that may require filling or exclusion from certain analyses.
- Other columns have no missing entries, suggesting robust data quality for ratings.

#### 5. Categorical Distributions:
- **Language**: The dataset primarily consists of English entries (1306 occurrences), indicating a linguistic bias.
- **Type**: Predominance of movies (2211 occurrences) suggests a focus on film data, with limited representation from other categories.

#### 6. General Observations:
- The dataset exhibits a broad variance in certain numerical features, especially within the anomaly detection metrics, which may warrant further investigation into the underlying causes.
- The presence of strong correlation between overall and quality ratings can inform future enhancements in content quality based on user feedback.
- Notable data gaps in critical columns suggest areas of improvement for future dataset collection efforts, particularly focusing on consistently reporting the director or producer (column "by").
- The analysis indicates the dataset's applications are ideally suited for quality analysis, anomaly detection, and understanding correlation dynamics among ratings, particularly within movie evaluations.

Overall, the dataset presents valuable insight, although it would benefit from addressing missing values and focusing on enhancing categorical diversity to improve robustness in analysis.

### Correlation
![Correlation Heatmap](media\correlation_matrix.png)

### Outliers
Outlier detection results:
![Box Plot of Outliers](media\boxplot.png)

## Recommnedations
Based on the information and summary you've provided about your dataset, here are several recommendations to improve the dataset management and potential analysis:

### Data Cleaning and Processing

1. **Missing Data Handling**: 
   - You have 99 missing values in the `date` column and 262 missing values in the `by` column. Consider the following approaches:
     - For the `date` column, if the missing data is systematic, you may analyze the reasons for this absence. If possible, consider backfilling or forward-filling based on surrounding entries or deducing the dates based on context.
     - For the `by` column, you could either fill these missing values with a placeholder (e.g., "Unknown") or exclude records with missing entries if the number of affected rows is not significant.

2. **Data Type Conversion**:
   - Convert the `date` column from an object type to a datetime type for easier temporal analysis.
   - Ensure that any categorical variables (e.g., `language`, `type`) are stored as `category` data types to optimize storage and performance.

3. **Duplicate Entries**: 
   - Check for duplicate entries in the dataset based on key identifying columns (e.g., `title` might be a good candidate). If duplicates are found, decide whether to aggregate them or keep only one instance based on your analysis goals.

### Statistical Analysis and Insights

4. **Exploratory Data Analysis (EDA)**:
   - Conduct visualizations (e.g., histograms, box plots) for the numerical columns to understand their distribution and check for outliers.
   - Analyze categorical columns (`language`, `type`) to explore their frequency and relationships with the `overall` and `quality` scores.

5. **Descriptive Statistics**:
   - Gather detailed summary statistics for the `overall`, `quality`, `repeatability`, and anomaly detection columns. Understanding their distributions can help in building predictive models later.

### Model Development

6. **Anomaly Detection**: 
   - Since your dataset includes columns indicating anomalies (`Anomaly`, `DBSCAN_Anomaly`, and `SVM_Anomaly`), consider analyzing the efficiency of these methods. Perform a comparison on model performance in detecting anomalies and consider whether to use these features in further predictive modeling.

7. **Predictive Modeling**:
   - Using the `overall` and `quality` scores as potential target variables, consider developing models to predict these scores based on the other features. Feature scaling and transformations may improve model performance.

### Documentation and Reporting

8. **Metadata Documentation**:
   - Maintain thorough documentation on the dataset, including descriptions of each column, potential sources of missing data, processing steps taken, and justifications for decisions made during preprocessing. This is essential for reproducibility.

### General Recommendations

9. **Regular Updates**: 
   - If the dataset frequently updates, establish a routine for checking data quality, including reviewing missing values, duplicate records, and outlier data.

10. **Data Integrity Check**: 
    - Implement validation checks to ensure that the anomalies and scores are reasonable considering the context of the data to maintain accuracy in analysis.

### Conclusion

Taking these steps will not only clean and prepare your data for current analysis but also set a solid foundation for future projects involving this dataset. Always remember to iteratively validate your findings and adjust your models based on new insights.

