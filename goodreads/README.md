# Data Analysis of goodreads.csv
This report presents an analysis of a dataset comprising 10,000 entries and 26 attributes related to books. The dataset includes key identifiers like book IDs and Goodreads IDs, along with essential details such as authors, publication years, average ratings, and various rating counts. Notably, missing data are present in several columns, including ISBN and original titles, which may affect certain analyses. Additionally, the dataset contains columns denoting anomalies detected through various methods, allowing for insights into unusual patterns within the data. Overall, this analysis aims to extract valuable insights and trends pertinent to the literary landscape represented in this dataset.
### Summary of Dataset Dimensions

- **Total Rows:** 10,000
- **Total Columns:** 26

### Column Information

- **Data Types:**
  - `int64`: 10 columns (e.g., `book_id`, `goodreads_book_id`)
  - `float64`: 6 columns (e.g., `isbn13`, `average_rating`)
  - `object`: 10 columns (e.g., `isbn`, `authors`, `title`)

### Key Statistics

- **Numerical Columns:** Various statistical measures including:
  - Mean, standard deviation, minimum, maximum, and percentiles (25%, 50%, and 75%).
- **Categorical Columns:** Information includes unique counts and frequency (for applicable columns).

### Missing Data

- Several columns have missing values:
  - `isbn`: 700 missing entries
  - `isbn13`: 585 missing entries
  - `original_publication_year`: 21 missing entries
  - `original_title`: 585 missing entries
  - `language_code`: 1,084 missing entries

This succinct format provides a clear overview of the dataset's structure and important attributes.

## Contents
- [Missing Values Summary](#missing-values-summary)
- [Anomalies Detected](#anomalies-detected)
- [Graphs](#graphs)
- [Analysis Results](#analysis-results)
- [Recommnedations](#Recommnedations)

## Missing Values Summary
The table below shows the count of missing values for each column in the dataset.
| Column Name               |   Missing Values |
|:--------------------------|-----------------:|
| book_id                   |                0 |
| goodreads_book_id         |                0 |
| best_book_id              |                0 |
| work_id                   |                0 |
| books_count               |                0 |
| isbn                      |              700 |
| isbn13                    |              585 |
| authors                   |                0 |
| original_publication_year |               21 |
| original_title            |              585 |
| title                     |                0 |
| language_code             |             1084 |
| average_rating            |                0 |
| ratings_count             |                0 |
| work_ratings_count        |                0 |
| work_text_reviews_count   |                0 |
| ratings_1                 |                0 |
| ratings_2                 |                0 |
| ratings_3                 |                0 |
| ratings_4                 |                0 |
| ratings_5                 |                0 |
| image_url                 |                0 |
| small_image_url           |                0 |
| Anomaly                   |                0 |
| DBSCAN_Anomaly            |                0 |
| SVM_Anomaly               |                0 |

## Anomalies Detected
Anomalies were detected using three methods. The results are summarized below:

### Isolation Forest
- Number of anomalies detected: **500**
- Method: Identifies anomalies by isolating data points through recursive partitioning.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Number of anomalies detected: **8792**
- Method: Identifies anomalies as points in low-density regions using density-based clustering.

### One-Class SVM (Support Vector Machine)
- Number of anomalies detected: **1656**
- Method: Learns a decision boundary to separate normal data points from anomalies.

## Graphs
Here are some key visualizations:
![Histogram](goodreads\histogram.png)  

## Analysis Results
### Comprehensive Analysis of the Dataset

#### 1. Correlation Observations:
   - **Positive Correlations**: 
     - Strong positive correlation between `ratings_count` and `work_ratings_count` (0.964) and also with `work_text_reviews_count` (0.764).
     - `ratings_5` shows a perfect correlation of 1.000 with itself and moderate correlation with `ratings_4` (0.934).
     - The `Anomaly` column shows a moderate positive correlation with `book_id` (0.366).
   - **Negative Correlations**: 
     - Notable negative correlations include `ratings_count` (-0.373) and `work_ratings_count` (-0.383) with `books_count`, indicating that higher numbers of books may result in lower ratings or vice versa.
     - The correlation between `SVM_Anomaly` and several rating categories is weak, suggesting that anomalies identified by this means may not heavily depend on ratings.

#### 2. Feature Distributions for Numerical Data:
   - **Mean, Std, and Range**:
     - The average `average_rating` is 0.900, indicating a generally favorable response to the books. However, the presence of a standard deviation of 0.436 indicates variability in ratings.
     - `books_count` has a mean of 4.2 with a standard deviation (std) of 2.87, suggesting that most books have a relatively low count of ratings.
     - The `ratings_count` has an impressive mean of 4705.77 with a high standard deviation of 6325.69, implying availability of books with high variance in reader engagement.
   
#### 3. Identification of Outliers or Extreme Values:
   - **Presence of Missing Data**:
     - The dataset has missing values in key columns: `isbn` (700), `isbn13` (585), `original_publication_year` (21), `original_title` (585), and `language_code` (1084).
   - Outliers can be suspected in `DBSCAN_Anomaly` which has a maximum value of 111, suggesting extreme values that may require attention or review.

#### 4. Trends in Missing Data and Categorical Distributions:
   - The `language_code` has a significant number of missing values (1084), which could impact analyses involving languages.
   - Concerning the `original_publication_year`, 21 missing values might skew historical analyses. Understanding why these records are missing (e.g., newly published books) can influence future predictions.
   - The higher frequency of missing `isbn` and `original_title` suggests possible data entry issues or sources where these data points were not provided.

#### 5. General Observations:
   - The dataset appears to be well-structured, although missing data requires consideration during analysis or any predictive modeling to maintain data integrity.
   - The high variance observed in total ratings suggests that while certain books might draw significant readership, others fail to capture similar engagement, indicating possible opportunities for marketing or recommendation strategies.
   - The anomaly detection columns (`Anomaly`, `DBSCAN_Anomaly`, `SVM_Anomaly`) suggest an ongoing effort to identify outliers, but further investigation into these anomalies could shed light on what drives these outliers and whether they relate strongly to any of the features.
   - The high number of unique `goodreads_book_id` and `best_book_id` compared to lower ratings and reviews indicates potential platform engagement discrepancies, suggesting a need for deeper engagement strategies based on book identification.

Overall, while the dataset comprises a wealth of information, careful attention to missing data points and analysis of behavior trends will enhance understanding and predictive analytics within the dataset.

### Correlation
![Correlation Heatmap](goodreads\correlation_matrix.png)

### Outliers
Outlier detection results:
![Box Plot of Outliers](goodreads\boxplot.png)

## Recommnedations
Based on the dataset information and summary you've provided, here are some recommendations and analyses you might consider performing:

### 1. Data Cleaning
   - **Handle Missing Values**: 
     - For the `isbn`, `isbn13`, and `original_title` columns, consider whether to fill missing values with a placeholder (e.g., "Unknown") or potentially drop those rows if they are significantly affecting analysis.
     - For `original_publication_year`, filling with the median or mode could be appropriate, or using forward/backward fill methods if relevant.
     - The `language_code` has a considerable number of missing values (1,084). Analyze if these can be imputed with commonly used languages or if such books are infrequent enough to drop.

### 2. Data Transformation
   - **Convert Data Types**: 
     - Convert `isbn13` from float to string, as ISBNs should not be treated as numbers (leading zeros may be lost).
     - Review the `language_code` and consider converting it to a categorical type for better analysis.

### 3. Exploratory Data Analysis (EDA)
   - **Descriptive Statistics**: Generate statistics for relevant numerical columns to understand distributions, skewness, etc.
   - **Visualizations**:
     - Plot the distribution of `average_rating`, `ratings_count`, and `work_ratings_count` to understand user engagement.
     - Use bar charts to analyze `rating_1` to `rating_5` to visualize how ratings are distributed.
     - Explore the `average_rating` in relation to `books_count` and `ratings_count` for insights into potential trends.

### 4. Anomaly Detection Analysis
   - **Evaluate Anomalies**: Since there are anomaly detection columns (`Anomaly`, `DBSCAN_Anomaly`, `SVM_Anomaly`), compare the identified anomalies to understand common characteristics. Perhaps create summary statistics or visualizations showing how these books differ from non-anomalous ones.
   - **Classification**: Consider building a model using anomaly indicators as labels to assess if other features can predict anomalies.

### 5. Feature Engineering
   - **Create New Features**: 
     - Calculate `average_rating_per_rating` by dividing `average_rating` by `ratings_count` to find which books have high ratings but few reviews.
     - Consider creating a boolean feature indicating if a book has received a certain threshold of reviews to segment highly-rated yet less-recognized books.

### 6. Predictive Modeling
   - **Build Predictive Models**: If you have a specific target variable (like predicting ratings or identifying anomalies), you could define a machine learning problem. Possible models could include regression for ratings, classification for anomalies, or clustering for grouping similar books.
   
### 7. Analyze Authors and Titles
   - **Text Analysis**: Conduct natural language processing (NLP) on `authors` and `original_title` to derive insights. Possibly quantify the impact of author popularity on ratings to see if well-known authors correlate with higher ratings.
   - **Most Common Authors/Titles**: Analyze which authors or titles are most prevalent in the dataset and assess their average ratings.

### 8. Insights on Language Code
   - **Language Analysis**: Investigate the correlation between language and ratings. Are certain languages associated with higher-rated books? This could present interesting publishing insights.

### 9. Report Findings
   - **Summarize Insights**: After your analyses, create a report or presentation summarizing key findings, trends in ratings, anomalies, and recommendations for readers or publishers.

### 10. Data Export
   - If you create any new variables or derived datasets, consider exporting them for future analyses or sharing with stakeholders.

By addressing these areas, you can derive meaningful insights from the dataset, enhance its usability, and build models or analyses that can aid in decision-making or enhance understanding of the book landscape.

