# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",                # For HTTP requests
#   "pandas",               # For data manipulation and analysis
#   "seaborn",              # For data visualization (statistical plots)
#   "matplotlib",           # For general-purpose plotting
#   "openai==0.28",         # For interacting with OpenAI API (pinned to 0.28 to avoid errors)
#   "scikit-learn",         # For machine learning algorithms
#   "tabulate",             # For pretty-printing tables
#   "numpy"                 # For numerical computations (you are using np in the code)
# ]
# ///

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import sys
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from tabulate import tabulate
from PIL import Image

aiproxy_token = os.environ.get("AIPROXY_TOKEN")
if not aiproxy_token:
    print("Error: AIPROXY_TOKEN is not set. Please provide the token.")
    sys.exit(1)

openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
openai.api_key = aiproxy_token

if len(sys.argv) != 2:
    print("Usage: uv run autolysis.py <dataset.csv>")
    sys.exit(1)

filename = sys.argv[1]

# Create output directory based on input filename
def create_output_directory(input_filename):
    # Remove .csv extension and create directory
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    output_dir = base_name
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

ENCODINGS_TO_TRY = ['utf-8', 'ISO-8859-1', 'latin1', 'utf-16', 'cp1252', 'windows-1252']

def load_dataset(filename):
    for encoding in ENCODINGS_TO_TRY:
        try:
            data = pd.read_csv(filename, encoding=encoding)
            print(f"Dataset loaded successfully with {encoding} encoding, "
                  f"containing {data.shape[0]} rows and {data.shape[1]} columns.")
            return data
        except UnicodeDecodeError:
            print(f"Error: Encoding issue with {encoding}. Trying next encoding...")
        except FileNotFoundError:
            print(f"Error: The file {filename} was not found. Please check the file path.")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            print(f"Error: The file {filename} is empty. Please provide a valid CSV file.")
            sys.exit(1)
        except Exception as e:
            print(f"Error: An unexpected error occurred while loading the file {filename}. {e}")
            sys.exit(1)
    
    print(f"Error: Could not read the file {filename} with any of the tried encodings.")
    sys.exit(1)

def explore_data(data):
    return data.describe(include='all')

def generate_correlation_matrix(data, output_dir):
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        print("No numeric columns available for correlation.")
        return
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix")
    
    # Save in the specified output directory
    output_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved correlation heatmap as '{output_path}'")

def generate_histograms(data, output_dir, column='average_rating'):
    numeric_data = data.select_dtypes(include=[np.number])
    if column in numeric_data.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(numeric_data[column], kde=True, color='blue')
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        
        # Save in the specified output directory
        output_path = os.path.join(output_dir, "histogram.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Histogram saved as '{output_path}'")
    else:
        print(f"Column '{column}' not found in the dataset.")


def generate_boxplot(data, output_dir, column='average_rating'):
    numeric_data = data.select_dtypes(include=[np.number])
    if column in numeric_data.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=numeric_data[column], color='red')
        plt.title(f"Boxplot of {column}")
        plt.xlabel(column)
        
        # Save in the specified output directory
        output_path = os.path.join(output_dir, "boxplot.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Boxplot saved as '{output_path}'")
    else:
        print(f"Column '{column}' not found in the dataset.")

def anomaly_detection(data):
    numeric_data = data.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy='median')
    numeric_data_imputed = imputer.fit_transform(numeric_data)
    iso_forest = IsolationForest(contamination=0.05)
    anomalies = iso_forest.fit_predict(numeric_data_imputed)
    data['Anomaly'] = anomalies
    return data

def dbscan_anomaly_detection(data):
    numeric_data = data.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy='median')
    numeric_data_imputed = imputer.fit_transform(numeric_data)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(numeric_data_imputed)
    data['DBSCAN_Anomaly'] = labels
    return data

def one_class_svm_anomaly_detection(data):
    numeric_data = data.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy='median')
    numeric_data_imputed = imputer.fit_transform(numeric_data)
    svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    labels = svm.fit_predict(numeric_data_imputed)
    data['SVM_Anomaly'] = labels
    return data

def gpt_api_call(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error during GPT API call: {e}")
        return "GPT API call failed due to an error."

def analyze_data_with_llm(data, filename="data.csv"):

    data_info = data.info()
    data_summary = data.describe(include='all') 
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        print("No numeric columns available for correlation.")
    else:
        corr_matrix = numeric_data.corr()
    
    starting_prompt = f"""
    Here is the dataset information and summary:
    {data_info}
    This dataset contains {data.shape[0]} rows and {data.shape[1]} columns. 
    The data types of the columns are as follows:
    {data.dtypes}
    The dataset includes the following numerical and categorical columns, with key statistics (mean, min, max, std, etc.):
    {data_summary}
    Missing data information:
    {data.isnull().sum()}"""

    data_info_prompt = f"""
    {starting_prompt}
    Please analyze the dataset based on the following detailed information and write me a short introduction paragraph for my
    data analysis report in few lines.
    """
    data_info_details = gpt_api_call(data_info_prompt)

    file_details_prompt = f"""
    {starting_prompt}
    Please provide a concise summary of the dataset's dimensions. This file has {data.shape[0]} rows and {data.shape[1]} columns.
    File: {filename}
    write in points
    """
    file_details = gpt_api_call(file_details_prompt)

    analysis_results_prompt = f"""
    {starting_prompt}
    Provide a comprehensive analysis of the dataset, focusing on the following aspects:
    - Correlations between numeric and categorical variables for this you can take a look at {corr_matrix}
    - Feature distributions for numerical data (mean, std, range).
    - Identification of outliers or extreme values.
    - Trends in missing data or categorical distributions.
    - Any general observations you find interesting based on the data summary provided above.
    Please write in points in professional way.
    """
    analysis_results = gpt_api_call(analysis_results_prompt)

    recommendations_prompt = f"""
    {starting_prompt}
    Please suggest recommendations on this dataset.
    """
    recommendations = gpt_api_call(recommendations_prompt)

    return {
        'data_info_details': data_info_details,
        'file_details': file_details,
        'analysis_results': analysis_results,
        'recommendations': recommendations,
    }

def safe_get(data, key, default="N/A"):
    if isinstance(data, dict):
        return data.get(key, default)
    elif isinstance(data, str):
        return data if key in data else default
    return default

def generate_readme(dataset_name, analysis_results, data, output_dir):
    # Modify file paths to use output directory
    def update_image_path(base_filename):
        return os.path.join(output_dir, base_filename)

    data_info_details = analysis_results.get('data_info_details', "N/A")
    file_details = analysis_results.get('file_details', "N/A")
    recommendations = analysis_results.get('recommendations', "N/A")

    missing_values_summary = data.isnull().sum().reset_index()
    missing_values_summary.columns = ['Column Name', 'Missing Values']
    missing_values_table = tabulate(missing_values_summary, headers='keys', tablefmt='pipe', showindex=False)

    anomaly_data = anomaly_detection(data.copy())
    dbscan_data = dbscan_anomaly_detection(data.copy())
    svm_data = one_class_svm_anomaly_detection(data.copy())

    isolation_anomalies_count = sum(anomaly_data['Anomaly'] == -1)
    dbscan_anomalies_count = sum(dbscan_data['DBSCAN_Anomaly'] == -1)
    svm_anomalies_count = sum(svm_data['SVM_Anomaly'] == -1)
    
    analysis_results_data = analysis_results.get('analysis_results', {})

    readme_content = f"""# Data Analysis of {dataset_name}
{data_info_details}
{file_details}

## Contents
- [Missing Values Summary](#missing-values-summary)
- [Anomalies Detected](#anomalies-detected)
- [Graphs](#graphs)
- [Analysis Results](#analysis-results)
- [Recommnedations](#Recommnedations)

## Missing Values Summary
The table below shows the count of missing values for each column in the dataset.
{missing_values_table}

## Anomalies Detected
Anomalies were detected using three methods. The results are summarized below:

### Isolation Forest
- Number of anomalies detected: **{isolation_anomalies_count}**
- Method: Identifies anomalies by isolating data points through recursive partitioning.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Number of anomalies detected: **{dbscan_anomalies_count}**
- Method: Identifies anomalies as points in low-density regions using density-based clustering.

### One-Class SVM (Support Vector Machine)
- Number of anomalies detected: **{svm_anomalies_count}**
- Method: Learns a decision boundary to separate normal data points from anomalies.

## Graphs
Here are some key visualizations:
![{"Histogram"}]({update_image_path("histogram.png")})  

## Analysis Results
{analysis_results_data}

### Correlation
![{"Correlation Heatmap"}]({update_image_path("correlation_matrix.png")})

### Outliers
Outlier detection results:
![{"Box Plot of Outliers"}]({update_image_path("boxplot.png")})

## Recommnedations
{recommendations}

"""

    def resize_image(image_path, width=500):
        try:
            with Image.open(image_path) as img:
                aspect_ratio = img.height / img.width
                new_height = int(width * aspect_ratio)
                img = img.resize((width, new_height))
                img.save(image_path) 
        except FileNotFoundError:
            print(f"Image {image_path} not found, skipping resizing.")

    # Resize images in the output directory
    resize_image(os.path.join(output_dir, "histogram.png"))
    resize_image(os.path.join(output_dir, "correlation_matrix.png"))
    resize_image(os.path.join(output_dir, "boxplot.png"))
    
    # Save README in the output directory
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"Generated README for {dataset_name} in {readme_path}.")

# Rest of the functions remain the same as in the original script
# (anomaly_detection, dbscan_anomaly_detection, one_class_svm_anomaly_detection, 
# analyze_data_with_llm, etc.)

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    # Create output directory based on input filename
    output_dir = create_output_directory(filename)

    # Load dataset
    data = load_dataset(filename)

    # Explore data and generate visualizations in the output directory
    explore_data(data)
    generate_correlation_matrix(data, output_dir)
    generate_histograms(data, output_dir)
    generate_boxplot(data, output_dir)

    # Perform anomaly detection
    data_with_anomalies = anomaly_detection(data)
    data_with_anomalies = dbscan_anomaly_detection(data_with_anomalies)
    data_with_anomalies = one_class_svm_anomaly_detection(data_with_anomalies)
    
    # Analyze data with LLM
    analysis_results = analyze_data_with_llm(data, filename)

    # Generate README in the output directory
    generate_readme(filename, analysis_results, data, output_dir)
    print("Analysis complete and README generated.")