# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "openai==0.28",
#   "charset_normalizer",
#   "scikit-learn",
#   "scipy"
# ]
# ///

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import openai
from charset_normalizer import detect
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ensure the environment variable for AI Proxy token is set
AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

# Set proxy API base URL
PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def analyze_dataset(file_path):
    """Analyzes the dataset and returns a Pandas DataFrame and analysis results."""
    # Detect file encoding
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        detected_encoding = detect(raw_data)['encoding']

    # Load the dataset with Pandas
    try:
        df = pd.read_csv(file_path, encoding=detected_encoding)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # Perform analysis
    analysis = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.apply(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }

    # Additional analysis for categorical data (sampling for efficiency)
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        analysis['categorical_overview'] = {}
        for col in categorical_columns:
            value_counts = df[col].value_counts()
            analysis['categorical_overview'][col] = {
                "unique_values": df[col].nunique(),
                "most_common": value_counts.idxmax() if not value_counts.empty else None,
                "most_common_count": value_counts.max() if not value_counts.empty else None
            }

    return df, analysis

def perform_clustering(df, output_dir):
    """Performs KMeans clustering on numeric data and saves the results."""
    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 1:
        sampled_data = df[numeric_columns].dropna().sample(frac=0.1, random_state=42)  # Use 10% of rows
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(sampled_data)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)

        sampled_data['Cluster'] = clusters
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=sampled_data.iloc[:, 0], y=sampled_data.iloc[:, 1], hue='Cluster', palette='viridis', data=sampled_data)
        plt.title("KMeans Clustering")
        plt.xlabel(numeric_columns[0])
        plt.ylabel(numeric_columns[1])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "kmeans_clustering.png"))
        plt.close()

def generate_time_series_analysis(df, output_dir):
    """Generates time series analysis if a datetime column exists."""
    datetime_columns = df.select_dtypes(include=['datetime', 'datetime64']).columns
    if len(datetime_columns) > 0:
        for col in datetime_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            time_series_df = df[[col]].dropna()
            time_series_df.set_index(col, inplace=True)

            # Aggregate by month and count occurrences
            monthly_data = time_series_df.resample('M').size()

            plt.figure(figsize=(10, 10))
            monthly_data.plot()
            plt.title(f"Time Series Analysis: {col}")
            plt.xlabel("Time (Monthly)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"time_series_analysis_{col}.png"))
            plt.close()

def generate_visualizations(df, output_dir):
    """Generates the most interesting visualizations from the dataset and saves them as PNG files."""
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Generate a correlation heatmap if applicable
    if len(numeric_columns) > 1:
        corr = df[numeric_columns].corr()
        figsize = (15, 15) if corr.shape[0] > 10 else (10, 10)  # Adjust size for larger heatmaps
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()

    generate_time_series_analysis(df, output_dir)

def narrate_story(analysis, output_dir):
    """Generates a narrative about the dataset analysis using the LLM."""
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    # Summarize the analysis to reduce token consumption
    summarized_analysis = {
        "columns": analysis["columns"],
        "missing_values": analysis["missing_values"],
        "summary_stats_sample": "Summary stats omitted for efficiency."
    }

    if "categorical_overview" in analysis:
        summarized_analysis["categorical_overview_sample"] = "Categorical overview omitted for efficiency."

    prompt = (
        "You are a data scientist tasked with narrating insights about a dataset. "
        "Provide key findings, trends, and actionable recommendations based on this summarized analysis: "
        f"{summarized_analysis}."
    )

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data scientist narrating the story of a dataset."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500  # Limit token usage for efficiency
    }

    try:
        response = requests.post(PROXY_URL, headers=headers, json=payload)
        response.raise_for_status()
        story = response.json().get('choices', [{}])[0].get('message', {}).get('content', "No content returned.")
    except Exception as e:
        story = f"Error generating narrative: {e}"

    # Write the story to README.md
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write("# Dataset Analysis\n\n")
        f.write(story)
        f.write("\n\n## Visualizations\n")
        for file in os.listdir(output_dir):
            if file.endswith(".png"):
                f.write(f"![{file}]({file})\n")

def analyze_and_generate_output(file_path):
    """Main function to analyze the dataset and generate outputs."""
    if not file_path.endswith('.csv'):
        print("Error: The input file must be a .csv file.")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(".", base_name)
    os.makedirs(output_dir, exist_ok=True)

    df, analysis = analyze_dataset(file_path)
    generate_visualizations(df, output_dir)
    perform_clustering(df, output_dir)
    narrate_story(analysis, output_dir)

    return output_dir

def main():
    """Entry point for the script."""
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]

    if os.path.exists(file_path):
        output_dir = analyze_and_generate_output(file_path)
        print(f"Analysis completed. Results saved in directory: {output_dir}")
    else:
        print(f"File {file_path} not found!")

if __name__ == "__main__":
    main()
