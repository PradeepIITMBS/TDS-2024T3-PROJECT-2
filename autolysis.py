# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "openai==0.28",
#   "charset_normalizer",
#   "scikit-learn",
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
    """Analyzes the dataset and returns a DataFrame and analysis results."""
    # Detect file encoding
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        detected_encoding = detect(raw_data)['encoding']

    # Load the dataset with detected encoding
    try:
        df = pd.read_csv(file_path, encoding=detected_encoding)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # Perform generic analysis
    analysis = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.apply(str).to_dict(),
        "summary_stats": df.describe(include='all').to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }

    # Additional analysis for categorical data
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        analysis[f"{col}_value_counts"] = df[col].value_counts().to_dict()

    return df, analysis

def perform_clustering(df, output_dir):
    """Performs KMeans clustering on numeric data and saves the results."""
    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 1:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_columns].dropna())
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)

        df['Cluster'] = pd.Series(clusters, index=df[numeric_columns].dropna().index)
        plt.figure(figsize=(5.12, 5.12))  # Adjusted for 512x512 px
        sns.scatterplot(x=df[numeric_columns[0]], y=df[numeric_columns[1]], hue='Cluster', palette='viridis', data=df)
        plt.title("KMeans Clustering")
        plt.xlabel(numeric_columns[0])
        plt.ylabel(numeric_columns[1])
        plt.annotate("Cluster centers are identified by distinct groupings", (0.5, 0.1), xycoords='figure fraction', ha='center', fontsize=10, color='gray')
        plt.savefig(os.path.join(output_dir, "kmeans_clustering.png"))
        plt.close()

def generate_visualizations(df, output_dir):
    """Generates the most interesting visualizations from the dataset and saves them as PNG files."""
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Generate a correlation heatmap if applicable
    if len(numeric_columns) > 1:
        corr = df[numeric_columns].corr()
        plt.figure(figsize=(5.12, 5.12))  # Adjusted for 512x512 px
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()

    # Create a distribution plot for the most variable numeric column
    if len(numeric_columns) > 0:
        most_variable_col = df[numeric_columns].std().idxmax()
        plt.figure(figsize=(5.12, 5.12))  # Adjusted for 512x512 px
        sns.histplot(df[most_variable_col], kde=True)
        plt.title(f"Distribution of {most_variable_col}")
        plt.xlabel(most_variable_col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{most_variable_col}_distribution.png"))
        plt.close()

    # Create a bar plot for the most frequent categorical column (limited for readability)
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        most_frequent_col = max(categorical_columns, key=lambda col: df[col].nunique())
        top_values = df[most_frequent_col].value_counts().nlargest(10)  # Limit to top 10 for readability
        plt.figure(figsize=(5.12, 5.12))  # Adjusted for 512x512 px
        sns.barplot(y=top_values.index, x=top_values.values)
        plt.title(f"Top 10 Frequency of {most_frequent_col}")
        plt.ylabel(most_frequent_col)
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{most_frequent_col}_frequency.png"))
        plt.close()

def narrate_story(analysis, output_dir):
    """Generates a narrative about the dataset analysis using the LLM."""
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data scientist narrating the story of a dataset."},
            {"role": "user", "content": f"Here's the analysis of the dataset: {analysis}. Please provide actionable insights and highlight key findings such as potential relationships, outliers, or trends based on the data."}
        ]
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

    # Add brief descriptions for visualizations
    with open(os.path.join(output_dir, "README.md"), "a") as f:
        f.write("\n\n### Correlation Heatmap\n")
        f.write("The correlation heatmap shows relationships between numeric variables, highlighting strong positive or negative correlations.\n")
        f.write("\n### Most Variable Column Distribution\n")
        f.write("This plot highlights the distribution of the most variable numeric feature in the dataset. It provides insights into the spread and central tendencies of the data.\n")
        f.write("\n### Top 10 Frequency of Most Frequent Categorical Column\n")
        f.write("This bar plot showcases the frequency distribution of the top 10 categories in the most frequent categorical column, ensuring readability.\n")
        f.write("\n### KMeans Clustering\n")
        f.write("This scatter plot visualizes the results of KMeans clustering on numeric variables, revealing distinct groupings in the dataset.\n")
        f.write("Key insights from clustering include the grouping patterns which may represent different audience preferences or performance tiers.\n")

def analyze_and_generate_output(file_path):
    """Main function to analyze the dataset and generate outputs."""
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