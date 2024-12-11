# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "openai==0.28",
#   "charset_normalizer",
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

# Ensure the environment variable for AI Proxy token is set
AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

# Set proxy API base URL
PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def analyze_dataset(file_path):
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
    return df, analysis

def generate_visualizations(df, output_dir):
    # Generate a correlation heatmap if applicable
    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 1:
        corr = df[numeric_columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()

    # Create a distribution plot for the first numeric column
    if len(numeric_columns) > 0:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[numeric_columns[0]], kde=True)
        plt.title(f"Distribution of {numeric_columns[0]}")
        plt.savefig(os.path.join(output_dir, f"{numeric_columns[0]}_distribution.png"))
        plt.close()

def narrate_story(analysis, output_dir):
    # Generate a narrative using the API proxy
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data scientist narrating the story of a dataset."},
            {"role": "user", "content": f"Here's the analysis of the dataset: {analysis}"}
        ]
    }

    try:
        response = requests.post(PROXY_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise error for HTTP issues
        story = response.json().get('choices', [{}])[0].get('message', {}).get('content', "No content returned.")
    except Exception as e:
        story = f"Error generating narrative: {e}"

    # Write the story to README.md
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(story)

def analyze_and_generate_output(file_path):
    # Define output directory based on file name
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(".", base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Analyze dataset
    df, analysis = analyze_dataset(file_path)

    # Generate visualizations
    generate_visualizations(df, output_dir)

    # Narrate the story
    narrate_story(analysis, output_dir)

    return output_dir

def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]

    if os.path.exists(file_path):
        output_dir = analyze_and_generate_output(file_path)
        print(f"Analysis completed. Results saved in directory: {output_dir}")
    else:
        print(f"File {file_path} not found!")

if __name__ == "__main__":
    main()
