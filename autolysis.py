# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "openai",
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import asyncio
import httpx
import time

execution_start_time = time.time()
total_tokens_used = 0

AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable not set")

API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def ensure_csv_file(filename):
    if not filename.lower().endswith('.csv'):
        raise ValueError(f"Invalid file type: {filename}. Please provide a .csv file.")

def create_output_directory(filename):
    dir_name = os.path.splitext(os.path.basename(filename))[0]
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def load_data(filename):
    return pd.read_csv(filename, encoding='utf-8', encoding_errors='replace')

def analyze_data(df):
    summary = df.describe(include='all')
    missing_values = df.isnull().sum()
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    return summary, missing_values, correlation_matrix

def visualize_data(df, output_dir):
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Density plot for the first numeric column
    if len(numeric_columns) > 0:
        sns.kdeplot(df[numeric_columns[0]], fill=True)
        plt.title(f'Density Plot of {numeric_columns[0]}')
        plt.xlabel(numeric_columns[0])
        plt.ylabel('Density')
        plt.savefig(os.path.join(output_dir, 'density_plot.png'))
        plt.clf()

    # Scatter plot for the first two numeric columns
    if len(numeric_columns) > 1:
        sns.scatterplot(x=numeric_columns[0], y=numeric_columns[1], data=df)
        plt.title(f'Scatter Plot of {numeric_columns[0]} vs {numeric_columns[1]}')
        plt.xlabel(numeric_columns[0])
        plt.ylabel(numeric_columns[1])
        plt.savefig(os.path.join(output_dir, 'scatter_plot.png'))
        plt.clf()

    # Histogram for all numeric columns
    if len(numeric_columns) > 0:
        df[numeric_columns].hist(bins=15, figsize=(15, 10))
        plt.suptitle('Histograms of Numeric Columns')
        plt.savefig(os.path.join(output_dir, 'histogram.png'))
        plt.clf()

async def generate_story(summary, missing_values, correlation_matrix):
    global total_tokens_used
    prompt = f"""
    Analyze the following dataset summary and provide insights:
    Summary: {summary}
    Missing Values: {missing_values}
    Correlation Matrix: {correlation_matrix}

    Additionally, provide insights on the following visualizations:
    1. Density Plot: Describe the distribution of the first numeric column.
    2. Scatter Plot: Describe the relationship between the first two numeric columns.
    3. Histogram: Describe the distribution of all numeric columns.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(API_URL, headers=headers, json=data)
            response_data = response.json()
            tokens_used = response_data.get('usage', {}).get('total_tokens', 0)
            total_tokens_used += tokens_used
            return response_data['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error during API request: {e}")
            return "Unable to generate story due to an error."

def save_readme(story, output_dir):
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# Automated Analysis Report\n")
        f.write(story)
        f.write("\n## Density Plot\n")
        f.write("This plot shows the distribution of the first numeric column.\n")
        f.write("![Density Plot](density_plot.png)\n")
        f.write("\n## Scatter Plot\n")
        f.write("This plot shows the relationship between the first two numeric columns.\n")
        f.write("![Scatter Plot](scatter_plot.png)\n")
        f.write("\n## Histogram\n")
        f.write("This plot shows the distribution of all numeric columns.\n")
        f.write("![Histogram](histogram.png)\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    ensure_csv_file(filename)
    output_dir = create_output_directory(filename)
    df = load_data(filename)
    summary, missing_values, correlation_matrix = analyze_data(df)
    visualize_data(df, output_dir)
    story = asyncio.run(generate_story(summary, missing_values, correlation_matrix))
    save_readme(story, output_dir)

if __name__ == "__main__":
    try:
        main()

    finally:
        # Calculate total execution time
        execution_end_time = time.time()
        total_execution_time = execution_end_time - execution_start_time

        print("\nSummary:")
        print(f"Total Execution Time: {total_execution_time:.2f} seconds")
        print(f"Total Tokens Used: {total_tokens_used}")