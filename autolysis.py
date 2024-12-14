# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "pandas",
#   "matplotlib",
#   "httpx",
#   "chardet",
#   "ipykernel",
#   "openai",
#   "numpy",
#   "scipy",
#   "scikit-learn",
# ]
# ///

import os, sys, time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import chardet
from pathlib import Path
import asyncio
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

start_time=time.time()
# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Utility Functions
def get_token():
    """Retrieve the API token from environment variables."""
    try:
        return os.environ["AIPROXY_TOKEN"]
    except KeyError:
        print("Error: AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)

async def load_data(file_path):
    """Load CSV file with automatic encoding detection."""
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected file encoding: {encoding}")
    return pd.read_csv(file_path, encoding=encoding)

async def async_post_request(headers, data):
    """Perform an asynchronous HTTP POST request."""
    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

# Data Analysis Functions
def perform_advanced_statistical_analysis(df):
    """Perform advanced statistical analyses including normality tests, PCA, and clustering."""
    results = {}
    numeric_df = df.select_dtypes(include=['number'])
    
    # Normality Tests
    normality_tests = {
        col: stats.shapiro(df[col].dropna())[1] > 0.05 for col in numeric_df.columns
    }
    results['normality'] = normality_tests

    # Outlier Detection
    if not numeric_df.empty:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(scaled_data)
        results['outliers'] = np.where(outliers == -1)[0].tolist()

    # PCA
    if numeric_df.shape[1] > 1:
        pca = PCA()
        pca.fit(scaled_data)
        results['pca'] = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist()
        }

    # Clustering
    inertias = []
    for k in range(1, min(6, len(numeric_df.columns) + 1)):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
    results['clustering'] = inertias

    return results

async def analyze_data(df, token):
    """Perform basic and advanced data analysis."""
    if df.empty:
        print("Error: Dataset is empty.")
        sys.exit(1)

    numeric_df = df.select_dtypes(include=['number'])
    basic_analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict() if not numeric_df.empty else {}
    }

    advanced_analysis = perform_advanced_statistical_analysis(df)
    analysis = {**basic_analysis, **advanced_analysis}

    prompt = (
        "You are a senior data scientist. Provide an analysis of the following results:\n\n"
        f"Basic Summary: {basic_analysis['summary']}\n\n"
        f"Advanced Analysis: {advanced_analysis}\n\n"
        "Suggest further analysis techniques and insights based on these results."
    )

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    suggestions = await async_post_request(headers, data)
    return analysis, suggestions

# Visualization
def visualize_data(df, output_dir, analysis):
    """Generate and save visualizations."""
    numeric_columns = df.select_dtypes(include=['number']).columns
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in numeric_columns[:3]:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(output_dir / f"{col}_distribution.png")
        plt.close()

    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', square=True)
        plt.title("Correlation Heatmap")
        plt.savefig(output_dir / "correlation_heatmap.png")
        plt.close()

# Narrative Generation
async def generate_narrative(analysis, token, file_path):
    """Generate narrative based on analysis."""
    prompt = (
        f"You are a data scientist. Generate a report for the dataset {file_path.name}:\n\n"
        f"Analysis Results: {analysis}\n\n"
        "Provide insights, trends, and recommendations."
    )
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    return await async_post_request(headers, data)

# Main Workflow
async def main(file_path):
    file_path = Path(file_path)
    if not file_path.is_file():
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)

    token = get_token()
    df = await load_data(file_path)
    analysis, suggestions = await analyze_data(df, token)

    output_dir = Path(file_path.stem)
    visualize_data(df, output_dir, analysis)

    narrative = await generate_narrative(analysis, token, file_path)
    with open(output_dir / "README.md", "w") as f:
        f.write(narrative)

    print("Analysis complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <file_path>")
        sys.exit(1)

    asyncio.run(main(sys.argv[1]))
    print(time.time()-start_time)