# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "openai==0.28",
#   "charset_normalizer",
#   "scikit-learn",
#   "numpy",
#   "scipy"
# ]
# ///

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import openai
from charset_normalizer import detect
import requests
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from scipy import stats

class DataAnalyzer:
    def __init__(self, file_path, output_dir):
        """
        Initialize the data analyzer with a file path and output directory.
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.df = self._load_dataset()
        self.numeric_columns = self.df.select_dtypes(include=['number']).columns
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns

    def _load_dataset(self):
        """
        Load the dataset with automatic encoding detection.
        """
        with open(self.file_path, 'rb') as f:
            raw_data = f.read()
            detected_encoding = detect(raw_data)['encoding']

        try:
            df = pd.read_csv(self.file_path, encoding=detected_encoding)
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)

    def perform_advanced_analysis(self):
        """
        Perform comprehensive data analysis with multiple advanced techniques.
        """
        analysis = {
            "basic_info": {
                "columns": list(self.df.columns),
                "dtypes": self.df.dtypes.apply(str).to_dict(),
                "shape": self.df.shape
            },
            "summary_stats": self.df.describe(include='all').to_dict(),
            "missing_values": self.df.isnull().sum().to_dict()
        }

        # Outlier detection
        if len(self.numeric_columns) > 0:
            analysis["outliers"] = self._detect_outliers()

        # Feature importance
        if len(self.numeric_columns) > 1:
            analysis["feature_importance"] = self._calculate_feature_importance()

        # Distribution analysis
        analysis["distributions"] = self._analyze_distributions()

        # Advanced clustering
        if len(self.numeric_columns) > 1:
            analysis["clustering"] = self._advanced_clustering()

        return analysis

    # Other methods (_detect_outliers, _calculate_feature_importance, etc.)
    
    def _detect_outliers(self):
        """
        Detect outliers using multiple methods.
        
        Returns:
            dict: Outlier information for numeric columns
        """
        outliers = {}
        for col in self.numeric_columns:
            # Z-score method
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outliers[col] = {
                "z_score_outliers": sum(z_scores > 3),
                "iqr_outliers": sum(self._iqr_outliers(self.df[col]))
            }
        return outliers

    def _iqr_outliers(self, series):
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            series (pd.Series): Input numeric series
        
        Returns:
            numpy.ndarray: Boolean mask of outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)

    def _calculate_feature_importance(self):
        """
        Calculate feature importance using mutual information.
        
        Returns:
            dict: Feature importance scores
        """
        if len(self.numeric_columns) < 2:
            return {}

        X = self.df[self.numeric_columns[:-1]]
        y = self.df[self.numeric_columns[-1]]
        
        # Mutual Information Regression
        mi_scores = mutual_info_regression(X, y)
        return dict(zip(X.columns, mi_scores))

    def _analyze_distributions(self):
        """
        Analyze distributions of numeric columns.
        
        Returns:
            dict: Distribution characteristics
        """
        distributions = {}
        for col in self.numeric_columns:
            distributions[col] = {
                "skewness": stats.skew(self.df[col].dropna()),
                "kurtosis": stats.kurtosis(self.df[col].dropna())
            }
        return distributions

    def _advanced_clustering(self):
        """
        Perform advanced clustering using multiple algorithms.
        
        Returns:
            dict: Clustering results
        """
        # Prepare data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[self.numeric_columns].dropna())
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)

        # KMeans Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans_labels = kmeans.fit_predict(scaled_data)

        # DBSCAN Clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(scaled_data)

        return {
            "kmeans_n_clusters": len(np.unique(kmeans_labels)),
            "dbscan_n_clusters": len(np.unique(dbscan_labels[dbscan_labels != -1]))
        }
    
    def generate_visualizations(self, analysis):
        """
        Generate comprehensive data visualizations as separate PNG files.
        """
        # Correlation Heatmap
        if len(self.numeric_columns) > 1:
            plt.figure(figsize=(10, 8))
            corr = self.df[self.numeric_columns].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0,
                        square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"), dpi=300)
            plt.close()

        # Distribution of Most Variable Column
        if len(self.numeric_columns) > 0:
            plt.figure(figsize=(10, 6))
            most_variable_col = self.df[self.numeric_columns].std().idxmax()
            sns.histplot(self.df[most_variable_col], kde=True)
            plt.title(f"Distribution of {most_variable_col}")
            plt.xlabel(most_variable_col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{most_variable_col}_distribution.png"), dpi=300)
            plt.close()

        # Clustering Visualization
        if len(self.numeric_columns) > 1:
            plt.figure(figsize=(10, 8))
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.df[self.numeric_columns].dropna())
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans_labels = kmeans.fit_predict(scaled_data)

            scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_labels,
                                  cmap='viridis', alpha=0.7)
            plt.title("Clustering Visualization (PCA)")
            plt.xlabel("First Principal Component")
            plt.ylabel("Second Principal Component")
            plt.colorbar(scatter)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "clustering_visualization.png"), dpi=300)
            plt.close()

    def generate_narrative(self, analysis):
        """
        Generate a narrative using LLM based on the analysis.
        """
        prompt = f"""
        You are a data science storyteller. Analyze this dataset summary:

        Dataset Overview:
        - Columns: {list(self.df.columns)}
        - Shape: {self.df.shape}
        - Numeric Columns: {list(self.numeric_columns)}
        - Categorical Columns: {list(self.categorical_columns)}

        Key Analysis Insights:
        - Missing Values: {analysis.get('missing_values', {})}
        - Outliers: {analysis.get('outliers', {})}
        - Feature Importance: {analysis.get('feature_importance', {})}
        - Distributions: {analysis.get('distributions', {})}
        - Clustering: {analysis.get('clustering', {})}

        Write a compelling narrative that:
        1. Describes the dataset
        2. Highlights key findings
        3. Provides actionable insights
        4. Suggests potential further analysis or business implications

        Use markdown formatting. Be creative and engaging!
        """

        headers = {
            "Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a data science storyteller creating a narrative about a dataset."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000
        }

        try:
            response = requests.post(
                "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            narrative = response.json()['choices'][0]['message']['content']
            return narrative
        except Exception as e:
            return f"Error generating narrative: {e}"

def generate_visualizations(self, analysis):
    """
    Generate comprehensive data visualizations as separate PNG files.
    
    Args:
        analysis (dict): Analysis results to inform visualization
    """
    # Correlation Heatmap
    if len(self.numeric_columns) > 1:
        plt.figure(figsize=(10, 8))
        corr = self.df[self.numeric_columns].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"), dpi=300)
        plt.close()

    # Distribution of Most Variable Column
    if len(self.numeric_columns) > 0:
        plt.figure(figsize=(10, 6))
        most_variable_col = self.df[self.numeric_columns].std().idxmax()
        sns.histplot(self.df[most_variable_col], kde=True)
        plt.title(f"Distribution of {most_variable_col}")
        plt.xlabel(most_variable_col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{most_variable_col}_distribution.png"), dpi=300)
        plt.close()

    # Clustering Visualization
    if len(self.numeric_columns) > 1:
        plt.figure(figsize=(10, 8))
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[self.numeric_columns].dropna())
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans_labels = kmeans.fit_predict(scaled_data)
        
        scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_labels, 
                              cmap='viridis', alpha=0.7)
        plt.title("Clustering Visualization (PCA)")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "clustering_visualization.png"), dpi=300)
        plt.close()
    
def generate_narrative(self, analysis):
    """
    Generate a narrative using LLM based on the analysis.
    
    Args:
        analysis (dict): Analysis results to craft narrative
    
    Returns:
        str: Generated narrative
    """
    # Prepare a compact, informative prompt
    prompt = f"""
    You are a data science storyteller. Analyze this dataset summary:

    Dataset Overview:
    - Columns: {list(self.df.columns)}
    - Shape: {self.df.shape}
    - Numeric Columns: {list(self.numeric_columns)}
    - Categorical Columns: {list(self.categorical_columns)}

    Key Analysis Insights:
    - Missing Values: {analysis.get('missing_values', {})}
    - Outliers: {analysis.get('outliers', {})}
    - Feature Importance: {analysis.get('feature_importance', {})}
    - Distributions: {analysis.get('distributions', {})}
    - Clustering: {analysis.get('clustering', {})}

    Write a compelling narrative that:
    1. Describes the dataset
    2. Highlights key findings
    3. Provides actionable insights
    4. Suggests potential further analysis or business implications
    
    Use markdown formatting. Be creative and engaging!
    """

    headers = {
        "Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data science storyteller creating a narrative about a dataset."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000
    }

    try:
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", 
            headers=headers, 
            json=payload
        )
        response.raise_for_status()
        narrative = response.json()['choices'][0]['message']['content']
        
        # Append visualization description
        most_variable_col = self.df[self.numeric_columns].std().idxmax()
        narrative += "\n\n## Visualizations\n"
        narrative += f"![Correlation Heatmap](correlation_heatmap.png)\n"
        narrative += f"![{most_variable_col} Distribution]({most_variable_col}_distribution.png)\n"
        narrative += f"![Clustering Visualization](clustering_visualization.png)\n\n"
    
        return narrative
    except Exception as e:
        return f"Error generating narrative: {e}"

def analyze_and_generate_output(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(".", base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Set up the analyzer
    analyzer = DataAnalyzer(file_path, output_dir)
    
    # Perform analysis
    analysis = analyzer.perform_advanced_analysis()
    
    # Generate visualizations
    analyzer.generate_visualizations(analysis)
    
    # Generate narrative
    narrative = analyzer.generate_narrative(analysis)
    
    # Write narrative to README
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(narrative)

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