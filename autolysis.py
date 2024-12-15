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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from charset_normalizer import detect
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats

class AdvancedDataAnalyzer:
    def __init__(self, file_path, output_dir):
        """
        Initialize the advanced data analyzer with comprehensive analysis capabilities.
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.df = self._load_dataset()
        
        # Identify column types
        self.numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        self.datetime_columns = self.df.select_dtypes(include=['datetime64']).columns.tolist()

    def _load_dataset(self):
        """
        Load dataset with robust encoding detection.
        """
        with open(self.file_path, 'rb') as f:
            raw_data = f.read()
            detected_encoding = detect(raw_data)['encoding']

        try:
            # Attempt to parse dates automatically
            df = pd.read_csv(
                self.file_path, 
                encoding=detected_encoding,
                parse_dates=True,
                infer_datetime_format=True
            )
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)

    def perform_comprehensive_analysis(self):
        """
        Conduct a comprehensive, generic analysis applicable to most datasets.
        """
        # Handle missing values by imputing or notifying the user
        if self.df.isnull().values.any():
            print("Missing values detected. Applying default imputation (mean for numeric, mode for categorical).")
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            for col in self.categorical_columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            for col in self.datetime_columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        analysis = {
            "dataset_overview": {
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
                "column_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            },
            "missing_values": {
                "count": self.df.isnull().sum().to_dict(),
                "percentage": (self.df.isnull().sum() / len(self.df) * 100).to_dict()
            },
            "basic_statistics": {},
            "distribution_analysis": {},
            "correlation_analysis": {},
            "outlier_detection": {},
            "dimensionality_reduction": {}
        }

        # Numeric Columns Analysis
        if self.numeric_columns:
            # Basic Statistics
            analysis["basic_statistics"]["numeric"] = self.df[self.numeric_columns].describe().to_dict()

            # Distribution Analysis
            analysis["distribution_analysis"]["numeric"] = {
                col: {
                    "skewness": self.df[col].skew(),
                    "kurtosis": self.df[col].kurtosis(),
                    "normality_test": {
                        "shapiro_statistic": stats.shapiro(self.df[col].dropna())[0]
                        if len(self.df[col].dropna()) > 2 else None,
                        "p_value": stats.shapiro(self.df[col].dropna())[1]
                        if len(self.df[col].dropna()) > 2 else None
                    }
                } for col in self.numeric_columns
            }

            # Handle empty or nearly empty columns
            analysis["distribution_analysis"]["numeric"] = {
                col: analysis["distribution_analysis"]["numeric"][col]
                for col in self.numeric_columns
                if len(self.df[col].dropna()) > 2
            }

            # Other analyses...

        # Additional checks for missing values
        if not any(analysis["missing_values"]["count"].values()):
            print("No missing values detected.")

        return analysis

    def _find_high_correlations(self, correlation_matrix, threshold=0.7):
        """
        Find highly correlated feature pairs.
        """
        high_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    high_corr.append({
                        "features": [correlation_matrix.columns[i], correlation_matrix.columns[j]],
                        "correlation": correlation_matrix.iloc[i, j]
                    })
        return high_corr

    def _detect_iqr_outliers(self, series):
        """
        Detect outliers using Interquartile Range method.
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return {
            "count": len(outliers),
            "percentage": len(outliers) / len(series) * 100
        }

    def _detect_z_score_outliers(self, series, threshold=3):
        """
        Detect outliers using Z-score method.
        """
        z_scores = (series - series.mean()) / series.std()
        z_scores.index = series.index  # Ensure the index aligns
        outliers = series[z_scores > threshold]
        return {
            "count": len(outliers),
            "percentage": len(outliers) / len(series) * 100
        }

    def _perform_pca(self):
        """
        Perform Principal Component Analysis.
        """
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[self.numeric_columns])
        
        # Perform PCA
        pca = PCA()
        pca.fit(scaled_data)
        
        # Explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        return {
            "explained_variance_ratio": explained_variance.tolist(),
            "cumulative_variance_ratio": cumulative_variance.tolist(),
            "n_components_for_95_variance": np.argmax(cumulative_variance >= 0.95) + 1
        }

    def generate_visualizations(self, analysis):
        """
        Generate comprehensive visualizations and save them as separate PNG files.
        """
        # Correlation Heatmap
        if self.numeric_columns and len(self.numeric_columns) > 1:
            plt.figure(figsize=(10, 8))
            corr = self.df[self.numeric_columns].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title("Correlation Heatmap")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"), dpi=300)
            plt.close()

        # Time Series Analysis
        if self.datetime_columns:
            for col in self.datetime_columns:
                plt.figure(figsize=(10, 6))
                time_series = self.df.set_index(col).sort_index()
                time_series_numeric = time_series[self.numeric_columns]
                time_series_numeric.plot(ax=plt.gca(), legend=True)
                plt.title(f"Time Series Analysis ({col})")
                plt.xlabel("Date")
                plt.ylabel("Values")
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"time_series_{col}.png"), dpi=300)
                plt.close()

        # Cluster Analysis
        if self.numeric_columns and len(self.numeric_columns) > 1:
            kmeans = KMeans(n_clusters=3, random_state=42)
            scaled_data = StandardScaler().fit_transform(self.df[self.numeric_columns].dropna())
            clusters = kmeans.fit_predict(scaled_data)

            plt.figure(figsize=(10, 8))
            plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
            plt.title("Cluster Analysis")
            plt.xlabel("Feature 1 (Scaled)")
            plt.ylabel("Feature 2 (Scaled)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "cluster_analysis.png"), dpi=300)
            plt.close()

        # PCA Variance Plot
        if "dimensionality_reduction" in analysis and analysis["dimensionality_reduction"]:
            plt.figure(figsize=(10, 6))
            plt.plot(np.cumsum(analysis["dimensionality_reduction"]["explained_variance_ratio"]))
            plt.title("Cumulative Explained Variance")
            plt.xlabel("Number of Components")
            plt.ylabel("Cumulative Explained Variance")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "pca_variance.png"), dpi=300)
            plt.close()


    def ask_llm_for_insights(self, analysis):
        """
        Use LLM to generate insights based on comprehensive analysis.
        """
        # Prepare a detailed prompt for LLM
        prompt = f"""
        Analyze the following dataset characteristics:

        Dataset Overview:
        - Total Rows: {analysis['dataset_overview']['total_rows']}
        - Total Columns: {analysis['dataset_overview']['total_columns']}
        - Column Types: {json.dumps({k: str(v) for k, v in analysis['dataset_overview']['column_types'].items()})}

        Missing Values:
        {json.dumps(analysis['missing_values'], indent=2)}

        Numeric Columns Statistics:
        {json.dumps(analysis['basic_statistics'].get('numeric', {}), indent=2)}

        Correlation Insights:
        {json.dumps(analysis.get('correlation_analysis', {}), indent=2)}

        Outlier Detection:
        {json.dumps(analysis.get('outlier_detection', {}), indent=2)}

        Provide:
        1. Key dataset insights
        2. Potential data quality issues
        3. Recommended preprocessing steps
        4. Potential advanced analysis techniques
        5. Business or research implications
        """

        try:
            headers = {
                "Authorization": f"Bearer {os.environ.get('AIPROXY_TOKEN', '')}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are an expert data science story teller providing comprehensive dataset insights."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500
            }

            response = requests.post(
                "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", 
                headers=headers, 
                json=payload
            )
            insights = response.json()['choices'][0]['message']['content']
            return insights
        except Exception as e:
            return f"LLM insight generation failed: {e}"

def main():
    """
    Main function to run comprehensive data analysis.
    """
    if len(sys.argv) != 2 or not sys.argv[1].lower().endswith('.csv'):
        print("Usage: python script.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = AdvancedDataAnalyzer(file_path, output_dir)
    
    # Perform comprehensive analysis
    analysis = analyzer.perform_comprehensive_analysis()
    
    # Generate visualizations
    analyzer.generate_visualizations(analysis)
    
    # Get LLM insights
    llm_insights = analyzer.ask_llm_for_insights(analysis)
    
    # Write analysis and LLM insights
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write("# Comprehensive Data Analysis Report\n\n")
        f.write("## Dataset Overview\n")
        f.write(json.dumps(analysis['dataset_overview'], indent=2))
        f.write("\n\n## LLM Insights\n")
        f.write(llm_insights)

    print(f"Analysis complete. Report saved in {output_dir}/README.md")

if __name__ == "__main__":
    main()