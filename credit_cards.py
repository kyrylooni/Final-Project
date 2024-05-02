import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN
from rich.progress import Progress

# Function to load and preprocess the data
def load_and_preprocess_data(file_path):
    with Progress() as progress:
        task1 = progress.add_task("[cyan]Loading data...", total=1)
        df = pd.read_csv(file_path)
        progress.update(task1, advance=1)

        # Ignore the first column
        df = df.iloc[:, 1:]

        task2 = progress.add_task("[magenta]Handling missing values...", total=1)
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        progress.update(task2, advance=1)

        task3 = progress.add_task("[yellow]Standardizing data...", total=1)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df)
        progress.update(task3, advance=1)

        return data_scaled

# Function to perform PCA and clustering
def pca_and_clustering(data_scaled):
    with Progress() as progress:
        task4 = progress.add_task("[green]Applying PCA...", total=1)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data_scaled)
        progress.update(task4, advance=1)

        task5 = progress.add_task("[blue]Performing DBSCAN clustering...", total=1)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(principal_components)
        progress.update(task5, advance=1)

        return principal_components, clusters

# Visualization function with outlier loading from CSV
def visualize_clusters(principal_components, clusters):
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.6)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Cluster Visualization')

    # Load outliers from CSV
    outlier_data = pd.read_csv('outliers.csv')
    plt.scatter(outlier_data['PC1'], outlier_data['PC2'], c='red', s=100, label='Outlier', edgecolors='black')  # Highlight outliers

    plt.legend()
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.savefig('cluster_visualization_with_outliers.png', format='png', dpi=300)
    plt.show()

# Main function
def main():
    file_path = 'CC GENERAL.csv'
    data_scaled = load_and_preprocess_data(file_path)
    principal_components, clusters = pca_and_clustering(data_scaled)
    visualize_clusters(principal_components, clusters)

if __name__ == "__main__":
    main()




# Turn the outliers into a pandas dataframe and save the results to a CSV file
