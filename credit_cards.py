import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN
from rich.progress import Progress

def load_and_preprocess_data(file_path):
    with Progress() as progress:
        task1 = progress.add_task("[cyan]Loading data...", total=1)
        df = pd.read_csv(file_path)
        progress.update(task1, advance=1)

        # Save identifiers and process the rest of the data
        identifiers = df['CUST_ID']
        df = df.iloc[:, 1:]  # Assume CUST_ID is the first column

        task2 = progress.add_task("[magenta]Handling missing values...", total=1)
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        progress.update(task2, advance=1)

        task3 = progress.add_task("[yellow]Standardizing data...", total=1)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df)
        progress.update(task3, advance=1)

        return data_scaled, identifiers

def pca_and_clustering(data_scaled, identifiers):
    with Progress() as progress:
        task4 = progress.add_task("[green]Applying PCA...", total=1)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data_scaled)
        progress.update(task4, advance=1)

        task5 = progress.add_task("[blue]Performing DBSCAN clustering...", total=1)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(principal_components)
        progress.update(task5, advance=1)

        return principal_components, clusters, identifiers

def visualize_clusters(principal_components, clusters, identifiers):
    plt.figure(figsize=(14, 10))  # Increased figure size for better clarity

    # Separate regular points and outliers for different visualization styles
    regular_mask = clusters != -1
    outlier_mask = clusters == -1
    regular_data = principal_components[regular_mask]
    outliers = principal_components[outlier_mask]
    outlier_ids = identifiers[outlier_mask]

    # Plot regular data points
    plt.scatter(regular_data[:, 0], regular_data[:, 1], c='blue', label='Regular Data', alpha=0.5, s=50)

    # Plot outliers in red
    plt.scatter(outliers[:, 0], outliers[:, 1], c='red', label='Outliers', s=100, edgecolors='black')

    # Annotate outliers
    for idx, (x, y) in enumerate(outliers):
        plt.annotate(outlier_ids.iloc[idx], (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Cluster Visualization with Enhanced Outlier Visibility')
    plt.legend()
    plt.colorbar(label='Cluster ID')

    # Adjust axes to spread out the points
    x_min, x_max = regular_data[:, 0].min(), regular_data[:, 0].max()
    y_min, y_max = regular_data[:, 1].min(), regular_data[:, 1].max()
    x_pad = (x_max - x_min) * 0.1  # Padding around the x limits
    y_pad = (y_max - y_min) * 0.1  # Padding around the y limits
    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    plt.savefig('cluster_visualization_with_outliers.png', format='png', dpi=300)
    plt.grid(True)
    plt.show()

def main():
    file_path = 'CC GENERAL.csv'
    data_scaled, identifiers = load_and_preprocess_data(file_path)
    principal_components, clusters, identifiers = pca_and_clustering(data_scaled, identifiers)
    visualize_clusters(principal_components, clusters, identifiers)

if __name__ == "__main__":
    main()
