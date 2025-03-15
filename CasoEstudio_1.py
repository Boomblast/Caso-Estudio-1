import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster

import sklearn
import numpy as np
import scipy
from sklearn.metrics import pairwise_distances

warnings.filterwarnings('ignore')

# Clase para Análisis Exploratorio de Datos (EDA)
class EDA:
    def __init__(self, df):
        self.df = df

    def analyze(self):
        """Muestra información básica, valores nulos, boxplots y la matriz de correlación"""
        print("\n Información del dataset:")
        print(self.df.info())
        print("\n Primeras 5 filas:")
        print(self.df.head())
        print("\n Estadísticas descriptivas:")
        print(self.df.describe())

        # Visualizar valores nulos
        plt.figure(figsize=(10,5))
        sns.heatmap(self.df.isnull(), cbar=False, cmap="viridis")
        plt.title("Mapa de valores nulos")
        plt.show()

        # Boxplots para detectar outliers
        plt.figure(figsize=(15,8))
        self.df.boxplot(rot=45, grid=False)
        plt.title("Boxplots de las variables")
        plt.show()

        # Matriz de correlación
        plt.figure(figsize=(12,8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Matriz de Correlación")
        plt.show()

# Clase para Algoritmos No Supervisados
class UnsupervisedLearning:
    def __init__(self, df):
        self.df = df
        self.X_scaled = StandardScaler().fit_transform(df)  # Normalización automática
        

    def apply_kmeans(self, max_k=10, k=3):
        """Aplica K-Means, grafica el método del codo y muestra los clusters"""

        # Método del codo
        inertia = []
        for i in range(1, max_k+1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(self.X_scaled)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(8,5))
        plt.plot(range(1, max_k+1), inertia, marker='o', linestyle='--')
        plt.xlabel('Número de Clusters (k)')
        plt.ylabel('Inercia')
        plt.title('Método del Codo')
        plt.show()

        # Aplicar K-Means con el número seleccionado de clusters
        self.kmeans = KMeans(n_clusters=k, random_state=42)
        self.df['Cluster'] = self.kmeans.fit_predict(self.X_scaled)
        print(" K-Means aplicado correctamente")
        print(self.df.head())

        # Calcular y mostrar el Silhouette Score
        silhouette_avg = silhouette_score(self.X_scaled, self.df['Cluster'])
        print(f"\nSilhouette Score: {silhouette_avg:.3f}")
        print("Interpretación del Silhouette Score del K-Means:")
        print("- Cercano a 1: Clusters bien definidos")
        print("- Cercano a 0: Clusters solapados")
        print("- Cercano a -1: Posible asignación incorrecta de muestras a clusters")

        # Graficar clusters
        plt.scatter(self.df.iloc[:, 0], self.df.iloc[:, 1], c=self.df['Cluster'], cmap='viridis')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Clusters formados por K-Means")
        plt.show()

    def apply_hac(self):
        """Aplica Clustering Jerárquico (HAC) con valores por defecto y asigna clusters"""
        
        # Definir parámetros por defecto
        k = 4  # Número de clusters
        linkage_method = 'complete'  # Método de enlace
        max_k = 10  # Máximo número de clusters para el método del codo

        # Método del codo para HAC
        inertias = []
        for i in range(1, max_k + 1):
            hac = AgglomerativeClustering(n_clusters=i)
            hac.fit(self.X_scaled)
            # Calcular inercia manualmente ya que HAC no tiene el atributo inertia_
            cluster_centers = np.array([self.X_scaled[hac.labels_ == j].mean(axis=0) for j in range(i)])
            inertia = sum(np.min(np.sum((self.X_scaled - cluster_centers[:, np.newaxis])**2, axis=2)) for cluster_centers in [cluster_centers])
            inertias.append(inertia)

        # Visualizar método del codo
        plt.figure(figsize=(8,5))
        plt.plot(range(1, max_k + 1), inertias, marker='o', linestyle='--')
        plt.xlabel('Número de Clusters (k)')
        plt.ylabel('Inercia')
        plt.title('Método del Codo - HAC')
        plt.show()

        # Paso 1: Calcular linkage
        linked = linkage(self.X_scaled, method=linkage_method)

        # Paso 2: Visualizar el dendrograma
        plt.figure(figsize=(10, 5))
        dendrogram(linked)
        plt.title("Dendrograma - Clustering Jerárquico")
        plt.xlabel("Muestras")
        plt.ylabel("Distancia")
        plt.show()

        # Paso 3: Asignar los clusters
        clusters = fcluster(linked, k, criterion='maxclust')
        self.df['Cluster'] = clusters
        print("HAC aplicado correctamente")
        print(self.df.head())

        # Paso 4: Calcular y mostrar el Silhouette Score
        silhouette_avg = silhouette_score(self.X_scaled, clusters)
        print(f"\nSilhouette Score: {silhouette_avg:.3f}")
        print("Interpretación del Silhouette Score del HAC:")
        print("- Cercano a 1: Clusters bien definidos")
        print("- Cercano a 0: Clusters solapados")
        print("- Cercano a -1: Posible asignación incorrecta de muestras a clusters")

# Cargar los datos
data_path = "C:\\Users\\Jorge\\OneDrive\\Documentos\\Jorge\\LEAD university\\2025\\Mineria de datos avanzada\\Caso de estudio 1\\wine-clustering.csv"
df = pd.read_csv(data_path)

# Aplicar EDA
eda = EDA(df)
eda.analyze()

# Aplicar K-Means
unsupervised = UnsupervisedLearning(df)
unsupervised.apply_kmeans()
unsupervised.apply_hac()