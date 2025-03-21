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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import statistics
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
from tqdm import tqdm
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, confusion_matrix, roc_auc_score, mean_absolute_error, roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, classification_report

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

    def apply_pca(self, n_components=2):
        """Aplica PCA y visualiza los resultados"""
        # Aplicar PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(self.X_scaled)
        
        # Calcular varianza explicada
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # Visualizar varianza explicada
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        plt.xlabel('Componente Principal')
        plt.ylabel('Ratio de Varianza Explicada')
        plt.title('Varianza Explicada por Componente')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 
                marker='o', linestyle='--')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Varianza Explicada Acumulada')
        plt.title('Varianza Explicada Acumulada vs Componentes')
        plt.tight_layout()
        plt.show()
        
        # Crear DataFrame con resultados de PCA
        pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        
        # Si hay clusters previos, agregarlos al DataFrame
        if 'Cluster' in self.df.columns:
            pca_df['Cluster'] = self.df['Cluster']
            
            # Visualizar clusters en espacio PCA
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis')
            plt.xlabel('Primera Componente Principal')
            plt.ylabel('Segunda Componente Principal')
            plt.title('Visualización de Clusters en Espacio PCA')
            plt.colorbar(scatter, label='Cluster')
            plt.show()
        else:
            # Visualizar datos sin clusters
            plt.figure(figsize=(10, 6))
            plt.scatter(pca_df['PC1'], pca_df['PC2'])
            plt.xlabel('Primera Componente Principal')
            plt.ylabel('Segunda Componente Principal')
            plt.title('Visualización de Datos en Espacio PCA')
            plt.show()
            
        # Mostrar la contribución de las variables originales a los componentes principales
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=self.df.columns
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
        plt.title('Contribución de Variables a Componentes Principales')
        plt.show()
        
        # Guardar resultados en el objeto
        self.pca = pca
        self.pca_df = pca_df
        
        # Imprimir resumen
        print("\nResumen de PCA:")
        print(f"Varianza explicada por componente: {explained_variance_ratio}")
        print(f"Varianza explicada acumulada: {cumulative_variance_ratio}")
        
        return pca_df
# Clase para la regresion
class Regresion:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column
        self.features = pd.get_dummies(df.drop(columns=[target_column]), drop_first=True)
        self.target = df[target_column]
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(self.features)
        self.modelos = {
            'linear': LinearRegression(),
            'lasso': Lasso(),
            'ridge': Ridge(),
            'svr': SVR(),
            'decision_tree': DecisionTreeRegressor(),
            'random_forest': RandomForestRegressor()
        }
        self.resultados = []

    def ajustar_hiperparametros(self):
        parametros = {
            'lasso': {'alpha': np.logspace(-4, 0, 10)},
            'ridge': {'alpha': np.logspace(-4, 0, 10)},
            'svr': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}
        }
        
        for nombre, modelo in self.modelos.items():
            if nombre in parametros:
                print(f"Ajustando hiperparámetros para {nombre}...")
                grid = GridSearchCV(modelo, parametros[nombre], cv=5)
                grid.fit(self.features_scaled, self.target)
                self.modelos[nombre] = grid.best_estimator_

    def validar_modelos(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for nombre, modelo in tqdm(self.modelos.items(), desc="Entrenando modelos"):
            r2_scores = []
            mse_scores = []
            
            for train_index, val_index in kf.split(self.features_scaled):
                X_train, X_val = self.features_scaled[train_index], self.features_scaled[val_index]
                y_train, y_val = self.target.iloc[train_index], self.target.iloc[val_index]
                
                modelo.fit(X_train, y_train)
                predictions = modelo.predict(X_val)
                mse = mean_squared_error(y_val, predictions)
                r2 = r2_score(y_val, predictions)
                
                mse_scores.append(mse)
                r2_scores.append(r2)
            
            mse_promedio = np.mean(mse_scores)
            r2_promedio = np.mean(r2_scores)
            std_mse = np.std(mse_scores)
            std_r2 = np.std(r2_scores)
            
            self.resultados.append((nombre, mse_promedio, r2_promedio, std_mse, std_r2))

    def mostrar_top_3_por_familia(self):
        familias = {
            'lineales': ['linear', 'lasso', 'ridge'],
            'arboles': ['decision_tree', 'random_forest'],
            'otros': ['svr']
        }

        print("=== Top 3 por Familia ===")
        for familia, modelos in familias.items():
            resultados_familia = [r for r in self.resultados if r[0] in modelos]
            resultados_familia_ordenados = sorted(resultados_familia, key=lambda x: x[2], reverse=True)[:3]
            print(f"--- {familia.capitalize()} ---")
            for i, (modelo, mse_promedio, r2_promedio, std_mse, std_r2) in enumerate(resultados_familia_ordenados, 1):
                print(f"Top {i} - Modelo: {modelo}")
                print(f"R^2 promedio: {round(r2_promedio, 4)}")
                print(f"MSE promedio: {round(mse_promedio, 2)}")
                print(f"Desviación estándar R^2: {round(std_r2, 4)}")
                print(f"Desviación estándar MSE: {round(std_mse, 2)}")
                print("\n--------------------\n")

    def mostrar_top_3_general(self):
        resultados_ordenados = sorted(self.resultados, key=lambda x: x[2], reverse=True)[:3]

        print("=== Top 3 Modelos de Regresión ===")
        for i, (modelo, mse_promedio, r2_promedio, std_mse, std_r2) in enumerate(resultados_ordenados, 1):
            print(f"Top {i} - Modelo: {modelo}")
            print(f"R^2 promedio: {round(r2_promedio, 4)}")
            print(f"MSE promedio: {round(mse_promedio, 2)}")
            print(f"Desviación estándar R^2: {round(std_r2, 4)}")
            print(f"Desviación estándar MSE: {round(std_mse, 2)}")
            print("\n--------------------\n")        

# Cargar los datos

# Ruta de Jorge
#data_path = "C:\\Users\\Jorge\\OneDrive\\Documentos\\Jorge\\LEAD university\\2025\\Mineria de datos avanzada\\Caso de estudio 1\\wine-clustering.csv"
#df = pd.read_csv(data_path)
#Ruta Relativa
df = pd.read_csv("wine-clustering.csv")

# Aplicar EDA
eda = EDA(df)
#eda.analyze()

# Aplicar K-Means
unsupervised = UnsupervisedLearning(df)
#unsupervised.apply_kmeans()
#unsupervised.apply_hac()

# Aplicar PCA
unsupervised.apply_pca()

