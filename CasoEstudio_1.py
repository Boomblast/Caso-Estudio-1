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
from sklearn.preprocessing import LabelEncoder


import sklearn
import numpy as np
import scipy
from sklearn.metrics import pairwise_distances

warnings.filterwarnings('ignore')

class EDA:
    def __init__(self, df):
        self.df = df

    def analyze(self):
        """Muestra información básica, valores nulos, boxplots y la matriz de correlación"""
        print("\nInformación del dataset:")
        print(self.df.info())
        print("\nPrimeras 5 filas:")
        print(self.df.head())
        print("\nEstadísticas descriptivas:")
        print(self.df.describe(include='all'))  # Incluye estadísticas de categóricas

        # Manejo de variables categóricas
        self.handle_categorical()

        # Visualizar valores nulos
        plt.figure(figsize=(10,5))
        sns.heatmap(self.df.isnull(), cbar=False, cmap="viridis")
        plt.title("Mapa de valores nulos")
        plt.show()

        # Boxplots para detectar outliers
        plt.figure(figsize=(15,8))
        self.df.select_dtypes(include=[np.number]).boxplot(rot=45, grid=False)
        plt.title("Boxplots de las variables numéricas")
        plt.show()

        # Matriz de correlación
        plt.figure(figsize=(12,8))
        sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Matriz de Correlación")
        plt.show()

    def handle_categorical(self):
        """Maneja las variables categóricas para el análisis exploratorio"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print("\nVariables categóricas detectadas:")
            for col in categorical_cols:
                print(f"- {col}: {self.df[col].nunique()} categorías -> {self.df[col].unique()}")

                # Gráfico de barras para variables categóricas (excluyendo género)
                if col != 'genero':
                    plt.figure(figsize=(8,4))
                    sns.countplot(x=self.df[col], data=self.df, palette="coolwarm")
                    plt.xticks(rotation=45)
                    plt.title(f"Distribución de {col}")
                    plt.show()

            # Codificación específica para el dataset de diabetes
            if 'genero' in self.df.columns:
                self.df['genero'] = self.df['genero'].map({'female': 0, 'male': 1})
                print("\nCodificación de género:")
                print("female -> 0")
                print("male -> 1")
            
            if 'no_diabetes' in self.df.columns and 'diabetes' in self.df.columns:
                # Crear una nueva columna 'Outcome' basada en las columnas existentes
                self.df['Outcome'] = 0  # Inicializar con 0 (no_diabetes)
                self.df.loc[self.df['diabetes'] == 1, 'Outcome'] = 1  # Marcar como 1 donde hay diabetes
                
                # Eliminar las columnas originales
                self.df = self.df.drop(['no_diabetes', 'diabetes'], axis=1)
                print("\nCodificación de diabetes:")
                print("no_diabetes -> 0")
                print("diabetes -> 1")
            
            # Codificar otras variables categóricas con Label Encoding
            remaining_categorical = self.df.select_dtypes(include=['object']).columns
            if len(remaining_categorical) > 0:
                print("\nCodificando otras variables categóricas con Label Encoding...")
                from sklearn.preprocessing import LabelEncoder
                for col in remaining_categorical:
                    self.df[col] = LabelEncoder().fit_transform(self.df[col])

# Clase para Clustering
class Clustering:
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
        # Obtener las columnas numéricas del DataFrame original
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        if 'Cluster' in numeric_columns:
            numeric_columns = numeric_columns.drop('Cluster')
            
        # Make sure we're using the correct number of features
        pca_features = numeric_columns[:len(pca.components_[0])]

        # Create loadings DataFrame
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=pca_features
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

class Classification:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]
        self.X_scaled = StandardScaler().fit_transform(self.X)
        
    def train_test_split(self, test_size=0.2, random_state=42):
        """Divide los datos en conjuntos de entrenamiento y prueba"""
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state
        )
        print(f"\nTamaño del conjunto de entrenamiento: {len(self.X_train)}")
        print(f"Tamaño del conjunto de prueba: {len(self.X_test)}")
        
    def train_logistic_regression(self, threshold=0.5):
        """Entrena un modelo de regresión logística con probabilidad de corte"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        import seaborn as sns
        
        # Entrenar el modelo
        self.lr_model = LogisticRegression(random_state=42)
        self.lr_model.fit(self.X_train, self.y_train)
        
        # Obtener probabilidades
        y_prob = self.lr_model.predict_proba(self.X_test)[:, 1]
        
        # Aplicar threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calcular AUC
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Mostrar resultados
        print(f"\nResultados de Regresión Logística (threshold={threshold}):")
        report = classification_report(self.y_test, y_pred, output_dict=True)
        print(f"Recall: {report['1']['recall']:.3f}")
        print(f"F1-score: {report['1']['f1-score']:.3f}")
        print(f"AUC: {roc_auc:.3f}")
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - Regresión Logística (threshold={threshold})')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.show()
        
        # Curva ROC
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC - Regresión Logística')
        plt.legend(loc="lower right")
        plt.show()
        
    def train_random_forest(self, threshold=0.5):
        """Entrena un modelo de Random Forest con probabilidad de corte"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        import seaborn as sns
        
        # Entrenar el modelo
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(self.X_train, self.y_train)
        
        # Obtener probabilidades
        y_prob = self.rf_model.predict_proba(self.X_test)[:, 1]
        
        # Aplicar threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calcular AUC
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Mostrar resultados
        print(f"\nResultados de Random Forest (threshold={threshold}):")
        report = classification_report(self.y_test, y_pred, output_dict=True)
        print(f"Recall: {report['1']['recall']:.3f}")
        print(f"F1-score: {report['1']['f1-score']:.3f}")
        print(f"AUC: {roc_auc:.3f}")
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - Random Forest (threshold={threshold})')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.show()
        
        # Curva ROC
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC - Random Forest')
        plt.legend(loc="lower right")
        plt.show()
        
        # Importancia de características
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10,6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Importancia de Características - Random Forest')
        plt.show()
        
    def train_svm(self, threshold=0.5):
        """Entrena un modelo de Support Vector Machine con probabilidad de corte"""
        from sklearn.svm import SVC
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        import seaborn as sns
        
        # Entrenar el modelo
        self.svm_model = SVC(kernel='rbf', random_state=42, probability=True)
        self.svm_model.fit(self.X_train, self.y_train)
        
        # Obtener probabilidades
        y_prob = self.svm_model.predict_proba(self.X_test)[:, 1]
        
        # Aplicar threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calcular AUC
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Mostrar resultados
        print(f"\nResultados de SVM (threshold={threshold}):")
        report = classification_report(self.y_test, y_pred, output_dict=True)
        print(f"Recall: {report['1']['recall']:.3f}")
        print(f"F1-score: {report['1']['f1-score']:.3f}")
        print(f"AUC: {roc_auc:.3f}")
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - SVM (threshold={threshold})')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.show()
        
        # Curva ROC
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC - SVM')
        plt.legend(loc="lower right")
        plt.show()

    def cross_validate(self, model, cv=5):
        """Realiza validación cruzada en un modelo"""
        from sklearn.model_selection import cross_val_score
        
        scores = cross_val_score(model, self.X_scaled, self.y, cv=cv)
        print(f"\nResultados de Validación Cruzada ({cv}-fold):")
        print(f"Scores: {scores}")
        print(f"Score promedio: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

    def benchmark(self, threshold=0.5):
        """Crea un benchmark comparando todos los algoritmos de clasificación"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import roc_auc_score, recall_score, f1_score
        
        # Entrenar todos los modelos
        models = {
            'Regresión Logística': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True)
        }
        
        # Crear DataFrame para resultados
        results = []
        
        for name, model in models.items():
            # Entrenar modelo
            model.fit(self.X_train, self.y_train)
            
            # Obtener probabilidades y predicciones
            y_prob = model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calcular métricas
            auc = roc_auc_score(self.y_test, y_prob)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Agregar resultados
            results.append({
                'Modelo': name,
                'AUC': auc,
                'Recall': recall,
                'F1-score': f1
            })
        
        # Crear DataFrame con resultados
        benchmark_df = pd.DataFrame(results)
        
        # Mostrar resultados
        print("\nBenchmark de Modelos de Clasificación:")
        print("=====================================")
        print(benchmark_df.to_string(index=False))
        
        # Visualizar resultados
        plt.figure(figsize=(12, 6))
        metrics = ['AUC', 'Recall', 'F1-score']
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, benchmark_df[metric], width, label=metric)
        
        plt.xlabel('Modelos')
        plt.ylabel('Score')
        plt.title('Comparación de Métricas por Modelo')
        plt.xticks(x + width, benchmark_df['Modelo'])
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return benchmark_df

#Codigo para testear dataset de clustering
""""
data_path_clustering = "C:\\Users\\Jorge\\OneDrive\\Documentos\\Jorge\\LEAD university\\2025\\Mineria de datos avanzada\\Caso de estudio 1\\wine-clustering.csv"
df = pd.read_csv(data_path_clustering)

# Aplicar EDA para Clustering
eda = EDA(df)
eda.analyze()

# Aplicar Clustering
clustering = Clustering(df)
clustering.apply_kmeans()
clustering.apply_hac()
clustering.apply_pca()
"""

#Codigo para testear datasets de diabetes y potabilidad para clasificacion
"""
# Cargar los datos de classification
data_path_classification_diabetes = "C:\\Users\\Jorge\\OneDrive\\Documentos\\Jorge\\LEAD university\\2025\\Mineria de datos avanzada\\Caso de estudio 1\\diabetes_V2.csv"
df_classification_diabetes = pd.read_csv(data_path_classification_diabetes)

# EDA para Classification
eda_classification_diabetes = EDA(df_classification_diabetes)
eda_classification_diabetes.analyze()

# Cargar los datos de potabilidad
data_path_classification_potability = "C:\\Users\\Jorge\\OneDrive\\Documentos\\Jorge\\LEAD university\\2025\\Mineria de datos avanzada\\Caso de estudio 1\\potabilidad_V2.csv"
df_classification_potability = pd.read_csv(data_path_classification_potability)

# EDA para Potabilidad
eda_classification_potability = EDA(df_classification_potability)
eda_classification_potability.analyze()

# Ejemplo de uso de clasificación para el dataset de potabilidad
classification_potability = Classification(df_classification_potability, target_column='Potability')
classification_potability.train_test_split()
classification_potability.train_logistic_regression(threshold=0.5)
classification_potability.cross_validate(classification_potability.lr_model)
classification_potability.train_random_forest(threshold=0.5)
classification_potability.cross_validate(classification_potability.rf_model)
classification_potability.train_svm(threshold=0.5)
classification_potability.cross_validate(classification_potability.svm_model)

# Benchmark para potabilidad
print("\nBenchmark para Dataset de Potabilidad:")
potability_benchmark = classification_potability.benchmark(threshold=0.5)

# Ejemplo de uso de clasificación para el dataset de diabetes
classification_diabetes = Classification(df_classification_diabetes, target_column='diabetes')
classification_diabetes.train_test_split()
classification_diabetes.train_logistic_regression(threshold=0.5)
classification_diabetes.cross_validate(classification_diabetes.lr_model)
classification_diabetes.train_random_forest(threshold=0.5)
classification_diabetes.cross_validate(classification_diabetes.rf_model)
classification_diabetes.train_svm(threshold=0.5)
classification_diabetes.cross_validate(classification_diabetes.svm_model)

# Benchmark para diabetes
print("\nBenchmark para Dataset de Diabetes:")
diabetes_benchmark = classification_diabetes.benchmark(threshold=0.5)
"""