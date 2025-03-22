import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA
import numpy as np

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
        
        # Ensure target is numeric (0, 1) format to avoid errors
        target_series = df[target_column]
        if target_series.dtype == 'object':
            unique_values = target_series.unique()
            if len(unique_values) == 2:
                # Convert categorical target to binary numeric (0, 1)
                positive_class = sorted(unique_values)[1]  # Assume alphabetically later is positive
                target_mapping = {val: 1 if val == positive_class else 0 for val in unique_values}
                df = df.copy()  # To avoid modifying the original DataFrame
                df[target_column] = df[target_column].map(target_mapping)
                self.pos_label = 1
                print(f"Converted target '{target_column}' from {unique_values} to binary (0, 1). Positive class: {positive_class}")
        
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
        
        # Define pos_label if not already set
        pos_label = getattr(self, 'pos_label', 1)
        
        # Calcular AUC
        fpr, tpr, _ = roc_curve(self.y_test, y_prob, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        
        # Mostrar resultados
        print(f"\nResultados de Regresión Logística (threshold={threshold}):")
        report = classification_report(self.y_test, y_pred, output_dict=True)
        print(f"Recall: {report[str(pos_label)]['recall']:.3f}")
        print(f"F1-score: {report[str(pos_label)]['f1-score']:.3f}")
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
        
        # Define pos_label if not already set
        pos_label = getattr(self, 'pos_label', 1)
        
        # Calcular AUC
        fpr, tpr, _ = roc_curve(self.y_test, y_prob, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        
        # Mostrar resultados
        print(f"\nResultados de Random Forest (threshold={threshold}):")
        report = classification_report(self.y_test, y_pred, output_dict=True)
        print(f"Recall: {report[str(pos_label)]['recall']:.3f}")
        print(f"F1-score: {report[str(pos_label)]['f1-score']:.3f}")
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
        
        # Define pos_label if not already set
        pos_label = getattr(self, 'pos_label', 1)
        
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
            auc = roc_auc_score(self.y_test, y_prob, multi_class='ovr')
            recall = recall_score(self.y_test, y_pred, pos_label=pos_label)
            f1 = f1_score(self.y_test, y_pred, pos_label=pos_label)
            
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
    
class Regresion:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]
        self.X_scaled = StandardScaler().fit_transform(self.X)
        self.metrics = {}  # Diccionario para almacenar las métricas

    def train_test_split(self, test_size=0.2, random_state=42):
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state
        )
        print(f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")

    def train_linear_regression(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score

        self.lr_model = LinearRegression()
        self.lr_model.fit(self.X_train, self.y_train)

        y_pred = self.lr_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"Linear Regression MSE: {mse:.3f}, R2 Score: {r2:.3f}")
        
        # Guardar métricas
        self.metrics['Regresión Lineal'] = {'MSE': mse, 'R2': r2}
        
        # Visualizar resultados
        fig = self.plot_regression_results(y_pred, "Regresión Lineal", return_fig=True)
        
        return {'model': self.lr_model, 'metrics': {'mse': mse, 'r2': r2}, 'predictions': y_pred, 'fig': fig}

    def train_decision_tree_regressor(self):
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        self.dt_model = DecisionTreeRegressor(random_state=42)
        self.dt_model.fit(self.X_train, self.y_train)

        y_pred = self.dt_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"Decision Tree Regressor MSE: {mse:.3f}, R2 Score: {r2:.3f}")
        
        # Guardar métricas
        self.metrics['Árbol de Decisión'] = {'MSE': mse, 'R2': r2}
        
        # Visualizar resultados
        results_fig = self.plot_regression_results(y_pred, "Árbol de Decisión", return_fig=True)
        
        # Visualizar importancia de características
        importance_fig = self.plot_feature_importance(self.dt_model.feature_importances_, "Árbol de Decisión", return_fig=True)
        
        return {
            'model': self.dt_model, 
            'metrics': {'mse': mse, 'r2': r2}, 
            'predictions': y_pred, 
            'results_fig': results_fig,
            'importance_fig': importance_fig,
            'feature_importance': self.dt_model.feature_importances_
        }

    def train_random_forest_regressor(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(self.X_train, self.y_train)

        y_pred = self.rf_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"Random Forest Regressor MSE: {mse:.3f}, R2 Score: {r2:.3f}")
        
        # Guardar métricas
        self.metrics['Random Forest'] = {'MSE': mse, 'R2': r2}
        
        # Visualizar resultados
        results_fig = self.plot_regression_results(y_pred, "Random Forest", return_fig=True)
        
        # Visualizar importancia de características
        importance_fig = self.plot_feature_importance(self.rf_model.feature_importances_, "Random Forest", return_fig=True)
        
        return {
            'model': self.rf_model, 
            'metrics': {'mse': mse, 'r2': r2}, 
            'predictions': y_pred, 
            'results_fig': results_fig,
            'importance_fig': importance_fig,
            'feature_importance': self.rf_model.feature_importances_
        }

    def plot_regression_results(self, y_pred, model_name, return_fig=False):
        """Visualiza los resultados de la regresión"""
        fig = plt.figure(figsize=(10, 6))
        
        # Gráfico de dispersión de valores reales vs predichos
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        
        # Línea diagonal perfecta
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Predicción Perfecta')
        
        plt.xlabel('Valores Reales')
        plt.ylabel('Valores Predichos')
        plt.title(f'Valores Reales vs Predichos - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if return_fig:
            return fig
        else:
            plt.show()

    def plot_feature_importance(self, importance_scores, model_name, return_fig=False):
        """Visualiza la importancia de las características"""
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        fig = plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title(f'Importancia de Características - {model_name}')
        plt.xlabel('Importancia')
        plt.ylabel('Características')
        plt.tight_layout()
        
        if return_fig:
            return fig
        else:
            plt.show()

    def plot_metrics_comparison(self, return_fig=False):
        """Visualiza la comparación de métricas entre modelos"""
        if not self.metrics:
            print("No hay métricas para comparar. Ejecuta primero los modelos de regresión.")
            return None

        # Crear DataFrame con las métricas
        metrics_df = pd.DataFrame(self.metrics).T
        
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gráfico de MSE
        metrics_df['MSE'].plot(kind='bar', ax=ax1)
        ax1.set_title('Comparación de MSE entre Modelos')
        ax1.set_xlabel('Modelos')
        ax1.set_ylabel('MSE')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de R2
        metrics_df['R2'].plot(kind='bar', ax=ax2)
        ax2.set_title('Comparación de R² entre Modelos')
        ax2.set_xlabel('Modelos')
        ax2.set_ylabel('R²')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if return_fig:
            return fig, metrics_df
        else:
            plt.show()
            return None

class SeriesTemporales:
    def __init__(self, df, date_column):
        """Inicializa la clase con el DataFrame y la columna de fechas."""
        self.df = df.copy()
        self.date_column = date_column
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        self.df.set_index(date_column, inplace=True)

    def resumen_estadistico(self):
        """Devuelve un resumen estadístico de los datos."""
        resumen = self.df.describe()
        resumen.loc['valores_nulos'] = self.df.isnull().sum()
        resumen.loc['valores_unicos'] = self.df.nunique()
        return resumen

    def graficar_series(self, columnas=None, return_fig=False):
        """Grafica las series temporales para las columnas seleccionadas."""
        if columnas is None:
            columnas = self.df.select_dtypes(include=[np.number]).columns

        fig = plt.figure(figsize=(15, 8))
        for col in columnas:
            plt.plot(self.df.index, self.df[col], label=col)

        plt.title('Serie Temporal')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        
        if return_fig:
            return fig
        else:
            plt.show()

    def pronosticar(self, target_column, periods=30, changepoint_prior_scale=0.05):
        """Realiza un pronóstico usando Facebook Prophet."""
        data_prophet = pd.DataFrame({
            'ds': self.df.index,
            'y': self.df[target_column]
        }).reset_index(drop=True)

        modelo = Prophet(changepoint_prior_scale=changepoint_prior_scale)
        modelo.fit(data_prophet)

        futuro = modelo.make_future_dataframe(periods=periods)
        pronostico = modelo.predict(futuro)
        
        # Store the model for later plotting
        self.last_model = modelo
        self.last_forecast = pronostico

        return pronostico

    def graficar_pronostico(self, target_column, periods=30, return_fig=False):
        """Grafica la serie temporal con su pronóstico."""
        pronostico = self.pronosticar(target_column, periods)

        fig = plt.figure(figsize=(15, 8))
        plt.plot(self.df.index, self.df[target_column], label='Datos Históricos', color='blue')
        plt.plot(pronostico['ds'], pronostico['yhat'], label='Pronóstico', color='red', linestyle='--')
        plt.fill_between(pronostico['ds'], pronostico['yhat_lower'], pronostico['yhat_upper'], color='red', alpha=0.1)

        plt.title(f'Pronóstico de la Serie Temporal para {target_column}')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        
        if return_fig:
            return fig
        else:
            plt.show()
            
    def plot_components(self, return_fig=False):
        """Muestra los componentes del pronóstico (tendencia, estacionalidad, etc.)."""
        if not hasattr(self, 'last_model') or not hasattr(self, 'last_forecast'):
            print("Primero debes ejecutar pronosticar() antes de poder mostrar los componentes.")
            return None
            
        fig = self.last_model.plot_components(self.last_forecast)
        
        if return_fig:
            return fig
        else:
            plt.show()
            
    def plot_model(self, return_fig=False):
        """Muestra el gráfico completo del modelo Prophet."""
        if not hasattr(self, 'last_model') or not hasattr(self, 'last_forecast'):
            print("Primero debes ejecutar pronosticar() antes de poder mostrar el modelo.")
            return None
            
        fig = self.last_model.plot(self.last_forecast)
        
        if return_fig:
            return fig
        else:
            plt.show()

if __name__ == "__main__":
    # Example code for clustering
    data_path_clustering = "wine-clustering.csv"
    df = pd.read_csv(data_path_clustering)

    # Apply EDA for Clustering
    eda = EDA(df)
    eda.analyze()

    # Apply Clustering
    clustering = Clustering(df)
    clustering.apply_kmeans()
    clustering.apply_hac()
    clustering.apply_pca()

    # Example code for diabetes classification
    data_path_classification_diabetes = "diabetes_V2.csv"
    df_classification_diabetes = pd.read_csv(data_path_classification_diabetes)

    # EDA for Classification
    eda_classification_diabetes = EDA(df_classification_diabetes)
    eda_classification_diabetes.analyze()

    # Example usage of classification for diabetes dataset
    classification_diabetes = Classification(df_classification_diabetes, target_column='diabetes')
    classification_diabetes.train_test_split()
    classification_diabetes.train_logistic_regression(threshold=0.5)
    classification_diabetes.cross_validate(classification_diabetes.lr_model)
    classification_diabetes.train_random_forest(threshold=0.5)
    classification_diabetes.cross_validate(classification_diabetes.rf_model)
    classification_diabetes.train_svm(threshold=0.5)
    classification_diabetes.cross_validate(classification_diabetes.svm_model)

    # Benchmark for diabetes
    print("\nBenchmark for Diabetes Dataset:")
    diabetes_benchmark = classification_diabetes.benchmark(threshold=0.5)

    # Example code for potability classification
    data_path_classification_potability = "potabilidad_V2.csv"
    df_classification_potability = pd.read_csv(data_path_classification_potability)

    # EDA for Potability
    eda_classification_potability = EDA(df_classification_potability)
    eda_classification_potability.analyze()

    # Example usage of classification for potability dataset
    classification_potability = Classification(df_classification_potability, target_column='Potability')
    classification_potability.train_test_split()
    classification_potability.train_logistic_regression(threshold=0.5)
    classification_potability.cross_validate(classification_potability.lr_model)
    classification_potability.train_random_forest(threshold=0.5)
    classification_potability.cross_validate(classification_potability.rf_model)
    classification_potability.train_svm(threshold=0.5)
    classification_potability.cross_validate(classification_potability.svm_model)

    # Benchmark for potability
    print("\nBenchmark for Potability Dataset:")
    potability_benchmark = classification_potability.benchmark(threshold=0.5)

    # Example code for regression
    data_path_regression = "wine-clustering.csv"
    df_regression = pd.read_csv(data_path_regression)

    # EDA for Regression
    eda_regression = EDA(df_regression)
    eda_regression.analyze()

    # Example usage of regression
    regression = Regresion(df_regression, target_column='Proline')

    # Split data into training and test sets
    regression.train_test_split()

    # Train different regression models
    regression.train_linear_regression()
    regression.train_decision_tree_regressor()
    regression.train_random_forest_regressor()

    # Visualize metrics comparison
    regression.plot_metrics_comparison()

    # Example code for time series analysis
    data_path = "mock_kaggle.csv"
    df = pd.read_csv(data_path)
    # Calculate basic statistics
    stats = df.describe()

    # Add missing values
    stats.loc['missing_values'] = df.isnull().sum()

    # Add unique values
    stats.loc['unique_values'] = df.nunique()

    # Show results
    print("\nStatistical Summary:")
    print(stats) 

    # Create an instance of SeriesTemporales
    series_temporales = SeriesTemporales(df, date_column='data')

    # Plot time series for 'venta' and 'inventario' columns
    series_temporales.graficar_series(columnas=['venta', 'inventario'])

    # Plot forecast for 'venta' column for next 30 periods
    series_temporales.graficar_pronostico(target_column='venta', periods=30)

    # Get forecast
    pronostico = series_temporales.pronosticar(target_column='venta', periods=30, changepoint_prior_scale=0.1)

    # Show forecast information
    print("Forecast shape:", pronostico.shape)
    print("\nFirst rows of forecast:")
    print(pronostico.head())