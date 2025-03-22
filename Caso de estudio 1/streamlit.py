import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from CEvaluator import ModelEvaluator
from sklearn.model_selection import train_test_split
import numpy as np
from CasoEstudio_1 import EDA, Clustering, Classification, Regresion, SeriesTemporales

# Set page configuration
st.set_page_config(
    page_title="Análisis de valor de Agua",
    page_icon="💧",
    layout="wide"
)

# Title and description
st.title("📊 Dashboard de valor de Agua")
st.markdown("Análisis y visualización de datos de valor de agua")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('consumo_agua.csv', sep=';')
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# Function to process uploaded data
def process_uploaded_file(uploaded_file):
    try:
        # Try with different separators
        try:
            df = pd.read_csv(uploaded_file, sep=',')
        except:
            try:
                df = pd.read_csv(uploaded_file, sep=';')
            except:
                df = pd.read_csv(uploaded_file, sep='\t')
        
        # Check if date column exists and convert
        date_cols = [col for col in df.columns if 'fecha' in col.lower() or 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            for col in date_cols:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        return df
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None

df = load_data()

if df is not None:
    # Convert fecha to datetime first
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Create derived columns
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    
    # Create season mapping
    season_map = {
        12: 'Verano', 1: 'Verano', 2: 'Verano',
        3: 'Otoño', 4: 'Otoño', 5: 'Otoño',
        6: 'Invierno', 7: 'Invierno', 8: 'Invierno',
        9: 'Primavera', 10: 'Primavera', 11: 'Primavera'
    }
    df['estacion'] = df['mes'].map(season_map)
    
    # Add sidebar
    st.sidebar.header("Filtros")
    
    # Add date range filter
    st.sidebar.subheader("Rango de Fechas")
    fecha_min = df['fecha'].min().date()
    fecha_max = df['fecha'].max().date()
    
    fecha_inicio = st.sidebar.date_input(
        "Fecha Inicial",
        value=fecha_min,
        min_value=fecha_min,
        max_value=fecha_max
    )
    
    fecha_fin = st.sidebar.date_input(
        "Fecha Final",
        value=fecha_max,
        min_value=fecha_min,
        max_value=fecha_max
    )
    
    # Add value range filter
    st.sidebar.subheader("Rango de Valores")
    valor_min, valor_max = st.sidebar.slider(
        "Seleccionar Rango de Valores ($)",
        float(df['valor'].min()),
        float(df['valor'].max()),
        (float(df['valor'].min()), float(df['valor'].max())),
        format="$%.2f"
    )
    
    # Add season filter
    st.sidebar.subheader("Estación")
    estacion_seleccionada = st.sidebar.multiselect(
        "Seleccionar Estaciones",
        options=['Verano', 'Otoño', 'Invierno', 'Primavera'],
        default=['Verano', 'Otoño', 'Invierno', 'Primavera']
    )
    
    # Add year filter
    años = sorted(df['año'].unique())
    año_seleccionado = st.sidebar.selectbox(
        "Seleccionar Año",
        options=años,
        index=len(años)-1
    )
    
    # Add month filter
    meses = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    mes_seleccionado = st.sidebar.selectbox(
        "Seleccionar Mes",
        options=list(meses.keys()),
        format_func=lambda x: meses[x],
        index=0
    )
    
    # Update filtered data with all filters
    df_filtered = df[
        (df['año'] == año_seleccionado) &
        (df['mes'] == mes_seleccionado) &
        (df['fecha'].dt.date >= fecha_inicio) &
        (df['fecha'].dt.date <= fecha_fin) &
        (df['valor'] >= valor_min) &
        (df['valor'] <= valor_max) &
        (df['estacion'].isin(estacion_seleccionada))
    ]
    
    # Add reset filters button
    if st.sidebar.button("Restablecer Filtros"):
        st.rerun()
    
    # Add summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Resumen del período seleccionado")
    st.sidebar.metric(
        label="Valor total",
        value=f"${df_filtered['valor'].sum():,.2f}"
    )
    st.sidebar.metric(
        label="Promedio mensual",
        value=f"${df_filtered['valor'].mean():,.2f}"
    )

    # Add visualization selector in sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Visualizaciones")
    
    viz_options = {
        "Tendencia Temporal": "time_series",
        "Promedio Mensual": "monthly_avg",
        "Comparación Anual": "yearly_comparison",
        "Análisis Estacional": "seasonal",
        "Distribución de Valores": "distribution"
    }
    
    selected_viz = st.sidebar.multiselect(
        "Seleccionar Visualizaciones",
        options=list(viz_options.keys()),
        default=list(viz_options.keys())
    )

    # Add sidebar sections
    st.sidebar.markdown("---")
    sidebar_sections = st.sidebar.radio(
        "Secciones",
        ["Análisis de Datos", "Análisis Exploratorio", "Clustering", "Clasificación", "Regresión", "Series Temporales", "Entrenamiento de Modelos"]
    )

    if sidebar_sections == "Análisis de Datos":
        # Statistical Analysis Section
        st.header("📊 Análisis Numérico")
        
        # Show data preview first
        st.subheader("Vista previa de los datos filtrados")
        st.dataframe(df_filtered)
        
        # Basic statistics
        st.subheader("Estadísticas básicas")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Valor promedio",
                value=f"${df['valor'].mean():,.2f}"
            )
        
        with col2:
            st.metric(
                label="Valor máximo",
                value=f"${df['valor'].max():,.2f}"
            )
        
        with col3:
            st.metric(
                label="Valor mínimo",
                value=f"${df['valor'].min():,.2f}"
            )

        # Advanced statistics
        st.subheader("Análisis Estadístico Detallado")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Desviación Estándar", f"${df['valor'].std():,.2f}")
            st.metric("Mediana", f"${df['valor'].median():,.2f}")
        
        with col2:
            st.metric("Percentil 75", f"${df['valor'].quantile(0.75):,.2f}")
            st.metric("Percentil 25", f"${df['valor'].quantile(0.25):,.2f}")

        # Visual Analysis Section
        st.header("📈 Análisis Visual")

        if "Tendencia Temporal" in selected_viz:
            st.subheader("Tendencia de valor a lo largo del tiempo")
            fig = px.line(
                df,
                x='fecha',
                y='valor',
                title='Valor de agua a lo largo del tiempo',
                template='plotly_dark',
                color_discrete_sequence=['#00C9FF']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=20
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if "Promedio Mensual" in selected_viz:
            st.subheader("Valor promedio mensual")
            monthly_avg = df.groupby('mes')['valor'].mean().reset_index()
            
            fig_monthly = px.bar(
                monthly_avg,
                x='mes',
                y='valor',
                title='Valor promedio por mes',
                labels={'mes': 'Mes', 'valor': 'Valor promedio ($)'},
                template='plotly_dark',
                color_discrete_sequence=['#00E676']
            )
            fig_monthly.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=20
            )
            fig_monthly.update_xaxes(
                ticktext=list(meses.values()),
                tickvals=list(meses.keys())
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        if "Comparación Anual" in selected_viz:
            st.subheader("Comparación Año a Año")
            yearly_avg = df.groupby('año')['valor'].agg(['mean', 'sum']).reset_index()
            col1, col2 = st.columns(2)
            
            with col1:
                fig_yearly = px.line(
                    yearly_avg,
                    x='año',
                    y='mean',
                    title='Promedio Anual',
                    template='plotly_dark',
                    color_discrete_sequence=['#FF4081']
                )
                fig_yearly.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_font_size=20
                )
                st.plotly_chart(fig_yearly, use_container_width=True)
            
            with col2:
                fig_yearly_total = px.bar(
                    yearly_avg,
                    x='año',
                    y='sum',
                    title='Total Anual',
                    template='plotly_dark',
                    color_discrete_sequence=['#7C4DFF']
                )
                fig_yearly_total.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_font_size=20
                )
                st.plotly_chart(fig_yearly_total, use_container_width=True)

        if "Análisis Estacional" in selected_viz:
            st.subheader("Análisis Estacional")
            seasonal_avg = df.groupby('estacion')['valor'].mean().reset_index()
            
            fig_seasonal = px.pie(
                seasonal_avg,
                values='valor',
                names='estacion',
                title='Distribución Estacional del Consumo',
                template='plotly_dark',
                color_discrete_sequence=['#FF9800', '#4CAF50', '#2196F3', '#F44336']
            )
            fig_seasonal.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=20
            )
            st.plotly_chart(fig_seasonal, use_container_width=True)

        if "Distribución de Valores" in selected_viz:
            st.subheader("Distribución de Valores")
            num_bins = st.slider("Número de intervalos", min_value=10, max_value=50, value=20)
            
            fig_hist = px.histogram(
                df,
                x='valor',
                nbins=num_bins,
                title='Distribución de Valores',
                template='plotly_dark',
                color_discrete_sequence=['#00BFA5']
            )
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=20
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Download data option
        st.subheader("Descargar datos")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name="Valor_agua_processed.csv",
            mime="text/csv"
        )
    elif sidebar_sections == "Análisis Exploratorio":
        st.header("🔍 Análisis Exploratorio de Datos")
        
        # Add file upload widget
        st.subheader("Cargar un dataset personalizado (opcional)")
        uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"], key="eda_uploader")
        
        # Use uploaded dataset or default dataset
        if uploaded_file is not None:
            custom_df = process_uploaded_file(uploaded_file)
            if custom_df is not None:
                st.success(f"Dataset cargado correctamente con {custom_df.shape[0]} filas y {custom_df.shape[1]} columnas.")
                eda_df = custom_df
            else:
                st.warning("Usando el dataset por defecto debido a error en el archivo subido.")
                eda_df = df
        else:
            eda_df = df
        
        if st.button("Realizar Análisis Exploratorio"):
            with st.spinner("Realizando análisis exploratorio..."):
                # Create an instance of EDA
                eda = EDA(eda_df)
                
                # Create a placeholder for the EDA visualizations
                eda_plots = st.container()
                
                with eda_plots:
                    # Handle categorical variables first
                    st.subheader("Variables Categóricas")
                    categorical_cols = eda_df.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        st.write("Variables categóricas detectadas:")
                        for col in categorical_cols:
                            st.write(f"- {col}: {eda_df[col].nunique()} categorías")
                            fig = px.bar(eda_df[col].value_counts().reset_index(), 
                                        x='index', y=col, 
                                        title=f"Distribución de {col}")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No se detectaron variables categóricas en el dataset.")
                    
                    # Display boxplots for numeric features
                    st.subheader("Distribución de Variables Numéricas")
                    numeric_cols = eda_df.select_dtypes(include=[np.number]).columns
                    
                    for col in numeric_cols:
                        fig = px.box(eda_df, y=col, title=f"Boxplot de {col}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display correlation matrix
                    st.subheader("Matriz de Correlación")
                    corr = eda_df.select_dtypes(include=[np.number]).corr()
                    fig = px.imshow(corr, 
                                   text_auto=True, 
                                   color_continuous_scale='RdBu_r',
                                   title="Matriz de Correlación")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Check for null values
                    st.subheader("Valores Nulos")
                    null_counts = eda_df.isnull().sum()
                    if null_counts.sum() > 0:
                        fig = px.bar(null_counts[null_counts > 0].reset_index(), 
                                    x='index', y=0, 
                                    title="Cantidad de Valores Nulos por Columna")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("No se detectaron valores nulos en el dataset.")
    elif sidebar_sections == "Clustering":
        st.header("🔮 Análisis de Clustering")
        
        # Add file upload widget
        st.subheader("Cargar un dataset personalizado (opcional)")
        uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"], key="clustering_uploader")
        
        # Use uploaded dataset or default dataset
        if uploaded_file is not None:
            custom_df = process_uploaded_file(uploaded_file)
            if custom_df is not None:
                st.success(f"Dataset cargado correctamente con {custom_df.shape[0]} filas y {custom_df.shape[1]} columnas.")
                cluster_df = custom_df
            else:
                st.warning("Usando el dataset por defecto debido a error en el archivo subido.")
                cluster_df = df
        else:
            cluster_df = df
        
        # Select features for clustering
        st.subheader("Seleccionar características para clustering")
        numeric_cols = cluster_df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.multiselect("Seleccionar características", numeric_cols, default=numeric_cols[:3] if len(numeric_cols) > 2 else numeric_cols)
        
        if len(selected_features) < 2:
            st.warning("Por favor selecciona al menos 2 características para realizar clustering.")
        else:
            # Parameters for clustering
            st.subheader("Parámetros de Clustering")
            
            clustering_method = st.radio("Método de Clustering", ["K-Means", "Clustering Jerárquico"])
            
            if clustering_method == "K-Means":
                k_value = st.slider("Número de clusters (k)", min_value=2, max_value=10, value=3)
                max_k = st.slider("Máximo k para método del codo", min_value=5, max_value=15, value=10)
            
            # Apply PCA option
            apply_pca = st.checkbox("Aplicar PCA para visualización", value=True)
            
            if st.button("Realizar Clustering"):
                with st.spinner("Realizando clustering..."):
                    # Prepare data for clustering
                    X = cluster_df[selected_features]
                    
                    # Initialize clustering
                    clustering = Clustering(X)
                    
                    if clustering_method == "K-Means":
                        st.subheader("Resultado de K-Means Clustering")
                        clustering.apply_kmeans(max_k=max_k, k=k_value)
                        
                        # Add cluster column to original dataframe
                        df_result = cluster_df.copy()
                        df_result['Cluster'] = clustering.df['Cluster']
                        
                        # Display sample of results
                        st.write("Muestra de datos con clusters asignados:")
                        st.dataframe(df_result.head(10))
                        
                    elif clustering_method == "Clustering Jerárquico":
                        st.subheader("Resultado de Clustering Jerárquico")
                        clustering.apply_hac()
                        
                        # Add cluster column to original dataframe
                        df_result = cluster_df.copy()
                        df_result['Cluster'] = clustering.df['Cluster']
                        
                        # Display sample of results
                        st.write("Muestra de datos con clusters asignados:")
                        st.dataframe(df_result.head(10))
                    
                    # Apply PCA if selected
                    if apply_pca:
                        st.subheader("Visualización PCA")
                        pca_df = clustering.apply_pca(n_components=2)
                        
                        # Create scatter plot with PCA results
                        fig = px.scatter(
                            pca_df, x='PC1', y='PC2', color='Cluster',
                            title='Visualización de Clusters con PCA',
                            labels={'PC1': 'Primera Componente Principal', 'PC2': 'Segunda Componente Principal'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
    elif sidebar_sections == "Clasificación":
        st.header("🏷️ Análisis de Clasificación")
        
        # Add file upload widget
        st.subheader("Cargar un dataset personalizado (opcional)")
        uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"], key="classification_uploader")
        
        # Use uploaded dataset or default dataset
        if uploaded_file is not None:
            custom_df = process_uploaded_file(uploaded_file)
            if custom_df is not None:
                st.success(f"Dataset cargado correctamente con {custom_df.shape[0]} filas y {custom_df.shape[1]} columnas.")
                class_df = custom_df
            else:
                st.warning("Usando el dataset por defecto debido a error en el archivo subido.")
                class_df = df
        else:
            class_df = df
        
        # Select target variable
        st.subheader("Seleccionar variable objetivo")
        all_cols = class_df.columns.tolist()
        target_variable = st.selectbox("Variable objetivo para clasificación", all_cols)
        
        if target_variable:
            # Check if target is suitable for classification
            unique_values = class_df[target_variable].nunique()
            
            if unique_values > 10:
                st.warning(f"La variable seleccionada tiene {unique_values} valores únicos. Para clasificación se recomienda una variable con menos categorías.")
            
            # Select features
            feature_cols = [col for col in all_cols if col != target_variable]
            selected_features = st.multiselect("Seleccionar características", feature_cols, default=feature_cols[:min(5, len(feature_cols))])
            
            if len(selected_features) < 1:
                st.warning("Por favor selecciona al menos una característica para el modelo.")
            else:
                # Parameters
                test_size = st.slider("Tamaño del conjunto de prueba", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
                threshold = st.slider("Umbral de probabilidad para clasificación", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
                
                if st.button("Ejecutar Clasificación"):
                    with st.spinner("Entrenando modelos de clasificación..."):
                        # Prepare data
                        X = class_df[selected_features]
                        y = class_df[target_variable]
                        
                        # Check if target variable is categorical/string and convert to numeric
                        if y.dtype == 'object':
                            st.info(f"La variable objetivo '{target_variable}' contiene valores categóricos. Realizando conversión automática a valores numéricos.")
                            # Get unique values and convert to dictionary mapping
                            unique_values = y.unique()
                            if len(unique_values) == 2:
                                # Create mapping dictionary - assume the alphabetically later value is positive
                                positive_class = sorted(unique_values)[1]
                                y_mapping = {val: 1 if val == positive_class else 0 for val in unique_values}
                                st.write(f"Conversión: {y_mapping} (clase positiva: '{positive_class}')")
                                # Apply mapping
                                y = y.map(y_mapping)
                                can_continue = True
                            else:
                                st.error(f"La variable objetivo tiene {len(unique_values)} valores únicos. Para clasificación binaria se requieren exactamente 2 valores.")
                                can_continue = False
                        else:
                            can_continue = True
                        
                        # Create DataFrame for classification and proceed only if we can continue
                        if can_continue:
                            df_class = pd.concat([X, y], axis=1)
                            
                            try:
                                # Initialize Classification
                                classification = Classification(df_class, target_column=target_variable)
                                
                                # Split data
                                classification.train_test_split(test_size=test_size)
                                
                                # Train models
                                st.subheader("Resultados de Regresión Logística")
                                classification.train_logistic_regression(threshold=threshold)
                                
                                st.subheader("Resultados de Random Forest")
                                classification.train_random_forest(threshold=threshold)
                                
                                # Create benchmark
                                st.subheader("Benchmark de Modelos")
                                benchmark_df = classification.benchmark(threshold=threshold)
                                
                                # Display benchmark results
                                fig = px.bar(
                                    benchmark_df, 
                                    x='Modelo', 
                                    y=['AUC', 'Recall', 'F1-score'],
                                    title='Comparación de Métricas por Modelo',
                                    barmode='group'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Error en la clasificación: {e}")
    elif sidebar_sections == "Regresión":
        st.header("📈 Análisis de Regresión")
        
        # Add file upload widget
        st.subheader("Cargar un dataset personalizado (opcional)")
        uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"], key="regression_uploader")
        
        # Use uploaded dataset or default dataset
        if uploaded_file is not None:
            custom_df = process_uploaded_file(uploaded_file)
            if custom_df is not None:
                st.success(f"Dataset cargado correctamente con {custom_df.shape[0]} filas y {custom_df.shape[1]} columnas.")
                reg_df = custom_df
            else:
                st.warning("Usando el dataset por defecto debido a error en el archivo subido.")
                reg_df = df
        else:
            reg_df = df
        
        # Select target variable
        st.subheader("Seleccionar variable objetivo")
        numeric_cols = reg_df.select_dtypes(include=[np.number]).columns.tolist()
        target_variable = st.selectbox("Variable objetivo para regresión", numeric_cols)
        
        if target_variable:
            # Select features
            feature_cols = [col for col in numeric_cols if col != target_variable]
            selected_features = st.multiselect("Seleccionar características", feature_cols, default=feature_cols[:min(5, len(feature_cols))])
            
            if len(selected_features) < 1:
                st.warning("Por favor selecciona al menos una característica para el modelo.")
            else:
                # Parameters
                test_size = st.slider("Tamaño del conjunto de prueba", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
                
                if st.button("Ejecutar Regresión"):
                    with st.spinner("Entrenando modelos de regresión..."):
                        # Prepare data
                        X = reg_df[selected_features]
                        y = reg_df[target_variable]
                        
                        # Create DataFrame for regression
                        df_reg = pd.concat([X, y], axis=1)
                        
                        try:
                            # Initialize Regression
                            regression = Regresion(df_reg, target_column=target_variable)
                            
                            # Split data
                            regression.train_test_split(test_size=test_size)
                            
                            # Train models
                            st.subheader("Resultados de Regresión Lineal")
                            lr_results = regression.train_linear_regression()
                            
                            # Display metrics in columns
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Error Cuadrático Medio (MSE)", f"{lr_results['metrics']['mse']:.3f}")
                            with col2:
                                st.metric("Coeficiente de Determinación (R²)", f"{lr_results['metrics']['r2']:.3f}")
                            
                            # Display plot
                            st.pyplot(lr_results['fig'])
                            
                            st.subheader("Resultados de Árbol de Decisión")
                            dt_results = regression.train_decision_tree_regressor()
                            
                            # Display metrics in columns
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Error Cuadrático Medio (MSE)", f"{dt_results['metrics']['mse']:.3f}")
                            with col2:
                                st.metric("Coeficiente de Determinación (R²)", f"{dt_results['metrics']['r2']:.3f}")
                            
                            # Display plots
                            st.pyplot(dt_results['results_fig'])
                            st.subheader("Importancia de Características - Árbol de Decisión")
                            st.pyplot(dt_results['importance_fig'])
                            
                            st.subheader("Resultados de Random Forest")
                            rf_results = regression.train_random_forest_regressor()
                            
                            # Display metrics in columns
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Error Cuadrático Medio (MSE)", f"{rf_results['metrics']['mse']:.3f}")
                            with col2:
                                st.metric("Coeficiente de Determinación (R²)", f"{rf_results['metrics']['r2']:.3f}")
                            
                            # Display plots
                            st.pyplot(rf_results['results_fig'])
                            st.subheader("Importancia de Características - Random Forest")
                            st.pyplot(rf_results['importance_fig'])
                            
                            # Compare metrics
                            st.subheader("Comparación de Métricas")
                            comparison_fig, metrics_df = regression.plot_metrics_comparison(return_fig=True)
                            st.pyplot(comparison_fig)
                            
                            # Add a download button for the metrics data
                            csv = metrics_df.reset_index().rename(columns={'index': 'Modelo'}).to_csv(index=False)
                            st.download_button(
                                label="Descargar métricas como CSV",
                                data=csv,
                                file_name="metricas_regresion.csv",
                                mime="text/csv",
                            )
                            
                        except Exception as e:
                            st.error(f"Error en la regresión: {e}")
    elif sidebar_sections == "Series Temporales":
        st.header("⏳ Análisis de Series Temporales")
        
        # Add file upload widget
        st.subheader("Cargar un dataset personalizado (opcional)")
        uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"], key="time_series_uploader")
        
        # Use uploaded dataset or default dataset
        if uploaded_file is not None:
            custom_df = process_uploaded_file(uploaded_file)
            if custom_df is not None:
                st.success(f"Dataset cargado correctamente con {custom_df.shape[0]} filas y {custom_df.shape[1]} columnas.")
                ts_df = custom_df
            else:
                st.warning("Usando el dataset por defecto debido a error en el archivo subido.")
                ts_df = df
        else:
            ts_df = df
        
        # Check if date column exists
        date_columns = ts_df.select_dtypes(include=['datetime64']).columns.tolist()
        if not date_columns:
            # Try to identify potential date columns
            potential_date_cols = [col for col in ts_df.columns if 'fecha' in col.lower() or 'date' in col.lower() or 'time' in col.lower()]
            
            if potential_date_cols:
                date_column = st.selectbox("Seleccionar columna de fecha", potential_date_cols)
            else:
                date_column = st.selectbox("Seleccionar columna de fecha", ts_df.columns.tolist())
                
            if date_column:
                st.info(f"Intentando convertir '{date_column}' a formato de fecha.")
                try:
                    # Create a copy of the dataframe to avoid modifying the original
                    df_ts = ts_df.copy()
                    df_ts[date_column] = pd.to_datetime(df_ts[date_column])
                    has_date = True
                except:
                    st.error(f"No se pudo convertir '{date_column}' a formato de fecha.")
                    has_date = False
        else:
            date_column = st.selectbox("Seleccionar columna de fecha", date_columns)
            df_ts = ts_df.copy()
            has_date = True
        
        if has_date:
            # Select target for forecasting
            numeric_cols = ts_df.select_dtypes(include=[np.number]).columns.tolist()
            target_column = st.selectbox("Variable para pronosticar", numeric_cols)
            
            if target_column:
                # Parameters
                periods = st.slider("Períodos a pronosticar", min_value=5, max_value=365, value=30)
                changepoint_prior_scale = st.slider("Flexibilidad del modelo (changepoint_prior_scale)", 
                                                   min_value=0.01, max_value=0.5, value=0.05, step=0.01)
                
                if st.button("Realizar Pronóstico"):
                    with st.spinner("Generando pronóstico..."):
                        try:
                            # Initialize SeriesTemporales
                            series_temporales = SeriesTemporales(df_ts, date_column=date_column)
                            
                            # Plot time series
                            st.subheader("Serie Temporal")
                            series_fig = series_temporales.graficar_series(columnas=[target_column], return_fig=True)
                            st.pyplot(series_fig)
                            
                            # Generate forecast and plot
                            st.subheader("Pronóstico")
                            # Get forecast data first
                            pronostico = series_temporales.pronosticar(target_column=target_column, 
                                                                     periods=periods, 
                                                                     changepoint_prior_scale=changepoint_prior_scale)
                            
                            # Then get both prophet plots
                            forecast_fig = series_temporales.plot_model(return_fig=True)
                            st.pyplot(forecast_fig)
                            
                            # Show model components
                            st.subheader("Componentes del Modelo")
                            components_fig = series_temporales.plot_components(return_fig=True)
                            st.pyplot(components_fig)
                            
                            # Display additional statistics about the forecast
                            st.subheader("Estadísticas del Pronóstico")
                            
                            # Calculate and display trend direction
                            last_trend = pronostico['trend'].iloc[-1]
                            first_trend = pronostico['trend'].iloc[len(df_ts.index)-1]  # Start from end of training data
                            trend_change = last_trend - first_trend
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    label="Tendencia", 
                                    value=f"{trend_change:.2f}",
                                    delta=f"{'+' if trend_change > 0 else ''}{trend_change:.2f}"
                                )
                            
                            with col2:
                                # Calculate average seasonal effect
                                if 'weekly' in pronostico.columns:
                                    seasonal_effect = pronostico['weekly'].abs().mean()
                                    st.metric("Efecto Estacional Promedio", f"{seasonal_effect:.2f}")
                                elif 'yearly' in pronostico.columns:
                                    seasonal_effect = pronostico['yearly'].abs().mean()
                                    st.metric("Efecto Estacional Promedio", f"{seasonal_effect:.2f}")
                            
                            # Display forecast data
                            st.subheader("Datos del Pronóstico")
                            st.dataframe(pronostico[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))
                            
                            # Download forecast
                            csv = pronostico.to_csv(index=False)
                            st.download_button(
                                label="Descargar Pronóstico (CSV)",
                                data=csv,
                                file_name="pronostico.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Error en el pronóstico: {e}")
    elif sidebar_sections == "Entrenamiento de Modelos":
        st.header("🤖 Entrenamiento de Modelos")
        
        # Add file upload widget
        st.subheader("Cargar un dataset personalizado (opcional)")
        uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"], key="model_training_uploader")
        
        # Use uploaded dataset or default dataset
        if uploaded_file is not None:
            custom_df = process_uploaded_file(uploaded_file)
            if custom_df is not None:
                st.success(f"Dataset cargado correctamente con {custom_df.shape[0]} filas y {custom_df.shape[1]} columnas.")
                train_df = custom_df
            else:
                st.warning("Usando el dataset por defecto debido a error en el archivo subido.")
                train_df = df
        else:
            train_df = df
        
        # Prepare data for modeling
        if 'fecha' in train_df.columns:
            # Create temporal features
            train_df['año'] = train_df['fecha'].dt.year
            train_df['mes'] = train_df['fecha'].dt.month
            train_df['dia'] = train_df['fecha'].dt.day
            train_df['dia_semana'] = train_df['fecha'].dt.dayofweek
            
            # Let user select target and features
            st.subheader("Seleccionar variables")
            target_col = st.selectbox("Variable objetivo", train_df.columns.tolist(), index=train_df.columns.get_loc('valor') if 'valor' in train_df.columns else 0)
            
            # Select features for modeling
            feature_options = [col for col in train_df.columns if col != target_col]
            default_features = ['año', 'mes', 'dia', 'dia_semana'] if all(f in train_df.columns for f in ['año', 'mes', 'dia', 'dia_semana']) else feature_options[:min(4, len(feature_options))]
            feature_cols = st.multiselect("Características para el modelo", feature_options, default=default_features)
            
            if not feature_cols:
                st.warning("Por favor selecciona al menos una característica para el modelo.")
            else:
                X = train_df[feature_cols]
                y = train_df[target_col]
                
                # Split the data
                test_size = st.slider("Tamaño del conjunto de prueba", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Create tabs for different search methods
                search_method = st.radio(
                    "Método de búsqueda de hiperparámetros",
                    ["Búsqueda Genética", "Búsqueda Exhaustiva"]
                )
                
                if st.button("Entrenar Modelos"):
                    with st.spinner("Entrenando modelos..."):
                        # Initialize evaluator
                        evaluator = ModelEvaluator(X_train, X_test, y_train, y_test)
                        
                        # Perform search based on selected method
                        if search_method == "Búsqueda Genética":
                            results = evaluator.genetic_search()
                        else:
                            results = evaluator.exhaustive_search()
                        
                        # Display results
                        st.subheader("Resultados del Entrenamiento")
                        
                        for model_name, model_results in results.items():
                            st.write(f"### Modelo: {model_name}")
                            st.write("Mejores parámetros:")
                            st.json(model_results['best_params'])
                            
                            # Make predictions and calculate metrics
                            model = model_results['estimator']
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            mse = np.mean((y_test - y_pred) ** 2)
                            rmse = np.sqrt(mse)
                            mae = np.mean(np.abs(y_test - y_pred))
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("MSE", f"{mse:.2f}")
                            with col2:
                                st.metric("RMSE", f"{rmse:.2f}")
                            with col3:
                                st.metric("MAE", f"{mae:.2f}")
                            
                            # Plot actual vs predicted
                            fig = px.scatter(
                                x=y_test,
                                y=y_pred,
                                labels={'x': 'Valores Reales', 'y': 'Predicciones'},
                                title=f'Valores Reales vs Predicciones - {model_name}'
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=[y_test.min(), y_test.max()],
                                    y=[y_test.min(), y_test.max()],
                                    mode='lines',
                                    name='Línea Perfect Fit',
                                    line=dict(color='red', dash='dash')
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("El dataset no contiene la columna 'fecha' necesaria para el entrenamiento.")
else:
    st.warning("Por favor, asegúrate de que el archivo 'consumo_agua.csv' existe en el directorio del proyecto.")
