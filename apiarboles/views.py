from django.shortcuts import render
from django.http import HttpResponse
from graphviz import Source
from sklearn.tree import export_graphviz
import os
import pandas as pd
from io import StringIO
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import base64         
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier

# Función para cargar el conjunto de datos
def load_dataset():
    df = pd.read_csv('/data/TotalFeatures-ISCXFlowMeter.csv')
    return df

# Funciones adicionales para dividir el conjunto de datos y escalarlo
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

# Construcción de una función que realice el particionado completo
def train_val_test_split(df):
    train_set, val_test_set = train_test_split(df, test_size=0.3, random_state=42)
    val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=42)
    return train_set, val_set, test_set

def plot_decision_boundary(clf, X, y):
    mins = X.min(axis=0) - 1
    maxs = X.max(axis=0) + 1
    x1, x2 = np.meshgrid(np.linspace(mins[0], maxs[0], 100),
                         np.linspace(mins[1], maxs[1], 100))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
    plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="normal")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="adware")
    plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="malware")
    plt.xlabel('min_flowpktl', fontsize=14)
    plt.ylabel('flow_fin', fontsize=14, rotation=90)
    plt.legend()

def index(request):
    df = load_dataset()

    # Imprimir los primeros 10 registros en una tabla HTML
    head_table_html = df.head(20).to_html()

    # Obtener la información del DataFrame
    buffer = StringIO()
    df.info(buf=buffer, verbose=True, show_counts=True)  
    info_text = buffer.getvalue()

    # Obtener el conteo de clases
    class_counts_text = df['calss'].value_counts().to_string()

    # Copiamos el conjunto de datos y transformamos la variable de salida a numérica para calcular correlaciones
    X = df.copy()
    X['calss'] = X['calss'].factorize()[0]
    # Calculamos correlaciones
    corr_matrix = X.corr()
    correlation_result = corr_matrix.to_html()

    # Filtrar la matriz de correlación para obtener valores mayores a 0.05
    filtered_corr_matrix = corr_matrix[corr_matrix["calss"] > 0.05].to_html()

    # Dividir el conjunto de datos y escalarlo
    train_set, val_set, test_set = train_val_test_split(df)
    X_train, y_train = remove_labels(train_set, 'calss')
    X_val, y_val = remove_labels(val_set, 'calss')
    X_test, y_test = remove_labels(test_set, 'calss')
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    X_val_scaled = scaler.fit_transform(X_val)
    
    # Transformar a DataFrame de Pandas
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    # Imprimir los primeros 10 registros escalados
    scaled_table_html = X_train_scaled_df.head(10).to_html()

    # Obtener la descripción del conjunto de datos escalado
    X_train_scaled_desc = X_train_scaled_df.describe().to_html()

    # Modelo entrenado con el conjunto de datos sin escalar
    MAX_DEPTH = 20
    clf_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=42)
    clf_tree.fit(X_train, y_train)

    # Modelo entrenado con el conjunto de datos escalado
    clf_tree_scaled = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=42)
    clf_tree_scaled.fit(X_train_scaled, y_train)

    # Entrenamiento del modelo de bosques aleatorios sin escalar
    clf_forest = RandomForestClassifier(n_estimators=100, max_depth=MAX_DEPTH, random_state=42)
    clf_forest.fit(X_train, y_train)

    # Entrenamiento del modelo de bosques aleatorios escalado
    clf_forest_scaled = RandomForestClassifier(n_estimators=100, max_depth=MAX_DEPTH, random_state=42)
    clf_forest_scaled.fit(X_train_scaled, y_train)

    # Predicciones en el conjunto de datos de entrenamiento para bosques aleatorios sin escalar
    y_train_pred_forest = clf_forest.predict(X_train)
    y_train_prep_pred_forest = clf_forest_scaled.predict(X_train_scaled)

    # Comparación de resultados entre escalado y sin escalar con bosques aleatorios en el conjunto de datos de entrenamiento
    print("Random Forest - Training Set:")
    evaluate_result(y_train_pred_forest, y_train, y_train_prep_pred_forest, y_train, f1_score)

    
    # Evaluamos los resultados de entrenamiento
    f1_score_train_result = evaluate_result(y_train_pred_forest, y_train, y_train_prep_pred_forest, y_train, f1_score)

    # Predecimos con el conjunto de datos de validación
    y_val_pred = clf_tree.predict(X_val)
    y_val_prep_pred = clf_tree_scaled.predict(X_val_scaled)
    
    # Evaluamos los resultados de validación
    f1_score_val_result = evaluate_result(y_val_pred, y_val, y_val_prep_pred, y_val, f1_score)

    # Reducimos el número de atributos del conjunto de datos para visualizarlo mejor
    X_train_reduced = X_train[['min_flowpktl', 'flow_fin']]
    # Generamos un modelo con el conjunto de datos reducido
    clf_tree_reduced = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf_tree_reduced.fit(X_train_reduced, y_train)

    # Data preprocessing
    X = df.copy()
    X['calss'] = X['calss'].factorize()[0]
    train_set, val_set, test_set = train_val_test_split(X)
    X_train, y_train = remove_labels(train_set, 'calss')
    X_val, y_val = remove_labels(val_set, 'calss')
    X_test, y_test = remove_labels(test_set, 'calss')
    
    # Train model
    MAX_DEPTH = 20
    clf_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=42)
    clf_tree.fit(X_train, y_train)
    
    # Reducimos el número de atributos del conjunto de datos para visualizarlo mejor
    X_train_reduced = X_train[['min_flowpktl', 'flow_fin']]
    
    # Generamos un modelo con el conjunto de datos reducido
    clf_tree_reduced = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf_tree_reduced.fit(X_train_reduced, y_train)
    
    # Representamos gráficamente el límite de decisión construido
    plt.figure(figsize=(8, 6))
    plot_decision_boundary(clf_tree_reduced, X_train_reduced.values, y_train)
    
    # Guardamos la imagen en un buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    plt.close()

    # Generar el árbol de decisión
    export_graphviz(
        clf_tree_reduced,
        out_file="android_malware.dot",
        feature_names=X_train_reduced.columns,
        class_names=["benign", "adware", "malware"],
        rounded=True,
        filled=True
    )

    # Leer el contenido del archivo generado
    with open("android_malware.dot", "r") as file:
        tree_content = file.read()

    return render(request, '/templates/result.html', 
                           {'head_table_html': head_table_html, 
                           'info_text': info_text, 
                           'class_counts_text': class_counts_text, 
                           'correlation_result': correlation_result, 
                           'filtered_corr_matrix': filtered_corr_matrix, 
                           'scaled_table_html': scaled_table_html, 
                           'X_train_scaled_desc': X_train_scaled_desc, 
                           'clf_tree': clf_tree, 
                           'clf_tree_scaled': clf_tree_scaled, 
                           'f1_score_train_result': f1_score_train_result, 
                           'f1_score_val_result': f1_score_val_result, 
                           'clf_tree_reduced': clf_tree_reduced, 
                           'X_train_reduced': X_train_reduced,
                           'decision_plot': plot_data,
                           'tree_content': tree_content,
                           'clf_forest': clf_forest,
                           'clf_forest_scaled': clf_forest_scaled,
                           'f1_score_train_result_forest': f1_score_train_result}) 

def evaluate_result(y_pred, y_true, y_prep_pred, y_prep_true, metric):
    metric_normal = metric(y_pred, y_true, average='weighted')
    metric_prep = metric(y_prep_pred, y_prep_true, average='weighted')
    return f'F1 Score sin escalar: {metric_normal}, F1 Score escalado: {metric_prep}'
