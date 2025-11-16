import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, make_scorer
)
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import numpy as np  # Necesario para los rangos de parámetros

# --- Cargar Datos y Dividir (tu código) ---
try:
    loan_data = pd.read_csv("dataset.csv")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'dataset.csv'.")
    exit()

y = loan_data['target_variable']
X = loan_data.drop(['id', 'target_variable'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Filas para entrenar: {X_train.shape[0]}")
print(f"Filas para probar: {X_test.shape[0]}")
print("-" * 40)

# --- Definir el Scorer ---
# Queremos maximizar el F1-Score de la Clase 1
f1_clase1_scorer = make_scorer(f1_score, pos_label=1)

#------------------------------------------------------------------------
# MODELO 1: RANDOM FOREST (Tuning)
#------------------------------------------------------------------------
print("\n--- Optimizando: Random Forest ---")

# 1. Definir la 'parrilla' de parámetros a probar
# (Vamos a probar diferentes profundidades máximas)
param_grid_rf = {
    'max_depth': [10, 20, 30],
    'n_estimators': [100, 200, 300],
    'min_samples_leaf': [1, 3, 5, 7]
}

# 2. Crear el objeto GridSearchCV
# Usamos un n_estimators fijo (ej. 200) para que la búsqueda sea más rápida
grid_rf = GridSearchCV(
    estimator=RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ),
    param_grid=param_grid_rf, # La parrilla que definimos
    scoring=f1_clase1_scorer, # La métrica a optimizar
    cv=3,                     # 3-Fold Cross-Validation
    verbose=1,                 # Muestra el progreso
    n_jobs=-1
)

# 3. Entrenar (esto probará todas las combinaciones)
grid_rf.fit(X_train, y_train)

# 4. Guardar resultados y mostrar el mejor
print(f"Mejor F1-Score (Clase 1) para RF: {grid_rf.best_score_:.4f}")
print(f"Mejores Parámetros: {grid_rf.best_params_}")

# 5. Guardar resultados en un DataFrame para graficar
results_rf = pd.DataFrame(grid_rf.cv_results_)
print("\nResultados completos de RF:")
print(results_rf[['param_max_depth', 'mean_test_score', 'std_test_score']])

# 6. Graficar
plt.figure(figsize=(10, 6))
plt.plot(results_rf['param_max_depth'], results_rf['mean_test_score'], marker='o')
plt.title('F1-Score (Clase 1) vs. max_depth para Random Forest')
plt.xlabel('max_depth')
plt.ylabel('Mean F1-Score (Clase 1)')
plt.grid(True)
plt.savefig('rf_f1_vs_max_depth.png')
print("\nGráfica 'rf_f1_vs_max_depth.png' guardada.")
print("-" * 40)

#------------------------------------------------------------------------
# MODELO 2: GRADIENT BOOSTING (Tuning)
#------------------------------------------------------------------------
print("\n--- Optimizando: Gradient Boosting ---")

# 1. Definir la 'parrilla' de parámetros
param_grid_gb = {
    'max_depth': [3, 5, 7, 10],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2, 0.5]
}

# 2. Calcular 'sample_weights' (como antes)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 3. Crear el objeto GridSearchCV
grid_gb = GridSearchCV(
    estimator=GradientBoostingClassifier(
        random_state=42
    ),
    param_grid=param_grid_gb,
    scoring=f1_clase1_scorer,
    cv=3,
    verbose=1
)

# 4. Entrenar
# ¡Importante! Debemos pasar los 'sample_weight' al .fit() del grid
grid_gb.fit(X_train, y_train, sample_weight=sample_weights)

# 5. Guardar resultados y mostrar el mejor
print(f"Mejor F1-Score (Clase 1) para GB: {grid_gb.best_score_:.4f}")
print(f"Mejores Parámetros: {grid_gb.best_params_}")

# 6. Guardar resultados en DataFrame
results_gb = pd.DataFrame(grid_gb.cv_results_)
print("\nResultados completos de GB:")
print(results_gb[['param_max_depth', 'mean_test_score', 'std_test_score']])

# 7. Graficar
plt.figure(figsize=(10, 6))
plt.plot(results_gb['param_max_depth'], results_gb['mean_test_score'], marker='o', color='orange')
plt.title('F1-Score (Clase 1) vs. max_depth para Gradient Boosting')
plt.xlabel('max_depth')
plt.ylabel('Mean F1-Score (Clase 1)')
plt.grid(True)
plt.savefig('gb_f1_vs_max_depth.png')
print("\nGráfica 'gb_f1_vs_max_depth.png' guardada.")
print("-" * 40)


# --- Evaluación Final con los MEJORES Modelos ---
print("\n--- Evaluación final en Test Set con los mejores parámetros ---")

# 1. Evaluar el mejor RF
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print("\nReporte de Clasificación (Mejor Random Forest):")
print(classification_report(y_test, y_pred_rf))

# 2. Evaluar el mejor GB
best_gb = grid_gb.best_estimator_
y_pred_gb = best_gb.predict(X_test)
print("\nReporte de Clasificación (Mejor Gradient Boosting):")
print(classification_report(y_test, y_pred_gb))