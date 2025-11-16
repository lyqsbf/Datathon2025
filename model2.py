import matplotlib
matplotlib.use('Agg') # <-- ¡Importante! Poner esto primero

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import PartialDependenceDisplay

# --- Explainability ---
try:
    import shap
except:
    print("ADVERTENCIA: No se pudo importar la librería 'shap'.")
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except:
    print("ADVERTENCIA: No se pudo importar la librería 'lime'.")
    LimeTabularExplainer = None

try:
    from PyALE import ale
except:
    print("ADVERTENCIA: No se pudo importar la librería 'PyALE'.")
    ale = None

# --------------------------
# CONFIG
# --------------------------
OUT_DIR = "explainability_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# LOAD DATA
# --------------------------
try:
    loan_data = pd.read_csv("dataset.csv")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'dataset.csv'.")
    exit()

# 'cust_in_iberia' fue eliminada en el pre-procesamiento, lo mantenemos
if 'cust_in_iberia' in loan_data.columns:
    loan_data = loan_data.drop("cust_in_iberia", axis=1)

y = loan_data["target_variable"]
X = loan_data.drop(["id", "target_variable"], axis=1)

# Usamos las columnas de X (que ya están escaladas)
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train:", X_train.shape, "Test:", X_test.shape)


# -------------------------------------------------------------
# MODELO – RANDOM FOREST
# -------------------------------------------------------------
print("\n==== RANDOM FOREST ====")
model_rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred_rf))
print(classification_report(y_test, pred_rf))


# =============================================================
# EXPLAINABILITY SECTION
# =============================================================

# -------------------------------------------------------------
# Helper: save figure
# -------------------------------------------------------------
def save_fig(fig, filename):
    try:
        fig.tight_layout()
        path = os.path.join(OUT_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("Guardado:", path)
    except Exception as e:
        print(f"Error guardando figura {filename}: {e}")
        plt.close(fig)

# -------------------------------------------------------------
# SHAP GLOBAL + LOCAL [VERSIÓN COMPLETA Y PRECISA]
# -------------------------------------------------------------
def shap_explain(model, X_train, X_test, model_name, instance_index=0):
    if shap is None:
        print("SHAP no instalado.")
        return

    print(f"\n=== SHAP para {model_name} ===")
    
    print("--- Inicializando Explainer con X_train completo... ---")
    explainer = shap.Explainer(model, X_train)
    
    print("--- Calculando SHAP values para X_test completo (esto tardará)... ---")
    shap_values = explainer(X_test, check_additivity=False)
    
    print("--- Generando gráfico Beeswarm ---")
    fig = plt.figure()
    shap.summary_plot(shap_values[:, :, 1], X_test, show=False)
    save_fig(fig, f"{model_name}_shap_summary_beeswarm.png")

    print("--- Generando gráfico Bar ---")
    fig = plt.figure()
    shap.summary_plot(shap_values[:, :, 1], X_test, plot_type="bar", show=False)
    save_fig(fig, f"{model_name}_shap_bar.png")

    if instance_index >= len(X_test):
        print(f"Error: instance_index {instance_index} fuera de rango (tamaño: {len(X_test)})")
        return

    fig = plt.figure()
    try:
        print(f"--- Generando SHAP local para instancia {instance_index} ---")
        shap.plots.waterfall(shap_values[instance_index, :, 1], show=False)
        save_fig(fig, f"{model_name}_shap_local_waterfall_idx{instance_index}.png")
    except Exception as e:
        print(f"Error generando waterfall: {e}")
        plt.close(fig)

    print("--- Cálculo SHAP completado ---")
    return shap_values

# -------------------------------------------------------------
# SHAP INTERACTION PLOTS [CORREGIDO - ROBUSTO]
# -------------------------------------------------------------
def shap_interaction_plots(shap_values, X_data, model_name, features_to_plot):
    if shap is None:
        print("SHAP no instalado.")
        return
        
    print(f"\n=== Gráficos de Interacción SHAP para {model_name} ===")
    
    # shap_values es un objeto Explanation.
    # .values extrae los arrays de numpy
    # [:, :, 1] selecciona los valores de la clase 1 (Won)
    try:
        shap_values_class1 = shap_values.values[:, :, 1]
    except AttributeError:
        # Fallback por si 'shap_values' no es un objeto Explanation sino un array
        shap_values_class1 = shap_values[:, :, 1]
        
    
    for pair in features_to_plot:
        main_feat, color_feat = pair
        
        if main_feat not in X_data.columns or color_feat not in X_data.columns:
            print(f"Advertencia: No se encontró {main_feat} o {color_feat}. Saltando...")
            continue
            
        try:
            # [CORRECCIÓN] NO creamos fig = plt.figure()
            # Dejamos que shap.dependence_plot cree su propia figura
            shap.dependence_plot(
                main_feat,
                shap_values_class1, # Los valores SHAP crudos para la clase 1
                X_data,             # El DataFrame con los valores de las features
                interaction_index=color_feat, # La feature para el color
                show=False
            )
            
            # [CORRECCIÓN] Capturamos la figura que SHAP acaba de crear
            fig = plt.gcf()
            
            save_fig(fig, f"{model_name}_interaction_{main_feat}_vs_{color_feat}.png")
        except Exception as e:
            print(f"Error generando gráfico de interacción para {main_feat} vs {color_feat}: {e}")
            plt.close() # Cerramos por si acaso

# -------------------------------------------------------------
# LIME LOCAL [CORREGIDO - CON ÍNDICE]
# -------------------------------------------------------------
def lime_explain(model, X_train, X_test, model_name, instance_index=0):
    if LimeTabularExplainer is None:
        print("LIME no instalado.")
        return

    print(f"\n=== LIME para {model_name} (Instancia: {instance_index}) ===")

    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=["Lost (0)", "Won (1)"],
        mode="classification"
    )

    if instance_index >= len(X_test):
        print(f"Error: instance_index {instance_index} fuera de rango")
        return
        
    exp = explainer.explain_instance(
        data_row=X_test.iloc[instance_index], # <-- Usa el índice
        predict_fn=model.predict_proba,
        num_features=10
    )

    # Guarda con el nombre indexado
    json_path = os.path.join(OUT_DIR, f"{model_name}_lime_local_idx{instance_index}.json")
    with open(json_path, "w") as f:
        json.dump(exp.as_list(), f, indent=2)
    print("Guardado:", json_path)

    fig = exp.as_pyplot_figure()
    save_fig(fig, f"{model_name}_lime_local_idx{instance_index}.png")

# -------------------------------------------------------------
# PDP (Partial Dependence) [CORREGIDO - TODAS LAS FEATURES]
# -------------------------------------------------------------
def pdp_explain(model, X_test, model_name):
    print(f"\n=== PDP para {model_name} (TODAS las Features) ===")

    for feat in X_test.columns:
        
        fig = plt.figure(figsize=(7, 4))
        try:
            PartialDependenceDisplay.from_estimator(
                model,
                X_test,
                [feat],
                ax=plt.gca(),
                kind='average'
            )
            plt.title(f"PDP para {feat}")
            save_fig(fig, f"{model_name}_pdp_{feat}.png")
        except Exception as e:
            print(f"Error generando PDP en {feat}: {e}")     
            plt.close(fig)

# -------------------------------------------------------------
# FEATURE IMPORTANCE (solo RF)
# -------------------------------------------------------------
def rf_feature_importance(model, model_name):
    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=feature_names).sort_values()

    fig = plt.figure(figsize=(8, 7))
    imp_series.plot.barh()
    plt.title("Random Forest Feature Importance")
    save_fig(fig, f"{model_name}_feature_importance.png")
    
    return imp_series.sort_values(ascending=False).head(3).index.tolist()

# -------------------------------------------------------------
# CLASE WRAPPER PARA PYALE
# -------------------------------------------------------------
class_names = ["Lost (0)", "Won (1)"]
class ModelWrapper:
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn

    def predict(self, X):
        return self.predict_fn(X)
    
    @property
    def classes_(self):
        return class_names

# -------------------------------------------------------------
# ALE (Accumulated Local Effect) [CORRECCIÓN MANUAL ROBUSTA]
# -------------------------------------------------------------
def ale_explain(model, X_data, model_name, top_features):
    if ale is None:
        print("PyALE no instalado.")
        return

    print(f"\n=== ALE para {model_name} (Features: {top_features}) ===")

    try:
        model_pred_func = model.predict_proba
        pred_func = lambda x: model_pred_func(x)[:, 1]
    except AttributeError:
        pred_func = model.predict

    wrapped_model = ModelWrapper(pred_func)
    
    for feat in top_features:
        if feat not in X_data.columns:
            print(f"Advertencia: La feature '{feat}' no se encuentra en X_data para ALE.")
            continue
        
        fig = None 
        try:
            ale_df = ale(
                X=X_data,
                model=wrapped_model, 
                feature=[feat],
                feature_type='auto',
                grid_size=20,
                include_CI=False
            )
            
            fig = plt.figure(figsize=(7, 4))
            
            # --- Detección robusta de columnas X e Y ---
            y_col = None
            if 'eff' in ale_df.columns:
                y_col = 'eff'
            else:
                eff_cols = [c for c in ale_df.columns if 'eff' in c]
                if eff_cols:
                    y_col = eff_cols[0]
            
            x_col = None
            if feat in ale_df.columns:
                x_col = feat
            else:
                # Fallback: usar la primera columna que no sea 'eff'
                non_eff_cols = [c for c in ale_df.columns if 'eff' not in c]
                if non_eff_cols:
                    x_col = non_eff_cols[0]

            if x_col is None or y_col is None:
                raise ValueError(f"No se pudieron detectar las columnas X/Y en el DataFrame de ALE. Columnas: {ale_df.columns}")
            # --- Fin de la detección ---

            plt.plot(ale_df[x_col], ale_df[y_col])
            
            plt.title(f"ALE para {feat}")
            plt.xlabel(feat)
            plt.ylabel("ALE")
            save_fig(fig, f"{model_name}_ale_{feat}.png")

        except Exception as e:
            print(f"Error generando ALE en {feat}: {type(e).__name__} - {e}")     
            if fig:
                plt.close(fig)
                
# =============================================================
# EJECUTAR EXPLICABILIDAD
# =============================================================

# --- Explicando Random Forest ---
print("\n\n--- Explicando Random Forest ---")

# --- 1. CÁLCULO SHAP (LENTO) ---
print("\n--- Analizando Caso Local: Instancia 0 (SHAP) ---")
shap_values_rf = shap_explain(model_rf, X_train, X_test, "RandomForest", instance_index=0)

# --- 2. GRÁFICOS DE INTERACCIÓN SHAP ---
if shap_values_rf is not None:
    interactions_to_plot = [
        ('cust_interactions', 'cust_hitrate'), # Ricitos de Oro vs Hitrate
        ('cust_interactions', 'opp_old'),      # Ricitos de Oro vs Antigüedad
        ('opp_old', 'cust_hitrate')            # Antigüedad vs Hitrate
    ]
    shap_interaction_plots(shap_values_rf, X_test, "RandomForest", interactions_to_plot)


    # --- 3. GENERAR MÁS CASOS LOCALES (RÁPIDO) ---
    print("\n--- Analizando Caso Local: Instancia 5 (desde SHAP) ---")
    fig = plt.figure()
    try:
        shap.plots.waterfall(shap_values_rf[5, :, 1], show=False)
        save_fig(fig, f"RandomForest_shap_local_waterfall_idx5.png")
    except Exception as e:
        print(f"Error generando waterfall (idx 5): {e}")
        plt.close(fig)

# --- 4. LIME ---
print("\n--- Analizando Caso Local: Instancia 0 (LIME) ---")
lime_explain(model_rf, X_train, X_test, "RandomForest", instance_index=0)
print("\n--- Analizando Caso Local: Instancia 5 (LIME) ---")
lime_explain(model_rf, X_train, X_test, "RandomForest", instance_index=5)

# --- 5. EXPLICACIONES GLOBALES (PDP, ALE) ---
rf_top_features = rf_feature_importance(model_rf, "RandomForest")

# [CORREGIDO] Llamada a pdp_explain sin 'rf_top_features'
pdp_explain(model_rf, X_test, "RandomForest") 

ale_explain(model_rf, X_test, "RandomForest", rf_top_features)


print("\n=== EXPLICACIONES GENERADAS ===")
print("Todos los archivos están en la carpeta:", OUT_DIR)

# --- AVISO SOBRE DATOS ESCALADOS ---
print("\n--- AVISO IMPORTANTE ---")
print("Recuerda que los datos de tu dataset.csv están 'escalados'.")
print("Esto significa que valores como 'opp_old = 3.54' o 'cust_hitrate = -0.25' son correctos para el modelo,")
print("pero para presentarlos a negocio, necesitarías el dataset original (sin escalar) para interpretar qué significan esos números.")