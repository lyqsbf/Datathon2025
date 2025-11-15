import matplotlib
matplotlib.use('Agg') # <-- ¡Importante! Poner esto primero

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# Se eliminó LogisticRegression
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

if 'cust_in_iberia' in loan_data.columns:
    loan_data = loan_data.drop("cust_in_iberia", axis=1)

y = loan_data["target_variable"]
X = loan_data.drop(["id", "target_variable"], axis=1)

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
# SHAP GLOBAL + LOCAL [VERSIÓN RÁPIDA]
# -------------------------------------------------------------
def shap_explain(model, X_train, X_test, model_name, n_samples_test=None):
    if shap is None:
        print("SHAP no instalado.")
        return

    print(f"\n=== SHAP para {model_name} ===")

    # --- Optimización: Submuestreo ---
    if n_samples_test is not None and n_samples_test < len(X_test):
        print(f"--- Usando submuestra de {n_samples_test} filas para X_test ---")
        try:
            data_to_explain = shap.sample(X_test, n_samples_test, random_state=42)
        except AttributeError: # Fallback para versiones antiguas
            data_to_explain = shap.utils.sample(X_test, n_samples_test, random_state=42)
    else:
        print("--- Usando X_test completo ---")
        data_to_explain = X_test

    try:
        background_data = shap.sample(X_train, 100, random_state=42)
    except AttributeError:
        background_data = shap.utils.sample(X_train, 100, random_state=42)
    # --- Fin Optimización ---

    explainer = shap.Explainer(model, background_data)
    
    shap_values = explainer(data_to_explain, check_additivity=False)
    
    # --- SHAP SUMMARY PLOT (Beeswarm) ---
    fig = plt.figure()
    shap.summary_plot(shap_values[:, :, 1], data_to_explain, show=False)
    save_fig(fig, f"{model_name}_shap_summary_beeswarm.png")

    # --- SHAP BAR PLOT --- 
    fig = plt.figure()
    shap.summary_plot(shap_values[:, :, 1], data_to_explain, plot_type="bar", show=False)
    save_fig(fig, f"{model_name}_shap_bar.png")

    # --- SHAP LOCAL (primer ejemplo, clase 1) ---
    fig = plt.figure()
    try:
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
        save_fig(fig, f"{model_name}_shap_local_waterfall.png")
    except Exception as e:
        print(f"Error generando waterfall: {e}")
        plt.close(fig)


# -------------------------------------------------------------
# LIME LOCAL
# -------------------------------------------------------------
def lime_explain(model, X_train, X_test, model_name):
    if LimeTabularExplainer is None:
        print("LIME no instalado.")
        return

    print(f"\n=== LIME para {model_name} ===")

    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=["Lost (0)", "Won (1)"],
        mode="classification"
    )

    exp = explainer.explain_instance(
        data_row=X_test.iloc[0],
        predict_fn=model.predict_proba,
        num_features=10
    )

    json_path = os.path.join(OUT_DIR, f"{model_name}_lime_local.json")
    with open(json_path, "w") as f:
        json.dump(exp.as_list(), f, indent=2)
    print("Guardado:", json_path)

    fig = exp.as_pyplot_figure()
    save_fig(fig, f"{model_name}_lime_local.png")


# -------------------------------------------------------------
# PDP (Partial Dependence)
# -------------------------------------------------------------
def pdp_explain(model, X_test, model_name, top_features):
    print(f"\n=== PDP para {model_name} (Features: {top_features}) ===")

    for feat in top_features:
        if feat not in X_test.columns:
            print(f"Advertencia: La feature '{feat}' no se encuentra en X_test para PDP.")
            continue
            
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
# ALE (Accumulated Local Effect) - CORRECCIÓN 9
# -------------------------------------------------------------
def ale_explain(model, X_data, model_name, top_features):
    if ale is None:
        print("PyALE no instalado.")
        return

    print(f"\n=== ALE para {model_name} (Features: {top_features}) ===")

    # --- Wrapper obligatorio para PyALE ---
    class PredictorWrapper:
        def __init__(self, model):
            self.model = model

        # PyALE exige este método
        def predict(self, X):
            return self.model.predict_proba(X)[:, 1]

    wrapped_model = PredictorWrapper(model)

    # --- Generamos los ALE para cada feature ---
    for feat in top_features:
        if feat not in X_data.columns:
            print(f"Advertencia: La feature '{feat}' no está en X_data.")
            continue

        try:
            fig = plt.figure(figsize=(7, 4))

            ax = ale(
                X=X_data,
                model=wrapped_model,   # <-- ahora sí tiene .predict()
                feature=feat,
                feature_type='auto',
                grid_size=20,
                include_CI=False
            )

            ax.set_title(f"ALE para {feat}")
            save_fig(fig, f"{model_name}_ale_{feat}.png")

        except Exception as e:
            print(f"Error generando ALE en {feat}: {e}")
            plt.close(fig)



# =============================================================
# EJECUTAR EXPLICABILIDAD
# =============================================================

# --- Explicando Random Forest ---
print("\n\n--- Explicando Random Forest ---")
#shap_explain(model_rf, X_train, X_test, "RandomForest")
#lime_explain(model_rf, X_train, X_test, "RandomForest")
rf_top_features = rf_feature_importance(model_rf, "RandomForest")
#pdp_explain(model_rf, X_test, "RandomForest", rf_top_features)
ale_explain(model_rf, X_test, "RandomForest", rf_top_features)


print("\n=== EXPLICACIONES GENERADAS ===")
print("Todos los archivos están en la carpeta:", OUT_DIR)