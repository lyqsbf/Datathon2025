import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight

# Cargar tus datos
try:
    loan_data = pd.read_csv("dataset.csv")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'dataset.csv'.")
    # Salir o manejar el error
    exit()

# 1. Definir X e y
y = loan_data['target_variable']
X = loan_data.drop(['id', 'target_variable'], axis=1)

# 2. Dividir los datos (¡CON ESTRATIFICACIÓN!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # <--- IMPORTANTE: Asegura que la proporción de clases se mantenga
)

print(f"Filas para entrenar: {X_train.shape[0]}")
print(f"Filas para probar: {X_test.shape[0]}")
print("-" * 40)

#------------------------------------------------------------------------
# MODELO 2: RANDOM FOREST
#------------------------------------------------------------------------
print("\n--- Evaluando: Random Forest ---")

# 1. Crear instancia del modelo
model_rf = RandomForestClassifier(
    n_estimators=1000,
    class_weight="balanced", # <--- IMPORTANTE: Para manejar el desbalanceo
    n_jobs=-1,               # <--- (Opcional) Para usar toda la CPU
    random_state=42
) 

# 2. Entrenar el modelo
model_rf.fit(X_train, y_train)

# 3. Evaluar
y_pred_rf = model_rf.predict(X_test) # <--- Variable única para la predicción

print(f"Precisión (Accuracy): {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
print("\nReporte de Clasificación (Random Forest):")
print(classification_report(y_test, y_pred_rf))
print("\nMatriz de Confusión (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))

# --- CÁLCULO ESPECÍFICO DE F1-SCORE ---

# Opción A: F1-Score Ponderado (el resumen más común)
# 'weighted' da más peso al f1 de la clase 0 (porque hay más muestras)
f1_ponderado = f1_score(y_test, y_pred_rf, average='weighted')
print(f"\nF1-Score Ponderado (Weighted) de RF: {f1_ponderado:.4f}")

# Opción B: F1-Score Macro (promedio simple de F1s)
# 'macro' trata ambas clases por igual, sin importar el desbalanceo
f1_macro = f1_score(y_test, y_pred_rf, average='macro')
print(f"F1-Score Macro de RF: {f1_macro:.4f}")

# Opción C: F1-Score de la Clase 1 (la minoritaria)
# 'average=None' devuelve los F1s de ambas clases en una lista [f1_clase0, f1_clase1]
f1_clase_1 = f1_score(y_test, y_pred_rf, average=None)[1]
print(f"F1-Score (Clase 1) de RF: {f1_clase_1:.4f}")

#------------------------------------------------------------------------
# MODELO 6: GRADIENT BOOSTING CLASSIFIER
#------------------------------------------------------------------------
print("\n--- Evaluando: Gradient Boosting Classifier ---")
print("Aviso: ¡Esto también puede tardar un poco...!")

# 1. Crear instancia del modelo
model_gb = GradientBoostingClassifier(
    n_estimators=1000,    # 1000 árboles
    random_state=42,
    # Nota: 'class_weight' y 'n_jobs' no son soportados por este modelo.
    # El desbalanceo se maneja en el .fit() con 'sample_weight'.
    learning_rate=0.05,    # Tasa de aprendizaje (importante para tuning)
    max_depth=15           # Profundidad de los árboles (importante para tuning)
) 

# 2. Calcular los pesos para manejar el desbalanceo
# Esto crea un array de pesos, dando más importancia a la clase 1
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)

# 3. Entrenar el modelo
# Pasamos los pesos de muestra aquí
model_gb.fit(X_train, y_train, sample_weight=sample_weights)
print("¡Modelo Gradient Boosting entrenado!")

# 4. Evaluar
y_pred_gb = model_gb.predict(X_test) # <--- Variable única

print(f"Precisión (Accuracy): {accuracy_score(y_test, y_pred_gb) * 100:.2f}%")
print("\nReporte de Clasificación (Gradient Boosting):")
print(classification_report(y_test, y_pred_gb))
print("\nMatriz de Confusión (Gradient Boosting):")
print(confusion_matrix(y_test, y_pred_gb))

# --- CÁLCULO ESPECÍFICO DE F1-SCORE ---

# Opción A: F1-Score Ponderado
f1_ponderado = f1_score(y_test, y_pred_gb, average='weighted')
print(f"\nF1-Score Ponderado (Weighted) de GB: {f1_ponderado:.4f}")

# Opción B: F1-Score Macro
f1_macro = f1_score(y_test, y_pred_gb, average='macro')
print(f"F1-Score Macro de GB: {f1_macro:.4f}")

# Opción C: F1-Score de la Clase 1
f1_clase_1 = f1_score(y_test, y_pred_gb, average=None)[1]
print(f"F1-Score (Clase 1) de GB: {f1_clase_1:.4f}")

# Opción C: F1-Score de la Clase 1 (la minoritaria)
# 'average=None' devuelve los F1s de ambas clases en una lista [f1_clase0, f1_clase1]
f1_clase_1 = f1_score(y_test, y_pred_rf, average=None)[1]
print(f"F1-Score (Clase 1) de RF: {f1_clase_1:.4f}")