import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


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

#-------------------------------------------------------------------------
# MODELO 1: REGRESIÓN LOGÍSTICA
#-------------------------------------------------------------------------
print("\n--- Evaluando: Regresión Logística ---")

# 1. Crear instancia del modelo
model_lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",  # <--- IMPORTANTE: Para manejar el desbalanceo
    random_state=42
) 

# 2. Entrenar el modelo
model_lr.fit(X_train, y_train)

# 3. Evaluar
y_pred_lr = model_lr.predict(X_test) # <--- Variable única para la predicción

print(f"Precisión (Accuracy): {accuracy_score(y_test, y_pred_lr) * 100:.2f}%")
print("\nReporte de Clasificación (Regresión Logística):")
print(classification_report(y_test, y_pred_lr))
print("\nMatriz de Confusión (Regresión Logística):")
print(confusion_matrix(y_test, y_pred_lr))

print("\n" + "-" * 40)

# --- CÁLCULO ESPECÍFICO DE F1-SCORE ---

# Opción A: F1-Score Ponderado (el resumen más común)
# 'weighted' da más peso al f1 de la clase 0 (porque hay más muestras)
f1_ponderado = f1_score(y_test, y_pred_lr, average='weighted')
print(f"\nF1-Score Ponderado (Weighted) de LR: {f1_ponderado:.4f}")

# Opción B: F1-Score Macro (promedio simple de F1s)
# 'macro' trata ambas clases por igual, sin importar el desbalanceo
f1_macro = f1_score(y_test, y_pred_lr, average='macro')
print(f"F1-Score Macro de LR: {f1_macro:.4f}")

# Opción C: F1-Score de la Clase 1 (la minoritaria)
# 'average=None' devuelve los F1s de ambas clases en una lista [f1_clase0, f1_clase1]
f1_clase_1 = f1_score(y_test, y_pred_lr, average=None)[1]
print(f"F1-Score (Clase 1) de LR: {f1_clase_1:.4f}")

#------------------------------------------------------------------------
# MODELO 2: RANDOM FOREST
#------------------------------------------------------------------------
print("\n--- Evaluando: Random Forest ---")

# 1. Crear instancia del modelo
model_rf = RandomForestClassifier(
    n_estimators=100,
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
# MODELO 3: SUPPORT VECTOR MACHINE (SVM)
#------------------------------------------------------------------------
print("\n--- Evaluando: Support Vector Machine (SVC) ---")
print("Aviso: ¡Esto puede tardar varios minutos...!")

# --- INICIO DEL CÓDIGO FALTANTE ---

# 1. Crear instancia del modelo
model_svc = SVC(
    class_weight="balanced",  # Para el desbalanceo
    random_state=42
) 

# 2. Entrenar el modelo
model_svc.fit(X_train, y_train)
print("¡Modelo SVM entrenado!")

# --- FIN DEL CÓDIGO FALTANTE ---

# 3. Evaluar
y_pred_svc = model_svc.predict(X_test) # <--- Esta línea ahora funcionará

print("\nReporte de Clasificación (SVM):")
print(classification_report(y_test, y_pred_svc))
print("\nMatriz de Confusión (SVM):")
print(confusion_matrix(y_test, y_pred_svc))

# --- CÁLCULO ESPECÍFICO DE F1-SCORE ---

# Opción A: F1-Score Ponderado (el resumen más común)
# 'weighted' da más peso al f1 de la clase 0 (porque hay más muestras)
f1_ponderado = f1_score(y_test, y_pred_svc, average='weighted')
print(f"\nF1-Score Ponderado (Weighted) de SVM: {f1_ponderado:.4f}")

# Opción B: F1-Score Macro (promedio simple de F1s)
# 'macro' trata ambas clases por igual, sin importar el desbalanceo
f1_macro = f1_score(y_test, y_pred_svc, average='macro')
print(f"F1-Score Macro de SVM: {f1_macro:.4f}")

# Opción C: F1-Score de la Clase 1 (la minoritaria)
# 'average=None' devuelve los F1s de ambas clases en una lista [f1_clase0, f1_clase1]
f1_clase_1 = f1_score(y_test, y_pred_svc, average=None)[1]
print(f"F1-Score (Clase 1) de SVM: {f1_clase_1:.4f}")

#------------------------------------------------------------------------
# MODELO 4: ÁRBOL DE DECISIÓN (DECISION TREE)
#------------------------------------------------------------------------
print("\n--- Evaluando: Árbol de Decisión (Decision Tree) ---")

# 1. Crear instancia del modelo
model_dt = DecisionTreeClassifier(
    max_depth=7,             # <--- Límite MUY importante para evitar sobreajuste.
                             #      Puedes probar a quitarlo, o poner 5, 10...
    class_weight="balanced", # <--- Para el desbalanceo
    random_state=42
)

# 2. Entrenar el modelo
model_dt.fit(X_train, y_train)

# 3. Evaluar
y_pred_dt = model_dt.predict(X_test) # <--- Variable única

print("\nReporte de Clasificación (Árbol de Decisión):")
print(classification_report(y_test, y_pred_dt))
print("\nMatriz de Confusión (Árbol de Decisión):")
print(confusion_matrix(y_test, y_pred_dt))

# --- CÁLCULO ESPECÍFICO DE F1-SCORE ---

# Opción A: F1-Score Ponderado (el resumen más común)
# 'weighted' da más peso al f1 de la clase 0 (porque hay más muestras)
f1_ponderado = f1_score(y_test, y_pred_svc, average='weighted')
print(f"\nF1-Score Ponderado (Weighted) de DT: {f1_ponderado:.4f}")

# Opción B: F1-Score Macro (promedio simple de F1s)
# 'macro' trata ambas clases por igual, sin importar el desbalanceo
f1_macro = f1_score(y_test, y_pred_svc, average='macro')
print(f"F1-Score Macro de DT: {f1_macro:.4f}")

# Opción C: F1-Score de la Clase 1 (la minoritaria)
# 'average=None' devuelve los F1s de ambas clases en una lista [f1_clase0, f1_clase1]
f1_clase_1 = f1_score(y_test, y_pred_svc, average=None)[1]
print(f"F1-Score (Clase 1) de DT: {f1_clase_1:.4f}")