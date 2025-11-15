import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

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