import pandas as pd

# Cargar el dataset
file_path = "dataset.csv"
try:
    df = pd.read_csv(file_path)
    
    # Columnas de interés
    competitor_columns = ['competitor_Z', 'competitor_X', 'competitor_Y', 'cust_in_iberia']
    
    # Calcular la media (probabilidad de éxito) para cada columna
    # Para una columna binaria (0/1), la media es igual a la 
    # proporción de 1s, que es la probabilidad de éxito.
    probabilities = df[competitor_columns].mean()
    
    print("Probabilidad de éxito (media) para cada competidor:")
    print(probabilities)
    
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta '{file_path}'")
except KeyError:
    print(f"Error: Una o más columnas {competitor_columns} no se encontraron.")
except Exception as e:
    print(f"Ocurrió un error: {e}")
