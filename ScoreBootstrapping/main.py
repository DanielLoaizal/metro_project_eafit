#  %%
import pandas as pd
from ide_utils.input_tables import load_dask
from ide_utils.output_table import send_to_output_table
from sklearn.utils import resample
import numpy as np

# Cargar datos
df = load_dask("Scores").compute()

def bootstrap_median(data, column, n_bootstrap=1000, alpha=0.05):
    medians = []
    for _ in range(n_bootstrap):
        sample = resample(data[column], replace=True, n_samples=len(data))
        medians.append(np.median(sample))
    # Calcular percentiles para el intervalo de confianza
    lower_bound = np.percentile(medians, 100 * alpha / 2)
    upper_bound = np.percentile(medians, 100 * (1 - alpha / 2))
    return np.median(data[column]), lower_bound, upper_bound

# Crear un nuevo DataFrame para almacenar los resultados
result_df = pd.DataFrame(columns=['Linea', 'Mediana_RMSE', 'Inf_RMSE', 'Sup_RMSE', 'Mediana_R2', 'Inf_R2', 'Sup_R2'])

# Iterar sobre las líneas del DataFrame original
for linea in df['Linea'].unique():
    # Filtrar el DataFrame por la línea actual
    linea_data = df[df['Linea'] == linea]
    # Calcular la mediana y rango de confianza para RMSE y Rdos
    rmse_stats = bootstrap_median(linea_data, 'mse')
    rdos_stats = bootstrap_median(linea_data, 'r2')
    # Agregar resultados al nuevo DataFrame utilizando concat
    result_df = pd.concat([result_df, pd.DataFrame({
        'Linea': [linea],
        'Mediana_RMSE': [rmse_stats[0]],
        'Inf_RMSE': [rmse_stats[1]],
        'Sup_RMSE': [rmse_stats[2]],
        'Mediana_R2': [rdos_stats[0]],
        'Inf_R2': [rdos_stats[1]],
        'Sup_R2': [rdos_stats[2]]
    })], ignore_index=True)

# Mostrar el nuevo DataFrame con los resultados
print(result_df)

send_to_output_table(result_df, table="summary")
