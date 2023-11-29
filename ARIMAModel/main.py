# %%
###############################################################################
from ide_utils.input_tables import load_dask
from ide_utils.output_table import send_to_output_table

###############################################################################
# from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import itertools
import json
import os

###############################################################################
warnings.filterwarnings("ignore")
from ARIMAModel import ARIMAModel

###############################################################################
def imputacion(data):
    if data['Qty_passangers']==0:
        return data['Mediana']
    else:
        return data['Qty_passangers']

df = load_dask("Trust").compute()
df.sort_values(by=['Fecha', 'Horas'], inplace=True)
df.reset_index(drop=True, inplace=True)
df['Qty_passangers'] = df.apply(imputacion, axis=1)
df = df.drop(['uploaded', 'festivos','window_daily', 'Mediana', 'promedio',
             'relacion', 'dia_semana','window_weekly', 'Mediana_semanal',
             'promedio_semanal','relacion_semanal', '__index_level_0__'], axis=1)


resultados_df = pd.DataFrame()
scores_total = pd.DataFrame()
train_until = '2023-06-23'

###############################################################################
# Nombre de la carpeta de salida
output_folder = "output_data"  

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for linea in df.loc[~df.Linea.isin(['T-A', 'M', 'J', 'K']),'Linea'].unique():
    eval_linea = {}  # Nuevo diccionario para cada línea

    # Crear la carpeta para la línea si no existe
    line_folder = os.path.join(output_folder, str(linea))
    if not os.path.exists(line_folder):
        os.makedirs(line_folder)

    for hora in df.Horas.unique():
        print(f"Linea: {linea} - Hora: {hora}")
        print("=" * 100)
        scores = pd.DataFrame([[linea, hora]], columns=['Linea', 'Horas'])
        arima_model_instance = ARIMAModel(df, linea=linea, hora=hora, fecha_columna='Fecha', valor_columna='Qty_passangers', frecuencia='D', train_until=train_until)
        prediction_df = arima_model_instance.test_df
        scores['pvalue'] = arima_model_instance.test_stationarity()
        model_fit, best_aic, best_order = arima_model_instance.train_arima_model()
        prediction_df['predictions'] = arima_model_instance.apply_arima_to_test(model_fit).to_list()
        scores['best_aic'] = best_aic
        scores['best_order'] = [list(best_order)]
        # Scores
        r2, mse = arima_model_instance.score(prediction_df['predictions'])
        scores['r2'] = r2
        scores['mse'] = mse

        # Guardando la salida
        resultados_df = pd.concat([resultados_df, prediction_df])
        scores_total = pd.concat([scores_total, scores])
        
        eval_linea[str(hora)] = {'predictions': prediction_df['predictions'].values.tolist(), 'scores': [r2, mse], 'p-value': scores['pvalue'], 'aic' : best_aic, 'best-order' : best_order}

resultados_df['Fecha'] = resultados_df['Fecha'].astype(str)
send_to_output_table(resultados_df, table='resultados_v2')
send_to_output_table(scores_total, table='Scores_v2')

# ###############################################################################