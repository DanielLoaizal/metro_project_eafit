from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

class ARIMAModel:
    def __init__(self, df, linea=None, hora=None, fecha_columna='fecha', valor_columna='afluencia', frecuencia='D', train_until=None):
        self.train_df = df.loc[df.Fecha<=train_until]
        self.test_df = df.loc[df.Fecha>train_until]
        self.train_df = self.compactar_datos(self.train_df, linea, hora, fecha_columna, valor_columna, frecuencia)
        self.test_df = self.compactar_datos(self.test_df, linea, hora, fecha_columna, valor_columna, frecuencia)
        self.train_data = self.train_df[valor_columna]
        self.test_data = self.test_df[valor_columna]
        self.df = self.compactar_datos(df, linea, hora, fecha_columna, valor_columna, frecuencia)
        self.X = self.df[valor_columna]

    def compactar_datos(self, df, linea=None, hora=None, fecha_columna='fecha', valor_columna='afluencia', frecuencia='D'):
        """
        Función para compactar datos temporales en un DataFrame según una frecuencia específica.

        Parámetros:
        - df: pandas DataFrame
            DataFrame que contiene los datos a ser compactados.
        - linea: str o None, opcional
            Valor para filtrar el DataFrame por la columna 'Linea'. Si es None, no se aplica ningún filtro por línea.
        - hora: int o None, opcional
            Valor para filtrar el DataFrame por la columna 'Horas'. Si es None, no se aplica ningún filtro por hora.
        - fecha_columna: str, opcional (por defecto: 'fecha')
            Nombre de la columna que contiene las fechas en el DataFrame.
        - valor_columna: str, opcional (por defecto: 'afluencia')
            Nombre de la columna que contiene los valores a sumar durante la compactación.
        - frecuencia: str, opcional (por defecto: 'D')
            Frecuencia a la cual compactar los datos. Utiliza códigos de frecuencia de pandas (por ejemplo, 'D' para días, 'H' para horas).

        Salida:
        - pandas DataFrame
            DataFrame compactado con las sumas de los valores de 'valor_columna' según la frecuencia especificada.

        Notas:
        - La función realiza una copia del DataFrame original para evitar modificaciones no deseadas.
        - Se pueden proporcionar valores para 'linea' y 'hora' para filtrar los datos antes de la compactación.
        - La columna de fechas ('fecha_columna') debe estar en formato datetime para realizar la compactación correctamente.
        - La frecuencia determina el intervalo temporal para la compactación (por ejemplo, 'D' para días, 'H' para horas).
        """ 
        df_copy = df.copy()

        # Filtrar por línea y hora si se proporcionan
        if linea is not None:
            df_copy = df_copy[df_copy['Linea'] == linea]
        if hora is not None:
            df_copy = df_copy[df_copy['Horas'] == hora]

        df_copy[fecha_columna] = pd.to_datetime(df_copy[fecha_columna])
        df_copy.set_index(fecha_columna, inplace=True)

        # Compactar los datos utilizando la frecuencia especificada
        df_compactado = df_copy.groupby(['Linea', 'Horas', pd.Grouper(freq=frecuencia)]).sum().reset_index()

        return df_compactado

    def test_stationarity(self):
        """ Realizar la prueba de Dickey-Fuller """
        result = adfuller(self.X)
        print('Prueba de Dickey-Fuller:')
        print(f'Estadística de prueba: {result[0]}')
        print(f'P-valor: {result[1]}')
        print(f'Valores críticos: {result[4]}')
        return result[1]

    def train_arima_model(self):
        """
        Entrena un modelo ARIMA utilizando la librería statsmodels. 
        ARIMA(p, d, q).

        Parámetros:
        - data: Serie temporal a modelar.
        - p: Orden del componente autorregresivo.
        - d: Orden de diferenciación para hacer estacionaria la serie temporal.
        - q: Orden del componente de media móvil.

        
        El AIC (Akaike Information Criterion) es un criterio de información que se utiliza para comparar 
        modelos estadísticos. Fue propuesto por Hirotugu Akaike y se utiliza comúnmente en el contexto de
        la selección de modelos, especialmente en el campo de la estadística y el análisis de series temporales.
        La idea detrás del AIC es encontrar un equilibrio entre la complejidad del modelo y su capacidad para 
        explicar los datos. Cuanto menor sea el valor del AIC, mejor se considera el modelo 
        en términos de ajuste y complejidad.

        Retorna:
        - Modelo ARIMA ajustado.
        """

        # Definir rangos para p, d, q
        p_range = range(6, 9)
        d_range = range(0, 2)
        q_range = range(6, 9)

        # Generar todas las combinaciones posibles de p, d, q
        order_combinations = list(itertools.product(p_range, d_range, q_range))

        best_aic = float('inf')  # Inicializar con un valor grande
        best_order = None

        # Iterar sobre todas las combinaciones y ajustar modelos ARIMA
        for order in order_combinations:
            try:
                arima_model = ARIMA(self.train_data, order=order, enforce_stationarity=False, enforce_invertibility=False)
                arima_model_fit = arima_model.fit()
                current_aic = arima_model_fit.aic

                if current_aic < best_aic:
                    best_aic = current_aic
                    best_order = order
            except ValueError as e:
                print(f'Error ajustando modelo ARIMA para orden {order}: {e}')
                continue

        # Ajustar el mejor modelo ARIMA con los órdenes encontrados
        try:
            best_arima_model = ARIMA(self.train_data, order=best_order)
            best_arima_model_fit = best_arima_model.fit()

            # Imprimir los órdenes óptimos y el AIC asociado
            print(f'Mejores órdenes: {best_order}')
            print(f'Mejor AIC: {best_aic}')

            # Visualizar autocorrelación y autocorrelación parcial del residuo
            self.plot_acf_pacf(best_arima_model_fit.resid)

            return best_arima_model_fit, best_aic, best_order
        except ValueError as e:
            print(f'Error ajustando el mejor modelo ARIMA: {e}')
            return None

    def plot_acf_pacf(self, resid):
        """
        Función para visualizar los gráficos de autocorrelación (ACF) y autocorrelación parcial (PACF) de los residuos.

        Parámetros:
        - resid: array-like
            Una serie de tiempo de residuos para la cual se desea visualizar los gráficos.

        Salida:
        - Figura de los gráficos ACF y PACF de los residuos.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        # Autocorrelación
        plot_acf(resid, lags=20, ax=ax1)
        ax1.set_title('Autocorrelación')
        ax1.grid(True)
        # Autocorrelación parcial
        plot_pacf(resid, lags=20, ax=ax2)
        ax2.set_title('Autocorrelación Parcial')
        ax2.grid(True)
        return fig

    def apply_arima_to_test(self, model_fit):
        """
        Aplica un modelo ARIMA previamente ajustado a los datos de prueba y devuelve las predicciones.

        Parámetros:
        - model_fit: Modelo ARIMA previamente ajustado.

        Retorna:
        - Predicciones del modelo ARIMA en los datos de prueba.
        """
        if model_fit is None:
            print("No se proporcionó un modelo ARIMA válido.")
            return None

        # Aplicar el modelo a los datos de prueba
        predictions = model_fit.forecast(steps=len(self.test_data))

        return predictions      

    def plot_predictions_vs_actual(self, predictions):
        """
        Visualiza la comparación entre los datos reales y las predicciones del modelo.

        Parámetros:
        - predictions: Predicciones del modelo ARIMA en los datos de prueba.
        - test_data: Datos reales de prueba.

        Salida:
        - Figura del gráfico de comparación entre Datos Reales y Predichos.
        """
        # Crear un DataFrame para visualización
        df_vis = pd.DataFrame({'Actual': self.test_data, 'Predicted': predictions}, index=self.test_data.index)

        # Graficar los datos reales y las predicciones
        fig = plt.figure(figsize=(13, 4))
        plt.plot(df_vis['Actual'], label='Actual', marker='o')
        plt.plot(df_vis['Predicted'], label='Predicted', marker='o')
        plt.title('Comparación entre Datos Reales y Predichos')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        return fig

    def score(self, predictions):
      # Calcular R²
      r2 = r2_score(self.test_data, predictions)
      print(f"Coeficiente de determinación (R²): {r2}")
      # Calcular MSE
      mse = mean_squared_error(self.test_data, predictions)
      print(f"Error cuadrático medio (RMSE): {np.sqrt(mse)}")
      return r2, np.sqrt(mse)