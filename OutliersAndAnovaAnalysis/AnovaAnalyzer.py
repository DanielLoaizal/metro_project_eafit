import pandas as pd
from scipy.stats import f_oneway

class AnovaAnalyzer:
    def __init__(self, df):
        self.df = df
        self.resultado_anova = None

    def realizar_anova(self):
        """
        Realiza un análisis de varianza (ANOVA) para determinar si hay diferencias significativas
        en la cantidad de pasajeros entre los días de la semana para cada combinación única de línea y hora.

        Retorna:
        - df_resultados (pd.DataFrame): DataFrame que contiene los resultados de las pruebas de ANOVA.

        Explicación:
        Un valor p menor a 0.05 indica que hay evidencia suficiente para rechazar la hipótesis nula.
        En este contexto, la hipótesis nula (H0) asume que las medias de la cantidad de pasajeros son iguales
        para todos los días de la semana en una línea y hora específicas. La hipótesis alternativa (H1)
        sugiere que al menos una de las medias es diferente.

        Por lo tanto:
        - Si Valor_p < 0.05: Se rechaza H0, lo que sugiere que hay diferencias significativas en la cantidad
          de pasajeros entre los días de la semana para esa combinación específica de línea y hora.
        - Si Valor_p >= 0.05: No hay evidencia suficiente para rechazar H0, lo que indica que no hay
          diferencias significativas en la cantidad de pasajeros entre los días de la semana.
        """
        # Manipulación de fechas
        self.df['Fecha'] = pd.to_datetime(self.df['Fecha'])
        self.df['dia_semana'] = self.df['Fecha'].dt.day_name()

        # Inicialización de la lista de resultados de ANOVA
        resultados_anova = []

        # Bucle para realizar pruebas de ANOVA para cada combinación de línea y hora
        for linea in self.df['Linea'].unique():
            for hora in self.df['Horas'].unique():
                # Filtrar el DataFrame por línea y hora
                subset = self.df[(self.df['Linea'] == linea) & (self.df['Horas'] == hora)]

                # Crear grupos para la prueba de ANOVA
                grupos = [subset[subset['dia_semana'] == dia]['Qty_passangers'] for dia in self.df['dia_semana'].unique()]

                # Realizar la prueba de ANOVA
                resultado_anova = f_oneway(*grupos)

                # Almacenar los resultados en la lista
                resultados_anova.append({
                    'Linea': linea,
                    'Hora': hora,
                    'Estadistica': resultado_anova.statistic,
                    'Valor_p': resultado_anova.pvalue
                })

        # Convertir la lista de resultados en un DataFrame
        self.resultado_anova = pd.DataFrame(resultados_anova)

        return self.resultado_anova[self.resultado_anova['Valor_p'] > 0.05]