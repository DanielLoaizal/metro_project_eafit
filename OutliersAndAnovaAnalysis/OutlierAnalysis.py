import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class AnalisisValoresAtipicos:
    def __init__(self, data, threshold_outliers=1.5):
        self.df = data
        self.threshold = threshold_outliers
        self.resultados = self.identificar_valores_atipicos_iqr()
        self.filas_con_atipicos = self.filtrar_filas_con_atipicos()
        
    def identificar_valores_atipicos_iqr(self):
        filas_atipicas = {}
        for columna in self.df.columns:
            Q1 = self.df[columna].quantile(0.25)
            Q3 = self.df[columna].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - self.threshold * IQR
            limite_superior = Q3 + self.threshold * IQR
            atipicos = self.df[(self.df[columna] < limite_inferior) | (self.df[columna] > limite_superior)]
            for indice, fila in atipicos.iterrows():
                direccion = "por encima" if fila[columna] > limite_superior else "por debajo"
                info_atipico = f"{columna} ({direccion})"
                if indice in filas_atipicas:
                    filas_atipicas[indice].append(info_atipico)
                else:
                    filas_atipicas[indice] = [info_atipico]
        filas_resultado = []
        for indice, variables in filas_atipicas.items():
            fila = self.df.loc[indice].copy()
            fila['variables_atipicas'] = '; '.join(variables)
            filas_resultado.append(fila)
        df_atipico = pd.DataFrame(filas_resultado)
        return df_atipico

    def filtrar_filas_con_atipicos(self):
        return self.df[self.df.index.isin(self.resultados.index)]