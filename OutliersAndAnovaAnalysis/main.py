# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ide_utils.input_tables import load_dask
from ide_utils.output_table import send_to_output_table
from OutlierAnalysis import AnalisisValoresAtipicos
from AnovaAnalyzer import AnovaAnalyzer

# Traer Datos
data = load_dask("Trust").compute()

def reemplazar_cero_por_mediana(row):
    if row['Qty_passangers'] == 0:
        return row['Mediana']
    else:
        return row['Qty_passangers']

data['Qty_passangers'] = data.apply(reemplazar_cero_por_mediana, axis=1)

data = data.drop(['uploaded',
       'window_daily', 'Mediana', 'promedio', 'relacion', 'dia_semana',
       'window_weekly', 'Mediana_semanal', 'promedio_semanal',
       'relacion_semanal', '__index_level_0__'], axis=1)

################# AnovaAnalyzer #############################
analyzer = AnovaAnalyzer(data)
resultado_anova = analyzer.realizar_anova()
resultado_anova.sort_values(by=['Valor_p'], ascending=True)
print("=========== ANOVA ===========")
print(resultado_anova)

################# AnalisisValoresAtipicos ###################

# Estructurar Datos
pivot_data = data.pivot_table(
    index=['Fecha', 'Linea','festivos'],
    columns='Horas',
    values='Qty_passangers',
    fill_value=0
).reset_index()

# Lineas
lineas = data['Linea'].unique()

# DataFrame Final
df_final = pd.DataFrame()

# Filtrar Lineas y Aplicar Proceso
for linea in lineas:
    pivot_data_linea = pivot_data[pivot_data['Linea'] == linea]
    pivot_data_linea_reduced = pivot_data_linea.drop(['Fecha','Linea','festivos'], axis=1)

    # Crear DataFrame de Atipicos
    analisis_de_atipicos = AnalisisValoresAtipicos(pivot_data_linea_reduced, threshold_outliers=1.5)
    resultados = analisis_de_atipicos.resultados

    # Recuperar Fecha, Linea y Festivos y Obtener Nuevas Caracteristicas
    if not resultados.empty:
        resultados_linea_merge = resultados.merge(pivot_data_linea[['Fecha','Linea','festivos']], left_index=True, right_index=True)
        resultados_linea_merge['Num_var_atipicas'] =  resultados_linea_merge['variables_atipicas'].apply(lambda x : len(x.split(';')))
        resultados_linea_merge['mes_dia'] =  resultados_linea_merge['Fecha'].apply(lambda x : str(x.month)+'-'+str(x.day))
        resultados_linea_merge['conteo_mes_dia'] = resultados_linea_merge.groupby('mes_dia')['mes_dia'].transform(lambda x: x.count())
    
    # Almacenar resultados, Filtrarlos/Generar Output
    df_final = pd.concat([df_final, resultados_linea_merge])

# Dataframe Final
df_final = df_final.reset_index(drop=True)
fontsize = 18

# Plot 1
# Establecer tamaño de fuente y tipo de fuente
plt.figure(figsize=(10, 6))
sns.countplot(x='Linea', data=df_final)
plt.title('Número de Registros por Línea', fontsize=fontsize+2)
plt.xlabel('Línea', fontsize=fontsize)
plt.ylabel('Cantidad de Registros', fontsize=fontsize)
plt.grid(True)
plt.show()

# Calcular media de variables_atipicas para cada linea por festivos'
grouped_data = df_final.groupby(['Linea', 'festivos'])['Num_var_atipicas'].mean().reset_index()

# Plot 2
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Linea', y='Num_var_atipicas', hue='festivos', data=grouped_data)
plt.title('Promedio de Variables Atípicas por Línea y Tipo de Día', fontsize=fontsize+2)
plt.xlabel('Línea', fontsize=fontsize)
plt.ylabel('Promedio de Variables Atípicas', fontsize=fontsize)
plt.grid(True)
plt.show()

# Procesamiento inicial: Extraer las variables atípicas y su frecuencia
df_final['variables_atipicas'] = df_final['variables_atipicas'].astype(str)
atipicos_desglosados = df_final['variables_atipicas'].str.split(';').explode()
frecuencia_atipicos = atipicos_desglosados.value_counts()
# Crear una tabla pivote para cada línea y contar la frecuencia de variables atípicas
pivot_table_lineas = pd.pivot_table(df_final, values='variables_atipicas', index=['Linea'], aggfunc=lambda x: x.value_counts().index[0] if len(x) > 0 else np.nan)
# Mostrar las 10 variables atípicas más representativas para cada línea con la fecha
for linea in pivot_table_lineas.index:
    print('='*80)
    top_10_atipicos_linea = df_final[df_final['Linea'] == linea].groupby(['Fecha', 'variables_atipicas']).size().reset_index(name='Count')
    top_10_atipicos_linea = top_10_atipicos_linea.sort_values(by='Count', ascending=False).head(10)
    print(f"\nTop 10 casos más representativos para la línea {linea}:\n{top_10_atipicos_linea}")

# Plot 3: Histograma de la cantidad de variables atípicas por registro
# plt.figure(figsize=(10, 6))
# sns.histplot(df_final['Num_var_atipicas'], bins=15, kde=True)
# plt.title('Distribución de la Cantidad de Variables Atípicas por Registro')
# plt.xlabel('Cantidad de Variables Atípicas')
# plt.ylabel('Frecuencia')
# plt.grid(True)
# plt.show()

# Plot 4: Frecuencia de variables atípicas específicas
plt.figure(figsize=(12, 8))
frecuencia_atipicos.head(20).plot(kind='bar')
plt.title('Top 20 Variables Atípicas más Frecuentes', fontsize=fontsize+2)
plt.xlabel('Variable Atípica', fontsize=fontsize)
plt.ylabel('Frecuencia', fontsize=fontsize)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Frecuencia de valores atípicos en días festivos vs días no festivos
frecuencia_festivos = df_final.groupby('festivos')['Num_var_atipicas'].sum()

# Frecuencia de valores atípicos por mes-día (para ver si hay fechas específicas con alta incidencia)
df_final['Fecha'] = pd.to_datetime(df_final['Fecha'])  # Asegurándose de que la columna Fecha es de tipo datetime
df_final['Mes'] = df_final['Fecha'].dt.month
frecuencia_por_mes = df_final.groupby('Mes')['Num_var_atipicas'].sum()

# Plot 5
# plt.figure(figsize=(6, 4))
# frecuencia_festivos.plot(kind='bar')
# plt.title('Frecuencia de Valores Atípicos en Días Festivos vs No Festivos')
# plt.xlabel('Es Festivo')
# plt.ylabel('Cantidad de Valores Atípicos')
# plt.xticks([0, 1], ['No', 'Sí'], rotation=0)
# plt.grid(True)
# plt.show()

# Plot 6
plt.figure(figsize=(12, 6))
frecuencia_por_mes.plot(kind='bar')
plt.title('Frecuencia de Valores Atípicos por Mes', fontsize=fontsize+2)
plt.xlabel('Mes', fontsize=fontsize)
plt.ylabel('Cantidad de Valores Atípicos', fontsize=fontsize)
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# Plot 7
# Extrayendo las top 20 variables atípicas
top_20_atipicos = atipicos_desglosados.value_counts().head(3).index

# Preparando un DataFrame para cada variable atípica
data_top_atipicos = []
for var in top_20_atipicos:
    # Filtrando el DataFrame original para cada variable atípica
    df_filtrado = df_final[df_final['variables_atipicas'].str.contains(var, regex=False)]

    # Conteo por mes y por festivo
    conteo_mes = df_filtrado.groupby('Mes').size()
    conteo_festivo = df_filtrado.groupby('festivos').size()

    # Añadiendo al conjunto de datos
    data_top_atipicos.append({'Variable Atípica': var, 'Conteo por Mes': conteo_mes, 'Conteo por Festivo': conteo_festivo})

# Creando visualizaciones para cada variable atípica
for data in data_top_atipicos:
    plt.figure(figsize=(15, 6))
    # Gráfico para el conteo por mes
    # plt.subplot(1, 2, 1)
    data['Conteo por Mes'].plot(kind='bar')#, color='skyblue')
    plt.title(f'Frecuencia de {data["Variable Atípica"]} por Mes', fontsize=fontsize+2)
    plt.xlabel('Mes', fontsize=fontsize)
    plt.ylabel('Frecuencia', fontsize=fontsize)
    plt.grid(True)

    # Gráfico para el conteo por festivo
    # plt.subplot(1, 2, 2)
    # data['Conteo por Festivo'].plot(kind='bar', color='salmon')
    # plt.title(f'Frecuencia de {data["Variable Atípica"]} en Días Festivos vs No Festivos')
    # plt.xlabel('Es Festivo')
    # plt.ylabel('Frecuencia')
    # plt.xticks([0, 1], ['No', 'Sí'])
    # plt.tight_layout()
    # plt.grid(True)
    # plt.show()

send_to_output_table(resultado_anova, table='Anova')
send_to_output_table(df_final, table='Outliers')