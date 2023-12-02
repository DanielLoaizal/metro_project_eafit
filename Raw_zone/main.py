# %%
import pandas as pd
import numpy as np
from ide_utils.input_tables import load_dask
from ide_utils.output_table import send_to_output_table


# Extrayendo los datos del API
afluencia_2021 = pd.read_excel("https://www.arcgis.com/sharing/rest/content/items/90bda39bd6d848de8c7eebd2b66d801e/data", header=2)
afluencia_2022 = pd.read_excel("https://www.arcgis.com/sharing/rest/content/items/1450e500041842a387444633ac656798/data", header=1)
afluencia_2023 = pd.read_excel("https://www.arcgis.com/sharing/rest/content/items/19f40fa21e204565a14482bee823ec44/data", header=1)

# Uniendo todos los datasets
all_data = pd.concat([afluencia_2021, afluencia_2022, afluencia_2023])

# Convirtiendo los nombres de las columnas en string
data_columns = all_data.columns

data_columns = list(map(lambda index: str(index).replace(':00:00', ''), data_columns))

# Asignando el verdadero valor de las columnas 0 y 1
data_columns[0] = 'Fecha'
data_columns[1] = 'Linea'
all_data = all_data.set_axis(data_columns, axis=1).drop(all_data.columns[-1], axis=1)

all_data['Fecha'] = all_data.Fecha.astype(str)


# Enviando a la salida del operador
send_to_output_table(all_data)