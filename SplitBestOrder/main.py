import pandas as pd
from ide_utils.input_tables import load_dask
from ide_utils.output_table import send_to_output_table

df = load_dask("scores").compute()

def extract_item_values(row):
    try:
        return [item['item'] for item in row['best_order']['list']]
    except (KeyError, TypeError):
        return None

df['item_values'] = df.apply(extract_item_values, axis=1)
df['p'] = df['item_values'].apply(lambda x : x[0])
df['d'] = df['item_values'].apply(lambda x : x[1])
df['q'] = df['item_values'].apply(lambda x : x[2])
df = df[['Linea', 'Horas', 'pvalue', 'best_aic', 'best_order', 'r2', 'mse', 'p', 'd', 'q']]

send_to_output_table(df, table="ScoresV2")