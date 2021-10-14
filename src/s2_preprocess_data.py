# %%
import pandas as pd

# %% tags=["parameters"]
upstream = {
    'load_data':{
        'nb': 's1_load_data.py',
        'data': '../data/raw/raw_data.csv'
    }
}
product = {
    'nb': 'pipeline_notebooks/s2_process_data.ipynb',
    'data': '../data/processed/processed_data.csv'
}

# %%
df = pd.read_csv(str(upstream['load_data']['data']))

# %%
df.head(2)

# %%
df["sepal area"] = df['sepal length (cm)'] *  df['sepal width (cm)']

# %%
df.head(2)

# %%
df.to_csv(str(product['data']))

# %%
