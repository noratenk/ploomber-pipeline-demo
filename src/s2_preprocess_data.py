# %%
import pandas as pd
import numpy as np
import tqdm
from sklearn.preprocessing import OneHotEncoder

# %%
pd.set_option('display.max_columns', None)

# %% tags=["parameters"]
upstream = {
    'load_data':{
        'nb': 's1_load_data.py',
        'data': '../data/raw_data.csv'
    }
}
product = {
    'nb': 'pipeline_notebooks/s2_process_data.ipynb',
    'data': '../data/processed_data.csv'
}

# %%
df = pd.read_csv(str(upstream['load_data']['data']), low_memory=False)

# %%
target='TradePrice'
df = df.dropna(subset = [target, 'Area'])

# %%
columns_to_drop = [
    'Renovation', 'Remarks', 'Period', 
    'Municipality', 'DistrictName', 'NearestStation', 
    'FloorPlan', 'TimeToNearestStation', 'MunicipalityCode', 
    'AreaIsGreaterFlag', 'UnitPrice', 'TotalFloorArea',
    'PricePerTsubo', 'Structure']
df = df.drop(columns=columns_to_drop)

# %%
df["Use_House"] = df['Use'].str.contains('House').fillna(False).astype(int)
df["Use_Office"] = df['Use'].str.contains('Office').fillna(False).astype(int)
df["Use_Shop"] = df['Use'].str.contains('Shop').fillna(False).astype(int)
df["Use_Factory"] = df['Use'].str.contains('Factory').fillna(False).astype(int)
df["Use_Parking"] = df['Use'].str.contains('Parking Lot').fillna(False).astype(int)
del df["Use"]

# %%
for col in df.select_dtypes(include=[bool]).columns.tolist():
    df[col] = df[col].astype(int)

# %%
categorical_columns = df.select_dtypes(include=[object]).columns.tolist()
for col in categorical_columns:
    df[col].fillna("Unknown", inplace=True)

# %%
numerical_columns = df.select_dtypes(include=[np.number, int, float]).columns.tolist()[1:]
for col in numerical_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# %%
assert df.isna().sum().sum()==0

# %%
for col in tqdm.tqdm(categorical_columns):
    df[col] = df[col].str.replace(',', ' ', regex=True)
    df[col] = df[col].str.replace('.', '', regex=True)
    df_cat = pd.get_dummies(df[col], drop_first=True)
    df_cat.columns = col+"_"+df_cat.columns
    df = df.join(df_cat)
    del df[col]

# %%
df.shape

# %%
df.head()

# %%
assert len(df.select_dtypes(include=[object]).columns.tolist())==0

# %%
assert df.loc[df["Area"]<1].shape[0]==0

# %%
df["TradePricePerArea"] = df["TradePrice"]/df["Area"]
df.drop(["TradePrice", "Area"], axis=1, inplace=True)

# %%
df.to_csv(str(product['data']), index=False)

# %%
