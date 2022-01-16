# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pickle
import os

import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics as sme
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt

# %% tags=["parameters"]
execution_time = datetime.datetime.now().strftime('%Y-%m-%d')
model_type = 'gbm'
n_estimators = 3
criterion = 'entropy'
learning_rate = 0.1
upstream = {
    'preprocess_data':{
        'nb': 's2_preprocess_data.py',
        'data': '../data/processed_data.csv'
    }
}
product = {
    'nb': f'../products/{execution_time}/pipeline_notebooks/s3.ipynb',
    'model': f'../products/{execution_time}/models/model.pickle'
}

# %%
df = pd.read_csv(str(upstream['preprocess_data']['data']))

# %%
df.head(1)

# %%
test_size=0.33
train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

# %%
print("Train shape: ", train_df.shape)
print("Test shape: ", test_df.shape)

# %%
input_variables = df.columns.tolist()
input_variables.remove('TradePricePerArea')
input_variables.remove('No')
target_variable = 'TradePricePerArea'

# %%
if model_type == 'random-forest':
    model = RandomForestRegressor(random_state=1, max_depth=2, n_estimators=n_estimators, criterion=criterion)
elif model_type == 'gbm':
    model = GradientBoostingRegressor(random_state=1, n_estimators=n_estimators, learning_rate = learning_rate)
else:
    raise ValueError("Model type is not implemented")

# %%
model.fit(train_df[input_variables], train_df[target_variable])

# %%
y_pred = model.predict(test_df[input_variables])

# %%
print("MAE: ", round(sme.mean_absolute_error(test_df[target_variable], y_pred), 2))
print("MAPE: ", round(sme.mean_absolute_percentage_error(test_df[target_variable], y_pred), 2))
print("R2: ", round(sme.r2_score(test_df[target_variable], y_pred), 2))
print("MSE: ", round(sme.mean_squared_error(test_df[target_variable], y_pred), 2))

# %%
os.makedirs(os.path.dirname(product['model']), exist_ok=True)
with open(product['model'], 'wb') as f:
    pickle.dump(model, f)

# %%
