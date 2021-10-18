# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn_evaluation import plot

# %% tags=["parameters"]
test_size = 0.33
upstream = {
    'preprocess_data':{
        'nb': 's2_preprocess_data.py',
        'data': '../data/processed_data.csv'
    }
}
product = {
    'nb': 'pipeline_notebooks/s3.ipynb',
    'model': '../models/model.pickle'
}

# %%
df = pd.read_csv(str(upstream['preprocess_data']['data']))

# %%
df.head()

# %%
train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

# %%
print("Train shape: ", train_df.shape)
print("Test shape: ", test_df.shape)

# %%
input_variables = df.columns.tolist()
input_variables.remove('target')
target_variable = 'target'

# %%
clf = RandomForestClassifier(random_state=1, max_depth=2, n_estimators=2)
clf.fit(train_df[input_variables], train_df[target_variable])

# %%
y_pred = clf.predict(test_df[input_variables])

print(classification_report(test_df[target_variable], y_pred))

# %%
plot.confusion_matrix(test_df[target_variable], y_pred)

# %%
with open(product['model'], 'wb') as f:
    pickle.dump(clf, f)

# %%
