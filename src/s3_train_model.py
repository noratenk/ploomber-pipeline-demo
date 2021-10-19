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
import os

import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn_evaluation import plot
import matplotlib.pyplot as plt

# %% tags=["parameters"]
execution_date = datetime.datetime.now().strftime('%Y-%m-%d')
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
    'nb': 'pipeline_notebooks/s3.ipynb',
    'model': f'../models/{execution_date}/model.pickle'
}

# %%
df = pd.read_csv(str(upstream['preprocess_data']['data']))

# %%
df.head()

# %%
test_size=0.33
train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

# %%
print("Train shape: ", train_df.shape)
print("Test shape: ", test_df.shape)

# %%
input_variables = df.columns.tolist()
input_variables.remove('target')
target_variable = 'target'

# %%
if model_type == 'random-forest':
    clf = RandomForestClassifier(random_state=1, max_depth=2, n_estimators=n_estimators, criterion=criterion)
elif model_type == 'gbm':
    clf = GradientBoostingClassifier(random_state=1, n_estimators=n_estimators, learning_rate = learning_rate)
else:
    raise ValueError("Model type is not implemented")

# %%
clf.fit(train_df[input_variables], train_df[target_variable])

# %%
y_pred = clf.predict(test_df[input_variables])

print(classification_report(test_df[target_variable], y_pred))

# %%
plot.confusion_matrix(test_df[target_variable], y_pred)
plt.show()

# %%
os.makedirs(os.path.dirname(product['model']), exist_ok=True)
with open(product['model'], 'wb') as f:
    pickle.dump(clf, f)

# %%
