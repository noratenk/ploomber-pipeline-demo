import pandas as pd
from sklearn import datasets

# + tags=["parameters"]
upstream = None
product = {
    'nb': 'notebooks/s1_load_data.ipynb',
    'data': '../data/raw/raw_data.csv'
}
# -

d = datasets.load_iris()

df = pd.DataFrame(d['data'])
df.columns = d['feature_names']
df['target'] = d['target']

df.head(2)

df.to_csv(str(product['data']), index=False)


