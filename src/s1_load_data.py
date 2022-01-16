# Resources:
# * Japan Real Estate Prices Dataset, Kaggle, nishio-dens, 2020. (https://www.kaggle.com/nishiodens/japan-real-estate-transaction-prices/metadata)
# * Japan Houses Price Prediction notebook, Kaggle. (https://www.kaggle.com/arkacze/japan-houses-price-prediction)

import pandas as pd
import os
import tqdm

# + tags=["parameters"]
upstream = None
product = {
    'nb': '../src/pipeline_notebooks/s1.ipynb',
    'data': '../data/raw_data.csv'
}
# -

path = '../data/training_raw/'
training_filenames = [filename for filename in os.listdir(path)  if filename.endswith('.csv')]

training_dfs = []
for filename in tqdm.tqdm(training_filenames):
    training_dfs.append(pd.read_csv(path+filename, low_memory=False))

df = pd.concat(training_dfs)

df.head(1)

df.to_csv(str(product['data']), index=False)

df.shape


