import pickle
import json
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from utils.dataloader import DataLoader
from settings. constants import VAL_CSV


with open('settings/specifications.json') as f:
    specifications = json.load(f)

x_columns = specifications['description']['X']
y_column = specifications['description']['y']

raw_val = pd.read_csv(VAL_CSV)
x_raw = raw_val[x_columns]

loader = DataLoader()
loader.fit(x_raw)
X = loader.load_data()
y = raw_val.SalePrice
loaded_model = pickle.load(open('models/GradientBoostingRegressor.pickle', 'rb'))
result = loaded_model.score(X, y)

print(result)
