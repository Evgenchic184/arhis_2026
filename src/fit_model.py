import importlib
import logging
import sys

import pandas as pd
import numpy as np

from sklearn.metrics import average_precision_score
from itertools import product
np.random.seed(42)

sys.path.insert(0, "./")

from src.utils import read_params

params = read_params()

def load_data():
    train = pd.read_parquet(params['datasets_path']['train'])
    val = pd.read_parquet(params['datasets_path']['val'])
    X_train, y_train = train.drop(columns=params['target_binary']), train[params['target_binary']]

    X_val, y_val = val.drop(columns=params['target_binary']), val[params['target_binary']]
    
    return X_train, y_train, X_val, y_val

def main():
    model_config = read_params("configs/model_v0.yaml")
    
    lib_model = model_config['lib_model']
    name_model = model_config['name_model']
    features = model_config['features']
    model_params = model_config['model_params']

    source_model = getattr(importlib.import_module(lib_model), name_model)

    X_train, y_train, X_val, y_val = load_data()

    order_params_name = model_params.keys()
    
    for grid_params in product(*[model_params[i] for i in order_params_name]):
        print(grid_params)
        cur_params = {"text_features": ["text_prepared"], "verbose": False}
        for i, j in zip(order_params_name, grid_params):
            cur_params[i] = j
        print(cur_params)
        model = source_model(model_params=cur_params, features=features)

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)
        print(average_precision_score(y_score=y_proba, y_true=y_val))
        


if __name__ == "__main__":
    main()