import importlib
import logging
import sys
import mlflow
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    
    model_config = read_params(args.config_path)
    
    lib_model = model_config['lib_model']
    name_model = model_config['name_model']
    features = model_config['features']
    model_params = model_config['model_params']

    source_model = getattr(importlib.import_module(lib_model), name_model)

    X_train, y_train, X_val, y_val = load_data()

    order_params_name = model_params.keys()
    mlflow.set_experiment(params['mlflow_exp_name'])
    for grid_params in product(*[model_params[i] for i in order_params_name]):
        with mlflow.start_run() as run:
            mlflow.set_tag("model_name", name_model)
            cur_params = {}
            for i, j in zip(order_params_name, grid_params):
                cur_params[i] = j
            print(cur_params)
            mlflow.log_params(cur_params)
            model = source_model(model_params=cur_params, features=features)
    
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)
            val_metric = average_precision_score(y_score=y_proba, y_true=y_val)
            train_metric = average_precision_score(y_score=model.predict_proba(X_train), y_true=y_train)
            
            mlflow.log_metric("pr_auc-val", val_metric)
            mlflow.log_metric("pr_auc-train", train_metric)
            mlflow.log_metric("overfitting", abs(train_metric - val_metric))


            val_metric = roc_auc_score(y_score=y_proba, y_true=y_val)
            train_metric = roc_auc_score(y_score=model.predict_proba(X_train), y_true=y_train)
            
            mlflow.log_metric("roc_auc-val", val_metric)
            mlflow.log_metric("roc_auc-train", train_metric)
            model.save(f"{name_model}")
            
            prec, rec, thr = precision_recall_curve(y_score=y_proba, y_true=y_val)
            fig = plt.figure(figsize=(6, 4))
            plt.plot(rec, prec)
            mlflow.log_figure(fig, "pr_auc-val.png")
        


if __name__ == "__main__":
    main()