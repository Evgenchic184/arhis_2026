import logging
import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

sys.path.insert(0, "./")

from src.utils import read_params

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

params = read_params()

def info_data(data, name):
    logging.info(f"{name} shape: {data.shape}, target distribution: \n{data[params['target_multi']].value_counts(normalize=True).to_markdown()}")

def split_data(data: pd.DataFrame):
    data_train, data_other = train_test_split(data, train_size=0.6, stratify=data[params['target_multi']])
    info_data(data_train, "train")
    data_val, data_test = train_test_split(data_other, train_size=0.5, stratify=data_other[params['target_multi']])
    info_data(data_val, "val")
    info_data(data_test, "test")
    return data_train, data_val, data_test

def main():
    data = pd.read_parquet(params["datasets_path"]["all_data"])
    data_train, data_val, data_test = split_data(data)

    data_train.to_parquet(params["datasets_path"]["train"])
    data_val.to_parquet(params["datasets_path"]["val"])
    data_test.to_parquet(params["datasets_path"]["test"])
    
    


if __name__ == "__main__":
    main()