import logging
import sys

import pandas as pd
import numpy as np
np.random.seed(42)

sys.path.insert(0, "./")

from src.utils import read_params
from src.transformations.text import preprocess_text, extract_text_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

params = read_params()

def read_data(path: str = './data/cyberbullying_tweets.csv') -> pd.DataFrame:
    data = pd.read_csv(path)
    logging.info(f"Data shape: {data.shape}")
    logging.info(f"New target distribution: \n{data[params['target_multi']].value_counts(normalize=True).to_markdown()}")
    return data

def bootstrap(data):
    index_0 = data[data[params['target_binary']] == 0].index
    amount = (data[params['target_binary']] == 1).sum() - (data[params['target_binary']] == 0).sum()
    index_to_add = np.random.choice(index_0, amount)
    new_blood = data.loc[index_to_add].reset_index(drop=True)
    data = pd.concat([data, new_blood]).reset_index(drop=True)
    return data
    

def prepare_target(data: pd.DataFrame) -> pd.DataFrame:
    data[params['target_binary']] = np.where(data[params['target_multi']] == 'not_cyberbullying', 0, 1)
    logging.info(f"New target distribution: \n{data[params['target_binary']].value_counts(normalize=True).to_markdown()}")
    data = bootstrap(data)
    logging.info(f"New target distribution after bootstrap: \n{data[params['target_binary']].value_counts(normalize=True).to_markdown()}")
    
    return data

def prepare_text_features(data: pd.DataFrame) -> pd.DataFrame:
    data[params['text_prepared']] = data[params['text']].apply(preprocess_text)
    
    text_features = data[params['text']].apply(extract_text_features)
    text_features_df = pd.DataFrame(text_features.tolist(), index=data.index)

    data = pd.concat([data, text_features_df], axis=1)
    
    logging.info(f"Added text features: {list(text_features_df.columns)}")
    return data

def save_data(data: pd.DataFrame) -> None:
    data.to_parquet(params["datasets_path"]["all_data"])    
    logging.info(f"Saved successufl. Data shape: {data.shape}")
    

def main():
    data = read_data()
    data = prepare_target(data)
    data = prepare_text_features(data)
    save_data(data)


if __name__ == "__main__":
    main()