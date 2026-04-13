import pandas as pd
import logging
import sys

sys.path.insert(0, "./")

from src.feature_store import OfflineFeatureStore


logging.basicConfig(level=logging.INFO)

def add_user_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:

    fs = OfflineFeatureStore(seed=seed)
    
    n_users = len(df) // 3
    df['user_id'] = fs.generate_user_ids(len(df), n_users)
    
    user_features = df['user_id'].apply(lambda uid: fs.get_user_features(uid))
    user_features_df = pd.DataFrame(user_features.tolist(), index=df.index)
    
    df = pd.concat([df, user_features_df], axis=1)
    
    logging.info(f"Added user features: {list(user_features_df.columns)}")
    logging.info(f"New users: {df['is_new_user'].sum()} / {len(df)} ({df['is_new_user'].mean():.1%})")
    logging.info(f"Reputation mean: {df['reputation_score'].mean():.3f} ± {df['reputation_score'].std():.3f}")
    
    return df


if __name__ == "__main__":
    df = pd.read_parquet('data/all_data.parquet')
    df = add_user_features(df)
    df.to_parquet('data/with_user_features.parquet')