import logging
import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from minio import Minio
from minio.error import S3Error

sys.path.insert(0, "./")

from src.utils import read_params

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

params = read_params()

MINIO_CONFIG = {
    "endpoint": os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    "access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    "secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    "bucket_name": os.getenv("MINIO_BUCKET_NAME", "data-splits"),
    "secure": os.getenv("MINIO_SECURE", "false").lower() == "true"
}

def info_data(data, name):
    logging.info(f"{name} shape: {data.shape}, target distribution: \n{data[params['target_multi']].value_counts(normalize=True).to_markdown()}")

def get_minio_client():
    try:
        client = Minio(
            MINIO_CONFIG["endpoint"],
            access_key=MINIO_CONFIG["access_key"],
            secret_key=MINIO_CONFIG["secret_key"],
            secure=MINIO_CONFIG["secure"]
        )
        logging.info(f"MinIO client initialized: {MINIO_CONFIG['endpoint']}")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize MinIO client: {e}")
        return None

def ensure_bucket_exists(client, bucket_name):
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logging.info(f"Created bucket: {bucket_name}")
        else:
            logging.info(f"Bucket already exists: {bucket_name}")
    except S3Error as e:
        logging.error(f"Error ensuring bucket exists: {e}")
        raise

def upload_file_to_minio(client, file_path, object_name, bucket_name=None):
    if bucket_name is None:
        bucket_name = MINIO_CONFIG["bucket_name"]
    
    try:
        ensure_bucket_exists(client, bucket_name)

        client.fput_object(
            bucket_name,
            object_name,
            file_path
        )
        logging.info(f"Uploaded '{file_path}' to '{bucket_name}/{object_name}'")
        return True
    except S3Error as e:
        logging.error(f"Failed to upload file to MinIO: {e}")
        return False

def split_data(data: pd.DataFrame):
    data_train, data_other = train_test_split(data, train_size=0.6, stratify=data[params['target_multi']])
    info_data(data_train, "train")
    data_val, data_test = train_test_split(data_other, train_size=0.5, stratify=data_other[params['target_multi']])
    info_data(data_val, "val")
    info_data(data_test, "test")
    return data_train, data_val, data_test

def main():
    data = pd.read_parquet(params["datasets_path"]["with_user_features"])
    data_train, data_val, data_test = split_data(data)

    data_train.to_parquet(params["datasets_path"]["train"])
    data_val.to_parquet(params["datasets_path"]["val"])
    data_test.to_parquet(params["datasets_path"]["test"])

    logging.info("Starting MinIO uploads...")
    client = get_minio_client()

    if client:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        upload_file_to_minio(
            client,
            params["datasets_path"]["train"],
            f"{timestamp}_train.parquet"
        )
        upload_file_to_minio(
            client,
            params["datasets_path"]["val"],
            f"{timestamp}_val.parquet"
        )
        upload_file_to_minio(
            client,
            params["datasets_path"]["test"],
            f"{timestamp}_test.parquet"
        )
        logging.info(f"All files uploaded to MinIO successfully with timestamp: {timestamp}")
    else:
        logging.warning("Skipping MinIO upload due to client initialization error")

    


if __name__ == "__main__":
    main()