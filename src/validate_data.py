import pandas as pd
import yaml
import json
import logging

logging.basicConfig(level=logging.INFO)

def load_contract(path: str = 'data_contract.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def validate_training_data(df: pd.DataFrame, contract: dict) -> dict:
    """Проверка обучающих данных против контракта"""
    report = {'passed': True, 'errors': [], 'warnings': []}

    required_cols = ['tweet_text', 'text_prepared', 'cyberbullying_bin',
                     'is_new_user', 'reputation_score', 'reports_last_24h']
    for col in required_cols:
        if col not in df.columns:
            report['errors'].append(f"Missing column: {col}")
            report['passed'] = False

    if len(df) < contract['training_data']['quality_requirements']['min_rows']:
        report['errors'].append(f"Too few rows: {len(df)}")
        report['passed'] = False

    target_ratio = df['cyberbullying_bin'].mean()
    min_ratio = contract['training_data']['quality_requirements']['target_distribution']['min_ratio_class_0']
    max_ratio = contract['training_data']['quality_requirements']['target_distribution']['max_ratio_class_0']
    
    if not (min_ratio <= target_ratio <= max_ratio):
        report['errors'].append(f"Target distribution out of range: {target_ratio:.2f}")
        report['passed'] = False

    if df['reputation_score'].min() < 0 or df['reputation_score'].max() > 1:
        report['errors'].append("reputation_score out of range [0, 1]")
        report['passed'] = False
    
    return report


if __name__ == "__main__":
    contract = load_contract()
    df = pd.read_parquet('data/with_user_features.parquet')
    report = validate_training_data(df, contract)
    
    with open('reports/validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    if not report['passed']:
        raise ValueError(f"Validation failed: {report['errors']}")
    
    logging.info("Validation passed.")