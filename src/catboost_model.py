from src.base_model import BaseModel
from catboost import CatBoostClassifier
from typing import Dict, List

class CatboostModel(BaseModel):
    def __init__(self, model_params: Dict, features: List, version: str=""):
        super().__init__(model_params, features, version)
        self.model = CatBoostClassifier(**self.model_params)

    def fit(self, X, y):
        X_ = X[self.features]
        self.model.fit(X_, y)
        return self
    
    def predict_proba(self, X):
        X_ = X[self.features]
        return self.model.predict_proba(X_)[:, 1]


