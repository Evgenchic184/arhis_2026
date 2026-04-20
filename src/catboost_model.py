from src.base_model import BaseModel
from catboost import CatBoostClassifier


class CatboostModel(BaseModel):
    def __init__(self, model_params: Dict, features: List, version: str):
        super().__init__(model_params, features, version)
        self.model = CatBoostClassifier(self.model_params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


