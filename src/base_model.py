from abc import abstractmethod
from typing import Dict, List
from pickle import load, dump

class BaseModel:
    def __init__(self, model_params: Dict, features: List, version: str = "0.0.1"):
        self.model_params = model_params
        self.features = features
        self.model = None
        self.version = version
    
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
    
    def _dict_to_save(self):
        return {
            "model": self.model
        }

    def save(self, path: str):
        dict_to_save = self._dict_to_save()
        with open(path, "wb") as f:
            dump(dict_to_save, f)
    
    def load(self, path: str):
        with open(path, "rb") as f:
            dict_to_load = load(f)
        for key in dict_to_load:
            setattr(self, key, dict_to_load[key])
        