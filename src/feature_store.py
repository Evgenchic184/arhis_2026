from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd


class FeatureStore(ABC):
    """Абстракция Feature Store для offline/online"""
    
    @abstractmethod
    def get_user_features(self, user_id: Optional[Union[str, int]]) -> Dict:
        """Получить признаки пользователя"""
        pass
    
    @abstractmethod
    def write_user_features(self, user_id: Union[str, int], features: Dict):
        """Записать признаки пользователя"""
        pass


class OfflineFeatureStore(FeatureStore):
    """
    Offline Store для обучения.
    Генерирует симулированные user-фичи с консистентностью по user_id.
    """
    
    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._cache: Dict[Union[str, int], Dict] = {}
    
    def get_user_features(self, user_id: Optional[Union[str, int]]) -> Dict:
        if user_id is None:
            return self._cold_start_features()
        
        user_key = str(user_id)
        
        if user_key not in self._cache:
            self._cache[user_key] = self._generate_user_profile(user_key)
        
        return self._cache[user_key]
    
    def write_user_features(self, user_id: Union[str, int], features: Dict):
        pass
    
    def _cold_start_features(self) -> Dict:
        """Дефолтные значения для новых пользователей"""
        return {
            'is_new_user': 1,
            'reputation_score': 0.5,
            'reports_last_24h': 0,
            'account_age_days': 0,
        }
    
    def _generate_user_profile(self, user_key: str) -> Dict:
        account_age = int(self._rng.exponential(scale=365))
        is_new = int(account_age < 7)
        
        base_rep = self._rng.beta(2, 2)
        reputation = base_rep * (0.7 if is_new else 1.0)

        reports_lambda = max(0, 3 - 2 * reputation)
        reports = int(self._rng.poisson(reports_lambda))
        
        return {
            'is_new_user': is_new,
            'reputation_score': float(np.clip(reputation + self._rng.normal(0, 0.1), 0, 1)),
            'reports_last_24h': reports,
            'account_age_days': account_age,
        }
    
    def generate_user_ids(self, n_rows: int, n_users: int) -> np.ndarray:
        return self._rng.integers(0, n_users, size=n_rows)


class OnlineFeatureStore(FeatureStore):
    """
    Online Store для инференса.
    """
    def __init__(self):
        self._store: Dict[str, Dict] = {}
    
    def get_user_features(self, user_id: Optional[Union[str, int]]) -> Dict:
        if user_id is None:
            return self._cold_start_features()
        
        user_key = str(user_id)
        if user_key not in self._store:
            return self._cold_start_features()
        
        return self._store[user_key]
    
    def write_user_features(self, user_id: Union[str, int], features: Dict):
        user_key = str(user_id)
        self._store[user_key] = features
    
    def _cold_start_features(self) -> Dict:
        return {
            'is_new_user': 1,
            'reputation_score': 0.5,
            'reports_last_24h': 0,
            'account_age_days': 0,
        }