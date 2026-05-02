import re
import pandas as pd
from typing import Dict, List
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from src.base_model import BaseModel

class LRModel(BaseModel):
    def __init__(self, model_params: Dict, features: List, version: str = ""):
        super().__init__(model_params, features, version)
        # Для текстовых данных оптимален MultinomialNB (или ComplementNB для несбалансированных данных)
        self.model = LogisticRegression(**self.model_params)
        
        # Векторизатор текстов
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',  # ⚠️ замените на 'russian' для русскоязычных текстов
            token_pattern=r'(?u)\b\w\w+\b',
            min_df=2,
            max_df=0.95,
            max_features=100
        )
        self.field_to_save.append('vectorizer')
        self._is_fitted = False

    def _clean_text(self, text: str) -> str:
        """Базовая очистка текста: нижний регистр, удаление пунктуации, цифр и лишних пробелов."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)   # Удаляем знаки препинания
        text = re.sub(r'\d+', '', text)       # Удаляем цифры
        text = re.sub(r'\s+', ' ', text).strip() # Нормализуем пробелы
        return text

    def _preprocess(self, X: pd.DataFrame) -> pd.Series:
        """Извлекает текстовые колонки, объединяет их (если несколько) и применяет очистку."""
        cols = self.features if isinstance(self.features, list) else [self.features]
        text_series = X[cols[0]].astype(str)
        for col in cols[1:]:
            text_series = text_series + " " + X[col].astype(str)
        return text_series.apply(self._clean_text)

    def fit(self, X, y):
        X_text = self._preprocess(X)
        X_vec = self.vectorizer.fit_transform(X_text)
        self.model.fit(X_vec, y)
        self._is_fitted = True
        return self
    
    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        X_text = self._preprocess(X)
        X_vec = self.vectorizer.transform(X_text)
        return self.model.predict_proba(X_vec)[:, 1]