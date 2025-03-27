import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, Dict, Any
from .base_model import BaseTransactionCategorizer

class RandomForestCategorizer(BaseTransactionCategorizer):
    """Random Forest implementation of the transaction categorizer."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """Prepare features from transaction data.
        
        Args:
            df: DataFrame containing transaction data
            training: Whether this is for training (fit_transform) or prediction (transform)
        """
        # Combine text features (description and extended details)
        text_features = df.apply(
            lambda row: f"{row['Description']} {row['Extended Details']}", 
            axis=1
        )
        
        # Transform text features
        if training:
            text_features = self.tfidf.fit_transform(text_features)
        else:
            text_features = self.tfidf.transform(text_features)
        
        # Prepare numeric features
        numeric_features = df[['Amount']].copy()
        numeric_features['Amount'] = numeric_features['Amount'].abs()  # Use absolute value
        
        # Transform numeric features
        if training:
            numeric_features = self.scaler.fit_transform(numeric_features)
        else:
            numeric_features = self.scaler.transform(numeric_features)
        
        # Combine features
        return np.hstack([text_features.toarray(), numeric_features])
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on the given data."""
        X_features = self.prepare_features(X, training=True)
        self.model.fit(X_features, y)
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict categories and confidence scores."""
        X_features = self.prepare_features(X, training=False)
        predictions = self.model.predict(X_features)
        probabilities = self.model.predict_proba(X_features)
        confidence_scores = np.max(probabilities, axis=1)
        return predictions, confidence_scores
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        model_data = {
            'model': self.model,
            'tfidf': self.tfidf,
            'scaler': self.scaler
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.tfidf = model_data['tfidf']
        self.scaler = model_data['scaler']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_type': 'RandomForest',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'feature_importance': dict(zip(
                self.tfidf.get_feature_names_out(),
                self.model.feature_importances_
            ))
        } 