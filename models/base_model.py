from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

class BaseTransactionCategorizer(ABC):
    """Base class for all transaction categorization models."""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on the given data.
        
        Args:
            X: DataFrame containing transaction features
            y: Series containing category labels
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict categories and confidence scores for transactions.
        
        Args:
            X: DataFrame containing transaction features
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary containing model metadata (type, parameters, etc.)
        """
        pass 