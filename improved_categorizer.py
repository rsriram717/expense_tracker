import os
import pandas as pd
import numpy as np
import re
import pickle
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from merchant_postprocessor import MerchantPostProcessor

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase and remove special characters
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text

# Custom transformer for text features
class TextFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            preprocessor=preprocess_text,
            ngram_range=(1, 2),
            max_features=500
        )
        
    def fit(self, X, y=None):
        self.vectorizer.fit(X['Description'].fillna(''))
        return self
    
    def transform(self, X):
        return self.vectorizer.transform(X['Description'].fillna(''))

# Custom transformer for numeric features
class NumericFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        # Extract amount as numeric feature
        amounts = self._extract_amounts(X)
        self.scaler.fit(amounts.reshape(-1, 1))
        return self
    
    def transform(self, X):
        amounts = self._extract_amounts(X)
        return self.scaler.transform(amounts.reshape(-1, 1))
    
    def _extract_amounts(self, X):
        # Convert to numeric and handle any errors
        return np.array(pd.to_numeric(X['Amount'].fillna(0), errors='coerce').fillna(0))

# Class to prepare all features in one step
class FeaturePreparation:
    def __init__(self):
        self.pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text', TextFeatures()),
                ('numeric', NumericFeatures())
            ]))
        ])
        
    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)
    
    def transform(self, X):
        return self.pipeline.transform(X)
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.pipeline, f)
    
    @staticmethod
    def load(filename):
        prep = FeaturePreparation()
        with open(filename, 'rb') as f:
            prep.pipeline = pickle.load(f)
        return prep

def train_model(data_dir='data/categorized', test_size=0.2, random_state=42, update_merchants=False):
    """
    Train a model on categorized transaction data.
    
    Args:
        data_dir: Directory containing categorized transaction data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        update_merchants: Whether to prompt for merchant category updates
    
    Returns:
        Tuple of (model, feature_preparation_pipeline)
    """
    # Find all CSV files in the data directory
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return None, None
    
    print(f"Loading categorized data from {data_dir}")
    
    # Load and combine all CSV files
    dataframes = []
    for file in csv_files:
        print(f"  Found: {os.path.basename(file)}")
        df = pd.read_csv(file)
        dataframes.append(df)
    
    # Combine all dataframes
    data = pd.concat(dataframes, ignore_index=True)
    
    # Check required columns
    required_cols = ['Description', 'Amount', 'Category']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None, None
    
    # Basic stats about the dataset
    print(f"Created dataset with {len(data)} transactions")
    category_counts = data['Category'].value_counts()
    print(f"Category distribution ({len(category_counts)} unique categories):")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    # Prepare features
    feature_prep = FeaturePreparation()
    X = feature_prep.fit_transform(data)
    y = data['Category']
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and feature preparation pipeline
    model_data = {
        'model': model,
        'feature_prep': feature_prep,
        'categories': list(model.classes_)
    }
    
    with open('improved_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved to improved_model.pkl")
    
    # Check if user wants to update merchant categories
    if update_merchants:
        try:
            update_merchant_categories(data)
        except Exception as e:
            print(f"Error updating merchant categories: {e}")
    
    return model, feature_prep

def update_merchant_categories(data):
    """
    Prompt user to update merchant categories based on model predictions
    """
    print("\nChecking for potential merchant category updates...")
    
    try:
        # Import the module here to avoid circular imports
        # and to make it optional for environments without user interaction
        import update_merchants
        
        # Ask if user wants to update merchant categories
        response = input("\nWould you like to update merchant categories based on training data? (y/n): ").strip().lower()
        
        if response == 'y':
            print("Starting merchant category update process...")
            update_merchants.interactive_update()
        else:
            print("Merchant category update skipped.")
            
            # Ask if user wants to run non-interactive update
            response = input("Would you like to generate a merchant suggestion report without applying changes? (y/n): ").strip().lower()
            if response == 'y':
                output_file = "merchant_suggestions.csv"
                suggestions = update_merchants.extract_merchant_suggestions()
                if suggestions is not None and len(suggestions) > 0:
                    suggestions.to_csv(output_file, index=False)
                    print(f"Saved {len(suggestions)} merchant suggestions to {output_file}")
                    print(f"You can review this file and apply changes manually later.")
                    print(f"To apply these changes later, use: python update_merchants.py")
    
    except ImportError:
        print("Update merchants module not available. Skipping merchant updates.")
    except Exception as e:
        print(f"Error during merchant update process: {e}")

def categorize_transactions(input_dir='data/to_categorize', output_dir='data/output', model_file='improved_model.pkl', use_postprocessor=True):
    # Check if model exists
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found. Please train the model first.")
        return
    
    # Load model
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    feature_prep = model_data['feature_prep']
    categories = model_data['categories']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find files to categorize
    files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(files)} file(s) to categorize")
    
    # Initialize merchant post-processor if enabled
    postprocessor = None
    if use_postprocessor:
        try:
            postprocessor = MerchantPostProcessor()
        except Exception as e:
            print(f"Error initializing merchant post-processor: {e}")
            print("Continuing without post-processing.")
    
    # Process each file
    for file in files:
        filename = os.path.basename(file)
        print(f"Processing: {filename}")
        
        # Load file
        df = pd.read_csv(file)
        
        # Check if already categorized
        if 'Category' in df.columns and 'Confidence' in df.columns:
            print(f"  File already contains categories, will update them.")
        
        # Prepare features
        X = feature_prep.transform(df)
        
        # Predict categories
        df['Category'] = model.predict(X)
        
        # Add prediction probabilities
        probabilities = model.predict_proba(X)
        
        # For each row, get the probability of the predicted class
        confidences = []
        for i, row in enumerate(probabilities):
            predicted_class_idx = list(categories).index(df['Category'].iloc[i])
            confidences.append(row[predicted_class_idx])
        
        df['Confidence'] = confidences
        
        # Apply merchant post-processing if enabled
        if postprocessor:
            print("Applying merchant post-processing...")
            df = postprocessor.process_transactions(df)
        
        # Save categorized data
        output_file = os.path.join(output_dir, f"improved_categorized_{filename}")
        df.to_csv(output_file, index=False)
        print(f"  Saved to: {output_file}")
        
        # Generate statistics
        print(f"  Categorized {len(df)} transactions")
        
        category_counts = df['Category'].value_counts()
        print(f"  Category distribution ({len(category_counts)} categories):")
        for category, count in category_counts.items():
            print(f"    {category}: {count}")
        
        # Confidence stats
        mean_confidence = df['Confidence'].mean()
        low_confidence = df[df['Confidence'] < 0.7]
        
        print(f"  Confidence statistics:")
        print(f"    Average confidence: {mean_confidence:.2f}")
        print(f"    Low confidence predictions: {len(low_confidence)} ({len(low_confidence)/len(df)*100:.1f}%)")

if __name__ == "__main__":
    # Train model
    model, feature_prep = train_model(update_merchants=True)
    
    # Categorize transactions
    if model is not None:
        categorize_transactions(use_postprocessor=True) 