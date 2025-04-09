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
from db_connector import get_engine

MODEL_VERSION_FILE = "data/model_version.txt"
MODELS_DIR = "models"

def get_next_model_version():
    """Reads the last model version, increments it, and saves it back."""
    version = 0
    try:
        if os.path.exists(MODEL_VERSION_FILE):
            with open(MODEL_VERSION_FILE, 'r') as f:
                version = int(f.read().strip())
    except (ValueError, FileNotFoundError):
        version = 0 # Start from 0 if file invalid or not found

    next_version = version + 1

    try:
        os.makedirs(os.path.dirname(MODEL_VERSION_FILE), exist_ok=True)
        with open(MODEL_VERSION_FILE, 'w') as f:
            f.write(str(next_version))
    except Exception as e:
        print(f"Warning: Could not write new model version to {MODEL_VERSION_FILE}: {e}")

    return next_version

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
        # Ensure Description exists and handle NaNs
        if 'Description' not in X.columns:
            raise ValueError("Input DataFrame must contain a 'Description' column.")
        self.vectorizer.fit(X['Description'].fillna(''))
        return self
    
    def transform(self, X):
        if 'Description' not in X.columns:
            raise ValueError("Input DataFrame must contain a 'Description' column.")
        return self.vectorizer.transform(X['Description'].fillna(''))

# Custom transformer for numeric features
class NumericFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        # Ensure Amount exists
        if 'Amount' not in X.columns:
            raise ValueError("Input DataFrame must contain an 'Amount' column.")
        amounts = self._extract_amounts(X)
        self.scaler.fit(amounts.reshape(-1, 1))
        return self
    
    def transform(self, X):
        if 'Amount' not in X.columns:
            raise ValueError("Input DataFrame must contain an 'Amount' column.")
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

def train_model(test_size=0.2, random_state=42):
    # --- Load Data From Database --- 
    try:
        engine = get_engine()
        # Read all finalized transactions from the database
        data = pd.read_sql('SELECT * FROM transactions', engine)
        print(f"Loaded {len(data)} records from the database.")
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None, None, None, None

    if data.empty:
        print("No training data found in the database. Please submit some categorized transactions first.")
        return None, None, None, None

    # --- Deduplication --- 
    # Define columns that identify a unique transaction
    # Adjust these if your identification criteria are different
    identifying_columns = ['transaction_date', 'description', 'amount', 'source_file']
    
    # Check if all identifying columns exist
    missing_id_cols = [col for col in identifying_columns if col not in data.columns]
    if missing_id_cols:
        print(f"Warning: Database table is missing columns needed for deduplication: {missing_id_cols}. Skipping deduplication.")
    elif 'timestamp' not in data.columns:
         print(f"Warning: Database table is missing 'timestamp' column needed for deduplication. Skipping deduplication.")
    else:
        print(f"Deduplicating based on: {identifying_columns} keeping the latest entry.")
        # Ensure timestamp is datetime for sorting
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        # Sort by identifying columns and then by timestamp descending
        data = data.sort_values(by=identifying_columns + ['timestamp'], ascending=[True]*len(identifying_columns) + [False])
        # Keep the first (most recent) entry for each unique transaction
        initial_count = len(data)
        data = data.drop_duplicates(subset=identifying_columns, keep='first')
        deduplicated_count = len(data)
        print(f"Deduplication complete. Kept {deduplicated_count} unique transactions (removed {initial_count - deduplicated_count}).")

    # --- Data Preparation --- 
    # Check required columns after loading and deduplication
    required_cols = ['description', 'amount', 'category']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"Error: Training data from database is missing required columns: {missing_cols}")
        print(f"Available columns: {list(data.columns)}")
        return None, None, None, None

    # Basic stats about the dataset
    print(f"Using {len(data)} unique transactions for training.")
    if 'category' in data.columns:
        category_counts = data['category'].value_counts()
        print(f"Category distribution ({len(category_counts)} unique categories):")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")
    else:
        print("Warning: 'category' column not found for distribution stats.")
        return None, None, None, None # Cannot train without category

    # Prepare features
    try:
        feature_prep = FeaturePreparation()
        # Pass standardized columns expected by transformers
        # Ensure the columns exist before renaming
        X_features = data[['description', 'amount']].rename(columns={'description': 'Description', 'amount': 'Amount'})
        X = feature_prep.fit_transform(X_features)
        y = data['category']
    except ValueError as e:
        print(f"Error during feature preparation: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"Unexpected error during feature preparation: {e}")
        return None, None, None, None


    # Split data for validation
    try:
        # Ensure there's enough data to split
        if len(data) < 2 or len(data['category'].unique()) < 2:
             print("Warning: Not enough data or categories to perform train/test split. Training on full dataset.")
             X_train, y_train = X, y
             X_test, y_test = X, y # Evaluate on training data
        else:
             X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
             )
    except ValueError as e:
        # Common issue if a category has only one sample
        print(f"Error during train/test split (potentially due to low sample count per category): {e}")
        print("Attempting training on full dataset...")
        X_train, y_train = X, y # Train on full data if split fails
        X_test, y_test = X, y # Evaluate on training data (not ideal, but better than crashing)

    # --- Train Model --- 
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    # --- Evaluate --- 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    # Only print classification report if test set is valid (different from train set)
    if X_test.shape[0] > 0 and (not hasattr(X_train, 'indices') or not np.array_equal(X_train.indices, X_test.indices) or not np.array_equal(y_train.values, y_test.values)):
         print("Classification report:")
         # Ensure target_names matches the unique classes in y_test for the report
         report_classes = sorted(y_test.unique().tolist())
         print(classification_report(y_test, y_pred, labels=report_classes, zero_division=0))
    else:
         print("Classification report skipped (evaluated on training data or test set empty).")


    # --- Model Saving with Versioning ---
    base_name = "random_forest"
    v_num = get_next_model_version()
    model_version = f"{base_name}_v{v_num}"
    model_filename = os.path.join(MODELS_DIR, f"{model_version}.pkl")

    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save model and feature preparation pipeline together
    model_data = {
        'model': model,
        'feature_prep': feature_prep,
        'categories': list(model.classes_),
        'model_version': model_version # Store version inside the file too
    }

    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_filename}")
    except Exception as e:
        print(f"Error saving model file {model_filename}: {e}")
        return None, None, None, None # Failed to save

    # Return model, feature_prep, version string, and filename
    return model, feature_prep, model_version, model_filename

# Function to find the latest model file
def find_latest_model_file(models_dir=MODELS_DIR, base_name="random_forest_v"):
    """Finds the model file with the highest version number."""
    model_files = glob.glob(os.path.join(models_dir, f"{base_name}*.pkl"))
    if not model_files:
        return None
    
    # Extract version numbers and find the max
    latest_file = max(model_files, key=lambda f: int(re.search(r'_v(\d+)\.pkl$', f).group(1)))
    return latest_file

def categorize_transactions(input_dir='data/to_categorize', output_dir='data/output', use_postprocessor=True):
    # --- Load Latest Model --- 
    latest_model_file = find_latest_model_file()
    if not latest_model_file:
         print(f"No trained model files found in {MODELS_DIR}. Please train the model first.")
         return {}, None, None # Return empty dict and None for versions
    
    print(f"Loading latest model: {latest_model_file}")
    try:
        with open(latest_model_file, 'rb') as f:
            model_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading model file {latest_model_file}: {e}")
        return {}, None, None

    model = model_data['model']
    feature_prep = model_data['feature_prep']
    categories = model_data['categories']
    # Get model version info from the loaded file, default if not found
    model_version = model_data.get('model_version', 'unknown_version') 
    # Use the actual file path used for loading as model_filename
    model_filename = latest_model_file 


    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find files to categorize
    files = glob.glob(os.path.join(input_dir, '*.csv'))

    if not files:
        print(f"No CSV files found in {input_dir}")
        return {}, model_version, model_filename # Return empty results but valid model info

    print(f"Found {len(files)} file(s) to categorize using model: {model_filename}")

    # Initialize merchant post-processor if enabled
    postprocessor = None
    if use_postprocessor:
        try:
            postprocessor = MerchantPostProcessor() # Assumes default merchant file path
        except Exception as e:
            print(f"Error initializing merchant post-processor: {e}")
            print("Continuing without post-processing.")

    results = {} # Store results per file: { filename: dataframe }

    # Process each file
    for file in files:
        input_filename = os.path.basename(file)
        print(f"Processing: {input_filename}")

        # Load file
        try:
            df = pd.read_csv(file)
            # Standardize column names early for consistency
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            # Store original filename for later DB storage
            df['source_file'] = input_filename 
        except Exception as e:
            print(f"  Error loading {input_filename}: {e}. Skipping.")
            continue

        # Prepare features using standardized columns
        try:
            # Ensure required columns exist before transformation
            required_cols_for_pred = ['description', 'amount']
            missing_cols = [col for col in required_cols_for_pred if col not in df.columns]
            if missing_cols:
                print(f"  Skipping {input_filename}: Missing required columns for prediction: {missing_cols}")
                continue

            # Use the correct internal column names for rename
            X_features = df[required_cols_for_pred].rename(columns={'description': 'Description', 'amount': 'Amount'})
            X = feature_prep.transform(X_features)
        except ValueError as e:
             print(f"  Error preparing features for {input_filename}: {e}. Skipping.")
             continue
        except Exception as e:
             print(f"  Unexpected error preparing features for {input_filename}: {e}. Skipping.")
             continue

        # Predict categories
        df['category'] = model.predict(X)

        # Add prediction probabilities
        probabilities = model.predict_proba(X)

        # Get confidence for the predicted class
        confidences = []
        predicted_categories = df['category'].tolist()
        category_to_index = {cat: i for i, cat in enumerate(categories)}

        for i, prob_row in enumerate(probabilities):
             predicted_cat = predicted_categories[i]
             if predicted_cat in category_to_index:
                 predicted_class_idx = category_to_index[predicted_cat]
                 confidences.append(prob_row[predicted_class_idx])
             else:
                 # Handle case where predicted category wasn't in the training classes
                 confidences.append(0.0)


        df['confidence'] = confidences

        # Apply merchant post-processing if enabled
        if postprocessor:
            print("Applying merchant post-processing...")
            # Ensure post-processor expects/returns lowercase columns
            try:
                 df = postprocessor.process_transactions(df) 
            except Exception as e:
                 print(f"Error during merchant post-processing: {e}. Skipping post-processing for this file.")

        # --- Save categorized data to output CSV for review ---
        # Map internal lowercase_underscore back to original-like Title Case for CSV output
        column_rename_map_output = {
             'transaction_date': 'Date', 
             'description': 'Description',
             'amount': 'Amount',
             'extended_details': 'Extended Details',
             'statement_description': 'Statement Description', 
             'appears_on_your_statement_as': 'Appears On Your Statement As',
             'category': 'Category',
             'confidence': 'Confidence',
             # source_file is internal, not needed in output CSV
        }
        # Create a copy for outputting, select and rename columns
        df_output = df.copy()
        # Ensure only columns present in df are renamed
        cols_to_rename = {k: v for k, v in column_rename_map_output.items() if k in df_output.columns}
        df_output = df_output.rename(columns=cols_to_rename)
        # Define the order/subset of columns for the output CSV
        output_columns_order = ['Date', 'Description', 'Amount', 'Extended Details', 'Appears On Your Statement As', 'Category', 'Confidence']
        # Filter df_output to only include columns that exist from the desired order
        final_output_cols = [col for col in output_columns_order if col in df_output.columns]
        df_output = df_output[final_output_cols]

        output_file_path = os.path.join(output_dir, f"improved_categorized_{input_filename}")
        try:
            df_output.to_csv(output_file_path, index=False)
            print(f"  Saved initial categorization for review to: {output_file_path}")
        except Exception as e:
            print(f"  Error saving output CSV {output_file_path}: {e}")


        # Add the processed dataframe (with internal names) to results dictionary
        results[input_filename] = df 

        # Generate statistics
        print(f"  Categorized {len(df)} transactions")
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            print(f"  Category distribution ({len(category_counts)} categories):")
            for category, count in category_counts.items():
                print(f"    {category}: {count}")

        if 'confidence' in df.columns:
             mean_confidence = df['confidence'].mean()
             low_confidence_count = (df['confidence'] < 0.7).sum()
             print(f"  Mean confidence: {mean_confidence:.3f}")
             print(f"  Transactions with confidence < 0.7: {low_confidence_count}")


    # Return the results dictionary and model identifiers
    return results, model_version, model_filename

if __name__ == "__main__":
    # Train model
    model, feature_prep, model_version, model_filename = train_model()
    
    # Categorize transactions
    if model is not None:
        results, model_version, model_filename = categorize_transactions(use_postprocessor=True) 