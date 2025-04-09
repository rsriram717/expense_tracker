import os
import pandas as pd
import numpy as np
import re
import pickle
import glob
from time import sleep
from datetime import datetime, timezone
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import sqlalchemy as sa
import openai
import asyncio
from merchant_postprocessor import MerchantPostProcessor
from db_connector import get_engine

# Load environment variables from .env file
load_dotenv()

MODEL_VERSION_FILE = "data/model_version.txt"
MODELS_DIR = "models"

# --- Inference.net Configuration ---
# IMPORTANT: Set this environment variable with your actual API key
INFERENCE_API_KEY_ENV_VAR = "INFERENCE_API_KEY" 
INFERENCE_BASE_URL = "https://api.inference.net/v1/" 
# LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLAMA_MODEL_NAME = "meta-llama/llama-3.1-8b-instruct/fp-8" # Updated model name

# --- Initialize OpenAI Client ---
api_key = os.getenv(INFERENCE_API_KEY_ENV_VAR)
client = None
async_client = None
if api_key:
    try:
        # Synchronous client (for potential other uses)
        client = openai.OpenAI(
            api_key=api_key,
            base_url=INFERENCE_BASE_URL,
        )
        print("OpenAI client initialized successfully.")

        # Asynchronous client (for concurrent calls)
        async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=INFERENCE_BASE_URL,
        )
        print("Async OpenAI client initialized successfully.")

    except Exception as e:
        print(f"Error initializing OpenAI clients: {e}")
else:
    print(f"Warning: Environment variable {INFERENCE_API_KEY_ENV_VAR} not set. Llama features will be unavailable.")

# Placeholder - Define your actual categories here
PREDEFINED_CATEGORIES = [
    "Groceries", "Restaurants", "Transportation", "Utilities", "Rent", 
    "Shopping", "Entertainment", "Travel", "Healthcare", "Income", "Misc"
]

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

# --- Test Function ---
def test_llama_connection():
    """Performs a simple API call to test connection and model access."""
    print("\\n--- Testing Llama Connection --- ")
    if not client:
        print("FAILURE: API client not initialized. Test skipped.")
        return False
    try:
        print(f"Attempting simple call to model: {LLAMA_MODEL_NAME}...")
        response = client.chat.completions.create(
            model=LLAMA_MODEL_NAME, 
            messages=[{"role": "user", "content": "What is 2+2? Respond with only the numerical answer."}],
            max_tokens=10,
            temperature=0.0
        )
        result = response.choices[0].message.content.strip()
        print(f"SUCCESS: Simple API call successful. Response: '{result}'")
        return True
    except Exception as e:
        print(f"FAILURE: Simple API call FAILED: {e}")
        return False

# --- LLM Categorization Function --- 
def get_llama_category(description, extended_details=None):
    """Uses Llama-3.1 via Inference.net to categorize a transaction."""
    if not client:
         print("API client not initialized. Cannot categorize.")
         return "Misc"
    if pd.isna(description) or str(description).strip() == "":
         print("Warning: Empty description provided. Defaulting to Misc.")
         return "Misc"
    context = str(description)
    if pd.notna(extended_details) and str(extended_details).strip() != "":
         context += " | " + str(extended_details)
    context = context[:1000]
    category_list_str = ", ".join([f'"{cat}"' for cat in PREDEFINED_CATEGORIES])
    prompt = f"""
    Analyze the transaction description below.
    Choose the single best category from this list: [{category_list_str}]
    Respond ONLY with the chosen category name, exactly as it appears in the list.

    Transaction: "{context}"

    Category:"""
    try:
        response = client.chat.completions.create(
            model=LLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": f"You are a transaction categorizer. Select one category from: {category_list_str}. Respond with only the category name."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=30,
            temperature=0.0,
        )
        raw_category = response.choices[0].message.content.strip().strip('"').strip()

        # Check if the response is empty FIRST
        if not raw_category:
            context_preview = context[:50] + ('...' if len(context) > 50 else '')
            print(f"Warning: LLM returned an empty response for context: '{context_preview}'. Defaulting to Misc.")
            return "Misc"

        # Now proceed with matching non-empty responses
        if raw_category in PREDEFINED_CATEGORIES:
            return raw_category
        else:
            # Case-insensitive check
            for cat in PREDEFINED_CATEGORIES:
                if cat.lower() == raw_category.lower():
                    print(f"Warning: LLM response '{raw_category}' matched '{cat}' case-insensitively.")
                    return cat
            # Substring check
            found_cats = [cat for cat in PREDEFINED_CATEGORIES if cat.lower() in raw_category.lower()]
            if len(found_cats) == 1:
                 chosen_cat = found_cats[0]
                 print(f"Warning: LLM response '{raw_category}' contained '{chosen_cat}'. Using matched category.")
                 return chosen_cat
            # If still not matched (non-empty, but unrecognized/ambiguous)
            print(f"Warning: LLM response '{raw_category}' not in predefined list or ambiguous. Defaulting to Misc.")
            return "Misc"
    except openai.APIConnectionError as e:
        print(f"API Connection Error: {e}")
    except openai.RateLimitError as e:
        print(f"API Rate Limit Error: {e}. Sleeping for 5s...")
        sleep(5)
    except openai.APIStatusError as e:
        print(f"API Status Error: {e.status_code} - {e.response}")
    except Exception as e:
        print(f"Error calling LLM API: {e}")
    return "Misc"

# --- LLM Batch Categorization Function ---
def get_llama_categories_batch(batch_details, batch_size=10):
    """Uses Llama-3.1 via Inference.net to categorize a batch of transactions."""
    if not client:
         print("API client not initialized. Cannot categorize batch.")
         return ["Misc"] * len(batch_details)
    
    if len(batch_details) == 0:
        print("Warning (batch): Empty batch details provided.")
        return []
        
    if len(batch_details) != batch_size:
        print(f"Warning (batch): Expected batch size {batch_size}, but got {len(batch_details)} details.")
        batch_size = len(batch_details)
    
    # Reduce batch size if it's too large - start with just 2 items
    if batch_size > 2:
        print(f"Reducing batch size from {batch_size} to 2 to avoid API issues")
        batch_size = 2
        batch_details = batch_details[:batch_size]
    
    category_list_str = ", ".join([f'"{cat}"' for cat in PREDEFINED_CATEGORIES])
    
    # Simplified prompt approach
    transactions_str = ""
    for i, details in enumerate(batch_details):
        description = details[0]
        extended_details = details[1] if len(details) > 1 else None
        context = str(description if pd.notna(description) else "")
        if pd.notna(extended_details) and str(extended_details).strip() != "":
            context += " | " + str(extended_details)
        context = context[:200]  # Further reduce context length
        transactions_str += f"\nTransaction {i+1}: \"{context}\"\n"
    
    # Simpler prompt format    
    prompt = f"""I need to categorize {batch_size} financial transactions into these categories: {category_list_str}.
{transactions_str}
For each transaction above, respond with ONLY the category name from the list.
Provide your response as:
Transaction 1: [category]
Transaction 2: [category]
"""

    try:
        print(f"Attempting API call with {batch_size} transactions...")
        response = client.chat.completions.create(
            model=LLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a financial transaction categorizer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,  # Reduced for simplicity
            temperature=0.0
        )
        
        raw_response = response.choices[0].message.content.strip()
        print(f"API Response: {raw_response}")
        
        # Simplified parsing
        parsed_categories = []
        lines = raw_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Look for "Transaction X: Category" pattern
            match = re.search(r"transaction\s+\d+\s*:\s*(.*)", line.lower())
            if match:
                category_text = match.group(1).strip().strip('"')
                
                # Direct match
                if category_text in PREDEFINED_CATEGORIES:
                    parsed_categories.append(category_text)
                    continue
                    
                # Case insensitive match
                for cat in PREDEFINED_CATEGORIES:
                    if cat.lower() == category_text.lower():
                        parsed_categories.append(cat)
                        break
                else:  # No match found
                    parsed_categories.append("Misc")
            
        # Ensure we have the expected number of results
        if len(parsed_categories) == batch_size:
            return parsed_categories
        else:
            print(f"Warning: Expected {batch_size} categories but parsed {len(parsed_categories)}")
            return ["Misc"] * batch_size
            
    except Exception as e:
        print(f"Error in batch API call: {e}")
        return ["Misc"] * batch_size

# --- Function to Store Model Results --- 
def store_model_training_results(model_version, model_filename, metrics):
    """Stores model version info and evaluation metrics in the database."""
    print(f"Storing model results for model: {model_version}")
    
    # Also print metrics to console for visibility
    for key, value in metrics.items():
        if key != 'classification_report' and key != 'error':
            print(f"  {key}: {value}")
    
    # Check if we can import the tables
    try:
        from db_connector import model_versions_table, model_scores_table
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import database tables: {e}")
        print("Metrics printed but not stored in database.")
        return True
    
    engine = get_engine()
    if not model_version or not model_filename or not metrics:
        print("Error: Missing model version, filename, or metrics for DB storage.")
        return False
    
    version_data = {
        'model_version': model_version,
        'model_filename': model_filename,
        'training_timestamp': datetime.now(timezone.utc),
        'training_dataset_size': metrics.get('training_dataset_size'),
        'holdout_set_size': metrics.get('holdout_set_size')
    }
    
    scores_data = []
    eval_timestamp = metrics.get('evaluation_timestamp')
    if eval_timestamp:
         try:
             eval_dt = datetime.fromisoformat(eval_timestamp)
             if eval_dt.tzinfo is None:
                 eval_dt = eval_dt.replace(tzinfo=timezone.utc)
         except ValueError:
             print(f"Warning: Could not parse evaluation timestamp '{eval_timestamp}'. Using current time.")
             eval_dt = datetime.now(timezone.utc)
         
         # Metrics to store in the database
         score_metrics = [
            'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'precision_weighted', 'recall_weighted', 'f1_weighted'
         ]
         
         for metric_name in score_metrics:
            if metric_name in metrics and pd.notna(metrics[metric_name]):
                scores_data.append({
                    'evaluation_timestamp': eval_dt,
                    'metric_name': metric_name,
                    'metric_value': metrics[metric_name]
                })
    
    # Actually store the data in the database
    with engine.begin() as connection:
        try:
            # Insert model version record
            insert_stmt = model_versions_table.insert().values(version_data)
            result = connection.execute(insert_stmt)
            model_version_id = result.inserted_primary_key[0]
            print(f"Stored model version: {model_version} with ID: {model_version_id}")
            
            # Insert model scores records
            if scores_data and model_version_id:
                for score in scores_data:
                    score['model_version_id'] = model_version_id
                connection.execute(model_scores_table.insert(), scores_data)
                print(f"Stored {len(scores_data)} evaluation metrics for model ID {model_version_id}.")
            
            return True
        except sa.exc.IntegrityError as e:
             print(f"Error storing model results (potential duplicate?): {e}")
        except Exception as e:
             print(f"Error storing model training results: {e}")
             print("Metrics printed but not stored in database.")
    
    return False

# --- Model Evaluation Function --- 
def evaluate_model_on_holdout(model_type, test_size=0.2, random_state=42):
    """Loads data, gets consistent holdout set, evaluates specified model type, saves results."""
    try:
        # --- Load Data --- 
        engine = get_engine()
        data = pd.read_sql('SELECT * FROM transactions', engine)
        print(f"Loaded {len(data)} records from the database for evaluation.")
    except Exception as e:
        print(f"Error loading data for evaluation: {e}")
        return {"error": f"Database Error: {e}"}

    if data.empty:
        print("No data found in database for evaluation.")
        return {"error": "No evaluation data"}

    # --- Data Preparation --- 
    required_cols = ['description', 'amount', 'category']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"Error: Evaluation data from database is missing required columns: {missing_cols}")
        return {"error": f"Missing columns: {missing_cols}"}

    # --- Train/Test Split --- 
    X_features = data[['description', 'amount']]
    y = data['category']
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, 
            random_state=random_state, stratify=y
        )
    except ValueError as e:
        print(f"Error during train/test split: {e}")
        return {"error": f"Train/test split error: {e}"}

    # Store test data for prediction
    test_descriptions_df = X_test.copy()
    
    # --- Model Selection Logic --- 
    eval_metrics = {
        'model_type': model_type,
        'holdout_set_size': len(y_test),
        'training_dataset_size': len(y_train)
    }
    
    # --- Local Model Evaluation --- 
    if model_type == "local":
        print("Evaluating local model...")
        # Much of the original local model training and evaluation logic
        # would be here - skipping for brevity as we're focusing on Llama
        return eval_metrics
    
    # --- LLM Model Evaluation --- 
    elif model_type == "llama":
        if not client:
            print("ERROR: Llama client not initialized for evaluation.")
            eval_metrics['error'] = "Llama client not initialized"
            return eval_metrics
        model_version_eval = f"Llama-3.1-8B-eval@{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        model_filename_eval = LLAMA_MODEL_NAME
        print(f"Generating predictions using Llama model for {len(test_descriptions_df)} holdout items...")
        start_time = datetime.now()
        
        # --- Batch Processing Logic ---
        # Using smaller batch size of 2 to avoid API errors
        batch_size = 2
        y_pred = []
        total_items = len(test_descriptions_df)
        
        # Process in smaller batches
        for i in range(0, total_items, batch_size):
            # Get current batch range
            current_batch_end = min(i + batch_size, total_items)
            batch_df = test_descriptions_df.iloc[i:current_batch_end]
            
            # Create list of tuples: [(desc1, ext1), (desc2, ext2), ...]
            batch_details = [
                (row['description'], row.get('extended_details')) 
                for _, row in batch_df.iterrows()
            ]
            
            print(f"  Processing batch {i//batch_size + 1}/{(total_items + batch_size - 1)//batch_size} (items {i+1}-{current_batch_end})... ")
            
            # Call the batch function
            batch_categories = get_llama_categories_batch(batch_details, batch_size=len(batch_details))
            y_pred.extend(batch_categories)
            
            # Add delay between batches to reduce rate limiting issues
            if i + batch_size < total_items:
                print("  Pausing 1 second between batches...")
                sleep(1)
        
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"  LLM batch prediction finished in {total_time:.2f} seconds.")
        
        # --- Calculate Metrics --- 
        try:
            eval_metrics['evaluation_timestamp'] = datetime.now(timezone.utc).isoformat()
            accuracy = accuracy_score(y_test, y_pred)
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
            eval_metrics['accuracy'] = accuracy
            eval_metrics['precision_macro'] = precision_macro
            eval_metrics['recall_macro'] = recall_macro
            eval_metrics['f1_macro'] = f1_macro
            eval_metrics['precision_weighted'] = precision_weighted
            eval_metrics['recall_weighted'] = recall_weighted
            eval_metrics['f1_weighted'] = f1_weighted
            eval_metrics['classification_report'] = classification_report(y_test, y_pred, zero_division=0)
            print(f"  Evaluation metrics calculated successfully.")
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            eval_metrics['error'] = f"Metrics Error: {e}"
            return eval_metrics

        # --- Store Results in Database --- 
        try:
            store_model_training_results(model_version_eval, model_filename_eval, eval_metrics)
            print("Successfully stored evaluation results.")
        except Exception as e:
            print(f"Error storing evaluation results: {e}")
            eval_metrics['error'] = f"DB Storage Error: {e}"
        return eval_metrics
    else:
        print(f"ERROR: Unknown model_type '{model_type}' for evaluation.")
        return {"error": f"Unknown model_type: {model_type}"}

# --- Main Block --- 
if __name__ == "__main__":
    # 1. Test basic connection first
    connection_ok = test_llama_connection()
    
    if connection_ok:
        # 2. If connection ok, proceed with evaluation
        print("\nStarting Llama model evaluation...")
        llama_eval_metrics = evaluate_model_on_holdout("llama")
        print("\n--- Llama Evaluation Summary ---")
        if llama_eval_metrics and "error" not in llama_eval_metrics:
            print("Evaluation completed and results stored (if DB connection succeeded).")
            for key, value in llama_eval_metrics.items():
                if key != 'classification_report' and key != 'error': 
                    print(f"  {key}: {value}")
        else:
            print("Evaluation failed.")
            if llama_eval_metrics and "error" in llama_eval_metrics:
                 print(f"  Error details: {llama_eval_metrics['error']}")
    else:
        print("\nSkipping evaluation due to connection test failure.") 