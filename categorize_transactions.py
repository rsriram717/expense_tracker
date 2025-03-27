import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import os
import glob
import argparse

class TransactionCategorizer:
    def __init__(self, model_path=None):
        """Initialize the categorizer, optionally loading a saved model."""
        self.model = None
        self.categories = None
        self.model_path = model_path or 'transaction_model.pkl'
        if os.path.exists(self.model_path):
            self.load_model()
    
    def preprocess_text(self, text):
        """Clean and normalize text for better matching."""
        if pd.isna(text):
            return ""
        # Convert to lowercase and remove special characters
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def prepare_features(self, df):
        """Create features for classification from transaction data."""
        # Combine description and statement description for better classification
        df['feature_text'] = df['Description'].fillna('') + ' ' + df['Appears On Your Statement As'].fillna('')
        df['feature_text'] = df['feature_text'].apply(self.preprocess_text)
        return df
    
    def train_from_directory(self, directory_path):
        """Train a model on all categorized CSV files in a directory."""
        print(f"Loading categorized data from {directory_path}...")
        
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {directory_path}")
        
        print(f"Found {len(csv_files)} CSV files for training")
        
        # Load and concatenate all CSV files
        all_data = []
        for file_path in csv_files:
            print(f"Loading {os.path.basename(file_path)}...")
            df = pd.read_csv(file_path)
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined dataset has {len(combined_df)} transactions")
        
        # Continue with training process
        return self.train_from_dataframe(combined_df)
    
    def train(self, categorized_data_path):
        """Train a new model on categorized transaction data."""
        if os.path.isdir(categorized_data_path):
            # If it's a directory, use train_from_directory
            return self.train_from_directory(categorized_data_path)
        
        # Otherwise, load a single file
        print(f"Loading categorized data from {categorized_data_path}...")
        df = pd.read_csv(categorized_data_path)
        return self.train_from_dataframe(df)
    
    def train_from_dataframe(self, df):
        """Train model directly from a dataframe."""
        # Extract unique categories
        self.categories = sorted(df['Category'].unique())
        print(f"Found {len(self.categories)} unique categories")
        
        # Prepare features for training
        df = self.prepare_features(df)
        
        # Create and train the model
        print("Training model...")
        X = df['feature_text']
        y = df['Category']
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a pipeline with TF-IDF and Random Forest
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        accuracy = self.model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")
        
        # Save the model
        self.save_model()
        
        return accuracy
    
    def categorize_directory(self, input_directory, output_directory):
        """Categorize all CSV files in a directory."""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Run train() first.")
        
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(input_directory, "*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {input_directory}")
        
        print(f"Found {len(csv_files)} CSV files to categorize")
        
        results = {}
        
        # Process each file
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_directory, f"categorized_{filename}")
            
            print(f"Categorizing {filename}...")
            result_df = self.categorize(file_path, output_path)
            results[filename] = result_df
            
        return results
    
    def categorize(self, transactions_path, output_path=None):
        """Categorize new transactions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Run train() first.")
        
        print(f"Loading new transactions from {transactions_path}...")
        df = pd.read_csv(transactions_path)
        original_df = df.copy()
        
        # Prepare features
        df = self.prepare_features(df)
        
        # Predict categories
        print("Categorizing transactions...")
        X = df['feature_text']
        predicted_categories = self.model.predict(X)
        
        # Add predictions to original dataframe
        original_df['Category'] = predicted_categories
        
        # Calculate confidence scores (probability of the predicted class)
        probabilities = self.model.predict_proba(X)
        confidence = [probabilities[i, np.where(self.model.classes_ == cat)[0][0]] 
                     for i, cat in enumerate(predicted_categories)]
        original_df['Confidence'] = confidence
        
        # Save categorized transactions
        if output_path:
            print(f"Saving categorized transactions to {output_path}")
            original_df.to_csv(output_path, index=False)
        
        return original_df
    
    def update_category(self, transactions_df, index, new_category):
        """Update a category for a specific transaction."""
        if new_category not in self.categories:
            self.categories.append(new_category)
            print(f"Added new category: {new_category}")
        
        transactions_df.at[index, 'Category'] = new_category
        return transactions_df
    
    def retrain_with_feedback(self, original_data_path, feedback_data):
        """Retrain the model incorporating user feedback."""
        if os.path.isdir(original_data_path):
            # If it's a directory, load all files
            csv_files = glob.glob(os.path.join(original_data_path, "*.csv"))
            
            if not csv_files:
                raise ValueError(f"No CSV files found in {original_data_path}")
            
            all_data = []
            for file_path in csv_files:
                df = pd.read_csv(file_path)
                all_data.append(df)
            
            original_df = pd.concat(all_data, ignore_index=True)
        else:
            # Otherwise, load a single file
            original_df = pd.read_csv(original_data_path)
        
        # Combine with feedback data
        combined_df = pd.concat([original_df, feedback_data], ignore_index=True)
        
        # Train on combined data
        self.train_from_dataframe(combined_df)
        
        return self.model
    
    def save_model(self):
        """Save model to disk."""
        model_data = {
            'model': self.model,
            'categories': self.categories
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.categories = model_data['categories']
            print(f"Model loaded from {self.model_path}")
            print(f"Model has {len(self.categories)} categories")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def generate_category_stats(self, df):
        """Generate statistics about categorized transactions."""
        category_counts = df['Category'].value_counts()
        
        # Calculate total amount per category
        category_amounts = df.groupby('Category')['Amount'].agg(['sum', 'mean', 'count'])
        category_amounts = category_amounts.sort_values(by='sum', ascending=False)
        
        return category_amounts


def main():
    parser = argparse.ArgumentParser(description='Categorize credit card transactions.')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--categorize', action='store_true', help='Categorize new transactions')
    parser.add_argument('--categorized-data', default='data/categorized', help='Path to categorized data for training (file or directory)')
    parser.add_argument('--new-transactions', default='data/to_categorize', help='Path to new transactions for categorization (file or directory)')
    parser.add_argument('--output', default='data/output', help='Path for output files or directory')
    parser.add_argument('--model', default='transaction_model.pkl', help='Path to save/load model')
    
    args = parser.parse_args()
    
    categorizer = TransactionCategorizer(model_path=args.model)
    
    if args.train:
        categorizer.train(args.categorized_data)
    
    if args.categorize:
        if categorizer.model is None:
            print("No model found. Training first...")
            categorizer.train(args.categorized_data)
        
        if os.path.isdir(args.new_transactions):
            results = categorizer.categorize_directory(args.new_transactions, args.output)
            
            # Generate and display statistics for all files
            for filename, df in results.items():
                stats = categorizer.generate_category_stats(df)
                print(f"\nCategory statistics for {filename}:")
                print(stats)
        else:
            output_path = args.output
            if os.path.isdir(output_path):
                basename = os.path.basename(args.new_transactions)
                output_path = os.path.join(output_path, f"categorized_{basename}")
            
            categorized_df = categorizer.categorize(args.new_transactions, output_path)
            
            # Generate and display statistics
            stats = categorizer.generate_category_stats(categorized_df)
            print("\nCategory statistics:")
            print(stats)
            
            print(f"\nCategorized {len(categorized_df)} transactions")
            print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main() 