import pandas as pd
import os
import glob
from datetime import datetime
from db_connector import store_finalized_transactions, transactions_table, initialize_db, get_engine, update_db_schema

# Configuration
CATEGORIZED_DIR = "data/categorized"

def migrate_data():
    """Loads data from CSVs in data/categorized, prepares it, and stores it in the DB."""
    print(f"Starting migration from {CATEGORIZED_DIR}...")
    
    csv_files = glob.glob(os.path.join(CATEGORIZED_DIR, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in data/categorized. Migration skipped.")
        return

    all_dfs = []

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Processing file: {filename}...")
        
        try:
            df = pd.read_csv(file_path)
            
            # 1. Determine Card Type from filename
            if "gold" in filename.lower():
                card_type = "Gold"
            elif "platinum" in filename.lower():
                card_type = "Platinum"
            else:
                card_type = None # Or raise an error if required
            
            # 2. Standardize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # 3. Prepare DataFrame for DB
            df_to_db = pd.DataFrame()
            
            # Map columns (handle potential missing columns in source CSV)
            df_to_db['transaction_date'] = pd.to_datetime(df.get('date', None), errors='coerce').dt.date
            df_to_db['description'] = df.get('description', None)
            df_to_db['amount'] = pd.to_numeric(df.get('amount', None), errors='coerce')
            df_to_db['extended_details'] = df.get('extended_details', None)
            df_to_db['statement_description'] = df.get('appears_on_your_statement_as', None)
            df_to_db['category'] = df.get('category', None)
            df_to_db['home'] = df.get('home', None)
            
            # Add derived/metadata fields
            df_to_db['card_type'] = card_type
            df_to_db['source_file'] = filename
            df_to_db['is_manually_categorized'] = True # Assume categorized files are final
            df_to_db['confidence'] = None
            df_to_db['model_version'] = None
            df_to_db['model_filename'] = None
            df_to_db['timestamp'] = datetime.utcnow()
            
            # Clean data - drop rows where essential info is missing AFTER mapping
            required_for_db = ['transaction_date', 'description', 'amount', 'category', 'source_file']
            initial_rows = len(df_to_db)
            df_to_db.dropna(subset=required_for_db, inplace=True)
            dropped_rows = initial_rows - len(df_to_db)
            if dropped_rows > 0:
                print(f"  Dropped {dropped_rows} rows due to missing essential data.")

            # Ensure correct types before potential storage
            for col in ['home', 'card_type', 'category', 'description', 'extended_details', 'statement_description', 'source_file']:
                 if col in df_to_db:
                     df_to_db[col] = df_to_db[col].astype(str).replace({'nan': None, 'None': None})
            
            if not df_to_db.empty:
                 all_dfs.append(df_to_db)
            else:
                 print(f"  No valid data found in {filename} after cleaning.")
                 
        except Exception as e:
            print(f"  Error processing {filename}: {e}. Skipping this file.")
            continue
            
    # Store all data at once if any valid data was processed
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Final check: Select only columns present in the DB table definition
        db_cols = [c.name for c in transactions_table.columns if c.name != 'id'] # Exclude auto-inc id
        final_df_to_store = combined_df[[col for col in db_cols if col in combined_df.columns]]
        
        print(f"\nAttempting to store {len(final_df_to_store)} total transactions...")
        try:
            store_finalized_transactions(final_df_to_store)
            print("Migration completed successfully.")
        except Exception as e:
            print(f"\nError storing combined data during migration: {e}")
            print("Migration failed.")
    else:
        print("\nNo valid data collected from CSV files. Nothing to store.")

if __name__ == "__main__":
    # Ensure DB and schema are up-to-date before migration
    print("Preparing database...")
    db_engine = get_engine()
    initialize_db(db_engine)
    update_db_schema(db_engine)
    print("Database ready.")
    
    # Run the migration
    migrate_data() 