import sqlalchemy as sa
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Boolean, Date, Text, DateTime, DECIMAL
from datetime import datetime

DB_PATH = 'sqlite:///data/transactions.db'

def get_engine(db_path=DB_PATH):
    """Returns a SQLAlchemy engine for the database."""
    return create_engine(db_path)

def initialize_db(engine):
    """Creates the transactions table if it doesn't exist."""
    metadata = MetaData()
    transactions = Table('transactions',
                       metadata,
                       Column('id', Integer, primary_key=True, autoincrement=True),
                       Column('transaction_date', Date, nullable=False),
                       Column('description', Text, nullable=False),
                       Column('amount', DECIMAL(10, 2), nullable=False),
                       Column('extended_details', Text, nullable=True),
                       Column('statement_description', Text, nullable=True),
                       Column('category', Text, nullable=False),
                       Column('is_manually_categorized', Boolean, nullable=False),
                       Column('confidence', Float, nullable=True),
                       Column('model_version', Text, nullable=True),
                       Column('model_filename', Text, nullable=True),
                       Column('source_file', Text, nullable=False),
                       Column('timestamp', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
                       )
    try:
        metadata.create_all(engine)
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")

def store_finalized_transactions(df):
    """
    Stores the finalized DataFrame into the transactions table.
    Assumes the DataFrame has the correct columns and metadata already set 
    (including 'is_manually_categorized', 'model_version', etc.).
    """
    engine = get_engine()
    required_cols = [
        'transaction_date', 'description', 'amount',
        'category', 'is_manually_categorized', 'source_file'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns for database storage: {missing_cols}")

    # Ensure correct data types before saving (especially for date)
    df_to_store = df.copy()
    if 'transaction_date' in df_to_store.columns:
        df_to_store['transaction_date'] = pd.to_datetime(df_to_store['transaction_date']).dt.date
    
    # Add timestamp if not present (though default should handle it)
    if 'timestamp' not in df_to_store.columns:
        df_to_store['timestamp'] = datetime.utcnow()

    try:
        # Append data to the table
        df_to_store.to_sql('transactions', engine, if_exists='append', index=False)
        print(f"Successfully stored {len(df_to_store)} transactions to the database.")
    except Exception as e:
        print(f"Error storing transactions to database: {e}")
        raise # Re-raise the exception after printing

# Initialize the database when this module is imported
if __name__ == "__main__":
    print("Initializing database from db_connector...")
    db_engine = get_engine()
    initialize_db(db_engine) 