import sqlalchemy as sa
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Boolean, Date, Text, DateTime, DECIMAL, ForeignKey
from datetime import datetime, timezone

DB_PATH = 'sqlite:///data/transactions.db'

def get_engine(db_path=DB_PATH):
    """Returns a SQLAlchemy engine for the database."""
    return create_engine(db_path)

def initialize_db(engine):
    """Creates all defined tables if they don't exist."""
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
        print("Database tables checked/created successfully.")
    except Exception as e:
        print(f"Error initializing database tables: {e}")

def store_finalized_transactions(df):
    """Stores the finalized transaction DataFrame into the transactions table."""
    engine = get_engine()
    required_cols = [
        'transaction_date', 'description', 'amount',
        'category', 'is_manually_categorized', 'source_file'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns for transactions storage: {missing_cols}")

    df_to_store = df.copy()
    if 'transaction_date' in df_to_store.columns:
        df_to_store['transaction_date'] = pd.to_datetime(df_to_store['transaction_date']).dt.date
    if 'timestamp' not in df_to_store.columns:
        df_to_store['timestamp'] = datetime.now(timezone.utc)
        
    for col in ['home', 'card_type', 'confidence', 'model_version', 'model_filename', 'extended_details', 'statement_description']:
         if col not in df_to_store.columns:
             df_to_store[col] = None
        
    db_cols = [c.name for c in transactions_table.columns]
    df_to_store = df_to_store[[col for col in db_cols if col in df_to_store.columns]]

    try:
        df_to_store.to_sql(transactions_table.name, engine, if_exists='append', index=False)
        print(f"Successfully stored {len(df_to_store)} transactions.")
    except Exception as e:
        print(f"Error storing transactions: {e}")
        raise

# Create metadata object
metadata = MetaData()

# Define model_versions_table
model_versions_table = Table(
    'model_versions', metadata,
    Column('id', Integer, primary_key=True),
    Column('model_version', String(50), unique=True, nullable=False),
    Column('model_filename', String(255), nullable=False),
    Column('training_timestamp', DateTime, nullable=False),
    Column('training_dataset_size', Integer),
    Column('holdout_set_size', Integer)
)

# Define model_scores_table
model_scores_table = Table(
    'model_scores', metadata,
    Column('id', Integer, primary_key=True),
    Column('model_version_id', Integer, ForeignKey('model_versions.id'), nullable=False),
    Column('evaluation_timestamp', DateTime, nullable=False),
    Column('metric_name', String(50), nullable=False),
    Column('metric_value', Float, nullable=False)
)

# Create the tables if they don't exist
def create_tables():
    """Create the model_versions and model_scores tables if they don't exist."""
    engine = get_engine()
    metadata.create_all(engine)

# Call create_tables when the module is imported
create_tables()

# Initialize the database when this module is imported
if __name__ == "__main__":
    print("Initializing database from db_connector...")
    db_engine = get_engine()
    initialize_db(db_engine) 