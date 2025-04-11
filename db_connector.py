import sqlalchemy as sa
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Boolean, Date, Text, DateTime, DECIMAL, ForeignKey
from datetime import datetime, timezone

DB_PATH = 'sqlite:///data/transactions.db'

def get_engine(db_path=DB_PATH):
    """Returns a SQLAlchemy engine for the database."""
    return create_engine(db_path)

metadata = MetaData()
transactions_table = Table('transactions',
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

def initialize_db(engine):
    """Creates all defined tables if they don't exist."""
    try:
        # Only create the main transactions table here
        transactions_table.create(engine, checkfirst=True)
        
        # Also ensure model tables exist
        model_versions_table.create(engine, checkfirst=True)
        model_scores_table.create(engine, checkfirst=True)
        
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
        
    # Use the globally defined transactions_table
    db_cols = [c.name for c in transactions_table.columns if c.name != 'id'] # Exclude primary key for insertion
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
# create_tables() # Keep this commented out for now

def clear_model_data():
    """Deletes all data from model_versions and model_scores tables."""
    engine = get_engine()
    with engine.begin() as connection:
        try:
            # Delete scores first due to foreign key constraint
            delete_scores = model_scores_table.delete()
            result_scores = connection.execute(delete_scores)
            print(f"Deleted {result_scores.rowcount} rows from model_scores.")
            
            # Delete versions
            delete_versions = model_versions_table.delete()
            result_versions = connection.execute(delete_versions)
            print(f"Deleted {result_versions.rowcount} rows from model_versions.")
            
            print("Model data cleared successfully.")
        except Exception as e:
            print(f"Error clearing model data: {e}")
            # Transaction rolls back automatically

def store_model_training_results(model_version, model_filename, metrics):
    """Stores model training results in the database.
    
    Args:
        model_version: The version identifier of the model
        model_filename: The filename of the saved model
        metrics: Dictionary containing model metrics
        
    Returns:
        True if successful, False otherwise
    """
    engine = get_engine()
    
    # Ensure model tables exist
    model_versions_table.create(engine, checkfirst=True)
    model_scores_table.create(engine, checkfirst=True)
    
    try:
        # Insert model version data
        with engine.begin() as connection:
            # Check if this model version already exists
            exists_query = sa.select(model_versions_table.c.id).where(
                model_versions_table.c.model_version == str(model_version)
            )
            existing_id = connection.execute(exists_query).scalar()
            
            # If this model version already exists, update it
            if existing_id:
                update_stmt = model_versions_table.update().where(
                    model_versions_table.c.id == existing_id
                ).values(
                    model_filename=model_filename,
                    training_timestamp=datetime.now(timezone.utc),
                    training_dataset_size=metrics.get('training_dataset_size', None),
                    holdout_set_size=metrics.get('holdout_set_size', None)
                )
                connection.execute(update_stmt)
                model_version_id = existing_id
                print(f"Updated existing model version: {model_version}")
            else:
                # Insert new model version
                insert_stmt = model_versions_table.insert().values(
                    model_version=str(model_version),
                    model_filename=model_filename,
                    training_timestamp=datetime.now(timezone.utc),
                    training_dataset_size=metrics.get('training_dataset_size', None),
                    holdout_set_size=metrics.get('holdout_set_size', None)
                )
                result = connection.execute(insert_stmt)
                model_version_id = result.inserted_primary_key[0]
                print(f"Inserted new model version: {model_version}")
            
            # Insert model scores
            metrics_to_store = {k: v for k, v in metrics.items() 
                               if k not in ['training_dataset_size', 'holdout_set_size', 'error']}
            
            # Delete existing scores for this model version
            delete_stmt = model_scores_table.delete().where(
                model_scores_table.c.model_version_id == model_version_id
            )
            connection.execute(delete_stmt)
            
            # Insert new scores
            for metric_name, metric_value in metrics_to_store.items():
                if isinstance(metric_value, (int, float)):
                    insert_stmt = model_scores_table.insert().values(
                        model_version_id=model_version_id,
                        evaluation_timestamp=datetime.now(timezone.utc),
                        metric_name=metric_name,
                        metric_value=float(metric_value)
                    )
                    connection.execute(insert_stmt)
            
            print(f"Stored {len(metrics_to_store)} metrics for model version {model_version}")
            return True
            
    except Exception as e:
        print(f"Error storing model training results: {e}")
        return False

# Initialize the database when this module is imported
if __name__ == "__main__":
    print("Initializing database from db_connector...")
    db_engine = get_engine()
    initialize_db(db_engine)
    
    # Example usage for clearing data (uncomment to run)
    # print("\nClearing model version and score data...")
    # clear_model_data() 