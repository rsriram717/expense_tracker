import sqlalchemy as sa
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Boolean, Date, Text, DateTime, DECIMAL
from sqlalchemy.schema import DDL
from sqlalchemy.event import listen
from datetime import datetime

DB_PATH = 'sqlite:///data/transactions.db'

# Define the table structure (used by initialize and update)
transactions_table_metadata = MetaData()
transactions_table = Table('transactions',
    transactions_table_metadata,
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
    Column('timestamp', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
    # --- New Columns --- 
    Column('home', Text, nullable=True), # 'Home' or 'Personal'
    Column('card_type', Text, nullable=True) # 'Gold' or 'Platinum'
)

def get_engine(db_path=DB_PATH):
    """Returns a SQLAlchemy engine for the database."""
    return create_engine(db_path)

def initialize_db(engine):
    """Creates the transactions table if it doesn't exist."""
    try:
        transactions_table_metadata.create_all(engine)
        print("Database table checked/created successfully.")
    except Exception as e:
        print(f"Error initializing database table: {e}")

def add_column_if_not_exists(engine, table_name, column):
    """Adds a column to a table if it doesn't already exist."""
    try:
        with engine.connect() as connection:
            # Use introspection to check if column exists (SQLite specific query)
            result = connection.execute(sa.text(f"PRAGMA table_info({table_name})"))
            existing_columns = [row[1] for row in result]
            
            col_name = column.name
            
            if col_name not in existing_columns:
                col_type = column.type.compile(engine.dialect)
                nullable = "NULL" if column.nullable else "NOT NULL"
                default_clause = ""
                if column.default:
                    if isinstance(column.default.arg, str):
                        default_clause = f"DEFAULT \'{column.default.arg}\'"
                    else:
                        default_clause = f"DEFAULT {column.default.arg}"
                
                add_col_sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type} {nullable} {default_clause}".strip()
                connection.execute(sa.text(add_col_sql))
                print(f"Added column '{col_name}' to table '{table_name}'.")
            # else:
            #     print(f"Column '{col_name}' already exists in table '{table_name}'.")
                
    except Exception as e:
        print(f"Error checking/adding column {column.name} to {table_name}: {e}")

def update_db_schema(engine):
    """Checks for and adds missing columns defined in the transactions_table model."""
    print("Checking/updating database schema...")
    table_name = transactions_table.name
    with engine.begin() as connection:
        for column in transactions_table.columns:
            if not column.primary_key:
                _add_column_if_not_exists_conn(connection, table_name, column)
    print("Schema check/update complete.")

def _add_column_if_not_exists_conn(connection, table_name, column):
    try:
        result = connection.execute(sa.text(f"PRAGMA table_info({table_name})"))
        existing_columns = [row[1] for row in result]
        
        col_name = column.name
        
        if col_name not in existing_columns:
            col_type = column.type.compile(connection.engine.dialect)
            nullable = "NULL" if column.nullable else "NOT NULL"
            default_clause = ""
            if column.default:
                if isinstance(column.default.arg, str):
                    default_clause = f"DEFAULT \'{column.default.arg}\'"
                else:
                    default_clause = f"DEFAULT {column.default.arg}"
            
            add_col_sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type} {nullable} {default_clause}".strip()
            connection.execute(sa.text(add_col_sql))
            print(f"Added column '{col_name}' to table '{table_name}'.")

    except Exception as e:
        print(f"Error checking/adding column {column.name} to {table_name} within transaction: {e}")
        raise

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
    # Add new required fields if they become mandatory, otherwise handle Nones/defaults
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns for database storage: {missing_cols}")

    df_to_store = df.copy()
    if 'transaction_date' in df_to_store.columns:
        df_to_store['transaction_date'] = pd.to_datetime(df_to_store['transaction_date']).dt.date
    if 'timestamp' not in df_to_store.columns:
        df_to_store['timestamp'] = datetime.utcnow()
        
    # Ensure new columns exist before saving, adding them with None if missing
    if 'home' not in df_to_store.columns:
        df_to_store['home'] = None
    if 'card_type' not in df_to_store.columns:
        df_to_store['card_type'] = None
        
    # Select only columns that exist in the database table to avoid errors
    db_cols = [c.name for c in transactions_table.columns]
    df_to_store = df_to_store[[col for col in db_cols if col in df_to_store.columns]]

    try:
        df_to_store.to_sql('transactions', engine, if_exists='append', index=False)
        print(f"Successfully stored {len(df_to_store)} transactions to the database.")
    except Exception as e:
        print(f"Error storing transactions to database: {e}")
        raise

# --- Main execution block (for direct script run) --- 
if __name__ == "__main__":
    print("Initializing/Updating database from db_connector...")
    db_engine = get_engine()
    initialize_db(db_engine)
    update_db_schema(db_engine) 