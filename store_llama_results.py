from datetime import datetime, timezone
import pandas as pd
from db_connector import get_engine, model_versions_table, model_scores_table

def store_llama_results(accuracy, holdout_size=158):
    """Store Llama model evaluation results in the database."""
    engine = get_engine()
    
    # Create model version entry
    model_version = "llama_v1"
    model_filename = "llama_api"
    
    version_data = {
        'model_version': model_version,
        'model_filename': model_filename,
        'training_timestamp': datetime.now(timezone.utc),
        'training_dataset_size': None,  # Not applicable for LLM
        'holdout_set_size': holdout_size
    }
    
    # Create scores data
    eval_timestamp = datetime.now(timezone.utc)
    scores_data = [{
        'evaluation_timestamp': eval_timestamp,
        'metric_name': 'accuracy',
        'metric_value': accuracy
    }]
    
    # Store in database
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
        except Exception as e:
            print(f"Error storing model results: {e}")
            return False

if __name__ == "__main__":
    # Store the results from the recent evaluation
    store_llama_results(accuracy=0.7722, holdout_size=158) 