import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import os
import numpy as np
from datetime import datetime
from improved_categorizer import train_model, categorize_transactions, find_latest_model_file, evaluate_model_on_holdout
from db_connector import store_finalized_transactions, initialize_db, get_engine, store_model_training_results

# Set page config for a modern look
st.set_page_config(
    page_title="Transaction Categorizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #0E6EFF;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #0052CC;
    }
    .stSelectbox>div>div>select {
        border-radius: 5px;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Transaction Categorizer")
st.markdown("Review and finalize transaction categories.")

# Initialize session state for output file and categorization state
if 'output_file' not in st.session_state:
    st.session_state.output_file = None
if 'categorization_run' not in st.session_state:
    st.session_state.categorization_run = False
if 'reload_categorized' not in st.session_state:
    st.session_state.reload_categorized = False

# Initialize Database
try:
    initialize_db(get_engine())
except Exception as e:
    st.error(f"Database Initialization Error: {e}")

# Initialize Session State
def init_session_state():
    defaults = {
        'categorization_results': None, # Stores dict {filename: df}
        'current_model_version': None,
        'current_model_filename': None,
        'selected_input_file': None, # Path object for the selected input CSV
        'current_display_df': None, # DF currently shown in data_editor (with Title Case cols)
        'original_categorized_df': None, # DF loaded after categorization (internal lowercase cols)
        'categorization_run_for_file': None, # Track which input file was categorized
        'data_saved_to_output': False, # Track if 'Save Changes' was clicked
        'output_file_path': None # Path to the temp output CSV
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Define available categories (consider loading from DB or config later)
categories = [
    "Food & Drink", "Transportation", "Entertainment", "Groceries",
    "Shopping", "Travel-Airline", "Travel-Lodging", "Travel-Other",
    "Clothes", "Subscriptions", "Home", "Pets", "Beauty",
    "Professional Services", "Medical", "Misc"
]

# Helper Functions
def load_and_prepare_input_df(file_path):
    """Loads a CSV, standardizes columns, filters, returns df with lowercase cols."""
    try:
        df = pd.read_csv(file_path)
        # Standardize column names: lowercase, replace space with underscore
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # Ensure essential columns exist
        required = ['description', 'amount']
        if not all(col in df.columns for col in required):
            st.error(f"Input file {file_path.name} missing required columns: {required}. Available: {list(df.columns)}")
            return None
            
        # Convert date column if exists
        if 'date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['date'], errors='coerce')
            # Keep original date if conversion fails for some rows
            df['transaction_date'] = df['transaction_date'].fillna(df['date'])
            df = df.drop(columns=['date']) # Use transaction_date internally
        elif 'transaction_date' in df.columns:
             df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        else:
            st.warning(f"Input file {file_path.name} missing a 'date' or 'transaction_date' column.")
            # Optionally create a default date or handle it downstream
            df['transaction_date'] = pd.NaT # Assign Not-a-Time if no date

        # Filter out specific descriptions
        if 'description' in df.columns:
             df = df[~df['description'].str.contains('MOBILE PAYMENT - THANK YOU', case=False, na=False)]

        # Add source file info
        df['source_file'] = file_path.name
        
        # Ensure category/confidence columns exist for editor, even if empty initially
        if 'category' not in df.columns:
            df['category'] = None
        if 'confidence' not in df.columns:
             df['confidence'] = np.nan

        return df

    except Exception as e:
        st.error(f"Error loading or preparing file {file_path.name}: {e}")
        return None

def format_df_for_display(df):
    """Formats the DataFrame for st.data_editor (Title Case columns)."""
    df_display = df.copy()
    # Rename internal columns to display-friendly names
    rename_map = {
        'transaction_date': 'Date',
        'description': 'Description',
        'amount': 'Amount',
        'extended_details': 'Extended Details',
        'statement_description': 'Statement Description',
        'appears_on_your_statement_as': 'Appears On Your Statement As',
        'category': 'Category',
        'confidence': 'Confidence'
    }
    df_display.rename(columns={k: v for k, v in rename_map.items() if k in df_display.columns}, inplace=True)
    
    # Select and order columns for display
    display_columns = ['Date', 'Description', 'Amount', 'Extended Details', 'Appears On Your Statement As', 'Category', 'Confidence']
    # Keep only columns that exist in the dataframe
    final_display_cols = [col for col in display_columns if col in df_display.columns]
    return df_display[final_display_cols]

# Sidebar for controls
with st.sidebar:
    st.header("Training (Local RandomForest)")
    if st.button("💪 Train New RF Model Version"):
        with st.spinner("Training RandomForest model..."):
            try:
                model, feature_prep, model_version, model_filename, metrics = train_model()
                if model and model_version and model_filename and metrics:
                    st.success(f"Successfully trained model: {model_filename}")
                    try:
                         store_model_training_results(model_version, model_filename, metrics)
                         st.success("Stored RF training results in database.")
                         st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                         st.metric("F1 (Macro)", f"{metrics.get('f1_macro', 0):.4f}")
                    except Exception as db_e:
                         st.error(f"Error storing RF training results in DB: {db_e}")
                elif metrics and "error" in metrics:
                     st.error(f"RF Model training failed: {metrics['error']}")
                else:
                    st.error("RF Model training failed. Check logs.")
            except Exception as e:
                st.error(f"Error during RF model training: {str(e)}")
                st.exception(e)

    st.divider()
    st.header("Categorization")
    
    # --- Model Selection --- 
    model_options = ["Local RandomForest", "Llama 3.1 (API)"]
    # Check if local model exists for default selection
    local_model_exists = find_latest_model_file() is not None
    default_index = 0 if local_model_exists else 1 # Default to Llama if no local model
    selected_model_display = st.radio(
         "Choose categorization model:", 
         model_options,
         index=default_index,
         key="model_select"
    )
    
    # Map display name to internal type used by categorize_transactions
    model_type_arg = "local" if selected_model_display == "Local RandomForest" else "llama"
    
    # Disable button if selected model isn't available
    run_disabled = (model_type_arg == "local" and not local_model_exists) or \
                   (model_type_arg == "llama" and client is None) # Assuming client check in improved_categorizer
                   
    if run_disabled and model_type_arg == "local":
        st.warning("No trained local model found. Please train one first.")
    if run_disabled and model_type_arg == "llama":
        st.warning(f"Llama API client not initialized. Set {INFERENCE_API_KEY_ENV_VAR} env var.")
        
    if st.button("🔄 Run Categorization", type="primary", disabled=run_disabled):
        with st.spinner(f"Categorizing transactions using {selected_model_display}..."):
            try:
                # Pass the selected model type
                results_dict, model_ver, model_fname = categorize_transactions(model_type=model_type_arg)
                
                if results_dict is not None:
                    st.session_state.categorization_results = results_dict
                    st.session_state.current_model_version = model_ver
                    st.session_state.current_model_filename = model_fname
                    st.session_state.categorization_run_for_file = st.session_state.selected_input_file.name if st.session_state.selected_input_file else None
                    st.session_state.data_saved_to_output = False
                    st.success(f"Categorization complete using {model_fname}!")
                    st.experimental_rerun()
                else:
                    st.error(f"Categorization failed using {selected_model_display}. Check logs.")
            except Exception as e:
                st.error(f"Error during {selected_model_display} categorization: {str(e)}")
                st.exception(e)
                
    st.divider()
    st.header("Evaluation")
    if st.button("📊 Evaluate Models on Holdout Set"):
        with st.spinner("Evaluating models..."):
            st.write("Evaluating Local RandomForest...")
            try:
                 rf_metrics = evaluate_model_on_holdout("local")
                 if rf_metrics and "error" not in rf_metrics:
                     st.success("Local RF evaluation complete and stored.")
                     st.metric("RF Accuracy", f"{rf_metrics.get('accuracy', 0):.4f}")
                 else:
                     st.error(f"Local RF evaluation failed: {rf_metrics.get('error', 'Unknown error')}")
            except Exception as e:
                 st.error(f"Error during local evaluation: {e}")
                 
            st.write("Evaluating Llama 3.1...")
            try:
                 llama_metrics = evaluate_model_on_holdout("llama")
                 if llama_metrics and "error" not in llama_metrics:
                     st.success("Llama 3.1 evaluation complete and stored.")
                     st.metric("Llama Accuracy", f"{llama_metrics.get('accuracy', 0):.4f}")
                 else:
                     st.error(f"Llama evaluation failed: {llama_metrics.get('error', 'Unknown error')}")
            except Exception as e:
                 st.error(f"Error during Llama evaluation: {e}")

# Main content area
tab1, tab2 = st.tabs(["Review Transactions", "Analytics (DB)"])

with tab1:
    st.header("Review & Finalize")
    
    input_path = Path("data/to_categorize")
    output_path = Path("data/output")
    output_path.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    if not input_path.exists() or not any(input_path.glob("*.csv")):
        st.warning("No input CSV files found in `data/to_categorize`. Please add files to categorize.")
    else:
        transaction_files = sorted(list(input_path.glob("*.csv")), key=os.path.getmtime, reverse=True)
        
        # File selector using filenames
        selected_filename = st.selectbox(
            "Select Input Transaction File",
            options=[f.name for f in transaction_files]
        )
        
        if selected_filename:
            selected_file_path = input_path / selected_filename
            st.session_state.selected_input_file = selected_file_path
            st.session_state.output_file_path = output_path / f"improved_categorized_{selected_filename}"

            # Decide which DataFrame to load: 
            # 1. Freshly categorized result (if run just now for this file)?
            # 2. Previously saved output CSV?
            # 3. Original input CSV (if neither of the above exist)?
            
            df_to_display = None
            internal_df = None # Keep the df with internal column names

            # Condition 1: Categorization just run for THIS file
            if st.session_state.categorization_run_for_file == selected_filename and selected_filename in st.session_state.categorization_results:
                st.info(f"Displaying results from categorization run with {st.session_state.current_model_filename}")
                internal_df = st.session_state.categorization_results[selected_filename].copy()
                st.session_state.original_categorized_df = internal_df.copy() # Store the state right after model run
                st.session_state.categorization_run_for_file = None # Reset flag 
            
            # Condition 2: Load from saved output CSV if it exists and wasn't just categorized
            elif st.session_state.output_file_path.exists():
                st.info(f"Loading previously categorized data from {st.session_state.output_file_path.name}")
                try:
                     # Load the output CSV (should have display-friendly names)
                     df_display_from_csv = pd.read_csv(st.session_state.output_file_path)
                     # Convert back to internal names for consistency before potential editing/saving
                     # This requires careful mapping back
                     reverse_rename_map = {
                        'Date': 'transaction_date', 'Description': 'description', 'Amount': 'amount',
                        'Extended Details': 'extended_details', 'Statement Description': 'statement_description',
                        'Appears On Your Statement As': 'appears_on_your_statement_as', 
                        'Category': 'category', 'Confidence': 'confidence'
                     }
                     internal_df = df_display_from_csv.rename(columns={k: v for k, v in reverse_rename_map.items() if k in df_display_from_csv.columns})
                     # Add source file info if missing from output csv
                     if 'source_file' not in internal_df.columns:
                         internal_df['source_file'] = selected_filename
                     # Store this as the baseline if loaded from output
                     st.session_state.original_categorized_df = internal_df.copy() 
                except Exception as e:
                     st.error(f"Error loading saved output file {st.session_state.output_file_path.name}: {e}")
                     internal_df = None
            
            # Condition 3: Load original input file if no categorized version available
            if internal_df is None:
                st.info(f"Loading original input file: {selected_filename}. Run categorization to get predictions.")
                internal_df = load_and_prepare_input_df(selected_file_path)
                st.session_state.original_categorized_df = internal_df.copy() if internal_df is not None else None

            # Display Data Editor
            if internal_df is not None:
                st.session_state.current_display_df = format_df_for_display(internal_df)
                
                st.markdown("Edit categories below as needed:")
                edited_df_display = st.data_editor(
                    st.session_state.current_display_df,
                    key=f"editor_{selected_filename}", # Unique key per file
                    column_config={
                        "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                        "Description": st.column_config.TextColumn("Description", width="medium"),
                        "Amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                        "Extended Details": st.column_config.TextColumn("Ext. Details", width="medium"),
                        "Appears On Your Statement As": st.column_config.TextColumn("Stmt Desc.", width="medium"),
                        "Category": st.column_config.SelectboxColumn("Category", options=categories, required=False),
                        "Confidence": st.column_config.NumberColumn("Confidence", format="%.3f", disabled=True),
                    },
                    hide_index=True,
                    use_container_width=True,
                    num_rows="dynamic"
                )

                # Action Buttons
                col1, col2 = st.columns(2)
                with col1:
                    # Save changes to the output CSV file
                    if st.button("💾 Save Changes to Output File"):
                        try:
                            # Save the *display* DataFrame to the output file
                            edited_df_display.to_csv(st.session_state.output_file_path, index=False)
                            st.session_state.data_saved_to_output = True
                            st.success(f"Changes saved to {st.session_state.output_file_path.name}")
                        except Exception as e:
                            st.error(f"Error saving changes to output file: {str(e)}")

                with col2:
                     # Submit finalized data to the database
                     if st.button("📤 Finalize and Submit to Database"):
                         with st.spinner("Processing and submitting to database..."):
                             try:
                                 # 1. Convert edited display DF back to internal column names
                                 reverse_rename_map = {
                                     'Date': 'transaction_date', 'Description': 'description', 'Amount': 'amount',
                                     'Extended Details': 'extended_details', 'Statement Description': 'statement_description',
                                     'Appears On Your Statement As': 'appears_on_your_statement_as', 
                                     'Category': 'category', 'Confidence': 'confidence'
                                 }
                                 final_internal_df = edited_df_display.rename(columns={k: v for k, v in reverse_rename_map.items() if k in edited_df_display.columns})
                                 
                                 # Ensure essential columns for DB exist
                                 db_essentials = ['transaction_date', 'description', 'amount', 'category']
                                 if not all(col in final_internal_df.columns for col in db_essentials):
                                      st.error(f"Cannot submit: Missing essential columns in edited data: {db_essentials}")
                                      raise ValueError("Missing essential columns")
                                      
                                 # Add source file if missing
                                 if 'source_file' not in final_internal_df.columns:
                                     final_internal_df['source_file'] = selected_filename
                                     
                                 # Add timestamp
                                 final_internal_df['timestamp'] = datetime.utcnow()

                                 # 2. Determine which rows were manually edited
                                 original_df = st.session_state.original_categorized_df
                                 if original_df is None:
                                     st.error("Cannot determine changes: Original data state not found.")
                                     raise ValueError("Original state missing")
                                     
                                 # Ensure indices align for comparison (important if rows added/deleted)
                                 final_internal_df = final_internal_df.reset_index(drop=True)
                                 original_df = original_df.reset_index(drop=True)
                                 # Merge to align rows based on common columns (adjust keys as needed)
                                 # Use a robust merge key if Description/Amount/Date might change slightly
                                 merge_keys = ['description', 'amount', 'transaction_date'] 
                                 # Handle cases where keys might not exist in both
                                 valid_merge_keys = [k for k in merge_keys if k in final_internal_df.columns and k in original_df.columns]
                                 if not valid_merge_keys:
                                     st.warning("Cannot reliably compare changes; assuming all are manual.")
                                     final_internal_df['is_manually_categorized'] = True
                                 else:
                                     comparison_df = pd.merge(final_internal_df, 
                                                               original_df[[*valid_merge_keys, 'category']], 
                                                               on=valid_merge_keys, 
                                                               how='left', 
                                                               suffixes=('', '_orig'))
                                     # Category is manually changed if it doesn't match original or original was NaN
                                     final_internal_df['is_manually_categorized'] = (comparison_df['category'] != comparison_df['category_orig']) | (comparison_df['category_orig'].isna())


                                 # 3. Set metadata based on manual categorization status
                                 manual_mask = final_internal_df['is_manually_categorized'] == True
                                 
                                 # For manually changed rows, nullify model info
                                 final_internal_df.loc[manual_mask, 'confidence'] = None
                                 final_internal_df.loc[manual_mask, 'model_version'] = None
                                 final_internal_df.loc[manual_mask, 'model_filename'] = None
                                 
                                 # For model-categorized rows (not changed), use session state model info
                                 final_internal_df.loc[~manual_mask, 'model_version'] = st.session_state.current_model_version
                                 final_internal_df.loc[~manual_mask, 'model_filename'] = st.session_state.current_model_filename
                                 # Confidence should already be present for these rows from categorization

                                 # 4. Prepare final columns for DB (match db_connector schema)
                                 db_columns = [
                                      'transaction_date', 'description', 'amount', 'extended_details', 
                                      'statement_description', 'category', 'is_manually_categorized',
                                      'confidence', 'model_version', 'model_filename', 'source_file', 'timestamp'
                                      # appears_on_your_statement_as is mapped to statement_description if needed
                                 ]
                                 df_for_db = pd.DataFrame(columns=db_columns) # Ensure all DB columns exist
                                 for col in db_columns:
                                     if col in final_internal_df.columns:
                                         df_for_db[col] = final_internal_df[col]
                                     else:
                                         # Add missing columns with default null values (or handle appropriately)
                                         if col in ['confidence', 'model_version', 'model_filename', 'extended_details', 'statement_description']:
                                             df_for_db[col] = None 
                                         # Handle other potential missing cols if needed
                                         
                                 # Ensure correct types before storing
                                 df_for_db['amount'] = pd.to_numeric(df_for_db['amount'], errors='coerce')
                                 df_for_db['confidence'] = pd.to_numeric(df_for_db['confidence'], errors='coerce')
                                 df_for_db['transaction_date'] = pd.to_datetime(df_for_db['transaction_date'], errors='coerce').dt.date
                                 df_for_db['is_manually_categorized'] = df_for_db['is_manually_categorized'].astype(bool)
                                 
                                 # Drop rows with NaN in essential fields after conversion
                                 df_for_db.dropna(subset=['transaction_date', 'description', 'amount', 'category', 'is_manually_categorized', 'source_file'], inplace=True)

                                 # 5. Store in Database
                                 store_finalized_transactions(df_for_db)
                                 st.success(f"Successfully submitted {len(df_for_db)} transactions to the database!")
                                 
                                 # Optional: Clean up output file after successful submission
                                 try:
                                     if st.session_state.output_file_path.exists():
                                          st.session_state.output_file_path.unlink()
                                          print(f"Removed temporary output file: {st.session_state.output_file_path.name}")
                                 except Exception as e:
                                     st.warning(f"Could not remove temporary output file {st.session_state.output_file_path.name}: {e}")
                                     
                             except Exception as e:
                                 st.error(f"Error submitting to database: {str(e)}")
                                 st.exception(e)
            else:
                 st.info("Load or categorize a file to review transactions.")

# Analytics Tab (Using Database)
with tab2:
    st.header("Database Analytics")
    try:
        engine = get_engine()
        db_data = pd.read_sql("SELECT * FROM transactions", engine)
        
        if db_data.empty:
            st.warning("No data found in the transaction database. Submit some finalized transactions first.")
        else:
            st.info(f"Displaying analytics based on {len(db_data)} records from the database.")
            # Convert types for plotting
            db_data['transaction_date'] = pd.to_datetime(db_data['transaction_date'])
            db_data['amount'] = pd.to_numeric(db_data['amount'])

            # Filters for DB data
            st.sidebar.divider()
            st.sidebar.header("Database Filters")
            min_db_date = db_data['transaction_date'].min().date()
            max_db_date = db_data['transaction_date'].max().date()
            start_db_date = st.sidebar.date_input("Start Date (DB)", value=min_db_date, min_value=min_db_date, max_value=max_db_date, key="db_start")
            end_db_date = st.sidebar.date_input("End Date (DB)", value=max_db_date, min_value=min_db_date, max_value=max_db_date, key="db_end")
            
            db_categories = sorted(db_data['category'].unique().tolist())
            selected_db_categories = st.sidebar.multiselect("Categories (DB)", options=db_categories, default=db_categories, key="db_cats")
            
            # Filter DB data
            filtered_db_data = db_data[
                (db_data['transaction_date'].dt.date >= start_db_date) & 
                (db_data['transaction_date'].dt.date <= end_db_date) &
                (db_data['category'].isin(selected_db_categories))
            ].copy() # Create a copy to avoid SettingWithCopyWarning

            # Basic Stats
            st.subheader("Summary Metrics (Database)")
            exp_col, inc_col, net_col = st.columns(3)
            db_expenses = filtered_db_data[filtered_db_data['amount'] < 0]['amount'].sum()
            db_income = filtered_db_data[filtered_db_data['amount'] > 0]['amount'].sum()
            exp_col.metric("Total Expenses", f"${abs(db_expenses):,.2f}")
            inc_col.metric("Total Income", f"${db_income:,.2f}")
            net_col.metric("Net Flow", f"${db_income + db_expenses:,.2f}")

            # Plots
            # Category distribution (Expenses Only)
            st.subheader("Expense Distribution by Category (Database)")
            expenses_db = filtered_db_data[filtered_db_data['amount'] < 0].copy()
            expenses_db['amount'] = expenses_db['amount'].abs()
            category_summary_db = expenses_db.groupby('category')['amount'].sum().reset_index().sort_values('amount', ascending=False)

            if not category_summary_db.empty:
                fig_pie_db = px.pie(
                    category_summary_db,
                    values='amount',
                    names='category',
                    title="Expense Category Distribution"
                )
                st.plotly_chart(fig_pie_db, use_container_width=True)
            
            # Monthly spending by category
            st.subheader("Monthly Spending by Category (Database)")
            expenses_db['month_year'] = expenses_db['transaction_date'].dt.to_period('M').astype(str)
            monthly_spending_db = expenses_db.groupby(['month_year', 'category'])['amount'].sum().reset_index()
            
            if not monthly_spending_db.empty:
                fig_bar_db = px.bar(
                    monthly_spending_db,
                    x='month_year',
                    y='amount',
                    color='category',
                    title="Monthly Expenses by Category",
                    labels={'amount': 'Total Amount ($)', 'month_year': 'Month'},
                    category_orders={"month_year": sorted(monthly_spending_db['month_year'].unique())}
                )
                st.plotly_chart(fig_bar_db, use_container_width=True)

            # Data Provenance
            st.subheader("Data Provenance")
            prov_col1, prov_col2 = st.columns(2)
            manual_count = filtered_db_data['is_manually_categorized'].sum()
            model_count = len(filtered_db_data) - manual_count
            prov_col1.metric("Manually Categorized", manual_count)
            prov_col2.metric("Model Categorized", model_count)
            
            st.write("Model Versions Used (in filtered data):")
            st.dataframe(filtered_db_data[filtered_db_data['model_version'].notna()]['model_version'].value_counts())

    except Exception as e:
        st.error(f"Error loading or displaying analytics from database: {e}")
        st.exception(e) 