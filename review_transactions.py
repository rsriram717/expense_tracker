import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import os
from datetime import datetime
from improved_categorizer import train_model, categorize_transactions

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
st.markdown("Review and manage your transaction categories with ease.")

# Initialize session state for output file and categorization state
if 'output_file' not in st.session_state:
    st.session_state.output_file = None
if 'categorization_run' not in st.session_state:
    st.session_state.categorization_run = False
if 'reload_categorized' not in st.session_state:
    st.session_state.reload_categorized = False

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Run Categorization", type="primary"):
        with st.spinner("Training model and categorizing transactions..."):
            try:
                # Train the model
                train_model()
                # Categorize transactions
                categorize_transactions()
                # Set flag to reload categorized data
                st.session_state.categorization_run = True
                st.session_state.reload_categorized = True
                st.success("Categorization complete!")
            except Exception as e:
                st.error(f"Error during categorization: {str(e)}")

# Main content area
tab1, tab2 = st.tabs(["Review Transactions", "Analytics"])

with tab1:
    # Load transactions
    transactions_path = Path("data/to_categorize")
    if not transactions_path.exists():
        st.error("No transactions found. Please add transactions to the 'data/to_categorize' directory.")
    else:
        # Get list of transaction files
        transaction_files = list(transactions_path.glob("*.csv"))
        if not transaction_files:
            st.error("No CSV files found in the transactions directory.")
        else:
            # File selector
            selected_file = st.selectbox(
                "Select Transaction File",
                options=transaction_files,
                format_func=lambda x: x.name
            )
            
            # Load and display transactions
            if selected_file:
                try:
                    # Read the CSV file
                    df = pd.read_csv(selected_file)
                    
                    # Convert date column to datetime if it exists
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    
                    # Define column name mappings (exact matches)
                    column_mappings = {
                        'date': 'Date',
                        'description': 'Description',
                        'amount': 'Amount',
                        'extended details': 'Extended Details',
                        'appears on your statement as': 'Appears On Your Statement As',
                        'category': 'Category'
                    }
                    
                    # Rename columns to match our expected format
                    df = df.rename(columns=column_mappings)
                    
                    # Filter out "MOBILE PAYMENT - THANK YOU" transactions
                    df = df[~df['Description'].str.contains('MOBILE PAYMENT - THANK YOU', case=False, na=False)]
                    
                    # Set up path for saving the output, but don't load from it
                    output_file = Path("data/output") / f"improved_categorized_{selected_file.name}"
                    st.session_state.output_file = output_file
                    
                    # Check if we should load categorized data after running the model
                    if st.session_state.reload_categorized and output_file.exists():
                        df = pd.read_csv(output_file)
                        df.columns = df.columns.str.lower() # Ensure lowercase column names
                        # Reset the flag so we don't reload again until categorization is run
                        st.session_state.reload_categorized = False
                    
                    # Add confidence column if it doesn't exist
                    if 'confidence' not in df.columns:
                        df['confidence'] = 1.0
                    
                    # Add category column if it doesn't exist
                    if 'category' not in df.columns:
                        df['category'] = None  # Start with no categories
                    
                    # Define categories
                    categories = [
                        "Food & Drink", "Transportation", "Entertainment", "Groceries",
                        "Shopping", "Travel-Airline", "Travel-Lodging", "Travel-Other",
                        "Clothes", "Subscriptions", "Home", "Pets", "Beauty",
                        "Professional Services", "Medical", "Misc"
                    ]
                    
                    # Select and order columns
                    columns = [
                        "date", "description", "amount", "extended details",
                        "appears on your statement as", "category", "confidence"
                    ]
                    
                    # Convert column names to lowercase for comparison
                    df.columns = df.columns.str.lower()
                    
                    # Ensure all required columns exist
                    missing_cols = [col for col in columns if col not in df.columns]
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                        st.write("Available columns:", ", ".join(df.columns))
                    else:
                        # Create an editable dataframe with only the specified columns
                        edited_df = st.data_editor(
                            df[columns],
                            column_config={
                                "date": st.column_config.TextColumn(
                                    "Date",
                                    width="medium",
                                ),
                                "description": st.column_config.TextColumn(
                                    "Description",
                                    width="medium",
                                ),
                                "amount": st.column_config.NumberColumn(
                                    "Amount",
                                    format="$%.2f",
                                ),
                                "extended details": st.column_config.TextColumn(
                                    "Extended Details",
                                    width="medium",
                                ),
                                "appears on your statement as": st.column_config.TextColumn(
                                    "Statement Description",
                                    width="medium",
                                ),
                                "category": st.column_config.SelectboxColumn(
                                    "Category",
                                    options=categories,
                                    required=True,
                                ),
                                "confidence": st.column_config.NumberColumn(
                                    "Confidence",
                                    format="%.2f",
                                    disabled=True,
                                ),
                            },
                            hide_index=True,
                            use_container_width=True,
                        )
                        
                        # Save and Submit buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Save button
                            if st.button("💾 Save Changes"):
                                try:
                                    # Save the edited dataframe
                                    edited_df.to_csv(output_file, index=False)
                                    st.success("Changes saved successfully!")
                                except Exception as e:
                                    st.error(f"Error saving changes: {str(e)}")
                        
                        with col2:
                            # Submit button
                            if st.button("📤 Submit to Training Data"):
                                try:
                                    # Create categorized directory if it doesn't exist
                                    categorized_dir = Path("data/categorized")
                                    categorized_dir.mkdir(parents=True, exist_ok=True)
                                    
                                    # Create filename for categorized data
                                    categorized_file = categorized_dir / f"categorized_{selected_file.name}"
                                    
                                    # Save the edited dataframe to categorized directory
                                    edited_df.to_csv(categorized_file, index=False)
                                    st.success("Successfully submitted to training data!")
                                except Exception as e:
                                    st.error(f"Error submitting to training data: {str(e)}")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")

with tab2:
    # Analytics section
    if st.session_state.output_file is not None and st.session_state.output_file.exists():
        try:
            df = pd.read_csv(st.session_state.output_file)
            
            if 'category' not in df.columns:
                st.warning("No categorized transactions available. Please run categorization first.")
            else:
                # Category distribution
                st.subheader("Category Distribution")
                category_counts = df['category'].value_counts()
                if not category_counts.empty:
                    fig = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title="Transaction Categories Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Monthly spending by category
                st.subheader("Monthly Spending by Category")
                if 'date' in df.columns and 'amount' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    monthly_spending = df.groupby([df['date'].dt.to_period('M'), 'category'])['amount'].sum().reset_index()
                    monthly_spending['date'] = monthly_spending['date'].astype(str)
                    
                    fig = px.bar(
                        monthly_spending,
                        x='date',
                        y='amount',
                        color='category',
                        title="Monthly Spending by Category",
                        barmode='stack'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confidence distribution
                if 'confidence' in df.columns:
                    st.subheader("Model Confidence Distribution")
                    fig = px.histogram(
                        df,
                        x='confidence',
                        nbins=20,
                        title="Distribution of Model Confidence Scores"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating analytics: {str(e)}")
    else:
        st.info("Run the categorization first to see analytics.") 