import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from datetime import datetime, timedelta
import glob

# Set page config for a modern look
st.set_page_config(
    page_title="Transaction Dashboard",
    page_icon="📊",
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
st.title("Transaction Analytics Dashboard")
st.markdown("Analyze your spending patterns and track financial trends.")

def load_data():
    # Find all CSV files in the data/categorized directory
    csv_files = glob.glob("data/categorized/*.csv")
    
    if not csv_files:
        st.error("No CSV files found in data/categorized directory.")
        return None
    
    # Load all data files
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            filename = os.path.basename(file)
            df['Source'] = filename  # Add source file as a column
            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    
    if not dfs:
        return None
    
    # Combine all data
    data = pd.concat(dfs, ignore_index=True)
    
    # Convert date column to datetime
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    
    # Make sure all column names are properly capitalized
    data.columns = [col if col.lower() in ['date', 'category', 'amount', 'source'] 
                    else col for col in data.columns]
    
    return data

# Load data
data = load_data()

if data is not None:
    # Sidebar for filters
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = data['Date'].min().date()
    max_date = data['Date'].max().date()
    
    # Default to last 3 months if possible
    default_start = max(min_date, (datetime.now() - timedelta(days=90)).date())
    
    start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    
    # Category filter
    categories = sorted(data['Category'].unique().tolist())
    selected_categories = st.sidebar.multiselect(
        "Categories",
        options=categories,
        default=categories
    )
    
    # Card member filter if available
    if 'Card Member' in data.columns:
        card_members = sorted(data['Card Member'].unique().tolist())
        selected_card_members = st.sidebar.multiselect(
            "Card Members",
            options=card_members,
            default=card_members
        )
    else:
        selected_card_members = None
    
    # Home/Personal filter if available
    if 'Home' in data.columns:
        home_options = sorted(data['Home'].unique().tolist())
        selected_home = st.sidebar.multiselect(
            "Home/Personal",
            options=home_options,
            default=home_options
        )
    else:
        selected_home = None
    
    # Apply filters
    filtered_data = data.copy()
    filtered_data = filtered_data[
        (filtered_data['Date'].dt.date >= start_date) & 
        (filtered_data['Date'].dt.date <= end_date) &
        (filtered_data['Category'].isin(selected_categories))
    ]
    
    if selected_card_members is not None:
        filtered_data = filtered_data[filtered_data['Card Member'].isin(selected_card_members)]
    
    if selected_home is not None:
        filtered_data = filtered_data[filtered_data['Home'].isin(selected_home)]
    
    # Display summary metrics
    st.subheader("Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate positive (income) and negative (expenses) transaction sums
    expenses = filtered_data[filtered_data['Amount'] < 0]['Amount'].sum()
    income = filtered_data[filtered_data['Amount'] > 0]['Amount'].sum()
    
    with col1:
        st.metric("Total Transactions", len(filtered_data))
    with col2:
        st.metric("Total Expenses", f"${abs(expenses):.2f}")
    with col3:
        st.metric("Total Income", f"${income:.2f}")
    with col4:
        net = income + expenses  # expenses is negative, so add to get net
        st.metric("Net", f"${net:.2f}", delta=f"${net:.2f}")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Spending by Category", "Time Trends", "Transaction List", "Insights"])
    
    with tab1:
        st.subheader("Spending by Category")
        
        # Prepare data for category analysis
        expenses_by_category = filtered_data[filtered_data['Amount'] < 0].groupby('Category')['Amount'].sum().abs().reset_index()
        expenses_by_category = expenses_by_category.sort_values('Amount', ascending=False)
        
        # Category visualization (pie chart and bar chart)
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                expenses_by_category, 
                values='Amount', 
                names='Category',
                title="Expense Distribution by Category",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(
                expenses_by_category.head(10), 
                x='Category', 
                y='Amount', 
                title="Top 10 Expense Categories",
                color='Category',
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_bar.update_layout(xaxis_title="Category", yaxis_title="Amount ($)")
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.subheader("Time Trends")
        
        # Time based analysis
        time_unit = st.radio(
            "Time Grouping",
            options=["Day", "Week", "Month"],
            horizontal=True,
            index=2
        )
        
        if time_unit == "Day":
            filtered_data['Time_Period'] = filtered_data['Date'].dt.date
        elif time_unit == "Week":
            filtered_data['Time_Period'] = filtered_data['Date'].dt.to_period('W').astype(str)
        else:  # Month
            filtered_data['Time_Period'] = filtered_data['Date'].dt.to_period('M').astype(str)
        
        # Spending over time
        spending_over_time = filtered_data.groupby(['Time_Period', 'Category'])['Amount'].sum().reset_index()
        spending_over_time = spending_over_time[spending_over_time['Amount'] < 0]  # Only expenses
        spending_over_time['Amount'] = spending_over_time['Amount'].abs()  # Make positive for visualization
        
        # Line chart
        fig_line = px.line(
            spending_over_time.groupby('Time_Period')['Amount'].sum().reset_index(),
            x='Time_Period',
            y='Amount',
            title=f"Total Spending Over Time (by {time_unit})",
            markers=True
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Stacked bar chart
        fig_bar = px.bar(
            spending_over_time,
            x='Time_Period',
            y='Amount',
            color='Category',
            title=f"Spending by Category Over Time (by {time_unit})",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        st.subheader("Transaction List")
        
        # Allow sorting and searching
        sort_by = st.selectbox(
            "Sort by",
            options=["Date (newest first)", "Date (oldest first)", "Amount (highest first)", "Amount (lowest first)"],
            index=0
        )
        
        search_term = st.text_input("Search in Description")
        
        # Apply sorting and filtering
        display_data = filtered_data.copy()
        
        if search_term:
            display_data = display_data[display_data['Description'].str.contains(search_term, case=False, na=False)]
        
        if sort_by == "Date (newest first)":
            display_data = display_data.sort_values("Date", ascending=False)
        elif sort_by == "Date (oldest first)":
            display_data = display_data.sort_values("Date", ascending=True)
        elif sort_by == "Amount (highest first)":
            display_data = display_data.sort_values("Amount", ascending=False)
        else:  # Amount (lowest first)
            display_data = display_data.sort_values("Amount", ascending=True)
        
        # Select columns to display
        display_columns = ['Date', 'Description', 'Amount', 'Category']
        if 'Card Member' in display_data.columns:
            display_columns.append('Card Member')
        if 'Home' in display_data.columns:
            display_columns.append('Home')
        
        st.dataframe(
            display_data[display_columns],
            use_container_width=True,
            column_config={
                "Date": st.column_config.DateColumn("Date"),
                "Amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                "Description": st.column_config.TextColumn("Description", width="large")
            }
        )
    
    with tab4:
        st.subheader("Spending Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top merchants by spend
            merchants = filtered_data[filtered_data['Amount'] < 0].copy()
            # Extract merchant name from description (simplistic approach)
            merchants['Merchant'] = merchants['Description'].str.split().str[:2].str.join(' ')
            merchant_spending = merchants.groupby('Merchant')['Amount'].sum().abs().reset_index()
            merchant_spending = merchant_spending.sort_values('Amount', ascending=False).head(10)
            
            fig_merchant = px.bar(
                merchant_spending,
                x='Amount',
                y='Merchant',
                orientation='h',
                title="Top 10 Merchants by Spend",
                color='Amount',
                color_continuous_scale='Viridis'
            )
            fig_merchant.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_merchant, use_container_width=True)
        
        with col2:
            # Weekday analysis
            if 'Date' in filtered_data.columns:
                filtered_data['Weekday'] = filtered_data['Date'].dt.day_name()
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekday_spending = filtered_data[filtered_data['Amount'] < 0].groupby('Weekday')['Amount'].sum().abs().reindex(weekday_order).reset_index()
                
                fig_weekday = px.bar(
                    weekday_spending,
                    x='Weekday',
                    y='Amount',
                    title="Spending by Day of Week",
                    color='Amount',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_weekday, use_container_width=True)
        
        # Category trends over time
        st.subheader("Category Trends")
        
        # Select top categories to analyze
        top_categories = expenses_by_category.head(5)['Category'].tolist()
        selected_trend_categories = st.multiselect(
            "Select categories to analyze trends",
            options=categories,
            default=top_categories[:3]
        )
        
        if selected_trend_categories:
            trend_data = spending_over_time[spending_over_time['Category'].isin(selected_trend_categories)]
            
            fig_trend = px.line(
                trend_data,
                x='Time_Period',
                y='Amount',
                color='Category',
                title="Spending Trends by Selected Categories",
                markers=True
            )
            st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.error("Unable to load transaction data. Please check the data/categorized directory.")

# Run the app with: streamlit run dashboard.py 