import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

# --- UI/UX Configurations ---
st.set_page_config(page_title="AI Sales Forecaster", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Sales Forecaster")
st.write("Upload your historical sales dataset, map your columns, and instantly generate machine learning forecasts.")

# --- Sidebar settings ---
st.sidebar.header("⚙️ Configuration Panel")
uploaded_file = st.sidebar.file_uploader("Upload Sales Data (CSV format)", type=['csv'])

if uploaded_file:
    try:
        # Load Data
        df = pd.read_csv(uploaded_file)
        
        # Dynamic Column Mapping
        st.sidebar.subheader("Map Your Columns")
        date_col = st.sidebar.selectbox("Select Date Column", df.columns)
        target_col = st.sidebar.selectbox("Select Target (Sales) Column", df.columns)
        
        forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=365, value=30)
        
        # Preprocessing Data
        df[date_col] = pd.to_datetime(df[date_col])
        daily_sales = df.groupby(date_col)[target_col].sum().reset_index()
        daily_sales = daily_sales.sort_values(by=date_col)
        
        # --- UI Tabs ---
        tab1, tab2, tab3 = st.tabs(["📋 Data Overview", "📊 Exploratory Analysis", "🤖 ML Forecasting"])
        
        with tab1:
            st.subheader("Raw Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", f"{len(df):,}")
            col2.metric("Total Sales Volume", f"{daily_sales[target_col].sum():,.2f}")
            col3.metric("Dataset Timeframe", f"{daily_sales[date_col].dt.date.min()} to {daily_sales[date_col].dt.date.max()}")
            
        with tab2:
            st.subheader("Historical Sales Trend")
            fig = px.line(daily_sales, x=date_col, y=target_col, template="plotly_white")
            fig.update_traces(line_color="#1f77b4")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            st.subheader("Generate Future Predictions")
            
            if st.button("Train Model & Forecast"):
                with st.spinner("Training Random Forest Algorithm..."):
                    
                    
                    def create_features(data, date_column):
                        data['Year'] = data[date_column].dt.year
                        data['Month'] = data[date_column].dt.month
                        data['Day'] = data[date_column].dt.day
                        data['DayOfWeek'] = data[date_column].dt.dayofweek
                        data['DayOfYear'] = data[date_column].dt.dayofyear
                        return data
                    
                    ml_data = create_features(daily_sales.copy(), date_col)
                    
                    X = ml_data[['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear']]
                    y = ml_data[target_col]
                    
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    
                    last_date = daily_sales[date_col].max()
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon)
                    future_df = pd.DataFrame({date_col: future_dates})
                    future_df = create_features(future_df, date_col)
                    
                    future_X = future_df[['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear']]
                    future_df['Predicted_Sales'] = model.predict(future_X)
                    
                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(x=daily_sales[date_col], y=daily_sales[target_col], mode='lines', name='Historical', line=dict(color='#1f77b4')))
                    fig_forecast.add_trace(go.Scatter(x=future_df[date_col], y=future_df['Predicted_Sales'], mode='lines', name='Forecast', line=dict(color='#ff7f0e', dash='dash')))
                    fig_forecast.update_layout(title="Sales Forecast vs Historical Trend", xaxis_title="Date", yaxis_title="Sales", template="plotly_white")
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    st.subheader("Export Results")
                    csv = future_df[[date_col, 'Predicted_Sales']].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Forecast as CSV", data=csv, file_name="sales_forecast.csv", mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Ensure your CSV has at least one valid Date column and one Numeric column for sales.")
else:
    st.info("👋 Welcome! Please upload your CSV file in the sidebar to get started.")