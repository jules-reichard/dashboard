import pandas as pd
import streamlit as st
import plotly.express as px
import yfinance as yf
from datetime import datetime
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Financial Market Dashboard",
    layout="wide"
)

st.title("Financial Market Dashboard 📈")

# --- DATA FETCHING ---
# Dictionary of tickers we discussed
tickers = {
    'S&P 500': '^GSPC',
    'Nasdaq 100': '^NDX',
    'VIX': '^VIX',
    'BTC-USD': 'BTC-USD',
    '10-Year Treasury Yield': '^TNX',
    '2-Year Treasury Yield': '^FVX',
    'CAC 40 (France)': '^FCHI',
    'DAX (Germany)': '^GDAXI',
    'Nikkei 225 (Japan)': '^N225',
    'Hang Seng (Hong Kong)': '^HSI',
    'Crude Oil': 'CL=F',
    'Gold': 'GC=F'
}

# Use Streamlit's caching to avoid re-downloading data on every interaction
@st.cache_data
def load_data(start_date, end_date):
    """Downloads historical data for all tickers."""
    try:
        data = yf.download(list(tickers.values()), start=start_date, end=end_date)
        if data.empty:
            st.error("No data downloaded. Check ticker symbols and date range.")
            return None, None

        # --- FIX STARTS HERE ---
        # Gracefully handle that some tickers (like '^VIX') only have 'Close'.
        # We will prioritize 'Adj Close' but fill any missing values with 'Close'.
        if 'Adj Close' in data.columns:
            combined_data = data['Adj Close'].combine_first(data['Close'])
        else:
            combined_data = data['Close']
        # --- FIX ENDS HERE ---

        # Rename columns to be user-friendly
        combined_data.columns = tickers.keys()
        
        # Calculate daily returns for AI analysis
        returns = combined_data.pct_change().dropna()
        
        return combined_data.dropna(how='all'), returns
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return None, None

# --- UI TABS ---
tab1, tab2 = st.tabs(["📊 Market Data & Charts", "🤖 AI Analyst Summary"])

# ==============================================================================
# TAB 1: MARKET DATA & CHARTS
# ==============================================================================
with tab1:
    st.header("Market Performance and Volatility Analysis")

    # --- Sidebar for user inputs ---
    st.sidebar.header("Dashboard Controls")
    start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", "today")

    # --- Load Data ---
    data, returns = load_data(start_date, end_date)

    if data is not None:
        # --- Chart 1: Normalized Performance Comparison ---
        st.subheader("Normalized Performance Comparison")
        # Normalize data to show growth from a common starting point (100)
        normalized_data = (data / data.iloc[0]) * 100
        
        assets_to_plot = st.multiselect(
            "Select assets to compare:",
            options=data.columns,
            default=['S&P 500', 'Nasdaq 100', 'BTC-USD', 'Gold']
        )
        if assets_to_plot:
            fig1 = px.line(normalized_data[assets_to_plot], title="Asset Growth Over Time (Normalized to 100)")
            st.plotly_chart(fig1, use_container_width=True)
        
        # --- Chart 2: S&P 500 vs. VIX (Fear Index) ---
        st.subheader("S&P 500 vs. VIX (Fear Index)")
        fig2 = px.line(data, x=data.index, y=['S&P 500', 'VIX'], 
                       title='S&P 500 vs. Volatility Index (VIX)',
                       labels={'value': 'Price / Index Level'})
        # Use a secondary y-axis for VIX since its scale is different
        fig2.update_traces(yaxis="y2", selector=dict(name='VIX'))
        st.plotly_chart(fig2, use_container_width=True)

        # --- Chart 3: Yield Curve Spread ---
        st.subheader("10-Year vs. 2-Year Treasury Yield Spread (Recession Indicator)")
        data['Yield Spread'] = data['10-Year Treasury Yield'] - data['2-Year Treasury Yield']
        fig3 = px.area(data, x=data.index, y='Yield Spread', title='Yield Curve Spread (10Y - 2Y)')
        fig3.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Inversion Threshold")
        st.plotly_chart(fig3, use_container_width=True)

        # --- Display Raw Data ---
        with st.expander("Show Raw Data Table"):
            st.dataframe(data.style.format("{:.2f}"))

# ==============================================================================
# TAB 2: AI ANALYST SUMMARY
# ==============================================================================
with tab2:
    st.header("Automated Market Analysis")
    st.write("This tab uses a Generative AI model to analyze the latest market data and provide a summary.")

    # --- Function to get AI summary (using Google's Gemini) ---
    def get_ai_summary(data_context):
        try:
            # Use st.secrets for API key management in deployed apps
            genai.configure(api_key=st.secrets["AIzaSyBGdozAwaLxw1fjKfxQEpKWLF_671KjYh4"])
            model = genai.GenerativeModel('gemini-pro')

            prompt = f"""
            You are an expert financial market analyst providing a daily briefing.
            Based ONLY on the data provided below, write a concise summary of the current market sentiment.

            Your analysis must cover:
            1. The general behavior of the stock market (using S&P 500 as the benchmark).
            2. The current market fear level (using the VIX). A VIX above 20 suggests heightened fear.
            3. The economic outlook based on the 10-Year Treasury Yield. Rising yields can signal inflation fears or economic strength, often acting as a headwind for stocks.
            4. Conclude by describing the relationship between these three indicators based on today's data (e.g., "Healthy Bull Market", "Inflationary Fear", "Panic/Crisis", or "Optimistic Recovery").

            **Today's Data:**
            {data_context}

            **Your Analysis:**
            """

            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Could not retrieve AI analysis. Error: {e}. Please ensure your GOOGLE_API_KEY is correctly set in secrets.toml."

    # --- Button to trigger the analysis ---
    if st.button("🤖 Analyze Current Market Situation"):
        if data is not None and not data.empty:
            with st.spinner("AI Analyst is processing the latest data..."):
                # Prepare the data context string for the AI
                latest_data = data.iloc[-1]
                latest_returns = returns.iloc[-1]
                
                context = f"""
                - Date: {latest_data.name.strftime('%Y-%m-%d')}
                - S&P 500 Close: {latest_data['S&P 500']:.2f} (Daily Change: {latest_returns['S&P 500']:.2%})
                - VIX Close: {latest_data['VIX']:.2f} (Daily Change: {latest_returns['VIX']:.2%})
                - 10-Year Treasury Yield: {latest_data['10-Year Treasury Yield']:.2f}% (Daily Change: {latest_returns['10-Year Treasury Yield']:.2%})
                """
                
                summary = get_ai_summary(context)
                st.markdown(summary)
        else:
            st.warning("Data could not be loaded. Please check your inputs on the 'Market Data' tab.")
