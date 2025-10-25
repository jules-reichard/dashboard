import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from transformers import pipeline
from datetime import date
import plotly.express as px

# Optional: uncomment if using FRED
# from fredapi import Fred

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Financial Dashboard",
    layout="wide",
    page_icon="📈"
)

# Custom background (dark theme)
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0D1117 0%, #1B2838 100%);
        color: white;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# FUNCTIONS
# -----------------------------

@st.cache_data(ttl=3600)
def get_market_data():
    tickers = {
        "S&P 500": "^GSPC",
        "VIX": "^VIX",
        "NASDAQ 100": "^NDX",
        "CAC 40": "^FCHI",
        "BTC": "BTC-USD",
        "Nikkei 225": "^N225",
        "Hang Seng": "^HSI"
    }
    data = {name: yf.download(symbol, period="6mo")["Close"] for name, symbol in tickers.items()}
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def get_interest_rate():
    # Placeholder value if FRED not configured
    # For live data, use FRED API (uncomment below and add your key)
    # fred = Fred(api_key=st.secrets["FRED_KEY"])
    # rate = fred.get_series_latest_release("DGS10")[-1]
    # return rate
    return 4.25  # Example static interest rate

def get_top_news(api_key):
    url = f'https://newsapi.org/v2/top-headlines?category=business&pageSize=10&apiKey={api_key}'
    articles = requests.get(url).json().get('articles', [])
    return [a['title'] for a in articles]

def analyze_sentiment(headlines):
    if not headlines:
        return [], 0, 0
    sentiment_pipeline = pipeline("sentiment-analysis")
    results = sentiment_pipeline(headlines)
    scores = [r['score'] if r['label'] == 'POSITIVE' else -r['score'] for r in results]
    avg_sentiment = np.mean(scores)
    success_percent = (sum(s > 0 for s in scores) / len(scores)) * 100
    return results, avg_sentiment, success_percent

# -----------------------------
# DASHBOARD
# -----------------------------

st.title("📊 Global Financial Dashboard")
st.caption(f"Updated: {date.today().strftime('%B %d, %Y')}")

# --- Market Data ---
st.header("Market Overview")
data = get_market_data()
st.line_chart(data)

st.subheader("📉 Correlation Matrix")
corr = data.corr()
st.dataframe(corr.style.background_gradient(cmap='coolwarm', axis=None))

# VIX and Market Relationship
st.subheader("📈 VIX vs S&P 500 Relationship")
vix_sp_corr = corr.loc["VIX", "S&P 500"]
st.metric("Correlation (VIX ↔ S&P 500)", f"{vix_sp_corr:.2f}")

# Interest Rates
rate = get_interest_rate()
st.metric("10-Year Treasury Rate (%)", rate)

# --- News Sentiment ---
st.header("🗞️ News Sentiment Analysis")
api_key = st.text_input("Enter your NewsAPI key:", type="password")

if api_key:
    with st.spinner("Fetching latest business news..."):
        headlines = get_top_news(api_key)
        st.write("**Top 10 Headlines:**")
        for h in headlines:
            st.write(f"- {h}")

        results, avg_sentiment, success_percent = analyze_sentiment(headlines)

        st.subheader("Sentiment Summary")
        col1, col2 = st.columns(2)
        col1.metric("Average Sentiment", f"{avg_sentiment:.2f}")
        col2.metric("Positive Forecast (%)", f"{success_percent:.1f}%")

        # Show details
        results_df = pd.DataFrame(results)
        st.write("Detailed Results", results_df)

        # Visualization
        fig = px.bar(results_df, x=results_df.index, y="score", color="label",
                     title="Sentiment by Article", text="label")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please enter your NewsAPI key above to fetch news sentiment.")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ in Streamlit • Data from Yahoo Finance, NewsAPI, and Transformers")
