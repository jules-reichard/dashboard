import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from transformers import pipeline
from datetime import date
import plotly.express as px

# Optional: uncomment if you have a FRED API key
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

    data = {}
    for name, symbol in tickers.items():
        try:
            df = yf.download(symbol, period="6mo", progress=False)
            if not df.empty:
                data[name] = df["Close"]
            else:
                st.warning(f"No data for {name} ({symbol})")
        except Exception as e:
            st.warning(f"Error fetching {name}: {e}")

    if not data:
        st.error("No market data could be retrieved. Please try again later.")
        return pd.DataFrame()

    df_all = pd.concat(data, axis=1)
    df_all.columns = df_all.columns.droplevel(0) if isinstance(df_all.columns, pd.MultiIndex) else df_all.columns
    return df_all

@st.cache_data(ttl=3600)
def get_interest_rate():
    # Placeholder value — replace with FRED if needed
    # fred = Fred(api_key=st.secrets["FRED_KEY"])
    # rate = fred.get_series_latest_release("DGS10")[-1]
    # return rate
    return 4.25  # Static fallback rate

def get_top_news(api_key):
    url = f'https://newsapi.org/v2/top-headlines?category=business&pageSize=10&apiKey={api_key}'
    try:
        res = requests.get(url)
        res.raise_for_status()
        articles = res.json().get('articles', [])
        return [a['title'] for a in articles]
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

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

if not data.empty:
    st.line_chart(data)

    st.subheader("📉 Correlation Matrix")
    corr = data.corr()
    st.dataframe(corr.style.background_gradient(cmap='coolwarm', axis=None))

    st.subheader("📈 VIX vs S&P 500 Relationship")
    vix_sp_corr = corr.loc["VIX", "S&P 500"]
    st.metric("Correlation (VIX ↔ S&P 500)", f"{vix_sp_corr:.2f}")

else:
    st.warning("No data available to display charts.")

# Interest Rates
rate = get_interest_rate()
st.metric("10-Year Treasury Rate (%)", rate)

# --- News Sentiment ---
st.header("🗞️ News Sentiment Analysis")
api_key = st.text_input("Enter your NewsAPI key:", type="password")

if api_key:
    with st.spinner("Fetching latest business news..."):
        headlines = get_top_news(api_key)
        if headlines:
            st.write("**Top 10 Headlines:**")
            for h in headlines:
                st.write(f"- {h}")

            results, avg_sentiment, success_percent = analyze_sentiment(headlines)

            st.subheader("Sentiment Summary")
            col1, col2 = st.columns(2)
            col1.metric("Average Sentiment", f"{avg_sentiment:.2f}")
            col2.metric("Positive Forecast (%)", f"{success_percent:.1f}%")

            results_df = pd.DataFrame(results)
            st.write("Detailed Results", results_df)

            fig = px.bar(
                results_df,
                x=results_df.index,
                y="score",
                color="label",
                title="Sentiment by Article",
                text="label"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No news articles were returned.")
else:
    st.info("Please enter your NewsAPI key above to fetch news sentiment.")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ in Streamlit • Data from Yahoo Finance, NewsAPI, and Transformers")
