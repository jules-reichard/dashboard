# app.py

# 1. Libaries
import pandas as pd
import streamlit as st
import plotly.express as px
import yfinance as yf
import datetime as dt
from datetime import datetime, timedelta
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="💹",
    layout="wide"
)

# --- DATA LOADING AND CACHING ---
# Use caching to avoid re-loading data on every interaction
@st.cache_data
def load_stock_data():
    # Defining date ranges
    today = datetime.today()
    ten_years_ago = today - timedelta(days=365 * 10)

    tickers = {
        'S&P 500': '^GSPC',
        'VIX': '^VIX',
        'NASDAQ 100': '^NDX',
        'CAC 40': '^FCHI',
        'Nikkei 225': '^N225',
        'Hang Seng': '^HSI'
    }
    
    # Use a single yf.download call for efficiency
    ticker_string = " ".join(tickers.values())
    raw_data = yf.download(ticker_string, start=ten_years_ago, end=today)
    
    # We only need the 'Close' prices for this dashboard
    close_prices = raw_data['Close'].copy()
    
    # Rename columns to be more friendly (e.g., '^GSPC' -> 'S&P 500')
    column_map = {v: k for k, v in tickers.items()}
    close_prices.rename(columns=column_map, inplace=True)
    
    # Handle potential missing values by forward-filling
    close_prices.ffill(inplace=True)
    
    return close_prices

# Function to fetch and analyze news sentiment
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_news_sentiment():
    rss_url = "https://www.cnbc.com/id/100003114/device/rss/rss.html"
    feed = feedparser.parse(rss_url)
    
    N = 30 # Number of headlines
    titles = [entry.title for entry in feed.entries[:N]]
    news_df = pd.DataFrame(titles, columns=['Title'])

    analyzer = SentimentIntensityAnalyzer()

    def get_vader_sentiment(text):
        score = analyzer.polarity_scores(text)['compound']
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    news_df['Sentiment'] = news_df['Title'].apply(get_vader_sentiment)
    return news_df


# --- APP LAYOUT ---

# Load the data
df_close = load_stock_data()

st.title("📈 Financial Markets Dashboard")
st.write("Data sourced from Yahoo Finance and CNBC over the last 10 years.")

# --- DISPLAY CHARTS AND DATA ---

st.header("Index Price Evolution (Normalized)")

# Normalize the data to compare performance
df_normalized = df_close / df_close.iloc[0]

# User selection for the main chart
indices_to_plot = st.multiselect(
    "Select indices to display:",
    options=df_normalized.columns,
    default=['S&P 500', 'NASDAQ 100', 'CAC 40']
)

if indices_to_plot:
    # Use Plotly for interactive charts
    fig = px.line(df_normalized[indices_to_plot], title='Normalized Close Prices')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please select at least one index to display the chart.")


# --- SENTIMENT ANALYSIS SECTION ---
st.header("📰 Latest Financial News Sentiment")
st.write("Sentiment analysis of the latest 30 headlines from CNBC's World News RSS feed.")

news_df = get_news_sentiment()
sentiment_counts = news_df['Sentiment'].value_counts()

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Sentiment Distribution")
    # Pie chart for sentiment distribution
    fig_sentiment = px.pie(
        sentiment_counts, 
        values=sentiment_counts.values, 
        names=sentiment_counts.index,
        color=sentiment_counts.index,
        color_discrete_map={'Positive':'green', 'Negative':'red', 'Neutral':'grey'}
    )
    fig_sentiment.update_layout(showlegend=False)
    st.plotly_chart(fig_sentiment, use_container_width=True)

with col2:
    st.subheader("Recent Headlines")
    # Display headlines with sentiment
    st.dataframe(
        news_df, 
        use_container_width=True,
        column_config={
            "Sentiment": st.column_config.TextColumn(
                "Sentiment",
            )
        },
        hide_index=True
    )

# You can add more sections from your notebook here, like the annual returns calculation
st.header("Annualized Returns & Volatility")
annual_returns = df_close.pct_change().mean() * 252
annual_volatility = df_close.pct_change().std() * (252**0.5)

summary_df = pd.DataFrame({
    'Annualized Return': annual_returns,
    'Annualized Volatility': annual_volatility
})

st.dataframe(summary_df.style.format("{:.2%}"))
