import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="Market Dashboard", layout="wide")
st.title("📊 Market Intelligence Dashboard")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("Controls")
tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "IBM"]
selected_ticker = st.sidebar.selectbox("Select Company", tickers)

# Set default date range
start_date_default = datetime.now() - timedelta(days=365)
end_date_default = datetime.now()

start_date = st.sidebar.date_input("Start Date", start_date_default)
end_date = st.sidebar.date_input("End Date", end_date_default)

# Create tabs
tabs = st.tabs(["📈 Stock Prices", "💰 Financial Statements", "📰 Sentiment Analysis"])

# ------------------------------
# 1️⃣ STOCK PRICE TAB
# ------------------------------
with tabs[0]:
    st.subheader(f"{selected_ticker} Stock Performance")

    @st.cache_data
    def load_data(ticker, start, end):
        # Ensure start date is before end date
        if start > end:
            st.error("Error: Start date must be before end date.")
            return pd.DataFrame()

        # Ensure end date is not in the future (yfinance constraint)
        if end > datetime.now().date():
            end = datetime.now().date() - timedelta(days=1)
        
        # Download stock data
        try:
            df = yf.download(ticker, start=start, end=end + timedelta(days=1)) # Add 1 day to end to include it
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            return pd.DataFrame()
            
        # Fallback if data is empty for the selected range
        if df.empty and (start != start_date_default or end != end_date_default):
            st.warning("No data found for the selected date range. Fetching last 1 year of data as fallback.")
            df = yf.download(ticker, period="1y")
            
        return df

    data = load_data(selected_ticker, start_date, end_date)

    if not data.empty:
        # Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close', line=dict(color='dodgerblue')))
        
        fig.update_layout(
            title=f"{selected_ticker} Closing Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Descriptive Statistics
        st.markdown("#### Descriptive Statistics")
        st.dataframe(data.describe())
    else:
        st.warning("No stock data available to display.")

# ------------------------------
# 2️⃣ FINANCIAL STATEMENTS TAB
# ------------------------------
with tabs[1]:
    st.subheader("💰 Company Financial Statements")

    @st.cache_data
    def get_financials(ticker):
        stock = yf.Ticker(ticker)
        return stock.income_stmt, stock.balance_sheet, stock.cashflow

    try:
        IS, BS, CF = get_financials(selected_ticker)

        st.markdown("### 🧾 Income Statement")
        st.dataframe(IS)

        st.markdown("### 📋 Balance Sheet")
        st.dataframe(BS)

        st.markdown("### 💵 Cash Flow Statement")
        st.dataframe(CF)

    except Exception as e:
        st.error(f"Error loading financial statements: {e}")
        st.warning("Note: yfinance may occasionally have trouble fetching financial statements for some tickers.")

# ------------------------------
# 3️⃣ SENTIMENT ANALYSIS TAB
# ------------------------------
with tabs[2]:
    st.subheader("📰 Market Sentiment (CNBC News)")

    @st.cache_data
    def fetch_cnbc_headlines():
        url = "https://www.cnbc.com/id/100003114/device/rss/rss.html"
        try:
            feed = feedparser.parse(url)
            headlines = pd.DataFrame({
                "Title": [entry.title for entry in feed.entries[:15]],
                "Link": [entry.link for entry in feed.entries[:15]]
            })
            return headlines
        except Exception as e:
            st.error(f"Error fetching RSS feed: {e}")
            return pd.DataFrame()

    headlines = fetch_cnbc_headlines()

    if not headlines.empty:
        analyzer = SentimentIntensityAnalyzer()

        headlines["Sentiment Score"] = headlines["Title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
        headlines["Sentiment"] = headlines["Sentiment Score"].apply(lambda x:
                                                                    "Positive" if x > 0.2 else
                                                                    "Negative" if x < -0.2 else
                                                                    "Neutral")

        avg_sentiment = headlines["Sentiment Score"].mean()

        st.metric("Average Sentiment (Recent CNBC Headlines)", f"{avg_sentiment:.2f}")
        
        # Display DataFrame with clickable links
        st.dataframe(
            headlines[["Title", "Sentiment", "Sentiment Score", "Link"]],
            column_config={
                "Link": st.column_config.LinkColumn("Article Link", display_text="Read Article")
            }
        )
        st.caption("News source: CNBC RSS Feed")
    else:
        st.warning("Could not fetch news headlines for sentiment analysis.")
