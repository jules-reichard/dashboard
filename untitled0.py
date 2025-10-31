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

start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

tabs = st.tabs(["📈 Stock Prices", "💰 Financial Statements", "📰 Sentiment Analysis"])

# ------------------------------
# 1️⃣ STOCK PRICE TAB
# ------------------------------
with tabs[0]:
    st.subheader(f"{selected_ticker} Stock Performance")

    @st.cache_data
    def load_data(ticker, start, end):
        df = yf.download(ticker, start=start, end=end)
        return df

    data = load_data(selected_ticker, start_date, end_date)

    if not data.empty:
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

        st.markdown("#### Descriptive Statistics")
        st.dataframe(data.describe())
    else:
        st.warning("No data found for this date range.")

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

# ------------------------------
# 3️⃣ SENTIMENT ANALYSIS TAB
# ------------------------------
with tabs[2]:
    st.subheader("📰 Market Sentiment (CNBC News)")

    @st.cache_data
    def fetch_cnbc_headlines():
        url = "https://www.cnbc.com/id/100003114/device/rss/rss.html"
        feed = feedparser.parse(url)
        return pd.DataFrame({
            "Title": [entry.title for entry in feed.entries[:15]],
            "Link": [entry.link for entry in feed.entries[:15]]
        })

    headlines = fetch_cnbc_headlines()
    analyzer = SentimentIntensityAnalyzer()

    headlines["Sentiment Score"] = headlines["Title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    headlines["Sentiment"] = headlines["Sentiment Score"].apply(lambda x:
                                                                "Positive" if x > 0.2 else
                                                                "Negative" if x < -0.2 else
                                                                "Neutral")

    avg_sentiment = headlines["Sentiment Score"].mean()

    st.metric("Average Sentiment (CNBC)", f"{avg_sentiment:.2f}")
    st.dataframe(headlines[["Title", "Sentiment", "Sentiment Score"]])

    st.caption("News source: CNBC RSS Feed")

