import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Market Dashboard", layout="wide")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("Dashboard Controls")
tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "IBM"]
selected_ticker = st.sidebar.selectbox("Select Company", tickers)

start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

# ------------------------------
# Fetch stock data
# ------------------------------
@st.cache_data
def load_data(ticker):
    return yf.download(ticker, start=start_date, end=end_date)

data = load_data(selected_ticker)

# ------------------------------
# Price Chart
# ------------------------------
st.title(f"{selected_ticker} Market Dashboard")

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
fig.update_layout(title=f"{selected_ticker} Stock Price", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Financial Statements
# ------------------------------
st.header("📊 Financial Statements")

@st.cache_data
def get_financials(ticker):
    stock = yf.Ticker(ticker)
    return stock.income_stmt, stock.balance_sheet, stock.cashflow

try:
    IS, BS, CF = get_financials(selected_ticker)

    st.subheader("Income Statement")
    st.dataframe(IS.style.format("{:.2f}"))

    st.subheader("Balance Sheet")
    st.dataframe(BS.style.format("{:.2f}"))

    st.subheader("Cash Flow Statement")
    st.dataframe(CF.style.format("{:.2f}"))

except Exception as e:
    st.error(f"Error loading financial statements: {e}")

# ------------------------------
# Basic Statistics
# ------------------------------
st.header("📈 Key Statistics")
st.write(data.describe())
