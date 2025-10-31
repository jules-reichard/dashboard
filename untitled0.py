import streamlit as st
import pandas as pd
import numpy as np  # Added for calculations
import plotly.graph_objects as go
import plotly.express as px  # Added for the pie chart
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
# This selection is for the Financial Statements tab
selected_ticker = st.sidebar.selectbox("Select Company (for Financials)", tickers)

# Set default date range
start_date_default = datetime.now() - timedelta(days=365)
end_date_default = datetime.now()

# Create tabs
tabs = st.tabs(["📈 Stock Prices", "💰 Financial Statements", "📰 Sentiment Analysis"])

# ------------------------------
# 1️⃣ STOCK PRICE TAB
# ------------------------------
with tabs[0]:
    st.subheader("Comparative Stock Performance") # Changed title

    # --- Date pickers moved from sidebar to this tab ---
    st.markdown("Select Date Range:")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", start_date_default)
    with col2:
        end_date = st.date_input("End Date", end_date_default)
    # --- End of moved date pickers ---

    @st.cache_data
    def load_all_data(start, end): # Changed function name
        """
        Loads data for all tickers in the global 'tickers' list.
        """
        # Ensure start date is before end date
        if start > end:
            st.error("Error: Start date must be before end date.")
            return pd.DataFrame()

        # Ensure end date is not in the future (yfinance constraint)
        if end > datetime.now().date():
            end = datetime.now().date() - timedelta(days=1)
        
        # Download stock data for all tickers
        try:
            # tickers list is defined in the sidebar section
            df = yf.download(tickers, start=start, end=end + timedelta(days=1)) # Add 1 day to end to include it
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            return pd.DataFrame()
            
        # Fallback if data is empty for the selected range
        if df.empty and (start != start_date_default or end != end_date_default):
            st.warning("No data found for the selected date range. Fetching last 1 year of data as fallback.")
            df = yf.download(tickers, period="1y")
            
        return df

    all_data = load_all_data(start_date, end_date) # Updated function call

    if not all_data.empty:
        # Get just the 'Close' prices
        close_data = all_data['Close']
        
        # Melt the dataframe to long format for Plotly Express
        # This makes it easy to plot multiple lines
        df_to_plot = close_data.reset_index().melt(id_vars='Date', value_name='Price', var_name='Ticker')

        # Plotly chart
        fig = px.line(
            df_to_plot,
            x='Date',
            y='Price',
            color='Ticker', # Creates a different line for each Ticker
            title='Closing Prices Comparison'
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Descriptive Statistics
        st.markdown("#### Descriptive Statistics (Close Prices)")
        st.dataframe(close_data.describe()) # Describe only the close prices
        
        # --- Annualized Metrics ---
        st.markdown("#### Annualized Metrics")
        
        # Calculate daily returns (for volatility)
        daily_returns = close_data.pct_change()
        
        N = 252  # Number of trading days in a year
        
        # --- Updated Annualized Return Calculation ---
        # Use geometric mean (based on start/end price) for a more accurate annualized return
        num_days = len(close_data)
        total_return_ratio = (close_data.iloc[-1] / close_data.iloc[0])
        annualized_returns = (total_return_ratio ** (N / num_days) - 1) * 100
        # --- End of Updated Calculation ---
        
        # Calculate annualized volatility (standard formula)
        annualized_vol = daily_returns.std() * np.sqrt(N) * 100
        
        # Combine into a DataFrame
        annual_stats_df = pd.DataFrame({
            'Annualized Return': annualized_returns,
            'Annualized Volatility': annualized_vol
        })
        
        # Display as formatted percentages
        st.dataframe(annual_stats_df.style.format("{:.2f}%"))

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

    # Helper function to add % change columns and format in thousands
    def format_financials_df(df):
        """
        Formats the financial statement dataframe to show values in thousands (k)
        and adds year-over-year percentage change columns.
        """
        if df.empty:
            return df, {}
        
        # yfinance columns are descending, so periods=-1 compares to the previous year (next column)
        df_pct = df.pct_change(axis=1, periods=-1)
        
        # Create a new dataframe to hold interleaved columns
        combined_df = pd.DataFrame()
        formatter = {}
        
        for col in df.columns:
            col_name_date = col.strftime('%Y-%m-%d')
            pct_col_name = f"{col.year} % Chg"
            
            # Divide value columns by 1000 for 'k' format
            combined_df[col_name_date] = df[col] / 1000
            # Format as thousands with 'k' suffix
            formatter[col_name_date] = lambda x: f'{x:,.0f}k' if pd.notna(x) else '-' 
            
            if col in df_pct.columns:
                combined_df[pct_col_name] = df_pct[col]
                formatter[pct_col_name] = '{:,.2%}' # Format as percentage
                
        return combined_df, formatter


    try:
        IS, BS, CF = get_financials(selected_ticker)

        # --- Income Statement ---
        st.markdown("### 🧾 Income Statement")
        if not IS.empty:
            is_display, is_formatter = format_financials_df(IS)
            st.dataframe(is_display.style.format(na_rep='-', formatter=is_formatter))
        else:
            st.warning("No Income Statement data available.")

        # --- Balance Sheet ---
        st.markdown("### 📋 Balance Sheet")
        if not BS.empty:
            bs_display, bs_formatter = format_financials_df(BS)
            st.dataframe(bs_display.style.format(na_rep='-', formatter=bs_formatter))
        else:
            st.warning("No Balance Sheet data available.")

        # --- Cash Flow Statement ---
        st.markdown("### 💵 Cash Flow Statement")
        if not CF.empty:
            cf_display, cf_formatter = format_financials_df(CF)
            st.dataframe(cf_display.style.format(na_rep='-', formatter=cf_formatter))
        else:
            st.warning("No Cash Flow Statement data available.")

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

        # --- Added Pie Chart ---
        sentiment_counts = headlines["Sentiment"].value_counts().reset_index(name='Count')
        
        fig_pie = px.pie(
            sentiment_counts, 
            names='Sentiment', 
            values='Count', 
            title='Sentiment Distribution',
            color='Sentiment',
            color_discrete_map={
                'Positive': 'seagreen', 
                'Negative': 'tomato', 
                'Neutral': 'silver'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # --- End of Added Pie Chart ---

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

