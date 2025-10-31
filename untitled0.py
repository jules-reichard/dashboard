import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import os
import json

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
    st.subheader("Comparative Stock Performance")

    # --- Date pickers moved from sidebar to this tab ---
    st.markdown("Select Date Range:")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", start_date_default)
    with col2:
        end_date = st.date_input("End Date", end_date_default)
    # --- End of moved date pickers ---

    # --- Add Ticker Selector ---
    # tickers list is defined in the sidebar section
    selected_tickers_chart = st.multiselect(
        "Select Stocks to Compare",
        options=tickers,
        default=tickers  # Default to all tickers
    )
    # --- End of Ticker Selector ---

    @st.cache_data
    def load_all_data(tickers_to_load, start, end): # Modified function signature
        """
        Loads data for all tickers passed in the 'tickers_to_load' list.
        """
        if not tickers_to_load:
             # Don't show a warning if the user just hasn't selected anything yet
             return pd.DataFrame() # Return empty if no tickers are selected
             
        # Ensure start date is before end date
        if start > end:
            st.error("Error: Start date must be before end date.")
            return pd.DataFrame()

        # Ensure end date is not in the future (yfinance constraint)
        if end > datetime.now().date():
            end = datetime.now().date() - timedelta(days=1)
        
        # Download stock data for all tickers
        try:
            # yfinance downloads up to (but not including) end date
            df = yf.download(tickers_to_load, start=start, end=end + timedelta(days=1)) 
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            return pd.DataFrame()
            
        # Fallback if data is empty for the selected range
        if df.empty and (start.date() != start_date_default.date() or end.date() != end_date_default.date()):
            st.warning("No data found for the selected date range. Fetching last 1 year of data as fallback.")
            df = yf.download(tickers_to_load, period="1y") # Use passed list
            
        return df

    # Check if tickers are selected before loading
    if selected_tickers_chart:
        all_data = load_all_data(selected_tickers_chart, start_date, end_date) # Updated function call

        if not all_data.empty:
            
            # --- Handle single vs. multiple tickers for 'Close' data ---
            if len(selected_tickers_chart) == 1:
                # If only one ticker, 'Close' is a Series. Convert to DataFrame.
                close_data = all_data[['Close']]
                close_data.columns = [selected_tickers_chart[0]] # Name the column correctly
            else:
                # If multiple tickers, 'Close' is already a DataFrame
                close_data = all_data['Close']
            # --- End of handle ---
            
            # Drop rows with any NaN values (e.g., holidays) before plotting
            close_data = close_data.dropna()

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
            daily_returns = close_data.pct_change().dropna()
            
            N = 252  # Number of trading days in a year
            
            # --- Annualized Return Calculation (Arithmetic Mean) ---
            annualized_returns = daily_returns.mean().apply(lambda x: ((1 + x)**N - 1) * 100)
            # --- End of Reverted Calculation ---
            
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
            if selected_tickers_chart: # Only show this if tickers were selected but no data came back
                st.warning("No stock data available to display for the selected tickers/date range.")
    else:
        # This message shows if the multiselect box is empty
        st.info("Select one or more tickers above to see the comparison chart and metrics.")


# ------------------------------
# 2️⃣ FINANCIAL STATEMENTS TAB
# ------------------------------
with tabs[1]:
    st.subheader(f"💰 {selected_ticker} Financial Statements")

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
            # --- Interactive Chart ---
            is_plot_df = IS.T.reset_index().sort_values(by='index')
            is_plot_df = is_plot_df.rename(columns={'index': 'Year'})
            is_plot_df['Year'] = is_plot_df['Year'].astype(str)
            is_metrics = is_plot_df.columns.drop('Year').tolist()
            default_is_metrics = [m for m in ['Total Revenue', 'Gross Profit', 'Net Income'] if m in is_metrics]
            
            selected_is_metrics = st.multiselect(
                "Select Income Statement metrics to plot",
                options=is_metrics,
                default=default_is_metrics,
                key="is_multiselect"
            )
            
            if selected_is_metrics:
                is_melted = is_plot_df.melt(id_vars='Year', value_vars=selected_is_metrics, var_name='Metric', value_name='Amount (k)')
                is_melted['Amount (k)'] = is_melted['Amount (k)'] / 1000  # Format in k
                
                fig_is = px.line(
                    is_melted, 
                    x='Year', 
                    y='Amount (k)', 
                    color='Metric', 
                    title=f"{selected_ticker} Income Statement Trends (in thousands)",
                    markers=True
                )
                st.plotly_chart(fig_is, use_container_width=True)
            # --- End Chart ---

            is_display, is_formatter = format_financials_df(IS)
            st.dataframe(is_display.style.format(na_rep='-', formatter=is_formatter))
        else:
            st.warning("No Income Statement data available.")

        # --- Balance Sheet ---
        st.markdown("### 📋 Balance Sheet")
        if not BS.empty:
            # --- Interactive Chart ---
            bs_plot_df = BS.T.reset_index().sort_values(by='index')
            bs_plot_df = bs_plot_df.rename(columns={'index': 'Year'})
            bs_plot_df['Year'] = bs_plot_df['Year'].astype(str)
            bs_metrics = bs_plot_df.columns.drop('Year').tolist()
            default_bs_metrics = [m for m in ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity'] if m in bs_metrics]

            selected_bs_metrics = st.multiselect(
                "Select Balance Sheet metrics to plot",
                options=bs_metrics,
                default=default_bs_metrics,
                key="bs_multiselect"
            )
            
            if selected_bs_metrics:
                bs_melted = bs_plot_df.melt(id_vars='Year', value_vars=selected_bs_metrics, var_name='Metric', value_name='Amount (k)')
                bs_melted['Amount (k)'] = bs_melted['Amount (k)'] / 1000
                
                fig_bs = px.line(
                    bs_melted, 
                    x='Year', 
                    y='Amount (k)', 
                    color='Metric', 
                    title=f"{selected_ticker} Balance Sheet Trends (in thousands)",
                    markers=True
                )
                st.plotly_chart(fig_bs, use_container_width=True)
            # --- End Chart ---

            bs_display, bs_formatter = format_financials_df(BS)
            st.dataframe(bs_display.style.format(na_rep='-', formatter=bs_formatter))
        else:
            st.warning("No Balance Sheet data available.")

        # --- Cash Flow Statement ---
        st.markdown("### 💵 Cash Flow Statement")
        if not CF.empty:
            # --- Interactive Chart ---
            cf_plot_df = CF.T.reset_index().sort_values(by='index')
            cf_plot_df = cf_plot_df.rename(columns={'index': 'Year'})
            cf_plot_df['Year'] = cf_plot_df['Year'].astype(str)
            cf_metrics = cf_plot_df.columns.drop('Year').tolist()
            default_cf_metrics = [m for m in ['Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditure'] if m in cf_metrics]

            selected_cf_metrics = st.multiselect(
                "Select Cash Flow metrics to plot",
                options=cf_metrics,
                default=default_cf_metrics,
                key="cf_multiselect"
            )

            if selected_cf_metrics:
                cf_melted = cf_plot_df.melt(id_vars='Year', value_vars=selected_cf_metrics, var_name='Metric', value_name='Amount (k)')
                cf_melted['Amount (k)'] = cf_melted['Amount (k)'] / 1000
                
                fig_cf = px.line(
                    cf_melted, 
                    x='Year', 
                    y='Amount (k)', 
                    color='Metric', 
                    title=f"{selected_ticker} Cash Flow Trends (in thousands)",
                    markers=True
                )
                st.plotly_chart(fig_cf, use_container_width=True)
            # --- End Chart ---

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

# --- Sentiment History Functions ---
HISTORY_FILE = "sentiment_history.json"

@st.cache_data(ttl=3600) # Cache the feed fetch for 1 hour
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

def load_sentiment_history():
    """Loads sentiment history from the JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_sentiment_history(history):
    """Saves the updated sentiment history to the JSON file."""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)
    except IOError as e:
        st.error(f"Error saving sentiment history: {e}")

def get_current_sentiment(headlines_df):
    """Calculates the current average sentiment from headlines."""
    if headlines_df.empty:
        return None, None
        
    analyzer = SentimentIntensityAnalyzer()
    headlines_df["Sentiment Score"] = headlines_df["Title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    headlines_df["Sentiment"] = headlines_df["Sentiment Score"].apply(lambda x:
                                                                "Positive" if x > 0.2 else
                                                                "Negative" if x < -0.2 else
                                                                "Neutral")
    avg_sentiment = headlines_df["Sentiment Score"].mean()
    return avg_sentiment, headlines_df

# --- End Sentiment History Functions ---


with tabs[2]:
    st.subheader("📰 Market Sentiment (CNBC News)")

    headlines = fetch_cnbc_headlines()
    
    if not headlines.empty:
        current_avg_sentiment, headlines_with_sentiment = get_current_sentiment(headlines)
        
        if current_avg_sentiment is not None:
            st.metric("Average Sentiment (Recent CNBC Headlines)", f"{current_avg_sentiment:.2f}")

            # --- Historical Sentiment Logic ---
            sentiment_history = load_sentiment_history()
            
            # Get today's date as a string
            today_str = datetime.now().strftime('%Y-%m-%d')
            
            # Check if we already saved today's score
            today_score_exists = any(entry['Date'] == today_str for entry in sentiment_history)
            
            if not today_score_exists:
                # Add the new score and timestamp
                new_entry = {
                    "Date": today_str,
                    "Timestamp": datetime.now().isoformat(),
                    "Score": current_avg_sentiment
                }
                sentiment_history.append(new_entry)
                save_sentiment_history(sentiment_history)
            
            # --- Display Historical Sentiment Chart ---
            if sentiment_history:
                history_df = pd.DataFrame(sentiment_history)
                history_df['Date'] = pd.to_datetime(history_df['Date'])
                history_df = history_df.sort_values(by='Date')
                
                fig_history = px.line(
                    history_df,
                    x='Date',
                    y='Score',
                    title='Historical Sentiment Trend (Updated Daily)',
                    markers=True
                )
                fig_history.update_layout(yaxis_title='Average Sentiment Score')
                st.plotly_chart(fig_history, use_container_width=True)
            # --- End Historical Chart ---

            # --- Added Pie Chart ---
            sentiment_counts = headlines_with_sentiment["Sentiment"].value_counts().reset_index(name='Count')
            
            fig_pie = px.pie(
                sentiment_counts, 
                names='Sentiment', 
                values='Count', 
                title='Recent Sentiment Distribution',
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
                headlines_with_sentiment[["Title", "Sentiment", "Sentiment Score", "Link"]],
                column_config={
                    "Link": st.column_config.LinkColumn("Article Link", display_text="Read Article")
                }
            )
            st.caption("News source: CNBC RSS Feed")
        else:
            st.warning("Could not calculate sentiment.")
    else:
        st.warning("Could not fetch news headlines for sentiment analysis.")

