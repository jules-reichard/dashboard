# app.py
# --- LIBRARIES ---
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="💹",
    layout="wide"
)

st.title("💹 Financial Markets Intelligence Dashboard")
st.write("Data sourced from **Yahoo Finance** and **CNBC RSS feeds**, updated daily.")

# --- DATA CACHING FUNCTIONS ---
@st.cache_data
def load_market_data():
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
    data = yf.download(list(tickers.values()), start=ten_years_ago, end=today)['Close']
    data.rename(columns={v: k for k, v in tickers.items()}, inplace=True)
    data.ffill(inplace=True)
    return data

@st.cache_data
def load_company_data():
    companies = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'IBM']
    IS, BS, CF = {}, {}, {}
    for c in companies:
        t = yf.Ticker(c)
        IS[c] = t.financials.T.pct_change()
        BS[c] = t.balance_sheet.T.pct_change()
        CF[c] = t.cashflow.T.pct_change()
    return IS, BS, CF

@st.cache_data(ttl=3600)
def get_news_sentiment():
    rss_url = "https://www.cnbc.com/id/100003114/device/rss/rss.html"
    feed = feedparser.parse(rss_url)
    titles = [entry.title for entry in feed.entries[:30]]
    news_df = pd.DataFrame(titles, columns=['Title'])
    analyzer = SentimentIntensityAnalyzer()
    def get_sentiment(text):
        score = analyzer.polarity_scores(text)['compound']
        if score >= 0.05: return 'Positive'
        elif score <= -0.05: return 'Negative'
        return 'Neutral'
    news_df['Sentiment'] = news_df['Title'].apply(get_sentiment)
    return news_df

# --- LOAD DATA ---
df_close = load_market_data()
IS, BS, CF = load_company_data()
news_df = get_news_sentiment()

# --- TABS ---
tabs = st.tabs(["📈 Market Overview", "🏦 Financial Statements", "🤖 PCA Clustering", "📰 News Sentiment"])

# --- TAB 1: MARKET OVERVIEW ---
with tabs[0]:
    st.header("Normalized Market Indices")
    df_norm = df_close / df_close.iloc[0]
    indices = st.multiselect("Select indices", df_norm.columns, default=['S&P 500','NASDAQ 100','CAC 40'])
    if indices:
        fig = px.line(df_norm[indices], title="Normalized Close Prices", labels={'value':'Normalized Price','index':'Date'})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Annualized Returns & Volatility")
    returns = df_close.pct_change()
    annual_ret = returns.mean() * 252
    annual_vol = returns.std() * (252**0.5)
    summary = pd.DataFrame({'Annual Return': annual_ret, 'Volatility': annual_vol}).sort_values('Annual Return', ascending=False)
    st.dataframe(summary.style.format("{:.2%}"))

# --- TAB 2: FINANCIAL STATEMENTS ---
with tabs[1]:
    st.header("Company Financial Growth (YoY %)")
    company = st.selectbox("Select a company", list(IS.keys()))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Income Statement")
        st.dataframe(IS[company].style.format("{:.2%}").background_gradient(cmap="RdYlGn"))
    with col2:
        st.subheader("Balance Sheet")
        st.dataframe(BS[company].style.format("{:.2%}").background_gradient(cmap="PuBuGn"))
    with col3:
        st.subheader("Cash Flow")
        st.dataframe(CF[company].style.format("{:.2%}").background_gradient(cmap="OrRd"))

# --- TAB 3: PCA + CLUSTERING ---
with tabs[2]:
    st.header("Market Structure Analysis (PCA + KMeans)")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_close.pct_change().dropna().T)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(reduced)
    cluster_df = pd.DataFrame(reduced, columns=['PC1','PC2'])
    cluster_df['Index'] = df_close.columns
    cluster_df['Cluster'] = clusters.astype(str)
    fig_pca = px.scatter(cluster_df, x='PC1', y='PC2', color='Cluster', text='Index',
                         title="PCA Clustering of Market Indices", color_discrete_sequence=px.colors.qualitative.Set2)
    fig_pca.update_traces(textposition='top center')
    st.plotly_chart(fig_pca, use_container_width=True)

# --- TAB 4: NEWS SENTIMENT ---
with tabs[3]:
    st.header("Latest CNBC Financial News Sentiment")
    counts = news_df['Sentiment'].value_counts()
    col1, col2 = st.columns([1,2])
    with col1:
        fig_sent = px.pie(values=counts.values, names=counts.index, title="Sentiment Distribution",
                          color=counts.index, color_discrete_map={'Positive':'green','Negative':'red','Neutral':'grey'})
        st.plotly_chart(fig_sent, use_container_width=True)
    with col2:
        st.dataframe(news_df, hide_index=True)
