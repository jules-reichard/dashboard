# -*- coding: utf-8 -*-
"""
Dashboard Financier - Comparaison d'actifs
Version améliorée avec métriques, drawdowns, actualités et multi-actifs
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="Dashboard Financier",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .stMetric:hover {
        background-color: #e0e2e6;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTES ET CONFIGURATION
# =============================================================================
ASSETS = {
    "Cryptomonnaies": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "SOL-USD": "Solana",
        "BNB-USD": "Binance Coin"
    },
    "Indices": {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "^DJI": "Dow Jones",
        "^FCHI": "CAC 40",
        "^STOXX50E": "Euro Stoxx 50"
    },
    "Actions Tech": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "NVDA": "NVIDIA",
        "META": "Meta"
    },
    "Matières premières": {
        "GC=F": "Or",
        "SI=F": "Argent",
        "CL=F": "Pétrole WTI"
    }
}

# Créer une liste plate pour les selectbox
ALL_ASSETS = {}
for category, assets in ASSETS.items():
    ALL_ASSETS.update(assets)

TICKER_TO_NAME = {ticker: name for ticker, name in ALL_ASSETS.items()}
NAME_TO_TICKER = {name: ticker for ticker, name in ALL_ASSETS.items()}

# =============================================================================
# FONCTIONS DE CHARGEMENT DE DONNÉES
# =============================================================================
@st.cache_data(ttl=3600)  # Cache d'une heure
def load_asset_data(ticker: str, years: int) -> pd.DataFrame:
    """Charge les données historiques d'un actif."""
    today = datetime.today()
    start_date = today - timedelta(days=365 * years)
    
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=today)
        if data.empty:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement de {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_asset_info(ticker: str) -> dict:
    """Récupère les informations détaillées d'un actif."""
    try:
        info = yf.Ticker(ticker).info
        return info
    except:
        return {}


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalise une série (z-score)."""
    return (series - series.mean()) / series.std()


def calculate_returns(prices: pd.Series) -> float:
    """Calcule le rendement total en pourcentage."""
    if len(prices) < 2:
        return 0.0
    return ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100


def calculate_volatility(prices: pd.Series, annualize: bool = True) -> float:
    """Calcule la volatilité (écart-type des rendements)."""
    returns = prices.pct_change().dropna()
    vol = returns.std()
    if annualize:
        vol *= np.sqrt(252)  # Annualisation
    return vol * 100


def calculate_drawdown(prices: pd.Series) -> pd.Series:
    """Calcule le drawdown depuis le plus haut historique."""
    peak = prices.cummax()
    drawdown = (prices - peak) / peak * 100
    return drawdown


def calculate_sharpe_ratio(prices: pd.Series, risk_free_rate: float = 0.04) -> float:
    """Calcule le ratio de Sharpe simplifié."""
    returns = prices.pct_change().dropna()
    excess_return = returns.mean() * 252 - risk_free_rate
    volatility = returns.std() * np.sqrt(252)
    if volatility == 0:
        return 0
    return excess_return / volatility


@st.cache_data(ttl=3600)
def get_news(ticker: str) -> list:
    """Récupère les actualités via RSS Yahoo Finance."""
    try:
        feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}")
        return feed.entries[:5]
    except:
        return []


# =============================================================================
# SIDEBAR - PARAMÈTRES
# =============================================================================
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    st.subheader("Sélection des actifs")
    
    # Premier actif
    asset1_name = st.selectbox(
        "Premier actif",
        options=list(NAME_TO_TICKER.keys()),
        index=list(NAME_TO_TICKER.keys()).index("Bitcoin")
    )
    asset1_ticker = NAME_TO_TICKER[asset1_name]
    
    # Deuxième actif
    asset2_name = st.selectbox(
        "Deuxième actif",
        options=list(NAME_TO_TICKER.keys()),
        index=list(NAME_TO_TICKER.keys()).index("S&P 500")
    )
    asset2_ticker = NAME_TO_TICKER[asset2_name]
    
    st.subheader("Période")
    years = st.slider("Nombre d'années", min_value=1, max_value=10, value=5)
    
    st.subheader("Options d'affichage")
    show_normalized = st.checkbox("Afficher les prix normalisés", value=True)
    show_drawdown = st.checkbox("Afficher les drawdowns", value=True)
    show_volume = st.checkbox("Afficher les volumes", value=False)
    
    st.markdown("---")
    
    if st.button("🔄 Rafraîchir les données", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("Dashboard créé avec Streamlit")
    st.caption(f"Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}")


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================
st.title("📈 Dashboard Financier")
st.markdown(f"**Comparaison:** {asset1_name} vs {asset2_name} sur {years} ans")

# Chargement avec spinner
with st.spinner("Chargement des données..."):
    df1 = load_asset_data(asset1_ticker, years)
    df2 = load_asset_data(asset2_ticker, years)

# Vérification des données
if df1.empty or df2.empty:
    st.error("Impossible de charger les données. Veuillez réessayer plus tard.")
    st.stop()


# =============================================================================
# MÉTRIQUES CLÉS
# =============================================================================
st.header("📊 Métriques clés")

col1, col2, col3, col4 = st.columns(4)

# Rendement Asset 1
return1 = calculate_returns(df1['Close'])
with col1:
    st.metric(
        label=f"Rendement {asset1_name}",
        value=f"{return1:.1f}%",
        delta=f"{return1:.1f}%" if return1 != 0 else None
    )

# Rendement Asset 2
return2 = calculate_returns(df2['Close'])
with col2:
    st.metric(
        label=f"Rendement {asset2_name}",
        value=f"{return2:.1f}%",
        delta=f"{return2:.1f}%" if return2 != 0 else None
    )

# Volatilité Asset 1
vol1 = calculate_volatility(df1['Close'])
with col3:
    st.metric(
        label=f"Volatilité {asset1_name}",
        value=f"{vol1:.1f}%"
    )

# Volatilité Asset 2
vol2 = calculate_volatility(df2['Close'])
with col4:
    st.metric(
        label=f"Volatilité {asset2_name}",
        value=f"{vol2:.1f}%"
    )

# Deuxième ligne de métriques
col5, col6, col7, col8 = st.columns(4)

# Prix actuels
with col5:
    current_price1 = df1['Close'].iloc[-1]
    st.metric(
        label=f"Prix actuel {asset1_name}",
        value=f"${current_price1:,.2f}"
    )

with col6:
    current_price2 = df2['Close'].iloc[-1]
    st.metric(
        label=f"Prix actuel {asset2_name}",
        value=f"${current_price2:,.2f}"
    )

# Sharpe Ratio
with col7:
    sharpe1 = calculate_sharpe_ratio(df1['Close'])
    st.metric(
        label=f"Sharpe {asset1_name}",
        value=f"{sharpe1:.2f}"
    )

with col8:
    sharpe2 = calculate_sharpe_ratio(df2['Close'])
    st.metric(
        label=f"Sharpe {asset2_name}",
        value=f"{sharpe2:.2f}"
    )


# =============================================================================
# GRAPHIQUE PRINCIPAL - COMPARAISON
# =============================================================================
st.markdown("---")
st.header("📈 Comparaison des performances")

# Préparation des données pour le graphique
df1_plot = df1[['Date', 'Close']].copy()
df2_plot = df2[['Date', 'Close']].copy()

if show_normalized:
    df1_plot['Close'] = normalize_series(df1['Close'])
    df2_plot['Close'] = normalize_series(df2['Close'])
    y_label = "Prix normalisé (z-score)"
else:
    y_label = "Prix ($)"

# Fusion des données
df1_plot = df1_plot.rename(columns={'Close': asset1_name})
df2_plot = df2_plot.rename(columns={'Close': asset2_name})

merged = pd.merge(df1_plot, df2_plot, on='Date', how='inner')
merged_melted = merged.melt(id_vars=['Date'], var_name='Actif', value_name='Prix')

# Création du graphique
fig_main = px.line(
    merged_melted,
    x='Date',
    y='Prix',
    color='Actif',
    title=f"{'Prix normalisés' if show_normalized else 'Prix'}: {asset1_name} vs {asset2_name}",
    color_discrete_map={asset1_name: '#FF6B6B', asset2_name: '#4ECDC4'},
    template='plotly_white'
)

fig_main.update_layout(
    xaxis_title="Date",
    yaxis_title=y_label,
    legend_title="Actif",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig_main, use_container_width=True)


# =============================================================================
# GRAPHIQUE DRAWDOWN
# =============================================================================
if show_drawdown:
    st.markdown("---")
    st.header("📉 Drawdowns (pertes depuis le plus haut)")
    
    df1['Drawdown'] = calculate_drawdown(df1['Close'])
    df2['Drawdown'] = calculate_drawdown(df2['Close'])
    
    fig_dd = go.Figure()
    
    fig_dd.add_trace(go.Scatter(
        x=df1['Date'],
        y=df1['Drawdown'],
        fill='tozeroy',
        name=asset1_name,
        line=dict(color='#FF6B6B'),
        fillcolor='rgba(255, 107, 107, 0.3)'
    ))
    
    fig_dd.add_trace(go.Scatter(
        x=df2['Date'],
        y=df2['Drawdown'],
        fill='tozeroy',
        name=asset2_name,
        line=dict(color='#4ECDC4'),
        fillcolor='rgba(78, 205, 196, 0.3)'
    ))
    
    fig_dd.update_layout(
        title="Drawdowns depuis les plus hauts historiques",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template='plotly_white',
        hovermode="x unified",
        height=400
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Stats drawdown
    col_dd1, col_dd2 = st.columns(2)
    with col_dd1:
        max_dd1 = df1['Drawdown'].min()
        st.metric(f"Drawdown max {asset1_name}", f"{max_dd1:.1f}%")
    with col_dd2:
        max_dd2 = df2['Drawdown'].min()
        st.metric(f"Drawdown max {asset2_name}", f"{max_dd2:.1f}%")


# =============================================================================
# GRAPHIQUE VOLUMES
# =============================================================================
if show_volume:
    st.markdown("---")
    st.header("📊 Volumes d'échange")
    
    fig_vol = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(f"Volume {asset1_name}", f"Volume {asset2_name}"))
    
    fig_vol.add_trace(
        go.Bar(x=df1['Date'], y=df1['Volume'], name=asset1_name, marker_color='#FF6B6B'),
        row=1, col=1
    )
    
    fig_vol.add_trace(
        go.Bar(x=df2['Date'], y=df2['Volume'], name=asset2_name, marker_color='#4ECDC4'),
        row=2, col=1
    )
    
    fig_vol.update_layout(
        height=500,
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)


# =============================================================================
# CORRÉLATION
# =============================================================================
st.markdown("---")
st.header("🔗 Analyse de corrélation")

# Corrélation glissante
window = 30  # Fenêtre de 30 jours

merged_corr = pd.merge(
    df1[['Date', 'Close']].rename(columns={'Close': 'Asset1'}),
    df2[['Date', 'Close']].rename(columns={'Close': 'Asset2'}),
    on='Date',
    how='inner'
)

# Rendements journaliers
merged_corr['Return1'] = merged_corr['Asset1'].pct_change()
merged_corr['Return2'] = merged_corr['Asset2'].pct_change()

# Corrélation glissante
merged_corr['Rolling_Corr'] = merged_corr['Return1'].rolling(window=window).corr(merged_corr['Return2'])

col_corr1, col_corr2 = st.columns([1, 2])

with col_corr1:
    overall_corr = merged_corr['Return1'].corr(merged_corr['Return2'])
    st.metric("Corrélation globale", f"{overall_corr:.3f}")
    
    st.markdown("""
    **Interprétation:**
    - **> 0.7**: Forte corrélation positive
    - **0.3 - 0.7**: Corrélation modérée
    - **-0.3 - 0.3**: Faible corrélation
    - **< -0.3**: Corrélation négative
    """)

with col_corr2:
    fig_corr = px.line(
        merged_corr,
        x='Date',
        y='Rolling_Corr',
        title=f"Corrélation glissante ({window} jours)",
        template='plotly_white'
    )
    fig_corr.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_corr.update_layout(
        yaxis_title="Corrélation",
        height=300
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# =============================================================================
# ACTUALITÉS
# =============================================================================
st.markdown("---")
st.header("📰 Actualités récentes")

col_news1, col_news2 = st.columns(2)

with col_news1:
    st.subheader(f"📰 {asset1_name}")
    news1 = get_news(asset1_ticker)
    if news1:
        for entry in news1:
            st.markdown(f"**[{entry.title}]({entry.link})**")
            if hasattr(entry, 'published'):
                st.caption(entry.published)
            st.markdown("---")
    else:
        st.info("Aucune actualité disponible")

with col_news2:
    st.subheader(f"📰 {asset2_name}")
    news2 = get_news(asset2_ticker)
    if news2:
        for entry in news2:
            st.markdown(f"**[{entry.title}]({entry.link})**")
            if hasattr(entry, 'published'):
                st.caption(entry.published)
            st.markdown("---")
    else:
        st.info("Aucune actualité disponible")


# =============================================================================
# DONNÉES BRUTES
# =============================================================================
st.markdown("---")
st.header("📋 Données brutes")

with st.expander(f"Voir les données {asset1_name}"):
    st.dataframe(df1, use_container_width=True)

with st.expander(f"Voir les données {asset2_name}"):
    st.dataframe(df2, use_container_width=True)

with st.expander("Voir les données fusionnées"):
    st.dataframe(merged, use_container_width=True)


# =============================================================================
# EXPORT
# =============================================================================
st.markdown("---")
st.header("💾 Exporter les données")

col_export1, col_export2 = st.columns(2)

with col_export1:
    csv1 = df1.to_csv(index=False)
    st.download_button(
        label=f"📥 Télécharger {asset1_name} (CSV)",
        data=csv1,
        file_name=f"{asset1_ticker}_{years}y.csv",
        mime="text/csv"
    )

with col_export2:
    csv2 = df2.to_csv(index=False)
    st.download_button(
        label=f"📥 Télécharger {asset2_name} (CSV)",
        data=csv2,
        file_name=f"{asset2_ticker}_{years}y.csv",
        mime="text/csv"
    )
