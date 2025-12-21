# -*- coding: utf-8 -*-
"""
Financial Dashboard - Multi-asset Comparison
Enhanced version with metrics, drawdowns, news, and multilingual support
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
# TRANSLATIONS
# =============================================================================
TRANSLATIONS = {
    "English": {
        # General
        "page_title": "Financial Dashboard",
        "main_title": "📈 Financial Dashboard",
        "comparison": "Comparison",
        "vs": "vs",
        "years": "years",
        "over": "over",
        
        # Sidebar
        "settings": "⚙️ Settings",
        "language": "🌐 Language",
        "asset_selection": "Asset Selection",
        "first_asset": "First asset",
        "second_asset": "Second asset",
        "period": "Period",
        "num_years": "Number of years",
        "display_options": "Display Options",
        "show_normalized": "Show normalized prices",
        "show_drawdown": "Show drawdowns",
        "show_volume": "Show volumes",
        "refresh_data": "🔄 Refresh data",
        "last_update": "Last update",
        "created_with": "Dashboard created with Streamlit",
        
        # Asset categories
        "cryptocurrencies": "Cryptocurrencies",
        "indices": "Indices",
        "tech_stocks": "Tech Stocks",
        "commodities": "Commodities",
        
        # Metrics
        "key_metrics": "📊 Key Metrics",
        "return": "Return",
        "volatility": "Volatility",
        "current_price": "Current Price",
        "sharpe": "Sharpe",
        
        # Charts
        "performance_comparison": "📈 Performance Comparison",
        "normalized_prices": "Normalized prices",
        "prices": "Prices",
        "normalized_price_zscore": "Normalized Price (z-score)",
        "price_usd": "Price ($)",
        "date": "Date",
        "asset": "Asset",
        
        # Drawdowns
        "drawdowns_title": "📉 Drawdowns (losses from peak)",
        "drawdowns_from_ath": "Drawdowns from all-time highs",
        "max_drawdown": "Max Drawdown",
        
        # Volume
        "trading_volumes": "📊 Trading Volumes",
        "volume": "Volume",
        
        # Correlation
        "correlation_analysis": "🔗 Correlation Analysis",
        "overall_correlation": "Overall Correlation",
        "rolling_correlation": "Rolling Correlation",
        "days": "days",
        "interpretation": "Interpretation",
        "strong_positive": "Strong positive correlation",
        "moderate": "Moderate correlation",
        "weak": "Weak correlation",
        "negative": "Negative correlation",
        
        # News
        "recent_news": "📰 Recent News",
        "no_news": "No news available",
        
        # Data
        "raw_data": "📋 Raw Data",
        "view_data": "View data",
        "merged_data": "View merged data",
        
        # Export
        "export_data": "💾 Export Data",
        "download": "📥 Download",
        
        # Errors
        "loading_data": "Loading data...",
        "error_loading": "Unable to load data. Please try again later.",
        "error_fetching": "Error fetching",
    },
    
    "Français": {
        # General
        "page_title": "Dashboard Financier",
        "main_title": "📈 Dashboard Financier",
        "comparison": "Comparaison",
        "vs": "vs",
        "years": "ans",
        "over": "sur",
        
        # Sidebar
        "settings": "⚙️ Paramètres",
        "language": "🌐 Langue",
        "asset_selection": "Sélection des actifs",
        "first_asset": "Premier actif",
        "second_asset": "Deuxième actif",
        "period": "Période",
        "num_years": "Nombre d'années",
        "display_options": "Options d'affichage",
        "show_normalized": "Afficher les prix normalisés",
        "show_drawdown": "Afficher les drawdowns",
        "show_volume": "Afficher les volumes",
        "refresh_data": "🔄 Rafraîchir les données",
        "last_update": "Dernière mise à jour",
        "created_with": "Dashboard créé avec Streamlit",
        
        # Asset categories
        "cryptocurrencies": "Cryptomonnaies",
        "indices": "Indices",
        "tech_stocks": "Actions Tech",
        "commodities": "Matières premières",
        
        # Metrics
        "key_metrics": "📊 Métriques clés",
        "return": "Rendement",
        "volatility": "Volatilité",
        "current_price": "Prix actuel",
        "sharpe": "Sharpe",
        
        # Charts
        "performance_comparison": "📈 Comparaison des performances",
        "normalized_prices": "Prix normalisés",
        "prices": "Prix",
        "normalized_price_zscore": "Prix normalisé (z-score)",
        "price_usd": "Prix ($)",
        "date": "Date",
        "asset": "Actif",
        
        # Drawdowns
        "drawdowns_title": "📉 Drawdowns (pertes depuis le plus haut)",
        "drawdowns_from_ath": "Drawdowns depuis les plus hauts historiques",
        "max_drawdown": "Drawdown max",
        
        # Volume
        "trading_volumes": "📊 Volumes d'échange",
        "volume": "Volume",
        
        # Correlation
        "correlation_analysis": "🔗 Analyse de corrélation",
        "overall_correlation": "Corrélation globale",
        "rolling_correlation": "Corrélation glissante",
        "days": "jours",
        "interpretation": "Interprétation",
        "strong_positive": "Forte corrélation positive",
        "moderate": "Corrélation modérée",
        "weak": "Faible corrélation",
        "negative": "Corrélation négative",
        
        # News
        "recent_news": "📰 Actualités récentes",
        "no_news": "Aucune actualité disponible",
        
        # Data
        "raw_data": "📋 Données brutes",
        "view_data": "Voir les données",
        "merged_data": "Voir les données fusionnées",
        
        # Export
        "export_data": "💾 Exporter les données",
        "download": "📥 Télécharger",
        
        # Errors
        "loading_data": "Chargement des données...",
        "error_loading": "Impossible de charger les données. Veuillez réessayer plus tard.",
        "error_fetching": "Erreur lors du chargement de",
    },
    
    "Español": {
        # General
        "page_title": "Panel Financiero",
        "main_title": "📈 Panel Financiero",
        "comparison": "Comparación",
        "vs": "vs",
        "years": "años",
        "over": "durante",
        
        # Sidebar
        "settings": "⚙️ Configuración",
        "language": "🌐 Idioma",
        "asset_selection": "Selección de activos",
        "first_asset": "Primer activo",
        "second_asset": "Segundo activo",
        "period": "Período",
        "num_years": "Número de años",
        "display_options": "Opciones de visualización",
        "show_normalized": "Mostrar precios normalizados",
        "show_drawdown": "Mostrar drawdowns",
        "show_volume": "Mostrar volúmenes",
        "refresh_data": "🔄 Actualizar datos",
        "last_update": "Última actualización",
        "created_with": "Panel creado con Streamlit",
        
        # Asset categories
        "cryptocurrencies": "Criptomonedas",
        "indices": "Índices",
        "tech_stocks": "Acciones tecnológicas",
        "commodities": "Materias primas",
        
        # Metrics
        "key_metrics": "📊 Métricas clave",
        "return": "Rendimiento",
        "volatility": "Volatilidad",
        "current_price": "Precio actual",
        "sharpe": "Sharpe",
        
        # Charts
        "performance_comparison": "📈 Comparación de rendimiento",
        "normalized_prices": "Precios normalizados",
        "prices": "Precios",
        "normalized_price_zscore": "Precio normalizado (z-score)",
        "price_usd": "Precio ($)",
        "date": "Fecha",
        "asset": "Activo",
        
        # Drawdowns
        "drawdowns_title": "📉 Drawdowns (pérdidas desde máximos)",
        "drawdowns_from_ath": "Drawdowns desde máximos históricos",
        "max_drawdown": "Drawdown máximo",
        
        # Volume
        "trading_volumes": "📊 Volúmenes de negociación",
        "volume": "Volumen",
        
        # Correlation
        "correlation_analysis": "🔗 Análisis de correlación",
        "overall_correlation": "Correlación global",
        "rolling_correlation": "Correlación móvil",
        "days": "días",
        "interpretation": "Interpretación",
        "strong_positive": "Fuerte correlación positiva",
        "moderate": "Correlación moderada",
        "weak": "Correlación débil",
        "negative": "Correlación negativa",
        
        # News
        "recent_news": "📰 Noticias recientes",
        "no_news": "No hay noticias disponibles",
        
        # Data
        "raw_data": "📋 Datos brutos",
        "view_data": "Ver datos",
        "merged_data": "Ver datos combinados",
        
        # Export
        "export_data": "💾 Exportar datos",
        "download": "📥 Descargar",
        
        # Errors
        "loading_data": "Cargando datos...",
        "error_loading": "No se pueden cargar los datos. Inténtelo de nuevo más tarde.",
        "error_fetching": "Error al cargar",
    },
    
    "中文": {
        # General
        "page_title": "金融仪表板",
        "main_title": "📈 金融仪表板",
        "comparison": "比较",
        "vs": "与",
        "years": "年",
        "over": "期间",
        
        # Sidebar
        "settings": "⚙️ 设置",
        "language": "🌐 语言",
        "asset_selection": "资产选择",
        "first_asset": "第一资产",
        "second_asset": "第二资产",
        "period": "时间段",
        "num_years": "年数",
        "display_options": "显示选项",
        "show_normalized": "显示标准化价格",
        "show_drawdown": "显示回撤",
        "show_volume": "显示成交量",
        "refresh_data": "🔄 刷新数据",
        "last_update": "最后更新",
        "created_with": "使用Streamlit创建的仪表板",
        
        # Asset categories
        "cryptocurrencies": "加密货币",
        "indices": "指数",
        "tech_stocks": "科技股",
        "commodities": "大宗商品",
        
        # Metrics
        "key_metrics": "📊 关键指标",
        "return": "回报率",
        "volatility": "波动率",
        "current_price": "当前价格",
        "sharpe": "夏普比率",
        
        # Charts
        "performance_comparison": "📈 绩效比较",
        "normalized_prices": "标准化价格",
        "prices": "价格",
        "normalized_price_zscore": "标准化价格 (z-score)",
        "price_usd": "价格 ($)",
        "date": "日期",
        "asset": "资产",
        
        # Drawdowns
        "drawdowns_title": "📉 回撤（从峰值的损失）",
        "drawdowns_from_ath": "从历史高点的回撤",
        "max_drawdown": "最大回撤",
        
        # Volume
        "trading_volumes": "📊 交易量",
        "volume": "成交量",
        
        # Correlation
        "correlation_analysis": "🔗 相关性分析",
        "overall_correlation": "总体相关性",
        "rolling_correlation": "滚动相关性",
        "days": "天",
        "interpretation": "解释",
        "strong_positive": "强正相关",
        "moderate": "中等相关",
        "weak": "弱相关",
        "negative": "负相关",
        
        # News
        "recent_news": "📰 最新消息",
        "no_news": "暂无新闻",
        
        # Data
        "raw_data": "📋 原始数据",
        "view_data": "查看数据",
        "merged_data": "查看合并数据",
        
        # Export
        "export_data": "💾 导出数据",
        "download": "📥 下载",
        
        # Errors
        "loading_data": "加载数据中...",
        "error_loading": "无法加载数据，请稍后再试。",
        "error_fetching": "获取数据时出错",
    },
    
    "Русский": {
        # General
        "page_title": "Финансовая панель",
        "main_title": "📈 Финансовая панель",
        "comparison": "Сравнение",
        "vs": "и",
        "years": "лет",
        "over": "за",
        
        # Sidebar
        "settings": "⚙️ Настройки",
        "language": "🌐 Язык",
        "asset_selection": "Выбор активов",
        "first_asset": "Первый актив",
        "second_asset": "Второй актив",
        "period": "Период",
        "num_years": "Количество лет",
        "display_options": "Параметры отображения",
        "show_normalized": "Показать нормализованные цены",
        "show_drawdown": "Показать просадки",
        "show_volume": "Показать объёмы",
        "refresh_data": "🔄 Обновить данные",
        "last_update": "Последнее обновление",
        "created_with": "Панель создана с помощью Streamlit",
        
        # Asset categories
        "cryptocurrencies": "Криптовалюты",
        "indices": "Индексы",
        "tech_stocks": "Технологические акции",
        "commodities": "Сырьевые товары",
        
        # Metrics
        "key_metrics": "📊 Ключевые показатели",
        "return": "Доходность",
        "volatility": "Волатильность",
        "current_price": "Текущая цена",
        "sharpe": "Шарп",
        
        # Charts
        "performance_comparison": "📈 Сравнение эффективности",
        "normalized_prices": "Нормализованные цены",
        "prices": "Цены",
        "normalized_price_zscore": "Нормализованная цена (z-score)",
        "price_usd": "Цена ($)",
        "date": "Дата",
        "asset": "Актив",
        
        # Drawdowns
        "drawdowns_title": "📉 Просадки (потери от максимума)",
        "drawdowns_from_ath": "Просадки от исторических максимумов",
        "max_drawdown": "Макс. просадка",
        
        # Volume
        "trading_volumes": "📊 Объёмы торгов",
        "volume": "Объём",
        
        # Correlation
        "correlation_analysis": "🔗 Анализ корреляции",
        "overall_correlation": "Общая корреляция",
        "rolling_correlation": "Скользящая корреляция",
        "days": "дней",
        "interpretation": "Интерпретация",
        "strong_positive": "Сильная положительная корреляция",
        "moderate": "Умеренная корреляция",
        "weak": "Слабая корреляция",
        "negative": "Отрицательная корреляция",
        
        # News
        "recent_news": "📰 Последние новости",
        "no_news": "Новостей нет",
        
        # Data
        "raw_data": "📋 Исходные данные",
        "view_data": "Посмотреть данные",
        "merged_data": "Посмотреть объединённые данные",
        
        # Export
        "export_data": "💾 Экспорт данных",
        "download": "📥 Скачать",
        
        # Errors
        "loading_data": "Загрузка данных...",
        "error_loading": "Невозможно загрузить данные. Попробуйте позже.",
        "error_fetching": "Ошибка при загрузке",
    },
    
    "العربية": {
        # General
        "page_title": "لوحة المعلومات المالية",
        "main_title": "📈 لوحة المعلومات المالية",
        "comparison": "مقارنة",
        "vs": "مقابل",
        "years": "سنوات",
        "over": "خلال",
        
        # Sidebar
        "settings": "⚙️ الإعدادات",
        "language": "🌐 اللغة",
        "asset_selection": "اختيار الأصول",
        "first_asset": "الأصل الأول",
        "second_asset": "الأصل الثاني",
        "period": "الفترة",
        "num_years": "عدد السنوات",
        "display_options": "خيارات العرض",
        "show_normalized": "عرض الأسعار المعيارية",
        "show_drawdown": "عرض التراجعات",
        "show_volume": "عرض أحجام التداول",
        "refresh_data": "🔄 تحديث البيانات",
        "last_update": "آخر تحديث",
        "created_with": "لوحة معلومات تم إنشاؤها باستخدام Streamlit",
        
        # Asset categories
        "cryptocurrencies": "العملات المشفرة",
        "indices": "المؤشرات",
        "tech_stocks": "أسهم التكنولوجيا",
        "commodities": "السلع",
        
        # Metrics
        "key_metrics": "📊 المؤشرات الرئيسية",
        "return": "العائد",
        "volatility": "التقلب",
        "current_price": "السعر الحالي",
        "sharpe": "شارب",
        
        # Charts
        "performance_comparison": "📈 مقارنة الأداء",
        "normalized_prices": "الأسعار المعيارية",
        "prices": "الأسعار",
        "normalized_price_zscore": "السعر المعياري (z-score)",
        "price_usd": "السعر ($)",
        "date": "التاريخ",
        "asset": "الأصل",
        
        # Drawdowns
        "drawdowns_title": "📉 التراجعات (الخسائر من الذروة)",
        "drawdowns_from_ath": "التراجعات من أعلى المستويات التاريخية",
        "max_drawdown": "أقصى تراجع",
        
        # Volume
        "trading_volumes": "📊 أحجام التداول",
        "volume": "الحجم",
        
        # Correlation
        "correlation_analysis": "🔗 تحليل الارتباط",
        "overall_correlation": "الارتباط الكلي",
        "rolling_correlation": "الارتباط المتحرك",
        "days": "يوم",
        "interpretation": "التفسير",
        "strong_positive": "ارتباط إيجابي قوي",
        "moderate": "ارتباط معتدل",
        "weak": "ارتباط ضعيف",
        "negative": "ارتباط سلبي",
        
        # News
        "recent_news": "📰 آخر الأخبار",
        "no_news": "لا توجد أخبار متاحة",
        
        # Data
        "raw_data": "📋 البيانات الخام",
        "view_data": "عرض البيانات",
        "merged_data": "عرض البيانات المدمجة",
        
        # Export
        "export_data": "💾 تصدير البيانات",
        "download": "📥 تحميل",
        
        # Errors
        "loading_data": "جاري تحميل البيانات...",
        "error_loading": "تعذر تحميل البيانات. يرجى المحاولة مرة أخرى لاحقاً.",
        "error_fetching": "خطأ في جلب",
    },
}

# Language configuration (RTL support)
RTL_LANGUAGES = ["العربية"]

def get_text(key: str) -> str:
    """Get translated text for the current language."""
    lang = st.session_state.get("language", "English")
    return TRANSLATIONS.get(lang, TRANSLATIONS["English"]).get(key, key)

def t(key: str) -> str:
    """Shorthand for get_text."""
    return get_text(key)


# =============================================================================
# PAGE CONFIGURATION (must be first Streamlit command)
# =============================================================================
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "language" not in st.session_state:
    st.session_state.language = "English"


# =============================================================================
# CUSTOM CSS - PIRATE MAP THEME
# =============================================================================
def apply_custom_css():
    """Apply custom CSS including RTL support and pirate map theme."""
    is_rtl = st.session_state.language in RTL_LANGUAGES
    direction = "rtl" if is_rtl else "ltr"
    text_align = "right" if is_rtl else "left"
    
    st.markdown(f"""
    <style>
        /* Main background - parchment/sand color */
        .stApp {{
            background-color: #F5E6C8;
        }}
        
        /* Metric cards - cream/ivory with brown border */
        [data-testid="stMetric"] {{
            background-color: #FDF8E8;
            padding: 20px;
            border-radius: 12px;
            border: 2px solid #C4A574;
            box-shadow: 3px 3px 8px rgba(61, 41, 20, 0.15);
        }}
        
        [data-testid="stMetric"]:hover {{
            background-color: #FFF9E6;
            border-color: #8B4513;
            box-shadow: 4px 4px 12px rgba(61, 41, 20, 0.25);
        }}
        
        /* Metric label - dark brown, readable */
        [data-testid="stMetric"] label {{
            color: #5D4023 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }}
        
        /* Metric value - dark brown */
        [data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: #3D2914 !important;
            font-weight: 700 !important;
        }}
        
        /* Metric delta - keep green/red for positive/negative */
        [data-testid="stMetric"] [data-testid="stMetricDelta"] svg {{
            stroke: #2E7D32;
        }}
        
        /* Sidebar - slightly darker parchment */
        [data-testid="stSidebar"] {{
            background-color: #EDD9B4;
            border-right: 3px solid #C4A574;
        }}
        
        [data-testid="stSidebar"] [data-testid="stMarkdown"] {{
            color: #3D2914;
        }}
        
        /* Headers - brown color */
        h1, h2, h3 {{
            color: #5D3A1A !important;
        }}
        
        /* Expanders - parchment style */
        [data-testid="stExpander"] {{
            background-color: #FDF8E8;
            border: 1px solid #C4A574;
            border-radius: 8px;
        }}
        
        /* Buttons - brown theme */
        .stButton > button {{
            background-color: #8B4513;
            color: #FDF8E8;
            border: none;
            border-radius: 8px;
        }}
        
        .stButton > button:hover {{
            background-color: #A0522D;
            color: #FFFFFF;
        }}
        
        /* Download buttons */
        .stDownloadButton > button {{
            background-color: #6B4423;
            color: #FDF8E8;
            border: 2px solid #8B4513;
        }}
        
        .stDownloadButton > button:hover {{
            background-color: #8B4513;
        }}
        
        /* Selectbox and inputs */
        [data-testid="stSelectbox"] {{
            background-color: #FDF8E8;
        }}
        
        /* Checkbox */
        [data-testid="stCheckbox"] label span {{
            color: #3D2914 !important;
        }}
        
        /* Info boxes */
        .stAlert {{
            background-color: #FDF8E8;
            border: 1px solid #C4A574;
        }}
        
        /* Markdown text */
        .stMarkdown {{
            color: #3D2914;
        }}
        
        /* Captions */
        .stCaption {{
            color: #6B5344 !important;
        }}
        
        /* RTL support */
        .main .block-container {{
            direction: {direction};
            text-align: {text_align};
        }}
        
        /* Divider lines */
        hr {{
            border-color: #C4A574;
        }}
        
        /* DataFrame/tables */
        [data-testid="stDataFrame"] {{
            background-color: #FDF8E8;
            border-radius: 8px;
        }}
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================
ASSETS = {
    "cryptocurrencies": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "SOL-USD": "Solana",
        "BNB-USD": "Binance Coin"
    },
    "indices": {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "^DJI": "Dow Jones",
        "^FCHI": "CAC 40",
        "^STOXX50E": "Euro Stoxx 50"
    },
    "tech_stocks": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "NVDA": "NVIDIA",
        "META": "Meta"
    },
    "commodities": {
        "GC=F": "Gold",
        "SI=F": "Silver",
        "CL=F": "Oil (WTI)"
    }
}

# Create flat list for selectbox
ALL_ASSETS = {}
for category, assets in ASSETS.items():
    ALL_ASSETS.update(assets)

TICKER_TO_NAME = {ticker: name for ticker, name in ALL_ASSETS.items()}
NAME_TO_TICKER = {name: ticker for ticker, name in ALL_ASSETS.items()}


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
@st.cache_data(ttl=3600)
def load_asset_data(ticker: str, years: int) -> pd.DataFrame:
    """Load historical data for an asset."""
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
        st.error(f"{t('error_fetching')} {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_asset_info(ticker: str) -> dict:
    """Get detailed asset information."""
    try:
        info = yf.Ticker(ticker).info
        return info
    except:
        return {}


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a series (z-score)."""
    return (series - series.mean()) / series.std()


def calculate_returns(prices: pd.Series) -> float:
    """Calculate total return in percentage."""
    if len(prices) < 2:
        return 0.0
    return ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100


def calculate_volatility(prices: pd.Series, annualize: bool = True) -> float:
    """Calculate volatility (standard deviation of returns)."""
    returns = prices.pct_change().dropna()
    vol = returns.std()
    if annualize:
        vol *= np.sqrt(252)
    return vol * 100


def calculate_drawdown(prices: pd.Series) -> pd.Series:
    """Calculate drawdown from all-time high."""
    peak = prices.cummax()
    drawdown = (prices - peak) / peak * 100
    return drawdown


def calculate_sharpe_ratio(prices: pd.Series, risk_free_rate: float = 0.04) -> float:
    """Calculate simplified Sharpe ratio."""
    returns = prices.pct_change().dropna()
    excess_return = returns.mean() * 252 - risk_free_rate
    volatility = returns.std() * np.sqrt(252)
    if volatility == 0:
        return 0
    return excess_return / volatility


@st.cache_data(ttl=3600)
def get_news(ticker: str) -> list:
    """Get news via Yahoo Finance RSS."""
    try:
        feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}")
        return feed.entries[:5]
    except:
        return []


# =============================================================================
# SIDEBAR - SETTINGS
# =============================================================================
with st.sidebar:
    # Language selector (always at top)
    st.subheader(t("language"))
    language = st.selectbox(
        "Language",
        options=list(TRANSLATIONS.keys()),
        index=list(TRANSLATIONS.keys()).index(st.session_state.language),
        label_visibility="collapsed"
    )
    
    if language != st.session_state.language:
        st.session_state.language = language
        st.rerun()
    
    st.markdown("---")
    st.header(t("settings"))
    
    st.subheader(t("asset_selection"))
    
    # First asset
    asset1_name = st.selectbox(
        t("first_asset"),
        options=list(NAME_TO_TICKER.keys()),
        index=list(NAME_TO_TICKER.keys()).index("Bitcoin")
    )
    asset1_ticker = NAME_TO_TICKER[asset1_name]
    
    # Second asset
    asset2_name = st.selectbox(
        t("second_asset"),
        options=list(NAME_TO_TICKER.keys()),
        index=list(NAME_TO_TICKER.keys()).index("S&P 500")
    )
    asset2_ticker = NAME_TO_TICKER[asset2_name]
    
    st.subheader(t("period"))
    years = st.slider(t("num_years"), min_value=1, max_value=10, value=5)
    
    st.subheader(t("display_options"))
    show_normalized = st.checkbox(t("show_normalized"), value=True)
    show_drawdown = st.checkbox(t("show_drawdown"), value=True)
    show_volume = st.checkbox(t("show_volume"), value=False)
    
    st.markdown("---")
    
    if st.button(t("refresh_data"), use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption(t("created_with"))
    st.caption(f"{t('last_update')}: {datetime.now().strftime('%H:%M:%S')}")


# =============================================================================
# DATA LOADING
# =============================================================================
st.title(t("main_title"))
st.markdown(f"**{t('comparison')}:** {asset1_name} {t('vs')} {asset2_name} {t('over')} {years} {t('years')}")

# Load with spinner
with st.spinner(t("loading_data")):
    df1 = load_asset_data(asset1_ticker, years)
    df2 = load_asset_data(asset2_ticker, years)

# Check data
if df1.empty or df2.empty:
    st.error(t("error_loading"))
    st.stop()


# =============================================================================
# KEY METRICS
# =============================================================================
st.header(t("key_metrics"))

col1, col2, col3, col4 = st.columns(4)

# Return Asset 1
return1 = calculate_returns(df1['Close'])
with col1:
    st.metric(
        label=f"{t('return')} {asset1_name}",
        value=f"{return1:.1f}%",
        delta=f"{return1:.1f}%" if return1 != 0 else None
    )

# Return Asset 2
return2 = calculate_returns(df2['Close'])
with col2:
    st.metric(
        label=f"{t('return')} {asset2_name}",
        value=f"{return2:.1f}%",
        delta=f"{return2:.1f}%" if return2 != 0 else None
    )

# Volatility Asset 1
vol1 = calculate_volatility(df1['Close'])
with col3:
    st.metric(
        label=f"{t('volatility')} {asset1_name}",
        value=f"{vol1:.1f}%"
    )

# Volatility Asset 2
vol2 = calculate_volatility(df2['Close'])
with col4:
    st.metric(
        label=f"{t('volatility')} {asset2_name}",
        value=f"{vol2:.1f}%"
    )

# Second row of metrics
col5, col6, col7, col8 = st.columns(4)

# Current prices
with col5:
    current_price1 = df1['Close'].iloc[-1]
    st.metric(
        label=f"{t('current_price')} {asset1_name}",
        value=f"${current_price1:,.2f}"
    )

with col6:
    current_price2 = df2['Close'].iloc[-1]
    st.metric(
        label=f"{t('current_price')} {asset2_name}",
        value=f"${current_price2:,.2f}"
    )

# Sharpe Ratio
with col7:
    sharpe1 = calculate_sharpe_ratio(df1['Close'])
    st.metric(
        label=f"{t('sharpe')} {asset1_name}",
        value=f"{sharpe1:.2f}"
    )

with col8:
    sharpe2 = calculate_sharpe_ratio(df2['Close'])
    st.metric(
        label=f"{t('sharpe')} {asset2_name}",
        value=f"{sharpe2:.2f}"
    )


# =============================================================================
# MAIN CHART - COMPARISON
# =============================================================================
st.markdown("---")
st.header(t("performance_comparison"))

# Prepare data for chart
df1_plot = df1[['Date', 'Close']].copy()
df2_plot = df2[['Date', 'Close']].copy()

if show_normalized:
    df1_plot['Close'] = normalize_series(df1['Close'])
    df2_plot['Close'] = normalize_series(df2['Close'])
    y_label = t("normalized_price_zscore")
else:
    y_label = t("price_usd")

# Merge data
df1_plot = df1_plot.rename(columns={'Close': asset1_name})
df2_plot = df2_plot.rename(columns={'Close': asset2_name})

merged = pd.merge(df1_plot, df2_plot, on='Date', how='inner')
merged_melted = merged.melt(id_vars=['Date'], var_name=t('asset'), value_name=t('prices'))

# Create chart with parchment theme colors
chart_title = f"{t('normalized_prices') if show_normalized else t('prices')}: {asset1_name} {t('vs')} {asset2_name}"
fig_main = px.line(
    merged_melted,
    x='Date',
    y=t('prices'),
    color=t('asset'),
    title=chart_title,
    color_discrete_map={asset1_name: '#8B4513', asset2_name: '#2E7D32'},
    template='plotly_white'
)

fig_main.update_layout(
    xaxis_title=t("date"),
    yaxis_title=y_label,
    legend_title=t("asset"),
    hovermode="x unified",
    height=500,
    paper_bgcolor='#FDF8E8',
    plot_bgcolor='#FDF8E8',
    font=dict(color='#3D2914'),
    title_font=dict(color='#5D3A1A')
)

st.plotly_chart(fig_main, use_container_width=True)


# =============================================================================
# DRAWDOWN CHART
# =============================================================================
if show_drawdown:
    st.markdown("---")
    st.header(t("drawdowns_title"))
    
    df1['Drawdown'] = calculate_drawdown(df1['Close'])
    df2['Drawdown'] = calculate_drawdown(df2['Close'])
    
    fig_dd = go.Figure()
    
    fig_dd.add_trace(go.Scatter(
        x=df1['Date'],
        y=df1['Drawdown'],
        fill='tozeroy',
        name=asset1_name,
        line=dict(color='#8B4513'),
        fillcolor='rgba(139, 69, 19, 0.3)'
    ))
    
    fig_dd.add_trace(go.Scatter(
        x=df2['Date'],
        y=df2['Drawdown'],
        fill='tozeroy',
        name=asset2_name,
        line=dict(color='#2E7D32'),
        fillcolor='rgba(46, 125, 50, 0.3)'
    ))
    
    fig_dd.update_layout(
        title=t("drawdowns_from_ath"),
        xaxis_title=t("date"),
        yaxis_title="Drawdown (%)",
        template='plotly_white',
        hovermode="x unified",
        height=400,
        paper_bgcolor='#FDF8E8',
        plot_bgcolor='#FDF8E8',
        font=dict(color='#3D2914'),
        title_font=dict(color='#5D3A1A')
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Drawdown stats
    col_dd1, col_dd2 = st.columns(2)
    with col_dd1:
        max_dd1 = df1['Drawdown'].min()
        st.metric(f"{t('max_drawdown')} {asset1_name}", f"{max_dd1:.1f}%")
    with col_dd2:
        max_dd2 = df2['Drawdown'].min()
        st.metric(f"{t('max_drawdown')} {asset2_name}", f"{max_dd2:.1f}%")


# =============================================================================
# VOLUME CHART
# =============================================================================
if show_volume:
    st.markdown("---")
    st.header(t("trading_volumes"))
    
    fig_vol = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(f"{t('volume')} {asset1_name}", f"{t('volume')} {asset2_name}"))
    
    fig_vol.add_trace(
        go.Bar(x=df1['Date'], y=df1['Volume'], name=asset1_name, marker_color='#8B4513'),
        row=1, col=1
    )
    
    fig_vol.add_trace(
        go.Bar(x=df2['Date'], y=df2['Volume'], name=asset2_name, marker_color='#2E7D32'),
        row=2, col=1
    )
    
    fig_vol.update_layout(
        height=500,
        template='plotly_white',
        showlegend=False,
        paper_bgcolor='#FDF8E8',
        plot_bgcolor='#FDF8E8',
        font=dict(color='#3D2914')
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)


# =============================================================================
# CORRELATION
# =============================================================================
st.markdown("---")
st.header(t("correlation_analysis"))

# Rolling correlation
window = 30

merged_corr = pd.merge(
    df1[['Date', 'Close']].rename(columns={'Close': 'Asset1'}),
    df2[['Date', 'Close']].rename(columns={'Close': 'Asset2'}),
    on='Date',
    how='inner'
)

# Daily returns
merged_corr['Return1'] = merged_corr['Asset1'].pct_change()
merged_corr['Return2'] = merged_corr['Asset2'].pct_change()

# Rolling correlation
merged_corr['Rolling_Corr'] = merged_corr['Return1'].rolling(window=window).corr(merged_corr['Return2'])

col_corr1, col_corr2 = st.columns([1, 2])

with col_corr1:
    overall_corr = merged_corr['Return1'].corr(merged_corr['Return2'])
    st.metric(t("overall_correlation"), f"{overall_corr:.3f}")
    
    st.markdown(f"""
    **{t('interpretation')}:**
    - **> 0.7**: {t('strong_positive')}
    - **0.3 - 0.7**: {t('moderate')}
    - **-0.3 - 0.3**: {t('weak')}
    - **< -0.3**: {t('negative')}
    """)

with col_corr2:
    fig_corr = px.line(
        merged_corr,
        x='Date',
        y='Rolling_Corr',
        title=f"{t('rolling_correlation')} ({window} {t('days')})",
        template='plotly_white'
    )
    fig_corr.add_hline(y=0, line_dash="dash", line_color="#8B4513")
    fig_corr.update_traces(line_color='#5D3A1A')
    fig_corr.update_layout(
        yaxis_title=t("overall_correlation"),
        height=300,
        paper_bgcolor='#FDF8E8',
        plot_bgcolor='#FDF8E8',
        font=dict(color='#3D2914'),
        title_font=dict(color='#5D3A1A')
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# =============================================================================
# NEWS
# =============================================================================
st.markdown("---")
st.header(t("recent_news"))

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
        st.info(t("no_news"))

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
        st.info(t("no_news"))


# =============================================================================
# RAW DATA
# =============================================================================
st.markdown("---")
st.header(t("raw_data"))

with st.expander(f"{t('view_data')} {asset1_name}"):
    st.dataframe(df1, use_container_width=True)

with st.expander(f"{t('view_data')} {asset2_name}"):
    st.dataframe(df2, use_container_width=True)

with st.expander(t("merged_data")):
    st.dataframe(merged, use_container_width=True)


# =============================================================================
# EXPORT
# =============================================================================
st.markdown("---")
st.header(t("export_data"))

col_export1, col_export2 = st.columns(2)

with col_export1:
    csv1 = df1.to_csv(index=False)
    st.download_button(
        label=f"{t('download')} {asset1_name} (CSV)",
        data=csv1,
        file_name=f"{asset1_ticker}_{years}y.csv",
        mime="text/csv"
    )

with col_export2:
    csv2 = df2.to_csv(index=False)
    st.download_button(
        label=f"{t('download')} {asset2_name} (CSV)",
        data=csv2,
        file_name=f"{asset2_ticker}_{years}y.csv",
        mime="text/csv"
    )
