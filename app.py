# ==========================================================
# InventoryAI Pro v1.0
# AI-Powered Retail Analytics & Demand Forecasting Platform
# ==========================================================

# -----------------------------
# IMPORTS
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="InventoryAI Pro",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background:#0F172A;
}

section[data-testid="stSidebar"]{
    background:#111827;
}

h1,h2,h3,h4,h5,h6,p,label{
    color:white !important;
}

div[data-testid="metric-container"]{
    background:#1E293B;
    border-radius:15px;
    padding:20px;
    border:1px solid #334155;
    box-shadow:0px 2px 12px rgba(0,0,0,0.25);
}

[data-testid="stFileUploader"]{
    background:#1E293B;
    border-radius:15px;
    padding:20px;
}

.stButton>button{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<h1 style='text-align:center'>
📦 InventoryAI Pro
</h1>

<p style='text-align:center;font-size:18px;color:#CBD5E1'>
AI-Powered Retail Inventory Analytics & Demand Forecasting
</p>
""", unsafe_allow_html=True)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def format_currency(value, symbol="$"):
    value = float(value)

    if value >= 1_000_000_000:
        return f"{symbol}{value/1_000_000_000:.1f}B"

    elif value >= 1_000_000:
        return f"{symbol}{value/1_000_000:.1f}M"

    elif value >= 1_000:
        return f"{symbol}{value/1_000:.1f}K"

    return f"{symbol}{value:,.0f}"


def find_column(columns, keywords):
    """
    Finds the first column that contains any keyword.
    """

    for col in columns:

        lower = col.lower()

        for key in keywords:

            if key in lower:

                return col

    return None


def risk_level(shortage):

    if shortage >= 100:
        return "High"

    elif shortage >= 50:
        return "Medium"

    else:
        return "Low"


# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "📂 Upload Inventory Dataset",
    type=["csv"]
)

if uploaded_file is None:

    st.info("Please upload a CSV dataset to continue.")

    st.stop()

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(uploaded_file)

df.columns = df.columns.str.strip()

st.success("Dataset loaded successfully.")

st.subheader("Dataset Preview")

st.dataframe(df.head())

# ==========================================================
# AUTOMATIC COLUMN DETECTION
# ==========================================================

product_guess = find_column(
    df.columns,
    ["product"]
)

category_guess = find_column(
    df.columns,
    ["category"]
)

region_guess = find_column(
    df.columns,
    ["region"]
)

inventory_guess = find_column(
    df.columns,
    ["inventory"]
)

sales_guess = find_column(
    df.columns,
    ["units sold","sales"]
)

forecast_guess = find_column(
    df.columns,
    ["forecast","demand"]
)

price_guess = find_column(
    df.columns,
    ["price"]
)

date_guess = find_column(
    df.columns,
    ["date"]
)

store_guess = find_column(
    df.columns,
    ["store"]
)

season_guess = find_column(
    df.columns,
    ["season"]
)

weather_guess = find_column(
    df.columns,
    ["weather"]
)

promotion_guess = find_column(
    df.columns,
    ["promotion","holiday"]
)

st.subheader("Column Mapping")

col1,col2,col3 = st.columns(3)

with col1:

    product_col = st.selectbox(
        "Product",
        df.columns,
        index=df.columns.get_loc(product_guess)
        if product_guess else 0
    )

    category_col = st.selectbox(
        "Category",
        df.columns,
        index=df.columns.get_loc(category_guess)
        if category_guess else 0
    )

    region_col = st.selectbox(
        "Region",
        df.columns,
        index=df.columns.get_loc(region_guess)
        if region_guess else 0
    )

with col2:

    inventory_col = st.selectbox(
        "Inventory",
        df.columns,
        index=df.columns.get_loc(inventory_guess)
        if inventory_guess else 0
    )

    sales_col = st.selectbox(
        "Sales",
        df.columns,
        index=df.columns.get_loc(sales_guess)
        if sales_guess else 0
    )

    forecast_col = st.selectbox(
        "Forecast",
        df.columns,
        index=df.columns.get_loc(forecast_guess)
        if forecast_guess else 0
    )

with col3:

    price_col = st.selectbox(
        "Price",
        df.columns,
        index=df.columns.get_loc(price_guess)
        if price_guess else 0
    )

    date_col = st.selectbox(
        "Date",
        df.columns,
        index=df.columns.get_loc(date_guess)
        if date_guess else 0
    )

    store_col = st.selectbox(
        "Store",
        df.columns,
        index=df.columns.get_loc(store_guess)
        if store_guess else 0
    )
