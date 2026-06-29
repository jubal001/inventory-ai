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

# ==========================================================
# CREATE ANALYSIS DATAFRAME
# ==========================================================

analysis_df = pd.DataFrame()

analysis_df["Date"] = df[date_col] if date_col else ""
analysis_df["Store"] = df[store_col].astype(str) if store_col else "Unknown"

analysis_df["Product"] = df[product_col].astype(str)
analysis_df["Category"] = df[category_col].astype(str)
analysis_df["Region"] = df[region_col].astype(str)

analysis_df["Inventory"] = pd.to_numeric(
    df[inventory_col],
    errors="coerce"
)

analysis_df["Sales"] = pd.to_numeric(
    df[sales_col],
    errors="coerce"
)

analysis_df["Forecast"] = pd.to_numeric(
    df[forecast_col],
    errors="coerce"
)

analysis_df["Price"] = pd.to_numeric(
    df[price_col],
    errors="coerce"
)

# Optional Columns
if season_guess:
    analysis_df["Season"] = df[season_guess]

if weather_guess:
    analysis_df["Weather"] = df[weather_guess]

if promotion_guess:
    analysis_df["Promotion"] = df[promotion_guess]

analysis_df = analysis_df.dropna()

# ==========================================================
# CALCULATED COLUMNS
# ==========================================================

analysis_df["Revenue"] = (
    analysis_df["Sales"] *
    analysis_df["Price"]
)

analysis_df["Shortage"] = (
    analysis_df["Forecast"] -
    analysis_df["Inventory"]
)

analysis_df["Risk Level"] = analysis_df["Shortage"].apply(
    lambda x: risk_level(max(x,0))
)

analysis_df["Restock Qty"] = (
    analysis_df["Forecast"]*1.20 -
    analysis_df["Inventory"]
).clip(lower=0)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📊 Dashboard Filters")

currency_symbol = st.sidebar.selectbox(
    "💱 Currency",
    ["$", "₦", "£", "€"],
    index=0
)

st.sidebar.markdown("---")

selected_products = st.sidebar.multiselect(
    "📦 Product",
    sorted(analysis_df["Product"].unique()),
    default=sorted(analysis_df["Product"].unique())
)

selected_categories = st.sidebar.multiselect(
    "🏷️ Category",
    sorted(analysis_df["Category"].unique()),
    default=sorted(analysis_df["Category"].unique())
)

selected_regions = st.sidebar.multiselect(
    "🌍 Region",
    sorted(analysis_df["Region"].unique()),
    default=sorted(analysis_df["Region"].unique())
)

selected_risk = st.sidebar.multiselect(
    "⚠️ Risk Level",
    ["High","Medium","Low"],
    default=["High","Medium","Low"]
)

# ==========================================================
# APPLY FILTERS
# ==========================================================

filtered_df = analysis_df[
    (analysis_df["Product"].isin(selected_products))
    &
    (analysis_df["Category"].isin(selected_categories))
    &
    (analysis_df["Region"].isin(selected_regions))
    &
    (analysis_df["Risk Level"].isin(selected_risk))
]

# ==========================================================
# KPI CALCULATIONS
# ==========================================================

total_revenue = filtered_df["Revenue"].sum()

inventory_value = (
    filtered_df["Inventory"] *
    filtered_df["Price"]
).sum()

forecast_revenue = (
    filtered_df["Forecast"] *
    filtered_df["Price"]
).sum()

products = filtered_df["Product"].nunique()

stores = filtered_df["Store"].nunique()

units_sold = filtered_df["Sales"].sum()

inventory = filtered_df["Inventory"].sum()

forecast = filtered_df["Forecast"].sum()

average_price = filtered_df["Price"].mean()

risk_products = filtered_df[
    filtered_df["Inventory"] <
    filtered_df["Forecast"]
]

risk_count = len(risk_products)

# ==========================================================
# AI DEMAND FORECASTING
# ==========================================================

model_df = filtered_df.copy()

categorical_cols = []

for col in [
    "Category",
    "Region",
    "Season",
    "Weather",
    "Promotion"
]:
    if col in model_df.columns:
        categorical_cols.append(col)

model_df = pd.get_dummies(
    model_df,
    columns=categorical_cols,
    drop_first=True
)

drop_cols = [
    "Forecast",
    "Date",
    "Product",
    "Store",
    "Risk Level"
]

X = model_df.drop(
    columns=drop_cols,
    errors="ignore"
)

y = model_df["Forecast"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)

# ==========================================================
# EXECUTIVE KPI DASHBOARD
# ==========================================================

st.subheader("📈 Executive Business Dashboard")

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Revenue",
    format_currency(
        total_revenue,
        currency_symbol
    )
)

c2.metric(
    "Forecast Revenue",
    format_currency(
        forecast_revenue,
        currency_symbol
    )
)

c3.metric(
    "Inventory Value",
    format_currency(
        inventory_value,
        currency_symbol
    )
)

c4.metric(
    "Products",
    f"{products:,}"
)

c5,c6,c7,c8 = st.columns(4)

c5.metric(
    "Stores",
    f"{stores:,}"
)

c6.metric(
    "Units Sold",
    f"{int(units_sold):,}"
)

c7.metric(
    "Products At Risk",
    f"{risk_count:,}"
)

c8.metric(
    "AI Accuracy",
    f"{accuracy*100:.1f}%"
)

# ==========================================================
# PRODUCT ANALYSIS
# ==========================================================

st.subheader("📦 Product Analysis")

selected_product = st.selectbox(
    "Select Product",
    sorted(filtered_df["Product"].unique())
)

product_df = filtered_df[
    filtered_df["Product"] == selected_product
]

st.dataframe(product_df)

prediction = product_df.iloc[[0]].copy()

prediction = pd.get_dummies(
    prediction,
    columns=categorical_cols,
    drop_first=True
)

prediction = prediction.reindex(
    columns=X.columns,
    fill_value=0
)

predicted = model.predict(prediction)[0]

st.metric(
    "AI Predicted Demand",
    f"{predicted:.0f}"
)

inventory_now = product_df["Inventory"].mean()

restock = max(
    predicted*1.2 - inventory_now,
    0
)

if restock>0:

    st.error(
        f"🚨 Restock Recommended: {int(restock)} units"
    )

else:

    st.success(
        "✅ Inventory level is sufficient."
    )
