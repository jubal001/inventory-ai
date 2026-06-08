import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Retail Inventory Analytics",
    layout="wide"
)

st.title("📊 Retail Inventory Analytics Dashboard")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload Retail Inventory CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your retail_store_inventory.csv file")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)

# Show columns
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# KPIs
# -----------------------------

required_columns = [
    "Inventory",
    "Units Sold",
    "Demand Forecast",
    "Product ID",
    "Category",
    "Region"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.write("Columns found:")
    st.write(df.columns.tolist())
    st.stop()
total_units_sold = df["Units Sold"].sum()
avg_demand = df["Demand Forecast"].mean()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Inventory",
    f"{int(total_inventory):,}"
)

col2.metric(
    "Units Sold",
    f"{int(total_units_sold):,}"
)

col3.metric(
    "Average Demand Forecast",
    f"{avg_demand:.2f}"
)

# -----------------------------
# Product Selection
# -----------------------------

st.subheader("📦 Product Analysis")

product_ids = sorted(df["Product ID"].unique())

selected_product = st.selectbox(
    "Select Product",
    product_ids
)

product_df = df[
    df["Product ID"] == selected_product
]

st.write(product_df)

# -----------------------------
# Product Metrics
# -----------------------------

inventory = product_df["Inventory"].mean()
sales = product_df["Units Sold"].mean()
forecast = product_df["Demand Forecast"].mean()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Inventory",
    int(inventory)
)

col2.metric(
    "Average Units Sold",
    int(sales)
)

col3.metric(
    "Demand Forecast",
    round(forecast, 2)
)

# -----------------------------
# Restock Recommendation
# -----------------------------

st.subheader("🤖 AI Restock Recommendation")

safety_buffer = 1.20

recommended_stock = forecast * safety_buffer

restock_qty = recommended_stock - inventory

if restock_qty > 0:
    st.error(
        f"🚨 Restock approximately {int(restock_qty)} units"
    )
else:
    st.success(
        "✅ Inventory level is healthy"
    )

# -----------------------------
# Category Analysis
# -----------------------------

st.subheader("🏷️ Category Analysis")

category_summary = (
    df.groupby("Category")["Units Sold"]
      .sum()
      .sort_values(ascending=False)
)

st.bar_chart(category_summary)

# -----------------------------
# Region Analysis
# -----------------------------

st.subheader("🌍 Regional Performance")

region_summary = (
    df.groupby("Region")["Units Sold"]
      .sum()
)

st.bar_chart(region_summary)

# -----------------------------
# Inventory vs Demand
# -----------------------------

st.subheader("📈 Inventory vs Demand")

comparison_df = product_df[
    ["Inventory", "Demand Forecast"]
]

st.line_chart(comparison_df)
