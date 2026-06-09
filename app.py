import streamlit as st
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Retail Inventory Analytics Dashboard",
    layout="wide"
)

st.title("📊 Retail Inventory Analytics Dashboard")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Retail Inventory CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your retail inventory CSV file.")
    st.stop()

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(uploaded_file)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

# Required columns from your dataset
required_columns = [
    "Product ID",
    "Category",
    "Region",
    "Inventory Level",
    "Units Sold",
    "Demand Forecast"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.write("Columns found:")
    st.write(df.columns.tolist())
    st.stop()

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

from sklearn.ensemble import
RandomForestRegressor
from sklearn.model_selection import
train_test_split

# -----------------------------
# KPI SECTION
# -----------------------------
st.subheader("📈 Business KPIs")

total_inventory = df["Inventory Level"].sum()
total_units_sold = df["Units Sold"].sum()
average_demand = df["Demand Forecast"].mean()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Inventory",
    f"{int(total_inventory):,}"
)

col2.metric(
    "Total Units Sold",
    f"{int(total_units_sold):,}"
)

col3.metric(
    "Average Demand Forecast",
    f"{average_demand:.2f}"
)

# -----------------------------
# AI FEATURES SECTION
# -----------------------------
st.subheader("🤖 AI Demand Forecasting")

model_df = df.copy()

# Convert category columns to numbers
model_df = pd.get_dummies(
    model_df,
    columns=["Category", "Region", "Seasonality"],
    drop_first=True
)

# Features
X = model_df.drop(
    columns=[
        "Demand Forecast",
        "Date",
        "Store ID",
        "Product ID",
        "Weather Condition"
    ],
    errors="ignore"
)

# Target
y = model_df["Demand Forecast"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

st.metric(
    "AI Model Accuracy (R²)",
    f"{score:.2f}"
)

# -----------------------------
# PRODUCT ANALYSIS
# -----------------------------
st.subheader("📦 Product Analysis")

products = sorted(df["Product ID"].astype(str).unique())

selected_product = st.selectbox(
    "Select Product",
    products
)

product_df = df[df["Product ID"].astype(str) == selected_product]

st.write(product_df)

prediction_row = product_df.iloc[[0]].copy()

prediction_row = pd.get_dummies(
    prediction_row,
    columns=["Category", "Region", "Seasonality"],
    drop_first=True
)

prediction_row = prediction_row.reindex(
    columns=X.columns,
    fill_value=0
)

predicted_demand = model.predict(prediction_row)[0]

st.subheader("🔮 AI Predicted Demand")

st.metric(
    "Predicted Demand",
    f"{predicted_demand:.2f}"
)
# -----------------------------
# PRODUCT METRICS
# -----------------------------
inventory = product_df["Inventory Level"].mean()
sales = product_df["Units Sold"].mean()
forecast = product_df["Demand Forecast"].mean()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Inventory Level",
    int(inventory)
)

col2.metric(
    "Units Sold",
    int(sales)
)

col3.metric(
    "Demand Forecast",
    round(forecast, 2)
)

# -----------------------------
# RESTOCK RECOMMENDATION
# -----------------------------
st.subheader("🤖 Restock Recommendation")

safety_buffer = 1.20

recommended_stock = predicted_demand * safety_buffer

restock_qty = recommended_stock - inventory

if restock_qty > 0:
    st.error(
        f"🚨 Restock Recommended: {int(restock_qty)} units"
    )
else:
    st.success(
        "✅ Current inventory is sufficient"
    )

# -----------------------------
# REVENUE ESTIMATION
# -----------------------------
df["Revenue"] = df["Units Sold"] * df["Price"]

total_revenue = df["Revenue"].sum()

st.metric(
    "Estimated Revenue",
    f"${total_revenue:,.2f}"
)

# -----------------------------
# tOP 10 BEST SELLING PRODUCTS
# -----------------------------
top_products = (
    df.groupby("Product ID")["Units Sold"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

# -----------------------------
# CATEGORY PERFORMANCE
# -----------------------------
st.subheader("🏷️ Category Performance")

category_summary = (
    df.groupby("Category")["Units Sold"]
      .sum()
      .sort_values(ascending=False)
)

st.bar_chart(category_summary)

# -----------------------------
# REGION PERFORMANCE
# -----------------------------
st.subheader("🌍 Regional Performance")

region_summary = (
    df.groupby("Region")["Units Sold"]
      .sum()
      .sort_values(ascending=False)
)

st.bar_chart(region_summary)

# -----------------------------
# INVENTORY VS DEMAND
# -----------------------------
st.subheader("📊 Inventory vs Demand")

comparison_df = product_df[
    ["Inventory Level", "Demand Forecast"]
]

st.line_chart(comparison_df)

# -----------------------------
# LOW STOCK ALERT
# -----------------------------
st.subheader("🏆 Top 10 Best Selling Products")
st.bar_chart(top_products)

st.subheader("⚠️ Low Stock Products")

low_stock = df[df["Inventory Level"] < 100]

st.dataframe(
    low_stock[
        ["Product ID",
         "Category",
         "Inventory Level"]
    ]
)

# -----------------------------
# DOWNLOAD PRODUCT DATA
# -----------------------------
csv = product_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Product Data",
    data=csv,
    file_name=f"{selected_product}_data.csv",
    mime="text/csv"
)
