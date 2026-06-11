import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.write("App started successfully!")

st.set_page_config(
    page_title="InventoryAI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<div style="
background: linear-gradient(135deg,#2563EB,#7C3AED);
padding:30px;
border-radius:20px;
margin-bottom:25px;
">
<h2 style="color:white;">
📈 Inventory Intelligence Platform
</h2>

<p style="color:white;">
Forecast demand, detect stock risks,
optimize inventory, and increase revenue
with AI-powered analytics.
</p>

</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
# 📦 InventoryAI

### Smart Retail Analytics

Powered by AI
""")

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #0F172A;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #374151;
}

/* Headers */
h1, h2, h3 {
    color: white;
    font-weight: 700;
}

/* Metrics */
div[data-testid="metric-container"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
}

/* Tables */
[data-testid="stDataFrame"] {
    background-color: #1E293B;
    border-radius: 15px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        90deg,
        #2563EB,
        #3B82F6
    );
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 25px;
    font-weight: bold;
}

/* Upload Area */
[data-testid="stFileUploader"] {
    background-color: #1E293B;
    border-radius: 15px;
    padding: 15px;
}

/* Inputs */
.stSelectbox,
.stTextInput {
    border-radius: 10px;
}

/* KPI Cards */
.kpi-card {
    background-color: #1E293B;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;color:white'>
📦 InventoryAI Dashboard
</h1>
<p style='text-align:center;color:#94A3B8'>
AI-Powered Inventory Forecasting & Business Analytics
</p>
""", unsafe_allow_html=True)

currency_symbol = st.sidebar.selectbox(
    "Currency",
    ["$", "₦", "£", "€"]
)

def format_currency(num):
    if num >= 1_000_000_000:
        return f"{currency_symbol}{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{currency_symbol}{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{currency_symbol}{num/1_000:.1f}K"
    else:
        return f"{currency_symbol}{num:,.0f}"

st.set_page_config(page_title="InventoryAI Dashboard", layout="wide")

st.title("📊 InventoryAI - Retail Analytics & Forecasting")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file.")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip()

def find_column(possible_names):
    for col in df.columns:
        if col.lower().strip() in [x.lower() for x in possible_names]:
            return col
    return None

product_guess = find_column(["Product ID","Product","SKU","Item","Item Code"])
category_guess = find_column(["Category","Product Category"])
region_guess = find_column(["Region","Location","Area"])
inventory_guess = find_column(["Inventory Level","Inventory","Stock","Stock Level"])
sales_guess = find_column(["Units Sold","Sales","Qty Sold","Quantity Sold"])
forecast_guess = find_column(["Demand Forecast","Forecast","Projected Demand"])
price_guess = find_column(["Price","Unit Price","Selling Price"])

st.subheader("🛠 Dataset Mapping")

product_col = st.selectbox("Product Column", df.columns,
                           index=df.columns.get_loc(product_guess) if product_guess else 0)
category_col = st.selectbox("Category Column", df.columns,
                            index=df.columns.get_loc(category_guess) if category_guess else 0)
region_col = st.selectbox("Region Column", df.columns,
                          index=df.columns.get_loc(region_guess) if region_guess else 0)
inventory_col = st.selectbox("Inventory Column", df.columns,
                             index=df.columns.get_loc(inventory_guess) if inventory_guess else 0)
sales_col = st.selectbox("Sales Column", df.columns,
                         index=df.columns.get_loc(sales_guess) if sales_guess else 0)
forecast_col = st.selectbox("Forecast Column", df.columns,
                            index=df.columns.get_loc(forecast_guess) if forecast_guess else 0)
price_col = st.selectbox("Price Column", df.columns,
                         index=df.columns.get_loc(price_guess) if price_guess else 0)

analysis_df = pd.DataFrame()
analysis_df["Product"] = df[product_col].astype(str)
analysis_df["Category"] = df[category_col].astype(str)
analysis_df["Region"] = df[region_col].astype(str)
analysis_df["Inventory"] = pd.to_numeric(df[inventory_col], errors="coerce")
analysis_df["Sales"] = pd.to_numeric(df[sales_col], errors="coerce")
analysis_df["Forecast"] = pd.to_numeric(df[forecast_col], errors="coerce")
analysis_df["Price"] = pd.to_numeric(df[price_col], errors="coerce")
analysis_df["Revenue"] = analysis_df["Sales"] * analysis_df["Price"]
analysis_df = analysis_df.dropna()

st.subheader("Dataset Preview")
st.dataframe(analysis_df.head())

st.subheader("📈 Business KPIs")
revenue = analysis_df["Revenue"].sum() if "Revenue" in analysis_df.columns else 0

products = analysis_df["Product"].nunique() if "Product" in analysis_df.columns else 0

forecast = analysis_df["Forecast"].sum() if "Forecast" in analysis_df.columns else 0

risk_count = len(
    analysis_df[
        analysis_df["Inventory"] < analysis_df["Forecast"]
    ]
) if (
    "Inventory" in analysis_df.columns and
    "Forecast" in analysis_df.columns
) else 0
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Revenue",
        format_currency(revenue)
    )

with col2:
    st.metric(
        "Forecast Revenue",
        format_currency(forecast_revenue)
    )

with col3:
    st.metric(
        "Products",
        format_currency(products)
    )

with col2:
    st.metric(
        "Risk",
        format_currency(risk_count)
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Inventory", f"{int(analysis_df['Inventory'].sum()):,}")
c2.metric("Total Sales", f"{int(analysis_df['Sales'].sum()):,}")
c3.metric("Average Forecast", f"{analysis_df['Forecast'].mean():.2f}")
c4.metric("Revenue", f"${analysis_df['Revenue'].sum():,.2f}")

st.subheader("🤖 AI Demand Forecasting")

model_df = pd.get_dummies(
    analysis_df[["Category","Region","Inventory","Sales","Price","Forecast"]],
    columns=["Category","Region"],
    drop_first=True
)

X = model_df.drop("Forecast", axis=1)
y = model_df["Forecast"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5
r2 = r2_score(y_test, preds)

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:.2f}")
m2.metric("RMSE", f"{rmse:.2f}")
m3.metric("R²", f"{r2:.2f}")

st.subheader("📦 Product Analysis")

products = sorted(analysis_df["Product"].unique())
selected_product = st.selectbox("Select Product", products)

product_df = analysis_df[analysis_df["Product"] == selected_product]

st.dataframe(product_df)

prediction_row = pd.get_dummies(
    product_df[["Category","Region","Inventory","Sales","Price"]].iloc[[0]],
    columns=["Category","Region"],
    drop_first=True
)

prediction_row = prediction_row.reindex(columns=X.columns, fill_value=0)

predicted_demand = model.predict(prediction_row)[0]

st.metric("🔮 Predicted Demand", f"{predicted_demand:.2f}")

inventory = product_df["Inventory"].mean()
recommended_stock = predicted_demand * 1.2
restock_qty = recommended_stock - inventory

st.subheader("🤖 Restock Recommendation")

if restock_qty > 0:
    st.error(f"🚨 Restock Recommended: {int(restock_qty)} units")
else:
    st.success("✅ Current inventory is sufficient")

st.subheader("⚠️ Inventory Risk Analysis")

at_risk = analysis_df[
    analysis_df["Inventory"] < analysis_df["Forecast"]
].copy()

if len(at_risk) > 0:

    at_risk["Shortage"] = (
        at_risk["Forecast"] -
        at_risk["Inventory"]
    )

    def risk_level(shortage):
        if shortage > 100:
            return "High"
        elif shortage > 50:
            return "Medium"
        return "Low"

    at_risk["Risk Level"] = (
        at_risk["Shortage"]
        .apply(risk_level)
    )

    at_risk["Recommended Order"] = (
        at_risk["Shortage"] * 1.2
    ).round()

    risk_table = at_risk[
        [
            "Product",
            "Category",
            "Inventory",
            "Forecast",
            "Shortage",
            "Risk Level",
            "Recommended Order"
        ]
    ]

    risk_table = risk_table.sort_values(
        by="Shortage",
        ascending=False
    )

    st.metric(
        "Products at Risk",
        len(risk_table)
    )

    st.dataframe(
        risk_table,
        use_container_width=True
    )

else:
    st.success("No inventory risks detected.")

st.subheader("💰 Revenue by Category")
st.bar_chart(
    analysis_df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
)

st.subheader("🌍 Revenue by Region")
st.bar_chart(
    analysis_df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
)

st.subheader("🏆 Top 10 Best Selling Products")
top_products = (
    analysis_df.groupby("Product")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
st.bar_chart(top_products)

st.subheader("🏷️ Category Performance")
st.bar_chart(
    analysis_df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
)

st.subheader("🌍 Regional Performance")
st.bar_chart(
    analysis_df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
)

st.subheader("📊 Inventory vs Demand")
st.line_chart(product_df[["Inventory", "Forecast"]])

if "Date" in df.columns:
    try:
        temp = df.copy()
        temp["Date"] = pd.to_datetime(temp["Date"])
        temp["Revenue"] = analysis_df["Revenue"].values[:len(temp)]
        daily = temp.groupby("Date")["Revenue"].sum()
        st.subheader("📈 Revenue Trend")
        st.line_chart(daily)
    except:
        pass

if "Store ID" in df.columns:
    try:
        temp = df.copy()
        temp["Sales"] = analysis_df["Sales"].values[:len(temp)]
        st.subheader("🏪 Store Performance")
        store_sales = temp.groupby("Store ID")["Sales"].sum()
        st.bar_chart(store_sales)
    except:
        pass

st.subheader("⚠️ Low Stock Products")
low_stock = analysis_df[analysis_df["Inventory"] < 100]
st.dataframe(low_stock)

csv = product_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Product Data",
    csv,
    file_name=f"{selected_product}_data.csv",
    mime="text/csv"
)
