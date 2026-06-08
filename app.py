import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Inventory AI", layout="wide")

st.title("📊 Smart Inventory & Sales Forecast System")

# =========================
# LOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()
    
df = pd.read_csv(uploaded_file)

# =========================
# PRODUCT SELECTION
# =========================
required_columns = ["day", "product", "sales", "stock"]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.write("Found columns:", df.columns.tolist())
    st.stop()

products = df["product"].unique()
product = st.selectbox("Select Product", products)

data = df[df["product"] == product].copy()

st.subheader(f"📦 Data for {product}")
st.write(data)

# =========================
# FEATURES
# =========================
data["lag_1"] = data["sales"].shift(1)
data["lag_2"] = data["sales"].shift(2)
data["moving_avg"] = data["sales"].rolling(3).mean()
data = data.dropna()

X = data[["day", "lag_1", "lag_2", "moving_avg"]]
y = data["sales"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# =========================
# KPI SECTION
# =========================
st.subheader("📊 Key Metrics")

col1, col2 = st.columns(2)
col1.metric("Total Sales", int(data["sales"].sum()))
col2.metric("Average Sales", round(data["sales"].mean(), 2))

# =========================
# STOCK INPUT
# =========================
st.subheader("📦 Current Stock")

stock = st.number_input("Enter current stock", min_value=0, value=int(data["stock"].iloc[-1]))

# =========================
# FORECAST
# =========================
st.subheader("🔮 Forecast (Next 14 Days)")

last_sales = data["sales"].iloc[-1]
last_lag2 = data["sales"].iloc[-2]

future_days = []
predictions = []

for i in range(1, 15):
    day = data["day"].max() + i

    lag_1 = last_sales
    lag_2 = last_lag2
    moving_avg = np.mean(data["sales"].tail(3))

    input_data = pd.DataFrame([[day, lag_1, lag_2, moving_avg]],
                              columns=["day", "lag_1", "lag_2", "moving_avg"])

    pred = model.predict(input_data)[0]

    future_days.append(day)
    predictions.append(pred)

    last_lag2 = last_sales
    last_sales = pred

forecast_df = pd.DataFrame({
    "Day": future_days,
    "Predicted Sales": predictions
})

st.line_chart(forecast_df.set_index("Day"))

# =========================
# RESTOCK ENGINE
# =========================
st.subheader("📦 Restock Recommendation")

total_demand = sum(predictions)
safety_stock = total_demand * 1.2  # 20% buffer

restock_qty = safety_stock - stock

if restock_qty > 0:
    st.error(f"🚨 Restock Needed: {int(restock_qty)} units of {product}")
else:
    st.success("✅ No Restock Needed")

# =========================
# DOWNLOAD
# =========================
csv = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Forecast",
    csv,
    "forecast.csv",
    "text/csv"
)
