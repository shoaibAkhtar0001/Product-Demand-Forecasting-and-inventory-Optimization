
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="DMart Demand & Inventory",
    page_icon="📦",
    layout="wide"
)

st.title("📦 DMart Demand Forecasting & Inventory Optimization")
st.caption(
    "Random Forest demand forecasts | Inventory planning | "
    "Illustrative inventory assumptions"
)

BASE_DIR = Path(__file__).parent
FORECAST_FILE = BASE_DIR / "daily_demand_forecast.csv"
INVENTORY_FILE = BASE_DIR / "inventory_optimization_results.csv"

# --------------------------------------------------
# LOAD EXISTING EXPORTED RESULTS
# --------------------------------------------------
if not FORECAST_FILE.exists() or not INVENTORY_FILE.exists():
    st.error(
        "Required CSV files are missing. Place both exported "
        "CSV files in the same folder as app.py."
    )
    st.stop()

forecast = pd.read_csv(FORECAST_FILE)
inventory = pd.read_csv(INVENTORY_FILE)

required_forecast = {
    "Date", "ProductCategory", "Forecast_Units"
}
required_inventory = {
    "ProductCategory", "Current_Stock",
    "Reorder_Point", "Recommended_Order_Units",
    "Stock_Status"
}

if not required_forecast.issubset(forecast.columns):
    st.error("The forecast CSV does not have the expected columns.")
    st.stop()

if not required_inventory.issubset(inventory.columns):
    st.error("The inventory CSV does not have the expected columns.")
    st.stop()

forecast["Date"] = pd.to_datetime(forecast["Date"])
forecast["Forecast_Units"] = pd.to_numeric(
    forecast["Forecast_Units"], errors="coerce"
)
forecast = forecast.dropna(
    subset=["Date", "Forecast_Units", "ProductCategory"]
)

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Dashboard Controls")

categories = sorted(
    forecast["ProductCategory"].unique().tolist()
)

category = st.sidebar.selectbox(
    "Select product category",
    categories
)

horizon = st.sidebar.selectbox(
    "Forecast horizon",
    [7, 14, 30],
    index=2
)

category_forecast = forecast[
    forecast["ProductCategory"] == category
].sort_values("Date")

if category_forecast.empty:
    st.warning("No forecast records are available.")
    st.stop()

# Use the beginning of the exported forecast window.
# This does not generate a new forecast.
forecast_start = category_forecast["Date"].min()
forecast_end = forecast_start + pd.Timedelta(days=horizon - 1)

selected_forecast = category_forecast[
    category_forecast["Date"].between(
        forecast_start, forecast_end
    )
].copy()

if selected_forecast.empty:
    st.warning("No forecast data is available for this horizon.")
    st.stop()

# --------------------------------------------------
# TOP SUMMARY CARDS
# --------------------------------------------------
total_units = selected_forecast["Forecast_Units"].sum()
average_daily = selected_forecast["Forecast_Units"].mean()

inventory_row = inventory[
    inventory["ProductCategory"] == category
]

st.subheader("Demand Forecast Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Product category", category)
c2.metric("Forecast horizon", f"{horizon} days")
c3.metric("Forecast demand", f"{total_units:,.0f} units")
c4.metric("Average daily demand", f"{average_daily:,.1f} units")

st.caption(
    f"Forecast window: {forecast_start:%d %b %Y} "
    f"to {selected_forecast['Date'].max():%d %b %Y}"
)

# --------------------------------------------------
# DEMAND FORECAST CHART
# --------------------------------------------------
st.divider()
st.subheader("📈 Predicted Daily Demand")

fig = px.line(
    selected_forecast,
    x="Date",
    y="Forecast_Units",
    markers=True,
    title=f"{category}: {horizon}-Day Demand Forecast",
    labels={
        "Date": "Date",
        "Forecast_Units": "Predicted units"
    }
)

fig.update_layout(
    hovermode="x unified",
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("View daily forecast records"):
    st.dataframe(
        selected_forecast,
        use_container_width=True,
        hide_index=True
    )

st.download_button(
    "Download selected forecast CSV",
    data=selected_forecast.to_csv(index=False).encode("utf-8"),
    file_name=f"forecast_{horizon}_days.csv",
    mime="text/csv"
)

# --------------------------------------------------
# INVENTORY OPTIMIZATION
# --------------------------------------------------
st.divider()
st.subheader("📦 Inventory Optimization")

if inventory_row.empty:
    st.warning(
        "No inventory assumptions exist for this category."
    )
else:
    item = inventory_row.iloc[0]

    i1, i2, i3, i4 = st.columns(4)

    i1.metric(
        "Illustrative current stock",
        f"{item['Current_Stock']:,.0f} units"
    )

    i2.metric(
        "Reorder point",
        f"{item['Reorder_Point']:,.0f} units"
    )

    i3.metric(
        "Recommended order",
        f"{item['Recommended_Order_Units']:,.0f} units"
    )

    i4.metric(
        "Stock status",
        str(item["Stock_Status"])
    )

    if str(item["Stock_Status"]).upper() == "REORDER":
        st.warning(
            "Replenishment is indicated under the current "
            "illustrative inventory assumptions."
        )
    else:
        st.success(
            "Current illustrative inventory is above the "
            "calculated reorder threshold."
        )

    # Compare current stock, reorder point and target stock
    chart_columns = [
        "Current_Stock", "Reorder_Point", "Target_Stock"
    ]

    if all(col in inventory.columns for col in chart_columns):
        comparison = pd.DataFrame({
            "Inventory metric": [
                "Current stock", "Reorder point", "Target stock"
            ],
            "Units": [
                item["Current_Stock"],
                item["Reorder_Point"],
                item["Target_Stock"]
            ]
        })

        fig2 = px.bar(
            comparison,
            x="Inventory metric",
            y="Units",
            text="Units",
            title=f"{category}: Stock and Replenishment Levels"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Inventory values are illustrative assumptions, not "
        "verified DMart stock or supplier data."
    )

# --------------------------------------------------
# ALL-CATEGORY SUMMARY
# --------------------------------------------------
st.divider()
st.subheader("📊 All-Category Inventory Summary")

st.dataframe(
    inventory,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download inventory recommendations CSV",
    data=inventory.to_csv(index=False).encode("utf-8"),
    file_name="inventory_recommendations.csv",
    mime="text/csv"
)

st.info(
    "This dashboard displays forecasts exported from your "
    "existing model. It does not retrain the model or generate "
    "new forecasts when you change the dashboard filters."
)