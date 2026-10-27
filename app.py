from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


st.set_page_config(
    page_title="Perceptoner Analytics",
    page_icon="🚁",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"


@st.cache_data
def load_data():
    sales = pd.read_csv(
        PROCESSED_DIR / "cleaned_sales.csv",
        parse_dates=["date", "week_start"],
    )

    weekly_sales = pd.read_csv(
        PROCESSED_DIR / "weekly_sales_summary.csv",
        parse_dates=["week_start"],
    )

    telemetry = pd.read_csv(
        PROCESSED_DIR / "cleaned_telemetry.csv",
        parse_dates=["Flight Date"],
    )

    risk_summary = pd.read_csv(
        PROCESSED_DIR / "telemetry_risk_summary.csv",
    )

    quality_report = pd.read_csv(
        OUTPUT_DIR / "data_quality_report.csv",
    )

    error_log_path = OUTPUT_DIR / "telemetry_error_log.csv"

    if error_log_path.exists():
        error_log = pd.read_csv(error_log_path)
    else:
        error_log = pd.DataFrame()

    return sales, weekly_sales, telemetry, risk_summary, quality_report, error_log


def create_forecast(weekly_sales):
    """Compare moving-average and linear-trend models using MAE."""

    data = weekly_sales.sort_values("week_start").copy()
    values = data["ordered_quantity"].to_numpy()

    test_size = max(4, int(len(values) * 0.20))
    train_values = values[:-test_size]
    test_values = values[-test_size:]

    window = min(4, len(train_values))
    moving_average_prediction = np.repeat(
        np.mean(train_values[-window:]),
        test_size,
    )

    train_x = np.arange(len(train_values)).reshape(-1, 1)
    test_x = np.arange(len(train_values), len(values)).reshape(-1, 1)

    trend_model = LinearRegression()
    trend_model.fit(train_x, train_values)
    trend_prediction = trend_model.predict(test_x)

    moving_average_mae = mean_absolute_error(
        test_values,
        moving_average_prediction,
    )
    trend_mae = mean_absolute_error(
        test_values,
        trend_prediction,
    )

    if moving_average_mae <= trend_mae:
        selected_model = "Moving Average"
        selected_mae = moving_average_mae
        forecast_value = float(np.mean(values[-window:]))
    else:
        selected_model = "Linear Trend"
        selected_mae = trend_mae

        full_x = np.arange(len(values)).reshape(-1, 1)
        final_model = LinearRegression()
        final_model.fit(full_x, values)

        next_x = np.array([[len(values)]])
        forecast_value = float(final_model.predict(next_x)[0])

    forecast_date = data["week_start"].max() + pd.Timedelta(days=7)

    historical_chart = data[
        ["week_start", "ordered_quantity"]
    ].rename(columns={"ordered_quantity": "Actual Orders"})

    historical_chart["Forecast"] = np.nan

    forecast_chart = pd.DataFrame(
        {
            "week_start": [forecast_date],
            "Actual Orders": [np.nan],
            "Forecast": [round(max(forecast_value, 0), 2)],
        }
    )

    chart_data = pd.concat(
        [historical_chart, forecast_chart],
        ignore_index=True,
    )

    return (
        selected_model,
        selected_mae,
        forecast_value,
        forecast_date,
        chart_data,
    )


try:
    (
        sales,
        weekly_sales,
        telemetry,
        risk_summary,
        quality_report,
        error_log,
    ) = load_data()

except FileNotFoundError:
    st.error(
        "Processed data files were not found. "
        "Run `python src\\data_cleaning.py` first."
    )
    st.stop()


st.title("Perceptoner Analytics Dashboard")
st.caption(
    "Drone sales, operational-risk and data-quality decision support"
)

page = st.sidebar.radio(
    "Choose a dashboard view",
    ["Sales Overview", "Operations Risk", "Data Quality"],
)


if page == "Sales Overview":
    (
        selected_model,
        selected_mae,
        forecast_value,
        forecast_date,
        chart_data,
    ) = create_forecast(weekly_sales)

    total_sales_value = sales["sales_value"].sum()
    delivered_percentage = (
        sales["order_status"]
        .str.lower()
        .eq("delivered")
        .mean()
        * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Validated drone orders", f"{len(sales):,}")
    col2.metric("Estimated sales value", f"${total_sales_value:,.2f}")
    col3.metric(
        "Next-week forecast",
        f"{forecast_value:.0f} orders",
    )
    col4.metric(
        "Delivered orders",
        f"{delivered_percentage:.1f}%",
    )

    st.subheader("Weekly Drone Demand and Forecast")
    st.caption(
        f"Selected model: {selected_model} | "
        f"Validation MAE: {selected_mae:.2f}"
    )

    st.line_chart(
        chart_data.set_index("week_start")[
            ["Actual Orders", "Forecast"]
        ]
    )

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Demand by Location")

        location_summary = (
            sales.groupby("location", as_index=False)
            .agg(
                ordered_quantity=(
                    "number_of_ordered_items",
                    "sum",
                )
            )
            .sort_values("ordered_quantity", ascending=False)
            .head(10)
        )

        st.bar_chart(
            location_summary.set_index("location")[
                "ordered_quantity"
            ]
        )

    with right_column:
        st.subheader("Demand by Ordering Channel")

        channel_summary = (
            sales.groupby("channel_of_ordering", as_index=False)
            .agg(
                ordered_quantity=(
                    "number_of_ordered_items",
                    "sum",
                )
            )
            .sort_values("ordered_quantity", ascending=False)
        )

        st.bar_chart(
            channel_summary.set_index("channel_of_ordering")[
                "ordered_quantity"
            ]
        )

    st.subheader("Order Status Summary")

    order_status_summary = (
        sales["order_status"]
        .value_counts()
        .rename_axis("order_status")
        .reset_index(name="order_count")
    )

    order_status_summary["percentage"] = (
        order_status_summary["order_count"]
        / order_status_summary["order_count"].sum()
        * 100
    ).round(2)

    st.dataframe(
        order_status_summary,
        use_container_width=True,
        hide_index=True,
    )


elif page == "Operations Risk":
    st.subheader("Operational Risk Overview")

    high_risk = risk_summary[
        risk_summary["risk_level"].astype(str) == "High"
    ].copy()

    medium_risk = risk_summary[
        risk_summary["risk_level"].astype(str) == "Medium"
    ].copy()

    col1, col2, col3 = st.columns(3)

    col1.metric("Flights analysed", f"{len(risk_summary):,}")
    col2.metric("High-risk flights", f"{len(high_risk):,}")
    col3.metric("Medium-risk flights", f"{len(medium_risk):,}")

    st.subheader("Risk Level Distribution")

    risk_counts = (
        risk_summary["risk_level"]
        .value_counts()
        .rename_axis("risk_level")
        .reset_index(name="flight_count")
    )

    st.bar_chart(
        risk_counts.set_index("risk_level")["flight_count"]
    )

    st.subheader("High-Risk Flights Requiring Review")

    if high_risk.empty:
        st.success("No high-risk flights were identified.")
    else:
        st.dataframe(
            high_risk.sort_values(
                "risk_points",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Operational Conditions by Flight Status")

    flight_status_summary = (
        telemetry.groupby("Flight Status", as_index=False)
        .agg(
            average_battery=(
                "Battery Remaining (%)",
                "mean",
            ),
            average_wind_speed=(
                "Wind Speed (m/s)",
                "mean",
            ),
            average_payload_ratio=(
                "payload_ratio",
                "mean",
            ),
        )
        .round(2)
    )

    st.dataframe(
        flight_status_summary,
        use_container_width=True,
        hide_index=True,
    )


else:
    st.subheader("Data Quality Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Sales records after cleaning",
        int(
            quality_report.loc[
                quality_report["dataset"] == "Sales",
                "cleaned_rows",
            ].iloc[0]
        ),
    )

    col2.metric(
        "Telemetry records after cleaning",
        int(
            quality_report.loc[
                quality_report["dataset"] == "Telemetry",
                "cleaned_rows",
            ].iloc[0]
        ),
    )

    col3.metric(
        "Malformed or blank telemetry rows",
        len(error_log),
    )

    st.subheader("Data Quality Report")

    st.dataframe(
        quality_report,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Telemetry Validation Log")

    if error_log.empty:
        st.success("No malformed telemetry rows were recorded.")
    else:
        st.dataframe(
            error_log,
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "The dashboard uses cleaned data only. "
        "The original raw source files remain unchanged."
    )