from pathlib import Path
import csv
import pandas as pd


# Project folders
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SALES_FILE = RAW_DIR / "electronics_shop_raw.xlsx"
TELEMETRY_FILE = RAW_DIR / "drone_telemetry_raw.csv"


def clean_sales_data():
    """Load, clean and filter the drone sales dataset."""

    sales = pd.read_excel(SALES_FILE)
    original_rows = len(sales)

    sales.columns = sales.columns.str.strip().str.lower()

    sales["product_name"] = sales["product_name"].astype(str).str.strip()
    sales["location"] = sales["location"].astype(str).str.strip().str.title()
    sales["channel_of_ordering"] = (
        sales["channel_of_ordering"].astype(str).str.strip().str.title()
    )
    sales["order_status"] = sales["order_status"].astype(str).str.strip().str.title()

    sales["product_price"] = pd.to_numeric(
        sales["product_price"], errors="coerce"
    )
    sales["number_of_ordered_items"] = pd.to_numeric(
        sales["number_of_ordered_items"], errors="coerce"
    )

    # Convert Excel serial numbers into real dates
    sales["date"] = pd.to_datetime(
        sales["date"],
        unit="D",
        origin="1899-12-30",
        errors="coerce",
    )

    duplicate_sales = sales.duplicated().sum()
    sales = sales.drop_duplicates()

    # Keep only Drone records
    drone_sales = sales[sales["product_name"].str.lower() == "drone"].copy()
    drone_rows_before_validation = len(drone_sales)

    invalid_sales = drone_sales[
        drone_sales["product_price"].isna()
        | drone_sales["number_of_ordered_items"].isna()
        | drone_sales["date"].isna()
        | (drone_sales["product_price"] <= 0)
        | (drone_sales["number_of_ordered_items"] <= 0)
    ].copy()

    drone_sales = drone_sales.drop(invalid_sales.index).copy()

    drone_sales["sales_value"] = (
        drone_sales["product_price"] * drone_sales["number_of_ordered_items"]
    )

    drone_sales["week_start"] = (
        drone_sales["date"].dt.to_period("W").apply(lambda period: period.start_time)
    )

    weekly_sales = (
        drone_sales.groupby("week_start", as_index=False)
        .agg(
            ordered_quantity=("number_of_ordered_items", "sum"),
            estimated_sales_value=("sales_value", "sum"),
            order_count=("productid", "count"),
        )
        .sort_values("week_start")
    )

    drone_sales.to_csv(PROCESSED_DIR / "cleaned_sales.csv", index=False)
    weekly_sales.to_csv(
        PROCESSED_DIR / "weekly_sales_summary.csv",
        index=False,
    )

    return {
        "original_sales_rows": original_rows,
        "duplicate_sales_rows": duplicate_sales,
        "drone_rows_before_validation": drone_rows_before_validation,
        "invalid_drone_sales_rows": len(invalid_sales),
        "cleaned_drone_sales_rows": len(drone_sales),
    }


def clean_telemetry_data():
    """Validate CSV structure, clean telemetry data and create risk indicators."""

    malformed_rows = []
    valid_rows = []

    with open(TELEMETRY_FILE, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        expected_fields = len(header)

        for row_number, row in enumerate(reader, start=2):
            if len(row) != expected_fields:
                malformed_rows.append(
                    {
                        "row_number": row_number,
                        "field_count": len(row),
                        "expected_field_count": expected_fields,
                        "row_content": " | ".join(row),
                    }
                )
            else:
                valid_rows.append(row)

    telemetry = pd.DataFrame(valid_rows, columns=header)
    original_valid_rows = len(telemetry)

    numeric_columns = [
        "Max Carry Weight (kg)",
        "Actual Carry Weight (kg)",
        "Altitude (meters)",
        "Flight Duration (minutes)",
        "Distance Flown (km)",
        "Battery Remaining (%)",
        "GPS Accuracy (meters)",
        "Wind Speed (m/s)",
    ]

    for column in numeric_columns:
        telemetry[column] = pd.to_numeric(telemetry[column], errors="coerce")

    telemetry["Flight Date"] = pd.to_datetime(
        telemetry["Flight Date"],
        errors="coerce",
    )

    telemetry["Flight Status"] = telemetry["Flight Status"].astype(str).str.strip()
    telemetry["Obstacles Encountered"] = (
        telemetry["Obstacles Encountered"].astype(str).str.strip().str.title()
    )

    invalid_telemetry = telemetry[
        telemetry["Battery Remaining (%)"].isna()
        | telemetry["Actual Carry Weight (kg)"].isna()
        | telemetry["Max Carry Weight (kg)"].isna()
        | telemetry["Flight Date"].isna()
        | (telemetry["Battery Remaining (%)"] < 0)
        | (telemetry["Battery Remaining (%)"] > 100)
        | (telemetry["Actual Carry Weight (kg)"] < 0)
        | (telemetry["Max Carry Weight (kg)"] <= 0)
    ].copy()

    telemetry = telemetry.drop(invalid_telemetry.index).copy()

    telemetry["payload_ratio"] = (
        telemetry["Actual Carry Weight (kg)"]
        / telemetry["Max Carry Weight (kg)"]
    ).round(2)

    telemetry["risk_points"] = 0

    telemetry.loc[
        telemetry["Flight Status"].str.lower() != "completed",
        "risk_points",
    ] += 1

    telemetry.loc[
        telemetry["Obstacles Encountered"].str.lower() == "yes",
        "risk_points",
    ] += 1

    telemetry.loc[
        telemetry["Battery Remaining (%)"] <= 25,
        "risk_points",
    ] += 1

    telemetry.loc[
        telemetry["Actual Carry Weight (kg)"]
        > telemetry["Max Carry Weight (kg)"],
        "risk_points",
    ] += 1

    telemetry["risk_level"] = pd.cut(
        telemetry["risk_points"],
        bins=[-1, 0, 1, 4],
        labels=["Low", "Medium", "High"],
    )

    telemetry.to_csv(
        PROCESSED_DIR / "cleaned_telemetry.csv",
        index=False,
    )

    telemetry[
        [
            "Drone ID",
            "Drone Model",
            "Battery Remaining (%)",
            "Actual Carry Weight (kg)",
            "Max Carry Weight (kg)",
            "Wind Speed (m/s)",
            "Obstacles Encountered",
            "Flight Status",
            "risk_points",
            "risk_level",
        ]
    ].to_csv(
        PROCESSED_DIR / "telemetry_risk_summary.csv",
        index=False,
    )

    pd.DataFrame(malformed_rows).to_csv(
        OUTPUT_DIR / "telemetry_error_log.csv",
        index=False,
    )

    return {
        "valid_telemetry_rows_before_cleaning": original_valid_rows,
        "malformed_or_blank_rows": len(malformed_rows),
        "invalid_telemetry_rows": len(invalid_telemetry),
        "cleaned_telemetry_rows": len(telemetry),
        "high_risk_flights": int((telemetry["risk_level"] == "High").sum()),
    }


def main():
    print("\n--- PERCEPTONER DATA CLEANING ---")

    sales_results = clean_sales_data()
    telemetry_results = clean_telemetry_data()

    quality_report = pd.DataFrame(
        [
            {
                "dataset": "Sales",
                "original_rows": sales_results["original_sales_rows"],
                "duplicate_rows": sales_results["duplicate_sales_rows"],
                "invalid_rows": sales_results["invalid_drone_sales_rows"],
                "cleaned_rows": sales_results["cleaned_drone_sales_rows"],
            },
            {
                "dataset": "Telemetry",
                "original_rows": telemetry_results[
                    "valid_telemetry_rows_before_cleaning"
                ],
                "duplicate_rows": 0,
                "invalid_rows": telemetry_results["invalid_telemetry_rows"],
                "cleaned_rows": telemetry_results["cleaned_telemetry_rows"],
            },
        ]
    )

    quality_report.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)

    print(f"Original sales rows: {sales_results['original_sales_rows']}")
    print(
        "Drone sales rows after cleaning: "
        f"{sales_results['cleaned_drone_sales_rows']}"
    )
    print(
        "Malformed or blank telemetry rows: "
        f"{telemetry_results['malformed_or_blank_rows']}"
    )
    print(
        "Cleaned telemetry rows: "
        f"{telemetry_results['cleaned_telemetry_rows']}"
    )
    print(f"High-risk flights: {telemetry_results['high_risk_flights']}")
    print("\nCleaning completed successfully.")
    print(f"Processed data folder: {PROCESSED_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()