import pandas as pd
from pathlib import Path

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

DATA_PATH = Path("../data/processed/modeling_dataset.csv")


@st.cache_data
def load_and_build_model():
    # Load base dataset
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    # Keep only rows with demand
    df = df.dropna(subset=["demand_mw"]).sort_values("timestamp").reset_index(drop=True)

    # Add lag features
    df["lag1"] = df["demand_mw"].shift(1)
    df["lag2"] = df["demand_mw"].shift(2)
    df["lag3"] = df["demand_mw"].shift(3)
    df = df.dropna(subset=["lag1", "lag2", "lag3"]).reset_index(drop=True)

    feature_cols = [
        "temperature",
        "dayofweek",
        "is_weekend",
        "is_holiday",
        "lag1",
        "lag2",
        "lag3",
    ]

    X = df[feature_cols]
    y = df["demand_mw"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Attach predictions only to the test period for plotting
    df_pred = df.iloc[len(X_train):].copy()
    df_pred["y_pred"] = y_pred

    return df, df_pred, mae, mape, feature_cols, model.feature_importances_


def main():
    st.title("Ontario Electricity Demand – Early-Warning Dashboard")
    st.caption("Daily 6:00 AM demand using IESO real-time totals with weather & lag features")

    df, df_pred, mae, mape, feature_cols, importances = load_and_build_model()

    # --- Sidebar filters ---
    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()

    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.sidebar.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

    if start_date > end_date:
        st.sidebar.error("Start date must be before end date.")

    # Filter prediction frame
    mask = (df_pred["timestamp"].dt.date >= start_date) & (df_pred["timestamp"].dt.date <= end_date)
    view = df_pred[mask]

    # --- KPIs ---
    col1, col2 = st.columns(2)
    col1.metric("MAE (MW)", f"{mae:,.0f}")
    col2.metric("MAPE (%)", f"{mape*100:,.2f}")

    st.markdown("### Actual vs Predicted Demand at 6:00 AM")

    if not view.empty:
        chart_data = view.set_index("timestamp")[["demand_mw", "y_pred"]]
        st.line_chart(chart_data)
    else:
        st.warning("No data in selected date range.")

    # --- Feature importance ---
    st.markdown("### Feature Importance (Random Forest)")
    fi = pd.DataFrame(
        {"feature": feature_cols, "importance": importances}
    ).sort_values("importance", ascending=False)

    st.bar_chart(fi.set_index("feature"))

    # --- Raw data preview ---
    with st.expander("Show raw prediction data"):
        st.dataframe(view[["timestamp", "demand_mw", "y_pred"]])


if __name__ == "__main__":
    main()