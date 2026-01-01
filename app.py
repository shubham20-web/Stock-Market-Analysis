import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------- Load assets ----------
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model.keras")

@st.cache_resource
def load_scaler_and_config():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("config.json", "r") as f:
        config = json.load(f)
    return scaler, config

model = load_lstm_model()
scaler, config = load_scaler_and_config()

LOOK_BACK = config["look_back"]
FEATURE_COLS = config["feature_columns"]
FREQ = config.get("freq", "B")

# ---------- Forecast function ----------
def forecast_30_days_lstm(df):
    data = df[FEATURE_COLS].copy()
    scaled = scaler.transform(data.values)

    if len(scaled) < LOOK_BACK:
        raise ValueError(f"Need at least {LOOK_BACK} rows, found {len(scaled)}.")

    window = scaled[-LOOK_BACK:, :]
    future_steps = 30
    future_ts_log_diff = []

    for _ in range(future_steps):
        inp = window[np.newaxis, :, :]
        next_scaled = model.predict(inp, verbose=0)[0, 0]

        dummy = np.zeros((1, scaled.shape[1]))
        dummy[0, -1] = next_scaled
        inv = scaler.inverse_transform(dummy)

        next_log_diff = inv[0, -1]
        future_ts_log_diff.append(next_log_diff)

        new_row = np.zeros((1, scaled.shape[1]))
        new_row[0, -1] = next_scaled
        window = np.vstack([window[1:], new_row])

    last_date = df.attrs["LastCloseDate"]

    future_dates = pd.date_range(
        start=last_date,
        periods=future_steps + 1,
        freq=FREQ
    )[1:]

    base_price = df["BasePriceForPlot"].iloc[-1]
    last_log_price = np.log(base_price)

    log_prices = [last_log_price]
    for d in future_ts_log_diff:
        log_prices.append(log_prices[-1] + d)

    price_forecast = np.exp(log_prices[1:])

    return pd.DataFrame(
        {"price_forecast": price_forecast},
        index=future_dates
    )

# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="AAPL 30-Day LSTM Forecast",
    layout="centered"
)

st.title("AAPL 30-Day Stock Price Forecast (LSTM)")
st.write(
    "Upload recent Apple stock data to generate a **30-business-day** forecast."
)

uploaded = st.file_uploader(
    "Upload CSV with Date column + features",
    type=["csv"]
)

if uploaded is not None:
    df = pd.read_csv(uploaded)

    # ---- SAFE DATE PARSING (FIXED) ----
    if "Date" not in df.columns:
        st.error("CSV must contain a 'Date' column.")
        st.stop()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        dayfirst=True
    )

    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    if df.empty:
        st.error("No valid rows found. Please check Date format.")
        st.stop()

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    st.success(
        f"Data loaded: {len(df)} rows "
        f"({df.index.min().date()} → {df.index.max().date()})"
    )

    # ---- User Inputs ----
    base_price_input = st.number_input(
        "Last actual close price (USD):",
        min_value=1.0,
        value=150.0
    )

    last_close_date_input = st.date_input(
        "Last actual close date:",
        value=df.index.max().date()
    )

    df["BasePriceForPlot"] = base_price_input
    df.attrs["LastCloseDate"] = pd.to_datetime(last_close_date_input)

    # ---- Forecast ----
    try:
        forecast_df = forecast_30_days_lstm(df)
    except Exception as e:
        st.error(str(e))
    else:
        st.subheader("30-Day Forecast")
        st.dataframe(forecast_df.round(2))

        fig, ax = plt.subplots()
        ax.plot(forecast_df.index, forecast_df["price_forecast"])
        ax.set_title("Forecasted Price Path")
        ax.set_ylabel("Price (USD)")
        ax.grid(True)
        st.pyplot(fig)

else:
    st.info("Upload a CSV file to start forecasting.")
