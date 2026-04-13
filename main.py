import os, time, requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz

from sklearn.ensemble import RandomForestClassifier

# ================= CONFIG =================
SYMBOL = "EURUSD=X"
IST = pytz.timezone("Asia/Kolkata")

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

CONF_THRESHOLD = 0.78
COOLDOWN = 1800   # 30 min (forces quality)

# ================= TELEGRAM =================
def send(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})


# ================= DATA =================
def get_data():
    df = yf.download(SYMBOL, interval="5m", period="5d")
    df = df.dropna()
    df = df.iloc[:-1]  # remove incomplete candle
    return df


# ================= FEATURES =================
def build_features(df):
    d = df.copy()

    d["ema9"]  = d["Close"].ewm(span=9).mean().shift(1)
    d["ema21"] = d["Close"].ewm(span=21).mean().shift(1)

    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = (100 - (100/(1+gain/(loss+1e-10)))).shift(1)

    d["return"] = d["Close"].pct_change().shift(1)
    d["volatility"] = d["Close"].pct_change().rolling(10).std().shift(1)

    d = d.dropna()
    return d


# ================= LABEL =================
def create_labels(df):
    threshold = 0.00025  # ~2.5 pips

    future = df["Close"].shift(-1)
    diff = future - df["Close"]

    df["target"] = None
    df.loc[diff > threshold, "target"] = 1
    df.loc[diff < -threshold, "target"] = 0

    df = df.dropna()
    return df


# ================= MODEL =================
def train(df):
    features = ["ema9","ema21","rsi","return","volatility"]

    split = int(len(df)*0.8)
    train = df.iloc[:split]

    X = train[features]
    y = train["target"]

    model = RandomForestClassifier(n_estimators=150, max_depth=6)
    model.fit(X,y)

    return model


# ================= SIGNAL =================
def generate(model, df):
    features = ["ema9","ema21","rsi","return","volatility"]

    latest = df.iloc[-1]
    X = latest[features].values.reshape(1,-1)

    probs = model.predict_proba(X)[0]
    pred = np.argmax(probs)
    conf = max(probs)

    # ===== TREND FILTER =====
    trend_up = latest["ema9"] > latest["ema21"]
    trend_down = latest["ema9"] < latest["ema21"]

    if pred == 1 and not trend_up:
        return None

    if pred == 0 and not trend_down:
        return None

    # ===== VOLATILITY FILTER =====
    if latest["volatility"] < df["volatility"].mean():
        return None

    # ===== RSI FILTER =====
    if pred == 1 and latest["rsi"] > 60:
        return None
    if pred == 0 and latest["rsi"] < 40:
        return None

    if conf < CONF_THRESHOLD:
        return None

    direction = "🟢 BUY" if pred==1 else "🔴 SELL"

    return direction, conf


# ================= SESSION =================
def is_session():
    now = datetime.now(IST)
    h = now.hour

    return (13 <= h <= 16) or (18 <= h <= 22)


# ================= MAIN =================
def run():
    last_signal = 0

    send("🚀 Smart Forex Bot Started")

    while True:
        try:
            if not is_session():
                time.sleep(60)
                continue

            df = get_data()
            df = build_features(df)
            df = create_labels(df)

            model = train(df)

            signal = generate(model, df)

            now = time.time()

            if signal and (now - last_signal > COOLDOWN):
                direction, conf = signal

                msg = f"""
⚡ HIGH QUALITY SIGNAL
Pair: EURUSD
Direction: {direction}
Confidence: {round(conf*100,2)}%
Time: {datetime.now(IST).strftime('%H:%M:%S')}
                """

                send(msg)
                last_signal = now

            time.sleep(60)

        except Exception as e:
            send(f"Error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    run()
