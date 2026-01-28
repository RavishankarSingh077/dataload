from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from data import load_data
from features import add_features
import os

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Constants
MODEL_INTRADAY = "model_intraday.pkl"
MODEL_DAILY = "model_daily.pkl"
CONFIDENCE_THRESHOLD = 0.6

def get_prediction(symbol, model_type="intraday"):
    # Select model and parameters
    if model_type == "daily":
        model_path = f"model_{symbol}_daily.pkl"
        if not os.path.exists(model_path):
             model_path = MODEL_DAILY # Fallback
        interval = "1d"
        period = "1y"
        expected_move_pct = 1.5 # 1.5% for daily
        time_hint = "next day"
    else:
        model_path = MODEL_INTRADAY
        interval = "5m"
        period = "5d"
        expected_move_pct = 0.2 # 0.2% for 15m
        time_hint = "15 mins"

    if not os.path.exists(model_path):
        return {"error": f"Model {model_path} not found. Please run train.py first."}

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        df = load_data(symbol, period=period, interval=interval)
        df = add_features(df)
        
        feature_cols = [
            "return", "volume_change", "rsi", "ema_9", "ema_20", "ema_50", 
            "ema_cross_9_20", "ema_cross_20_50", "dist_ema_9", "dist_ema_50", 
            "macd", "adx", "dist_ichimoku_a", "dist_ichimoku_base",
            "bb_high_diff", "bb_low_diff", "vwap_diff", "atr", "dist_atr",
            "obv_change", "mfi", "stoch_k", "stoch_d", "cci",
            "cmf", "force_index", "vpt"
        ]
        for i in range(1, 4):
            feature_cols.append(f"return_lag_{i}")
            feature_cols.append(f"volume_lag_{i}")
            
        X = df[feature_cols].iloc[-1:]

        # Predict
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
        prob_map = dict(zip(classes, probs))
        
        up_prob = float(prob_map.get(1, 0))
        not_up_prob = float(prob_map.get(0, 0))

        if up_prob >= CONFIDENCE_THRESHOLD:
            decision = f"BUY ({model_type.upper()})"
            signal_class = "buy"
        else:
            decision = "NO TRADE"
            signal_class = "neutral"

        # Determine currency symbol based on ticker suffix
        currency = "₹" if symbol.endswith(".NS") or symbol.endswith(".BO") else "$"

        last_price = float(df["Close"].iloc[-1])
        
        target_price = last_price * (1 + (expected_move_pct / 100))
        risk_price = last_price * (1 - (expected_move_pct / 100))

        return {
            "symbol": symbol,
            "up_prob": round(up_prob * 100, 2),
            "down_prob": 0, # Hidden in binary
            "no_trade_prob": round(not_up_prob * 100, 2),
            "decision": decision,
            "signal_class": signal_class,
            "last_price": round(last_price, 2),
            "target_price": round(target_price, 2),
            "risk_price": round(risk_price, 2),
            "expected_move": expected_move_pct,
            "time_hint": time_hint,
            "currency": currency
        }
    except Exception as e:
        return {"error": str(e)}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    symbol = request.form.get("symbol", "AAPL").upper()
    model_type = request.form.get("model_type", "intraday")
    result = get_prediction(symbol, model_type)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
