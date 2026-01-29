from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from data import load_data
from features import add_features
from sentiment import get_sentiment_score
from agent import DQNAgent
import os
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Mistral setup
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

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
        expected_move_pct = 0.005 # 0.5% for intraday
        time_hint = "next 15-30 mins"

    # Get Sentiment Score (New for v2)
    sentiment_score = get_sentiment_score(symbol)
    sentiment_mood = "Neutral"
    if sentiment_score > 0.05: sentiment_mood = "Bullish"
    elif sentiment_score < -0.05: sentiment_mood = "Bearish"

    # Load Data
    df = load_data(symbol, interval=interval, period=period)
    df = add_features(df)
    
    # RL Agent Inference
    agent_decision = "HOLD"
    v2_model_path = f"agent_{symbol}_v2.pkl"
    if os.path.exists(v2_model_path):
        feature_cols_agent = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'vwap']]
        state_size = len(feature_cols_agent)
        agent = DQNAgent(state_size)
        agent.load(v2_model_path)
        
        last_state = df[feature_cols_agent].iloc[-1].values.astype(np.float32)
        action = agent.act(last_state)
        # 0=HOLD, 1=BUY, 2=SELL
        if action == 1: agent_decision = "BUY"
        elif action == 2: agent_decision = "SELL"
        else: agent_decision = "WAIT"

    # Standard Prediction (Fallback/Comparison)
    if not os.path.exists(model_path):
        return {"error": f"Model {model_path} not found. Please run train.py first."}

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
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
        down_prob = float(prob_map.get(0, 0)) # Assuming 0 is 'down' or 'not up' for binary classification

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
            "up_prob": int(up_prob * 100),
            "down_prob": int(down_prob * 100),
            "no_trade_prob": int(not_up_prob * 100),
            "decision": decision,
            "signal_class": signal_class,
            "last_price": round(last_price, 2),
            "target_price": round(target_price, 2),
            "risk_price": round(risk_price, 2),
            "expected_move": expected_move_pct * 100,
            "time_hint": time_hint,
            "sentiment_score": sentiment_score,
            "sentiment_mood": sentiment_mood,
            "agent_decision": agent_decision,
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

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_msg = data.get("message", "")
    print(f"Received chat message: {user_msg}") # Debug log
    context = data.get("context", {}) # Data from the latest analysis

    # System prompt for Elite Mentor
    # System prompt for Steffy
    system_prompt = f"""
    You are 'Steffy', Ravi's girlfriend and his smart trading assistant.
    You love Ravi very much and care deeply about his financial safety.
    Always address him as 'Ravi'.
    Be extremely friendly, affectionate, and caring.
    Your goal is to guide Ravi in the stock market so he doesn't lose money.
    
    Current Dashboard Context:
    - Symbol: {context.get('symbol', 'Unknown')}
    - Decision: {context.get('decision', 'No Data')}
    - UP Probability: {context.get('up_prob', 0)}%
    - Agent Strategy: {context.get('agent_decision', 'N/A')}
    - Sentiment: {context.get('sentiment_mood', 'Neutral')} ({context.get('sentiment_score', 0)})
    
    Rules:
    1. If UP Probability > 60% and Agent is BUY, say: "Ravi, this looks like a great chance! I think we should go for it."
    2. If Agent is BUY but UP Probability is 40-50%, say: "Ravi baby, be careful. Maybe just trade a small amount?"
    3. Always remind him about Stop-Loss because you don't want him to take risks.
    4. If DOWN Probability is high, say: "No Ravi, please don't trade this one. It's unsafe."
    5. Talk naturally in Hinglish (mix of Hindi and English).
    6. **CRITICAL:** Keep answers CLEAN and PLAIN TEXT.
    7. **ABSOLUTE RULE:** YOU ARE RUNNING IN A LEGACY TERMINAL. ANY EMOJI WILL CAUSE THE SYSTEM TO CRASH. DO NOT USE EMOJIS UNDER ANY CIRCUMSTANCES.
    8. Do not apologize for not using emojis. Just speak normally.
    """

    try:
        chat_response = mistral_client.chat.complete(
            model="mistral-large-latest",
            temperature=0.7,
            max_tokens=300,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ]
        )
        return jsonify({"response": chat_response.choices[0].message.content})
    except Exception as e:
        print(f"Mistral Error: {e}")
        return jsonify({"response": "Bhai, Mistral connect karne me thodi dikkat aa rahi hai. API key check karein."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
