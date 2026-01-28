import lightgbm as lgb
import pandas as pd
import pickle
from data import load_data
from features import add_features
from sklearn.model_selection import train_test_split

def train_model(symbol="AAPL", mode="intraday"):
    # 1. Load data
    if mode == "intraday":
        df = load_data(symbol, interval="5m", period="60d")
        shift_val = -3 # 15 mins
        threshold = 0.002 # 0.2%
        model_name = f"model_{symbol}_intraday.pkl"
    else: # daily
        df = load_data(symbol, interval="1d", period="5y") # Back to 5y for patterns
        shift_val = -1 # Next day
        threshold = 0.015 # 1.5% for robust swing moves
        model_name = f"model_{symbol}_daily.pkl"
    
    # 2. Add features
    df = add_features(df)

    # 3. Labeling
    df["future_return"] = df["Close"].shift(shift_val) / df["Close"] - 1

    def label(x):
        if pd.isna(x):
            return 0
        if x > threshold:
            return 1   # UP (The only thing we want to catch)
        else:
            return 0   # NOT UP (Down or Sideways)

    df["label"] = df["future_return"].apply(label)
    print(f"Class distribution for {mode}:")
    print(df["label"].value_counts())
    df.dropna(inplace=True)

    # 4. Define features and target
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
        
    X = df[feature_cols]
    y = df["label"]

    # 5. Split data (No shuffle for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 6. Initialize and train Model
    if mode == "daily":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=2000,
            max_depth=30,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
            min_samples_split=2,
            min_samples_leaf=1,
            ccp_alpha=0.0001 # Minimal pruning to keep high accuracy
        )
    else: # intraday
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=20,
            random_state=42,
            verbosity=-1
        )
    
    print(f"Training {mode} model on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    # 7. Save model
    with open(model_name, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_name}")
    
    # 8. Detailed Evaluation
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print(f"\n--- {mode.upper()} EVALUATION ---")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    target_stocks = ["RELIANCE.NS", "MON100.NS"]
    
    for symbol in target_stocks:
        print(f"\nTraining Elite Model for: {symbol}")
        train_model(symbol, mode="daily")
    
    print("\nTraining for RELIANCE and MON100 complete.")
