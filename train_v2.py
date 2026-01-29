import pandas as pd
import numpy as np
from data import load_data
from features import add_features
from environment import TradingEnv
from agent import DQNAgent
from sentiment import get_sentiment_score
import os

def train_v2(symbol="MON100.NS", episodes=10):
    # 1. Load Data
    df_raw = load_data(symbol, interval="1d", period="5y")
    df = add_features(df_raw)
    
    # Add Sentiment as a feature (simplification for training)
    # In live, it's dynamic. Here we assume 0 or small variations.
    df['sentiment'] = 0.0 
    
    # Prepare features for RL
    feature_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'vwap']]
    df_train = df[feature_cols].copy()
    
    # 2. Setup Env and Agent
    env = TradingEnv(df)
    state_size = len(feature_cols)
    agent = DQNAgent(state_size)
    batch_size = 32
    
    print(f"--- Training Elite Pro v2 Agent for {symbol} ---")
    for e in range(episodes):
        state, _ = env.reset()
        for time in range(len(df)-1):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e+1}/{episodes}, Profit: {env.total_profit:.2f}, Epsilon: {agent.epsilon:.2f}")
                break
            if len(agent.memory) > batch_size and time % 10 == 0: # Fit every 10 steps
                agent.replay(batch_size)
        
    # 3. Save the v2 Model
    model_name = f"agent_{symbol}_v2.pkl"
    agent.save(model_name)
    print(f"Elite Pro v2 Model saved as {model_name}")

if __name__ == "__main__":
    elite_stocks = ["MON100.NS", "RELIANCE.NS", "BAJFINANCE.NS", "MAZDOCK.NS"]
    
    for symbol in elite_stocks:
        train_v2(symbol, episodes=5)
        
    print("\n--- All Elite Pro v2 Models Trained Locally ---")
