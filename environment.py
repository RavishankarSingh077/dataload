import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Custom Trading Environment for RL training.
    """
    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        
        # Action space: 0 = HOLD, 1 = BUY, 2 = SELL (for shorting)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: features from the dataframe
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = 0
        self.shares_held = 0
        self.total_profit = 0
        return self._get_observation(), {}

    def _get_observation(self):
        return self.df.iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Execute action
        if action == 1: # BUY
            if self.balance > current_price:
                self.shares_held += 1
                self.balance -= current_price
        elif action == 2: # SELL (Close long or Short)
            if self.shares_held > 0:
                self.balance += current_price
                self.shares_held -= 1
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Reward: Portfolio Value change
        portfolio_value = self.balance + (self.shares_held * current_price)
        reward = portfolio_value - self.initial_balance
        
        return self._get_observation(), reward, done, False, {}

