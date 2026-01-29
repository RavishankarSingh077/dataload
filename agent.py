import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from collections import deque

class DQNAgent:
    """
    Sniper Elite Agent: Uses a Q-Network to learn profitable actions.
    """
    def __init__(self, state_size, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Q-Network approximation using MLP
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            learning_rate_init=0.001,
            max_iter=1,
            warm_start=True
        )
        self._initialized = False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if not self._initialized or np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        X = []
        y = []
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                if self._initialized:
                    target = (reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0]))
                else:
                    target = reward
            
            if not self._initialized:
                # Initialize model with zero-like targets for the first time
                initial_targets = np.zeros(self.action_size)
                initial_targets[action] = target
                self.model.fit(state.reshape(1, -1), initial_targets.reshape(1, -1))
                self._initialized = True
            
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][action] = target
            X.append(state)
            y.append(target_f[0])
            
        self.model.fit(np.array(X), np.array(y))
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        import pickle
        with open(name, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, name):
        import pickle
        with open(name, 'rb') as f:
            self.model = pickle.load(f)
        self._initialized = True
