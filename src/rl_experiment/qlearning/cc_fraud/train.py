# 1. Environment Setup
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from tqdm import tqdm
import pandas as pd

TRAIN_INTERVAL = 1000
VAL_INTERVAL = 5000  # Steps between validations
VAL_SIZE = 1000      # Transactions per validation

def progressive_validation(q_network, env):
    val_states = []
    val_labels = []
    for _ in range(VAL_SIZE):
        state = env.X[np.random.randint(0, len(env.X))]
        val_states.append(state)
        val_labels.append(env.y[np.random.randint(0, len(env.y))])
    
    val_states = torch.FloatTensor(np.array(val_states))
    with torch.no_grad():
        preds = torch.argmax(q_network(val_states), dim=1)
    accuracy = (preds == torch.Tensor(val_labels)).float().mean()
    return accuracy.item()

# 2. Custom Gymnasium Environment
class FraudDetectionEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)  # 0=genuine, 1=fraud
        self.observation_space = gym.spaces.Box(low=-30, high=30, shape=(29,))  # PCA features
        
        # Load Kaggle dataset
        self.data = self._load_dataset("data/creditcard.csv")  
        self.max_steps = len(self.data)
        self.current_step = 0

    def reset(self, seed=None):
        self.current_step = 0
        return self.data[self.current_step], {}

    def step(self, action):
        reward = self._calculate_reward(action, self.data[self.current_step][-1])
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        return self.data[self.current_step - 1], reward, terminated, False, {}

    def _calculate_reward(self, action, true_label):
        # Based on Q-Credit Card Fraud Detector paper [5]
        if action == true_label:
            return 10 if action == 1 else 1  # Higher reward for correct fraud detection
        return -20 if (action == 0 and true_label == 1) else -5  # Severe penalty for false negatives
    
    def _load_dataset(self, path):
        data = pd.read_csv(path, index_col=False)
        print(data.shape)
        return data.values

# 3. Q-Network Architecture
class FraudQNetwork(nn.Module):
    def __init__(self, input_size=30, output_size=2):  # 28 PCA + Time + Amount
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # Added bottleneck layer
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is flattened correctly
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        return self.fc(x)

# 4. Training Loop with Experience Replay
def train_q_learning():
    print("Training Q-Learning agent...")
    env: FraudDetectionEnv = FraudDetectionEnv()
    q_network: FraudQNetwork = FraudQNetwork()
    optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001)
    memory = deque(maxlen=10000)
    
    epsilon = 1.0
    batch_size = 64
    
    for episode in tqdm(range(1000)):
        state, _ = env.reset()
        episode_reward = 0
        print(f"Episode {episode + 1}")
        
        limit = env.max_steps
        with tqdm(total=limit) as pbar:
            while True:
                # Epsilon-greedy action selection
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = q_network(torch.FloatTensor(state))
                        action = torch.argmax(q_values).item()
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                memory.append((state, action, reward, next_state, terminated))
                
                # Experience replay
                if env.current_step % TRAIN_INTERVAL == 0 and len(memory) >= batch_size:
                    print(f"experience replay: {env.current_step}")
                    batch = [memory[i] for i in np.random.choice(len(memory), batch_size, replace=False)]
                    
                    states = torch.FloatTensor(np.array([s for s, _, _, _, _ in batch]))
                    actions = [a for _, a, _, _, _ in batch]
                    rewards = [r for _, _, r, _, _ in batch]
                    next_states = torch.FloatTensor(np.array([ns for _, _, _, ns, _ in batch]))
                    dones = [d for _, _, _, _, d in batch]
                    
                    # Q-learning update
                    current_q = q_network(torch.FloatTensor(states[:, :-1]))
                    next_q = q_network(torch.FloatTensor(next_states[:, :-1]))
                    targets = current_q.clone()
                    
                    for i in range(batch_size):
                        if dones[i]:
                            targets[i][actions[i]] = rewards[i]
                        else:
                            targets[i][actions[i]] = rewards[i] + 0.99 * torch.max(next_q[i])
                    
                    loss = nn.MSELoss()(current_q, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                episode_reward += reward
                if terminated or truncated:
                    break
                    
                pbar.update(1)
            
        # Epsilon decay
        epsilon = max(0.01, epsilon * 0.995)

# 5. Usage
if __name__ == "__main__":
    train_q_learning()
