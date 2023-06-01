from collections import deque
import random
import torch 
import torch.nn as nn

class ReplayBuffer: 
    def __init__(self, max_size):
        self.buffer =  deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    

class Q_Network(nn.Module):

    def __init__(self, state_dim, action_dim, config, optimizer):
        super(Q_Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.model = self.build_model()
        self.optimizer = optimizer(self.model.parameters(), lr=config['lr'])
        self.loss = config['loss']
        if(config['loss'] == 'MSE'):
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.SmoothL1Loss()
        
        self.gamma = config['gamma']
    
    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim, self.config['layer1']),
            nn.ReLU(),
            nn.Linear(self.config['layer1'], self.config['layer2']),
            nn.ReLU(),
            nn.Linear(self.config['layer2'], self.action_dim),
            nn.Tanh()
        )
        return model
    
        
    
    def forward(self, state):
        return self.model(state)

    


class Agent:

    def __init__(self, observation_space, action_space, config): 
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        self.epsilon = config['epsilon']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.actor_config = config['actor_config']
        self.actor = self.build_actor()
        self.train_epochs = config['train_epochs']
    

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    def build_actor(self):
        return Q_Network(self.observation_space.shape[0], self.action_space.shape[0], self.actor_config, torch.optim.Adam)
    

    def compute_action(self, state):
        if random.random() < self.epsilon:
            return torch.tensor(self.action_space.sample())
        else:
            state = torch.tensor(state, dtype=torch.float)
            q_values = self.actor(state)
            return q_values
    
    def train(self, batch_size = 64):
        m = min(batch_size, len(self.replay_buffer))
        for _ in range(self.train_epochs):
            self.actor.optimizer.zero_grad()
            batch = self.replay_buffer.sample(m)
            state, action, reward, next_state, done = zip(*batch)

            state = torch.tensor([s.tolist() for s in state], dtype=torch.float)
            action = torch.tensor([a.tolist() for a in action], dtype=torch.long)
            reward = torch.tensor([r for r in reward], dtype=torch.float)
            next_state = torch.tensor([s.tolist() for s in next_state], dtype=torch.float)
            done = torch.tensor([d for d in done], dtype=torch.bool)
            q_values = self.actor.model(state)


            target_q_values = torch.zeros((m,self.action_space.shape[0]))
        
            # Calculate the target Q-values
            with torch.no_grad():
                next_q_values = self.actor.model(next_state)
                for i in range(m):
                    if done[i]:
                        target_q_values[i][action[i]] = reward[i]

                    else:
                        target_q_values[i][action[i]] = reward[i] + self.gamma * max(next_q_values[i])

            loss = self.actor.loss(q_values, target_q_values)
            loss.backward()
            self.actor.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()