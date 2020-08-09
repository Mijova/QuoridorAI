import pyspiel
import torch
import torch.nn as nn
import random

torch.set_default_dtype(torch.double)

BOARD_SIZE = 5
WALL_COUNT = 3
EPISODES = 100
EPSILON = 0.00
import torch
import torch.nn as nn


device = torch.device('cpu')

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        EXTRA_STATES = 2
        super(DQN, self).__init__()
        self.action_size = action_size

        h = 2 * action_size #hidden dimension
        num_of_channels = [3, 16, 64]
        kernel_sizes = [3, 3, 1, 1]

        self.convNN = nn.Sequential(
            nn.Conv2d(num_of_channels[0], num_of_channels[1], kernel_size=kernel_sizes[0]),
            nn.BatchNorm2d(num_of_channels[1]),
            nn.ReLU(),
            nn.Conv2d(num_of_channels[1], num_of_channels[2], kernel_size=kernel_sizes[1]),
            nn.BatchNorm2d(num_of_channels[2]),
            nn.ReLU())
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.neuralnet = nn.Sequential(
            nn.Linear(num_of_channels[-1] + EXTRA_STATES, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, self.action_size))

    def forward(self, x):
        y = self.convNN(x[0])
        y = self.avgpool(y)
        y = y.flatten(start_dim=1)
        y = torch.cat((y, x[1]), dim=1)
        y = self.neuralnet(y)
        return y

class DQNAgentReplay():
    def __init__(self, state_size, action_size):
        self.target = DQN(state_size, action_size)
        self.current = DQN(state_size, action_size)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        self.memory = deque(maxlen = 2000)
        self.batch_size =  64
        
        learning_rate = 0.0025
        self.optimizer = torch.optim.Adam(self.current.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.current(x)

    def update_target_model(self):
        self.target.load_state_dict(self.current.state_dict())

    def remember(self, state, action, reward, next_state):
        # Remember (Q,S,A,R,S')
        self.memory.append((state, action, reward, next_state))

    def backward(self,game):
        # Experience replay
        # 1. Create mini batch (size: self.batch_size) for training
        # 2. Update the current net -> use target net to evalute target
            # Tip: For best performance you can use torch gradient accumulation

        for state, action, reward, next_state in random.sample(self.memory, self.batch_size):
            state_action_q_values = self.current.forward(get_nn_input(game, state))
            if next_state.is_terminal():
                with torch.no_grad():
                    state_action_q_values_target = state_action_q_values.clone().detach()
                    state_action_q_values_target[0][action] = reward
            else:
                with torch.no_grad():
                    next_state_action_q_values = self.target.forward(get_nn_input(game, next_state))
                    if (next_state.current_player() == 1):
                        next_state_action_q_values  = torch.flip(next_state_action_q_values , [1])
                    state_action_q_values_target = state_action_q_values.clone().detach()
                    next_mask = torch.BoolTensor(next_state.legal_actions_mask())
                    next_legal_q_values = torch.masked_select(next_state_action_q_values, next_mask)
                    state_action_q_values_target[0][action] = reward - GAMMA * torch.max(next_legal_q_values)
            loss = self.loss_fn(state_action_q_values, state_action_q_values_target)
            loss.backward()
        self.optimizer.zero_grad()
        self.optimizer.step()

class DQNAgent():
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        
        learning_rate = 0.0025
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.model(x)

    def backward(self, y, y_target):
        loss = self.loss_fn(y, y_target)

        # Zero-out all the gradients
        self.optimizer.zero_grad()

        # Backward pass: compute gradient of the loss
        loss.backward()

        # Calling the step function on an Optimizer in order to apply the gradients (update)
        self.optimizer.step()

def fancy_mask(state):
    mask = torch.tensor(state.legal_actions_mask(), dtype=torch.float64)
    mask[mask==0] = float("-inf")
    return mask

def get_rotation_mask(board_size):
    rotation = []
    board_diam = 2*board_size - 1
    num_actions = board_diam**2 - 1
    for i in range(board_diam**2):
        x, y = i % board_diam, i // board_diam
        if (x % 2 == 0 and y % 2 == 0):
            rotation += [num_actions-i]
        elif (y % 2 == 0):
            rotation += [num_actions-i-2*board_diam]
        else:
            rotation += [num_actions-i-2]
    return rotation

def get_nn_input(game, state):
  obs = torch.tensor(state.observation_tensor()).view(game.observation_tensor_shape())
  if state.current_player() == 0:
    return obs[:3].unsqueeze(0), torch.tensor([[obs[3,1,1], obs[4,1,1]]])
  else:
    obs[:3] = torch.flip(obs[:3], [1, 2])
    obs[[0, 1]] = obs[[1, 0]]
    return obs[:3].unsqueeze(0), torch.tensor([[obs[4,1,1], obs[3,1,1]]])

ROTATION_MASK = get_rotation_mask(BOARD_SIZE)

def main():
    game = pyspiel.load_game(f"quoridor(ansi_color_output=true,board_size={BOARD_SIZE},wall_count={WALL_COUNT})")
    wins, draws, loses = (0, 0, 0)
    agent0 = torch.load("/mnt/QuoridorAI/Agents/Experiment5x5-20000", map_location='cpu')
    agent1 = torch.load("/mnt/QuoridorAI/Agents/Experiment5x5-20000", map_location='cpu')
    HUMAN = 0
    RANDOM = -1
    for episode in range(EPISODES):
        state = game.new_initial_state()
        while not state.is_terminal():
            if (state.current_player() == 0):
                #print(state)
                if HUMAN == 0:
                        print(state)
                        print(state.legal_actions())
                        action = int(input())
                else:
                        state_action_q_values = agent0.forward(get_nn_input(game, state))
                        #print((state_action_q_values*10**5).to(dtype=torch.int64).view((7,7)))
                        action = torch.argmax(state_action_q_values+fancy_mask(state)).item()
                        if RANDOM == 0 or action not in state.legal_actions() or random.random() <= EPSILON:
                            # print("RANDOM MOVE")
                            action = random.choice(state.legal_actions())
                state.apply_action(action)
            else:
                #print(state.current_player())
                if HUMAN == 1:
                    print(state)
                    print(state.legal_actions())
                    action = int(input())
                else:
                    state_action_q_values = agent1.forward(get_nn_input(game, state))[0][ROTATION_MASK]
                    action = torch.argmax(state_action_q_values+fancy_mask(state)).item()
                    if RANDOM == 1 or action not in state.legal_actions() or random.random() <= EPSILON:
                        action = random.choice(state.legal_actions())
                state.apply_action(action)
        rewards = state.rewards()
        if (rewards[0] == 1): wins += 1
        if (rewards[0] == 0): draws += 1
        if (rewards[0] == -1): loses += 1
    print(f"W:{wins}, D:{draws}, L:{loses}") 

if __name__ == '__main__':
    main()