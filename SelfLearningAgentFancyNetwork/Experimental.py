import random
import time
import torch
import torch.nn as nn
import pyspiel
from collections import deque

torch.set_default_dtype(torch.double)
random.seed(0)

# version 1.2
#Hyper params
EPISODES = 20000

#Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / EPISODES

#Discount rate
GAMMA = 0.99

# Game params
BOARD_SIZE = 4
WALL_COUNT = 1


#Switch nets frequency
SWITCH_FREQ = 50
#Update frequency
UPDATE_FREQ = 4

device = torch.device("cuda")


class DQNAgentReplay():
    def __init__(self, state_size, action_size):
        self.target = DQN(state_size, action_size).to(device)
        self.current = DQN(state_size, action_size).to(device)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        self.memory = deque(maxlen = 4000)
        self.batch_size =  256
        
        learning_rate = 0.0025
        self.optimizer = torch.optim.Adam(self.current.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.current(x)

    def update_target_model(self):
        self.target.load_state_dict(self.current.state_dict())

    def remember(self, state, action, player, next_state, ):
        # Remember (Q,S,A,R,S')
        self.memory.append((state, action, player, next_state))

    def backward(self,game):
        # Experience replay
        # 1. Create mini batch (size: self.batch_size) for training
        # 2. Update the current net -> use target net to evalute target
            # Tip: For best performance you can use torch gradient accumulation
        batch_len = min(self.batch_size, len(self.memory))
        batch = random.sample(self.memory, batch_len)
        board_diam = 2 * BOARD_SIZE - 1
        batch_nn_input = [torch.zeros([len(batch), 3, board_diam, board_diam]).to(device), torch.zeros([len(batch), 2]).to(device)]
        id = 0
        for state, action, player, next_state in batch:
            nn_input = get_nn_input(game, state)
            batch_nn_input[0][id] = nn_input[0]
            batch_nn_input[1][id] = nn_input[1]
            id = id + 1
        state_action_q_values = self.current.forward(batch_nn_input)
        with torch.no_grad():
          state_action_q_values_target = state_action_q_values.clone().detach()
          id = 0
          for state, action, player, next_state in batch:
            if next_state.is_terminal:
              passed_moves = next_state.move_number() - state.move_number()
              state_action_q_values_target[id][action] = (GAMMA**passed_moves) * next_state.rewards()[player]
            else:
              assert False
            id = id + 1
        loss = self.loss_fn(state_action_q_values, state_action_q_values_target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

def get_nn_input(game, state):
  obs = torch.tensor(state.observation_tensor()).view(game.observation_tensor_shape())
  if state.current_player() == 0:
    return obs[:3].unsqueeze(0).to(device), torch.tensor([[obs[3,0,0], obs[4,0,0]]]).to(device)
  else:
    obs[:3] = torch.flip(obs[:3], [1, 2])
    obs[[0, 1]] = obs[[1, 0]]
    return obs[:3].unsqueeze(0).to(device), torch.tensor([[obs[4,1,1], obs[3,0,0]]]).to(device)

def fancy_mask(state):
    mask = torch.tensor(state.legal_actions_mask(), dtype=torch.float64)
    mask[mask==0] = float("-inf")
    mask[mask==1] = 0
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

ROTATION_MASK = get_rotation_mask(BOARD_SIZE)

def main():
    print(f"quoridor(ansi_color_output=true,board_size={BOARD_SIZE},wall_count={WALL_COUNT})")
    game = pyspiel.load_game(f"quoridor(ansi_color_output=true,board_size={BOARD_SIZE},wall_count={WALL_COUNT})")
    board_diam = 2*BOARD_SIZE-1
    agent = DQNAgentReplay(2*BOARD_SIZE-1, game.num_distinct_actions())
    epsilon = EPSILON_START
    results = []
    wins, draws, loses = (0, 0, 0)
    start = time.time()
    for episode in range(1, EPISODES+1):
        #Start game/episode
        state = game.new_initial_state()

        if (episode > SWITCH_FREQ and episode % SWITCH_FREQ == 0):
            agent.update_target_model()

        #Loop inside one game episode
        while not state.is_terminal():
            pl = state.current_player()

            with torch.no_grad():
                state_action_q_values = agent.forward(get_nn_input(game, state))
            rotated_state_action_q_values = state_action_q_values.to(device) if state.current_player() == 0 else state_action_q_values[0][ROTATION_MASK].to(device)
            actual_action = torch.argmax(rotated_state_action_q_values+fancy_mask(state).to(device)).item()
            rotated_action = actual_action if state.current_player() == 0 else ROTATION_MASK[actual_action]
            if (actual_action not in state.legal_actions()): print(True, True)
            if actual_action not in state.legal_actions() or random.random() <= epsilon:
                actual_action = random.choice(state.legal_actions())
                rotated_action = actual_action if state.current_player() == 0 else ROTATION_MASK[actual_action]
            old_state = state.clone()
            state.apply_action(actual_action)

            agent.remember(old_state, rotated_action, pl, state)

            if episode > 10 and episode % UPDATE_FREQ == 0 and state.is_terminal():       
                agent.backward(game)
        rewards = state.rewards()
        if (rewards[0] == 1): wins += 1
        elif (rewards[0] == 0): draws += 1
        elif (rewards[0] == -1): loses += 1
        if (episode % 500 == 0):
            print("Episode: ", episode, epsilon)
            print(f"W:{wins}, D:{draws}, L:{loses}") 
            wins, draws, loses = (0, 0, 0)
        if epsilon > EPSILON_END:
            epsilon -= EPSILON_DECAY
        if (episode % 2000 == 0):
            torch.save(agent, f"Experiment{BOARD_SIZE}x{BOARD_SIZE}-{episode}")
    end = time.time()
    print(f"Execution lasted {end-start} seconds.")

if __name__ == '__main__':
    main()