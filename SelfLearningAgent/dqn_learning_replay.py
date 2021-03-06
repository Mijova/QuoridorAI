import random
import torch
import torch.nn as nn
import pyspiel
from dqn import DQN
from collections import deque

torch.set_default_dtype(torch.double)
random.seed(0)

# version 1.2
#Hyper params
EPISODES = 10001

#Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.2
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / EPISODES

#Discount rate
GAMMA = 0.99

# Game params
BOARD_SIZE = 4
WALL_COUNT = 1

#Switch nets frequency
SWITCH_FREQ = 10
#Update frequency
UPDATE_FREQ = 16


class DQNAgentReplay():
    def __init__(self, state_size, action_size):
        self.target = DQN(state_size, action_size)
        self.current = DQN(state_size, action_size)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        self.memory = deque(maxlen = 2000)
        self.batch_size =  128
        
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
                    state_action_q_values_target[0][action] = reward - GAMMA * torch.max(next_state_action_q_values+fancy_mask(next_state))
            loss = self.loss_fn(state_action_q_values, state_action_q_values_target)
            loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


def get_nn_input(game, state):
  obs = torch.tensor(state.observation_tensor()).view(game.observation_tensor_shape())
  if state.current_player() == 0:
    return obs[:3].unsqueeze(0), torch.tensor([[obs[3,0,0], obs[4,0,0]]])
  else:
    obs[:3] = torch.flip(obs[:3], [1, 2])
    obs[[0, 1]] = obs[[1, 0]]
    return obs[:3].unsqueeze(0), torch.tensor([[obs[4,1,1], obs[3,0,0]]])

def fancy_mask(state):
    mask = torch.tensor(state.legal_actions_mask(), dtype=torch.float64)
    mask[mask==0] = float("-inf")
    mask[mask==1] = 0
    return mask

def main():
    game = pyspiel.load_game(f"quoridor(ansi_color_output=true,board_size={BOARD_SIZE},wall_count={WALL_COUNT})")
    board_diam = 2*BOARD_SIZE-1
    agent = DQNAgentReplay(2*BOARD_SIZE-1, game.num_distinct_actions())
    epsilon = EPSILON_START
    results = []
    wins, draws, loses = (0, 0, 0)
    for episode in range(EPISODES):
        #Start game/episode
        state = game.new_initial_state()

        if (episode > SWITCH_FREQ and episode % SWITCH_FREQ == 0):
            agent.update_target_model()

        #Loop inside one game episode
        while not state.is_terminal():
            pl = state.current_player()

            with torch.no_grad():
                state_action_q_values = agent.forward(get_nn_input(game, state))
            rotated_state_action_q_values = state_action_q_values if state.current_player() == 0 else torch.flip(state_action_q_values.clone(), [1])
            
            actual_action = torch.argmax(rotated_state_action_q_values+fancy_mask(state)).item()
            rotated_action = actual_action if state.current_player() == 0 else (board_diam**2-1) - actual_action
            if (actual_action not in state.legal_actions()): print(True)
            if actual_action not in state.legal_actions() or random.random() <= epsilon:
                actual_action = random.choice(state.legal_actions())
                rotated_action = actual_action if state.current_player() == 0 else (board_diam**2-1) - actual_action
            old_state = state.clone()
            state.apply_action(actual_action)

            rewards = state.rewards()
            agent.remember(old_state, rotated_action, rewards[pl], state)

            if episode > 10 and episode % UPDATE_FREQ == 0 and not state.is_terminal():       
                agent.backward(game)

        if (rewards[0] == 1): wins += 1
        elif (rewards[0] == 0): draws += 1
        elif (rewards[0] == -1): loses += 1
        if (episode % 100 == 0):
            print("Episode: ", episode, epsilon)
            print(f"W:{wins}, D:{draws}, L:{loses}") 
            wins, draws, loses = (0, 0, 0)
        if epsilon > EPSILON_END:
            epsilon -= EPSILON_DECAY
        if (episode %  500 == 0):
            torch.save(agent, f"/mnt/QuoridorAI/Agents/SelfLearnedER{BOARD_SIZE}x{BOARD_SIZE}-{episode}")

if __name__ == '__main__':
    main()