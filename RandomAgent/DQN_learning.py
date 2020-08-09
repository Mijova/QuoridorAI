import random
import torch
import torch.nn as nn
from dqn import DQN
import pyspiel

torch.set_default_dtype(torch.double)
random.seed(0)

#Hyper params
EPISODES = 4001

#Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 0.99

#Discount rate
GAMMA = 0.95

# Game params
BOARD_SIZE = 3
WALL_COUNT = 1


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

def get_nn_input(game, state):
  obs = torch.tensor(state.observation_tensor()).view(game.observation_tensor_shape())
  if state.current_player() == 0:
    return obs[:3].unsqueeze(0), torch.tensor([[obs[3,1,1], obs[4,1,1]]])
  else:
    obs[:3] = torch.flip(obs[:3], [1, 2])
    obs[[0, 1]] = obs[[1, 0]]
    return obs[:3].unsqueeze(0), torch.tensor([[obs[4,1,1], obs[3,1,1]]])   



def main():
    game = pyspiel.load_game(f"quoridor(ansi_color_output=true,board_size={BOARD_SIZE},wall_count={WALL_COUNT})")
    board_diam = 2*BOARD_SIZE-1
    agent = DQNAgent(board_diam, game.num_distinct_actions())
    #agent = torch.load("/mnt/QuoridorAI/Agents/Turn2random")
    epsilon = EPSILON_START
    results = []
    wins, draws, loses = (0, 0, 0)
    for episode in range(EPISODES):
        #Start game/episode
        state = game.new_initial_state()

        PLAYER = random.choice([0,1])
        if (PLAYER==1):
            state.apply_action(random.choice(state.legal_actions()))

        #Loop inside one game episode
        while not state.is_terminal():
            state_action_q_values = agent.forward(get_nn_input(game, state))
            rotated_state_action_q_values = state_action_q_values if state.current_player() == 0 else torch.flip(state_action_q_values.clone(), [1])
            if random.random() <= epsilon:
                actual_action = random.choice(state.legal_actions())
                rotated_action = actual_action if state.current_player() == 0 else (board_diam**2-1) - actual_action
            else:
                actual_action = torch.argmax((rotated_state_action_q_values+1)*torch.tensor(state.legal_actions_mask())).item()
                rotated_action = actual_action if state.current_player() == 0 else (board_diam**2-1) - actual_action
            state.apply_action(actual_action)
            rewards = state.rewards()

            if not state.is_terminal():
                state.apply_action(random.choice(state.legal_actions()))
                rewards = state.rewards()
                if not state.is_terminal():
                    with torch.no_grad():
                        next_state_action_q_values = agent.forward(get_nn_input(game, state))
                        if (state.current_player() == 1):
                            next_state_action_q_values  = torch.flip(next_state_action_q_values , [1])
                        state_action_q_values_target = torch.tensor(state_action_q_values)
                        next_mask = torch.BoolTensor(state.legal_actions_mask())
                        next_legal_q_values = torch.masked_select(next_state_action_q_values, next_mask)
                        state_action_q_values_target[0][rotated_action] = rewards[PLAYER] + GAMMA * torch.max(next_legal_q_values)
                    agent.backward(state_action_q_values, state_action_q_values_target)               
            if state.is_terminal():
                    with torch.no_grad():
                        state_action_q_values_target = torch.tensor(state_action_q_values)
                        state_action_q_values_target[0][rotated_action] = rewards[PLAYER]
                    agent.backward(state_action_q_values, state_action_q_values_target)
        if (rewards[PLAYER] == 1): wins += 1
        if (rewards[PLAYER] == 0): draws += 1
        if (rewards[PLAYER] == -1): loses += 1
        if (episode % 200 == 0):
            print("Episode: ", episode)
            print(f"W:{wins}, D:{draws}, L:{loses}") 
            wins, draws, loses = (0, 0, 0)
        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY
    torch.save(agent, f"/mnt/QuoridorAI/Agents/BothTurnsRandom")


if __name__ == '__main__':
    main()

