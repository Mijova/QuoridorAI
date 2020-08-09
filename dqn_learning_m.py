import random
import numpy as np
import torch
import time
import torch.nn as nn
import pyspiel
from dqn_m import DQN

torch.set_default_dtype(torch.double)
random.seed(0)

#Hyper params
EPISODES = 5000

#Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 0.999

#Discount rate
GAMMA = 0.95

class DQNAgent():
    def __init__(self, action_size):
        self.model = DQN(action_size)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        
        learning_rate = 0.001
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

game = pyspiel.load_game("quoridor(ansi_color_output=true,board_size=3,wall_count=1)")
agent = DQNAgent(game.num_distinct_actions())

epsilon = EPSILON_START
results = []
start = time.time()

for episode in range(EPISODES):
    #Start game/episode
    state = game.new_initial_state()

    # Randomly switch players
    rlp = random.choice([0,1])
    hp = 1-rlp

    #Loop inside one game episode
    while not state.is_terminal():
        # First player move - AI agent
      if state.current_player()==rlp:
        p = state.current_player()
        state_action_q_values = agent.forward( torch.tensor(state.observation_tensor())\
                                              .view(game.observation_tensor_shape()).unsqueeze(0) )
        if random.random() <= epsilon:
            action = random.choice(state.legal_actions())
        else:
            mask = torch.BoolTensor(state.legal_actions_mask())

            legal_q_values = torch.masked_select(state_action_q_values, mask) 
            action = state.legal_actions()[torch.argmax(legal_q_values).item()]       

        state.apply_action(action)  # get next state
        rewards = state.rewards()
        rlp_reward = rewards[rlp]

        with torch.no_grad():
          if state.is_terminal():
            state_action_q_values_target = state_action_q_values.clone().detach()
            state_action_q_values_target[action] = rlp_reward
          else:
            next_state_action_q_values = agent.forward( torch.tensor(state.observation_tensor())\
                                              .view(game.observation_tensor_shape()).unsqueeze(0) )
            next_mask = torch.BoolTensor(state.legal_actions_mask())

            next_legal_q_values = torch.masked_select(next_state_action_q_values, next_mask) 
            next_action = state.legal_actions()[torch.argmax(next_legal_q_values).item()]

            state_action_q_values_target = state_action_q_values.clone().detach()
            state_action_q_values_target[action] = rlp_reward + GAMMA * torch.max(next_legal_q_values)

        agent.backward(state_action_q_values, state_action_q_values_target)

      ######################################################################    
      # Second player move - random player     
      elif state.current_player()==hp:
        p = state.current_player()
        action = random.choice(state.legal_actions())
        state.apply_action(action)
        rewards = state.rewards()
        hp_reward = rewards[hp]

      ######################################################################

      if state.is_terminal():
          results.append([rlp_reward, hp_reward])
          if episode%100==0:
            rl_score = np.mean(np.array(results)[-100:,0])
            print(f"episode: {episode}/{EPISODES}, player: {p}, current score: {rewards}, RL score: {rl_score}, e: {epsilon}") 
          break

    if epsilon > EPSILON_END:
        epsilon *= EPSILON_DECAY

end = time.time()
print("TIME")
print(end - start)
acc_wins = np.sum(np.array(results)==1, axis=0)
acc_wins_percent = acc_wins // EPISODES * 100
print(f"Accumulated wins:{acc_wins}, draw: {EPISODES-np.sum(acc_wins)}")
print(f"Accumulated wins:{acc_wins_percent}%, draw: {EPISODES-np.sum(acc_wins_percent)}")