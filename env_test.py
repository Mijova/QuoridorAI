import numpy as np
import torch
import pyspiel

game = pyspiel.load_game("quoridor(ansi_color_output=true,board_size=3,wall_count=3)")
state = game.new_initial_state()

wins, draws, loses = (0, 0, 0)
for i in range(200):
  state = game.new_initial_state()
  while not state.is_terminal():
    action = np.random.choice(state.legal_actions())
    state.apply_action(action)
    #print(str(state) + '\n')
  rewards = state.rewards()
  if (rewards[1] == 1): wins += 1
  if (rewards[1] == 0): draws += 1
  if (rewards[1] == -1): loses += 1
print(wins, draws, loses)