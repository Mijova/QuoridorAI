import numpy as np
import torch
import pyspiel

game = pyspiel.load_game("quoridor(ansi_color_output=true,board_size=4,wall_count=1)")
state = game.new_initial_state()
print(state.legal_actions())

while not state.is_terminal():
  action = np.random.choice(state.legal_actions())
  state.apply_action(action)
  print(state.get_game() + '\n')

def get_state_tensor(state, board_size):
  obs = torch.tensor(state.observation_tensor()).view((5, (2*board_size-1)**2))
  our_state = 2*obs[0] + 3*obs[1] + obs[2]
  return torch.cat((our_state, torch.tensor([torch.max(obs[3]), torch.max(obs[4])])))
