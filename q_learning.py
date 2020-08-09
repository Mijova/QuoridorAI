import torch
import pyspiel
import random

EPISODES = 2000
EPSILON = 0.15 #Exploration vs Exploitation
GAMMA = 0.9
LEARNING_RATE = 0.1
board_size = 3

def main():
    game = pyspiel.load_game("quoridor(ansi_color_output=true,board_size=3,wall_count=1)")
    average_cumulative_reward = 0.0

    # Q-table, for each env state have action space
    # (current example: 5x5 states, 49 actions per state)
    qtable = torch.zeros((2**27, game.num_distinct_actions(),), dtype=torch.float)

    cumulative_reward = 0.0
    wins, draws, loses = (0, 0, 0)
    # Loop over episodes
    for i in range(EPISODES):
        state = game.new_initial_state()
        
        # Loop over time-steps
        while not state.is_terminal():
            # 0 Currently you are in "state S (state)"
            # 1 Calculate action to be taken from state S. Use 'e-rand off-policy'
                # 1.1 Compute what the greedy action for the current state is
            state_tensor = get_state_tensor(state, board_size)
            action = torch.argmax((qtable[state_tensor] + EPISODES) * torch.tensor(state.legal_actions_mask()))

                # 1.2 Sometimes, the agent takes a random action, to explore the environment
            if random.random() < EPSILON:
                action = random.choice(state.legal_actions())
            
            # Perform the action
            state.apply_action(action)
            rewards = state.rewards()
            cumulative_reward += 1 if rewards[0] > 0 else 0

            if not state.is_terminal():
                # Random action
                state.apply_action(random.choice(state.legal_actions()))
                rewards = state.rewards()
                if (not state.is_terminal()):
                    new_state_tensor = get_state_tensor(state, board_size)
                    # Update the q-table
                    qtable[state_tensor][action] += LEARNING_RATE * (rewards[0] + torch.max(qtable[new_state_tensor]) * GAMMA - qtable[state_tensor][action])  
                else:
                    qtable[state_tensor][action] += LEARNING_RATE * (rewards[0] - qtable[state_tensor][action])  
            else:
                qtable[state_tensor][action] += LEARNING_RATE * (rewards[0] - qtable[state_tensor][action])
            
            if (i == EPISODES - 1):
                print(state)
        if (rewards[0] == 1): wins += 1
        if (rewards[0] == 0): draws += 1
        if (rewards[0] == -1): loses += 1
        if (i % 1000 == 0):
            print("Episode: ", i)
    print(f"W:{wins}, D:{draws}, L:{loses}") 


def get_state_tensor(state, board_size):
    sections = 5
    obs = torch.tensor(state.observation_tensor()).view((sections, (2*board_size-1)**2))
    obs.to(dtype=torch.int64)
    our_state = obs[0] + obs[1] + obs[2]
    binary = 2 ** torch.tensor(list(range(27)))
    x = torch.cat((our_state, torch.tensor([obs[3,1], obs[4,1]])))
    return torch.sum(x * binary, dtype=torch.int64)

if __name__ == '__main__':
    main()