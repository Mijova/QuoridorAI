import torch
import pyspiel
import random

EPISODES = 20001
GAMMA = 0.9
LEARNING_RATE = 0.10
board_size = 3

def main():
    EPSILON = 0.8 #Exploration vs Exploitation
    game = pyspiel.load_game("quoridor(ansi_color_output=true,board_size=3,wall_count=1)")
    average_cumulative_reward = 0.0

    # Q-table, for each env state have action space
    # (current example: 5x5 states, 49 actions per state)
    qtable = torch.zeros((num_of_states(3,2), game.num_distinct_actions(),), dtype=torch.float)
    cumulative_reward = 0.0
    wins, draws, loses = (0, 0, 0)
    # Loop over episodes
    for i in range(EPISODES):
        state = game.new_initial_state()
        state.apply_action(random.choice(state.legal_actions()))

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
            cumulative_reward += 1 if rewards[1] > 0 else 0

            if not state.is_terminal():
                # Random action
                state.apply_action(random.choice(state.legal_actions()))
                rewards = state.rewards()
                if (not state.is_terminal()):
                    new_state_tensor = get_state_tensor(state, board_size)
                    # Update the q-table
                    qtable[state_tensor][action] += LEARNING_RATE * (rewards[1] + torch.max(qtable[new_state_tensor]) * GAMMA - qtable[state_tensor][action])  
                else:
                    qtable[state_tensor][action] += LEARNING_RATE * (rewards[1] - qtable[state_tensor][action])  
            else:
                qtable[state_tensor][action] += LEARNING_RATE * (rewards[1] - qtable[state_tensor][action])
            
            if (i == EPISODES - 1):
                print(state)
        if (rewards[1] == 1): wins += 1
        if (rewards[1] == 0): draws += 1
        if (rewards[1] == -1): loses += 1
        if (i % 1000 == 0):
            EPSILON -= 0.05
            print("Episode: ", i)
            print(f"W:{wins}, D:{draws}, L:{loses}") 
            wins, draws, loses = (0, 0, 0)
            print(torch.sum(qtable != 0))

def get_state_tensor(state, board_size, max_walls=2):
    sections = 5
    board_diam = 2*board_size-1
    num_fields = board_diam**2
    obs = torch.tensor(state.observation_tensor()).view((sections, num_fields))
    p1 = torch.argmax(obs[0]).item()
    p1 = (p1 - (p1 // board_diam)*(board_size-1)) / 2
    p2 = torch.argmax(obs[1]).item()
    p2 = (p2 - (p2 // board_diam)*(board_size-1)) / 2
    w1 = obs[3,1]
    w2 = obs[4,1]
    walls = obs[2][1::2]
    binary = 2 ** torch.tensor(range(len(walls)))
    wall_state = torch.sum(walls * binary, dtype=torch.int64)
    state = ((wall_state*board_size + p1) * board_size + p2) * max_walls**2 + max_walls*w1 + w2
    return state.to(dtype=torch.int64)

def num_of_states(board_size, max_walls):
    num_fields = (2*board_size-1)**2
    num_walls = (num_fields-1)//2
    return 2**num_walls * board_size**4 * max_walls**2

if __name__ == '__main__':
    main()