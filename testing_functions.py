import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import deepst 

def plot_grid(grid_size, trajectories, map_values):
    fig, ax = plt.subplots()

    # Plot state values
    for i in range(grid_size):
        for j in range(grid_size):
            if map_values[i, j] > 0:
                color = 'green'
            elif map_values[i, j] < 0:
                color = 'purple'
            else:
                color = 'white'
            rect = plt.Rectangle((j, i), 1, 1, edgecolor='black', facecolor=color, alpha=0.3)
            ax.add_patch(rect)
            if color == 'green' or color == 'purple':
                ax.text(j + 0.5, i + 0.5, str(map_values[i, j]), color='black', ha='center', va='center')

    # Mark each state in the trajectory
    # Mark each state in the trajectory
    for trajectory in trajectories:
        prev_x, prev_y = None, None
        for i, (state, action) in enumerate(trajectory):
            # Convert the state to 2D coordinates
            x, y = np.unravel_index(state, (grid_size, grid_size))
            ax.plot(y + 0.5, x + 0.5, 'ro-')
            if prev_x is not None and prev_y is not None:
                # Display the action taken
                action_text = {
                    0: '↑',
                    1: '\n↓',
                    2: '←  ',
                    3: '  →'
                }
                color = 'black' if i < len(trajectory) - 1 else 'blue'
                ax.text(
                    y + 0.5,
                    x + 0.4,
                    action_text[action],
                    color=color,
                    ha='center',
                    va='center',
                    fontsize=15,
                    fontweight='bold'
                )
            prev_x, prev_y = x, y

    # Show the plot
    ax.set_xlim([0, grid_size])
    ax.set_ylim([0, grid_size])
    ax.invert_yaxis()  # Invert y axis for proper visualization
    plt.show()

def action_to_delta(action):
    dir = {
        0: np.array([-1, 0], dtype=np.int32),  # up
        1: np.array([1, 0], dtype=np.int32),  # down
        2: np.array([0, -1], dtype=np.int32),  # left
        3: np.array([0, 1], dtype=np.int32)  # right
    }
    return dir[action]

def retrain(agent, env, n_episodes):
    # Set epsilon to 0 for exploitation
    agent.epsilon = 0
    
    episode_reward1 = []
    episode_reward2 = []
    agent.trajectories = []

    for episode in range(n_episodes):
        agent.current_trajectory = []
        r1, r2, done = retrain_episode_run(agent, env)
        episode_reward1.append(r1)
        episode_reward2.append(r2)
        print("Epsilon = 0 | Episode = " + str(episode) + "| reward = " + str((r1, r2)))
        agent.trajectories.append(agent.current_trajectory)

    return episode_reward1, episode_reward2

def retrain_episode_run(agent, env):
    done = False
    reward1 = []
    reward2 = []
    state = env.reset()
    state = agent.flatten_observation(state)
    action = retrain_action_selection(agent, state)

    while not done and sum(reward2) > -5000:
        next_state, reward, done, _ = env.step(action)
        r1, r2 = reward
        reward1.append(r1)
        reward2.append(r2)
        next_state = agent.flatten_observation(next_state)
        next_action = retrain_action_selection(agent, next_state)

        if not done:
            next_value1 = agent.Q1[next_state, next_action]
            next_value2 = agent.Q2[next_state, next_action]
        else:
            next_value1 = 0
            next_value2 = 0

        old_value1 = agent.Q1[state, action]
        old_value2 = agent.Q2[state, action]
        new_value1 = ((1 - agent.alpha) * old_value1) + (agent.alpha * (r1 + (agent.gamma * next_value1)))
        new_value2 = ((1 - agent.alpha) * old_value2) + (agent.alpha * (r2 + (agent.gamma * next_value2)))
        agent.Q1[state, action] = new_value1
        agent.Q2[state, action] = new_value2

        agent.current_trajectory.append((state, action))

        state = next_state
        action = next_action

    return sum(reward1), sum(reward2), done

def retrain_action_selection(agent, state):
    actions = agent.get_non_dominated_actions(state)
    
    if 3 in actions and 1 in actions:  # If both right and down actions are available
        return agent.policy_always_right(actions)
    else:
        return np.random.choice(actions)