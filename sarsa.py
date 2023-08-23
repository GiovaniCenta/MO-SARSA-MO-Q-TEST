import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import deepst
from matplotlib.patches import Rectangle

class Agent:
    def __init__(self, n_states, n_actions, alpha, gamma, epsilon, epsilon_decay,epsilon_min):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q1 = np.zeros((n_states, n_actions))
        self.Q2 = np.zeros((n_states, n_actions))

    def flatten_observation(self, obs):
        if type(obs[1]) is dict:
            return int(np.ravel_multi_index((0), (env.size, env.size)))
        else:
            return int(np.ravel_multi_index(obs, (env.size, env.size)))

    def simple_cull(self, inputPoints, dominates):
        paretoPoints = set()
        candidateRowNr = 0
        dominatedPoints = set()
        while True:
            candidateRow = inputPoints[candidateRowNr]
            inputPoints.remove(candidateRow)
            rowNr = 0
            nonDominated = True
            while len(inputPoints) != 0 and rowNr < len(inputPoints):
                row = inputPoints[rowNr]
                if dominates(candidateRow, row):
                    inputPoints.remove(row)
                    dominatedPoints.add(tuple(row))
                elif dominates(row, candidateRow):
                    nonDominated = False
                    dominatedPoints.add(tuple(candidateRow))
                    rowNr += 1
                else:
                    rowNr += 1

            if nonDominated:
                paretoPoints.add(tuple(candidateRow))

            if len(inputPoints) == 0:
                break
        return paretoPoints, dominatedPoints

    def dominates(self, row, candidateRow):
        return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)

    def get_non_dominated_actions(self, state):
        Xs = self.Q1[state, :]
        Ys = self.Q2[state, :]
        points = np.column_stack((Xs, Ys))

        uniques = np.unique(points, axis=0)

        inputPoints = uniques.tolist()
        paretoPoints, _ = self.simple_cull(inputPoints, self.dominates)
        frontier = [p for p in paretoPoints]

        actions = []

        for pair in frontier:
            indexes = np.where((points[:, 0] == pair[0]) & (points[:, 1] == pair[1]))
            actions.append(indexes[0][0])

        return actions

    def action_selection(self, state):
        
        if np.random.random() > self.epsilon:
            
            actions = self.get_non_dominated_actions(state)
            action = random.choice(actions)
            #action = self.policy_always_right(actions)
        else:
            action = env.action_space.sample()
        return action

    def policy_always_right(self, actions):
        if 3 in actions:
            return 3
        else:
            return np.random.choice(actions)

    def episode_run(self, env):
        done = False
        reward1 = []
        reward2 = []
        state = env.reset()
        state = self.flatten_observation(state)
        action = self.action_selection(state)

        while not done and sum(reward2) > -5000:
            next_state, reward, done, _ = env.step(action)
            r1, r2 = reward
            reward1.append(r1)
            reward2.append(r2)
            next_state = self.flatten_observation(next_state)
            next_action = self.action_selection(next_state)

            if not done:
                next_value1 = self.Q1[next_state, next_action]
                next_value2 = self.Q2[next_state, next_action]
            else:
                next_value1 = 0
                next_value2 = 0

            old_value1 = self.Q1[state, action]
            old_value2 = self.Q2[state, action]
            new_value1 = ((1 - self.alpha) * old_value1) + (self.alpha * (r1 + (self.gamma * next_value1)))
            new_value2 = ((1 - self.alpha) * old_value2) + (self.alpha * (r2 + (self.gamma * next_value2)))
            self.Q1[state, action] = new_value1
            self.Q2[state, action] = new_value2

            self.current_trajectory.append((state, action))

            state = next_state
            action = next_action

        return sum(reward1), sum(reward2), done

    def train(self, env, n_episodes):
        episode_reward1 = []
        episode_reward2 = []
        self.trajectories = []

        for episode in range(n_episodes):
            self.current_trajectory = []
            r1, r2, done = self.episode_run(env)
            episode_reward1.append(r1)
            episode_reward2.append(r2)
            print("Episode = " + str(episode) + "| reward = " + str((r1, r2)))

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.trajectories.append(self.current_trajectory)

        return episode_reward1, episode_reward2

    def plot_rewards(self, n_episodes, reward1, reward2):
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(range(n_episodes), reward1, 'g-')
        ax2.plot(range(n_episodes), reward2, 'b-')

        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Treasure Reward', color='g')

        ax1.set_xlabel('Episode')
        ax2.set_ylabel('Time Penalty', color='b')

        plt.show()

    def plotGraph(self, episodes, rewards1, rewards2):
        fig, ax = plt.subplots()
        ax.plot(episodes, rewards1)
        ax.set_title('Treasure reward x Episodes')

        fig, ax2 = plt.subplots()
        ax2.plot(episodes, rewards2)
        ax2.set_title('Time penalty x Episodes')

        plt.show()

    def plot_nd_actions(self, nd_actions, values):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        grid = np.array(nd_actions).reshape(env.size, env.size)

        fig, ax = plt.subplots(figsize=(8,8))

        # Corrected directions for arrows
        dx = [0, 0, -1, 1]
        dy = [-1, 1, 0, 0]  # Flipped the signs for Up and Down

        # Plot state values and actions
        for i in range(env.size):
            for j in range(env.size):
                # Set the color of the state based on its value
                if values[i, j] > 0:
                    color = 'green'
                elif values[i, j] < 0:
                    color = 'purple'
                else:
                    color = 'white'
                
                rect = Rectangle((j, i), 1, 1, edgecolor='black', facecolor=color, alpha=0.3)
                ax.add_patch(rect)
                
                if color in ['green', 'purple']:
                    ax.text(j + 0.5, i + 0.5, str(values[i, j]), color='black', ha='center', va='center')
                else:
                    # If it's a white cell (no value), then draw the arrows
                    for dir in grid[i,j]:
                        ax.arrow(j + 0.5, i + 0.5, 0.4*dx[dir], 0.4*dy[dir], head_width=0.1, head_length=0.1, fc='k', ec='k')
                
        ax.set_xlim([0, env.size])
        ax.set_ylim([0, env.size])
        ax.invert_yaxis()  # Invert y axis for proper visualization
        plt.show()


if __name__ == "__main__":
    dst_map = deepst.CONCAVE_MAP
    env = deepst.DeepSeaTreasure(dst_map)
    n_states = env.size * env.size
    n_actions = 4
    alpha = 0.7
    gamma = 0.87
    epsilon = 1
    epsilon_decay = 0.9999
    n_episodes = 2000000
    agent = Agent(n_states, n_actions, alpha, gamma, epsilon, epsilon_decay,epsilon_min=0.05)
    reward1, reward2 = agent.train(env, n_episodes)
    #agent.plot_rewards(n_episodes, reward1, reward2)

    from testing_functions import plot_grid,retrain
    n_episodes = 1000
    #reward1, reward2 = retrain(agent, env, n_episodes)

    state = 0
    nd_actions = []
    for state in range(n_states):
        print("state = " + str(state))
        print("Up, Down, Left, Right")
        print(agent.Q1[state, :])
        print(agent.Q2[state, :])
        print("Non-dominated actions for state " + str(state))
        actions = agent.get_non_dominated_actions(state)
        nd_actions.append(actions)
        print(actions)
        print("========================================\n")

    #trajectories = [agent.trajectories[n_episodes - 1]]
    #plot_grid(grid_size=env.size, trajectories=trajectories, map_values=deepst.CONCAVE_MAP)
    agent.plot_nd_actions(nd_actions, dst_map)
