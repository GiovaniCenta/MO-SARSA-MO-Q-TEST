import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import deepst
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Agent:
    def __init__(self, n_states, n_actions, alpha, gamma, epsilon,epsilon_min):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon_min = epsilon_min
        
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
            action = np.random.choice(actions)
            #action = self.policy_always_right(actions)
        else:
            action = env.action_space.sample()
        return action
    
    

    def episode_run(self, env):
        done = False
        reward1 = []
        reward2 = []
        state = env.reset()
        
        #print(state)
        state = self.flatten_observation(state)
        #print(state)
        
        env.render()
        while not done and sum(reward2) > -5000:
            
            
            action = self.action_selection(state)
            

            next_state, reward, done, _ = env.step(action)
            r1, r2 = reward
            
            reward1.append(r1)
            reward2.append(r2)
            next_state = self.flatten_observation(next_state)
            

            if not done:
                old_value1 = self.Q1[state, action]
                old_value2 = self.Q2[state, action]
                next_max1 = np.max(self.Q1[next_state,:])
                next_max2 = np.max(self.Q2[next_state, :])
                print("next_max1 = " + str(next_max1))
                print("old_value1 = " + str(old_value1))
                print("r1 = " + str(r1))
                new_value1 = ((1 - self.alpha) * old_value1) + (self.alpha * (r1 + (self.gamma * next_max1)))
                print("new_value1 = " + str(new_value1))
                new_value2 = ((1 - self.alpha) * old_value2) + (self.alpha * (r2 + (self.gamma * next_max2)))
                self.Q1[state, action] = new_value1
                self.Q2[state, action] = new_value2
            else:
                self.Q1[state, action] = r1
                self.Q2[state, action] = r2

            self.current_trajectory.append((state, action))
            state = next_state
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

            self.epsilon *= 0.997
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
        
    def plotGraph(self,episodes,rewards1,rewards2):
        
        
        fig, ax = plt.subplots()
        ax.plot(episodes, rewards1)
        ax.set_title('Treasure reward x Episodes')
        #plt.show()
        
        
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



        
    def policy_run(self,actions,env):
        done = False
        reward1 = []
        reward2 = []
        state = env.reset()
        
        #print(state)
        state = self.flatten_observation(state)
        #print(state)
        
        
        while not done and sum(reward2) > -5000:
            action = actions[state]

            next_state, reward, done, _ = env.step(action)
            r1, r2 = reward
            reward1.append(r1)
            reward2.append(r2)
            next_state = self.flatten_observation(next_state)
            

            if not done:
                old_value1 = self.Q1[state, action]
                old_value2 = self.Q2[state, action]
                next_max1 = np.max(self.Q1[next_state,:])
                next_max2 = np.max(self.Q2[next_state, :])
                new_value1 = ((1 - self.alpha) * old_value1) + (self.alpha * (r1 + (self.gamma * next_max1)))
                new_value2 = ((1 - self.alpha) * old_value2) + (self.alpha * (r2 + (self.gamma * next_max2)))
                self.Q1[state, action] = new_value1
                self.Q2[state, action] = new_value2
            else:
                self.Q1[state, action] = r1
                self.Q2[state, action] = r2

            
            state = next_state
        return sum(reward1), sum(reward2), done
    
    def policy_always_right(self,actions):
        
        if 3 in actions:
            return 3
        else:
            return np.random.choice(actions)
        
    import matplotlib.pyplot as plt

# Data points for the Pareto frontier
def true_pareto_front():# Data points for the Pareto frontier
    true_points = [(1, -1), (2, -3), (3, -5), (5, -7), (8, -8), (16, -9), (24, -14), (50, -15), (74, -17), (124, -19)]
    points = [(1, -1),(2,-3),(3, -5), (5, -7), (8, -8), (16, -10),(24,-16),(50,-18),(74,-21), (124, -24)]
    # Separate the points into X and Y coordinates
    x_values, y_values = zip(*true_points)

    # Anomaly point
    #anomaly_point = (74, -25)

    # Points for the anomaly frontier
    #anomaly_frontier = [(1, -1), anomaly_point, (124, -19)]
    anomaly_frontier = points

    # Separate the points into X and Y coordinates for the anomaly frontier
    anomaly_x, anomaly_y = zip(*anomaly_frontier)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the points
    plt.scatter(x_values, y_values, color='blue',marker = 's', s=50, label='True Pareto Frontier Points')

    # Connect the points with lines
    plt.plot(x_values, y_values, color='lightblue', linestyle='-', linewidth=1,label = 'True Pareto Frontier')

    # Plot the anomaly point
    #plt.scatter(*anomaly_point, color='blue', marker = 'x',s=100, label='Anomaly Point', zorder=5)

    # Plot the anomaly frontier as a dashed line
    #plt.plot(anomaly_x, anomaly_y, color='green', linestyle='--', linewidth=1, label='Anomaly Frontier')
    plt.scatter(anomaly_x, anomaly_y, marker='x', color='red', s=100, label='PQL Frontier Points')
    plt.plot(anomaly_x, anomaly_y, color='red', linestyle='--', linewidth=1, label='PQL Frontier')

    # Add labels and title
    plt.xlabel('Treasure Reward')
    plt.ylabel('Time Penalty')
    plt.title('True Pareto Frontier vs PQL Frontier')

    # Add grid and legend
    
    plt.legend()

    # Show the plot
    plt.show()

def shift():
    import matplotlib.pyplot as plt

    # Original Pareto frontier points
    true_points = [(1, -2), (2, -4), (3, -8), (8, -11), (16, -13)]

    # Updated Pareto frontier points after adding a new point
    updated_points = [(1, 0), (2, -2), (3, -7), (8, -9), (16, -11)]

    # Separate the points into X and Y coordinates for both frontiers
    true_x, true_y = zip(*true_points)
    updated_x, updated_y = zip(*updated_points)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the original Pareto frontier points
    plt.scatter(true_x, true_y, color='blue', s=100, label='Original Pareto Frontier', zorder=5)
    plt.plot(true_x, true_y, color='blue', linestyle='-', linewidth=2)

    # Plot the updated Pareto frontier points
    plt.scatter(updated_x, updated_y, color='green', s=50, label='Updated Pareto Frontier', zorder=5)
    plt.plot(updated_x, updated_y, color='green', linestyle='--', linewidth=2)
    
    plt.scatter([1,2], [0,-2], color='red', marker = 'X',s=200, label='Anomaly point', zorder=5)

    # Add markers to show the distance between the shifts
    for true_point, updated_point in zip(true_points, updated_points):
        plt.plot([true_point[0], updated_point[0]], [true_point[1], updated_point[1]], color='black', linestyle='--', linewidth=1)

    # Add labels and title
    plt.xlabel('Treasure Reward')
    plt.ylabel('Time Penalty')
    plt.title('Shift in Pareto Frontier')

    # Add grid and legend
    
    plt.legend()

    # Show the plot
    plt.show()

def shift_corrected():
    import matplotlib.pyplot as plt

    # Original Pareto frontier points
    true_points = [(1, -2), (2, -4), (3, -8), (8, -11), (16, -13)]

    # Updated Pareto frontier points after adding a new point
    updated_points = [(1, -2), (2, -4), (3, -7), (8, -9), (16, -11)]

    # Separate the points into X and Y coordinates for both frontiers
    true_x, true_y = zip(*true_points)
    updated_x, updated_y = zip(*updated_points)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the original Pareto frontier points
    plt.scatter(true_x, true_y, color='blue', s=100, label='Original Pareto Frontier', zorder=5)
    plt.plot(true_x, true_y, color='blue', linestyle='-', linewidth=2)

    # Plot the updated Pareto frontier points
    plt.scatter(updated_x, updated_y, color='green', s=50, label='Updated Pareto Frontier', zorder=5)
    plt.plot(updated_x, updated_y, color='green', linestyle='--', linewidth=2)
    
    plt.scatter([1,2], [-2,-4], color='black',s=200, label='corrected anomaly point', zorder=5)

    # Add markers to show the distance between the shifts
    for true_point, updated_point in zip(true_points, updated_points):
        plt.plot([true_point[0], updated_point[0]], [true_point[1], updated_point[1]], color='black', linestyle='--', linewidth=1)

    # Add labels and title
    plt.xlabel('Treasure Reward')
    plt.ylabel('Time Penalty')
    plt.title('Shift in Pareto Frontier')

    # Add grid and legend
    
    plt.legend()

    # Show the plot
    plt.show()



if __name__ == "__main__":
    
    """    alpha = 0.4
    gamma = 1
    r1 = 1
    q_sa = 0
    maxx=1
    q = ((1 - alpha) * q_sa) + (alpha * (r1 + (gamma * maxx)))
    print(q)
    exit(8)"""
    true_pareto_front()
    #shift()
    #shift_corrected()
    exit(8)
    dst_map = deepst.TEST_MAP4x4
    env = deepst.DeepSeaTreasure(dst_map )
    n_states = env.size*env.size
    n_actions = 4
    alpha = 0.4
    gamma = 0.98
    epsilon = 1
    epsilon_decay = 0.996
    n_episodes = 3600
    agent = Agent(n_states, n_actions, alpha, gamma, epsilon,epsilon_min = 0.05)
    reward1, reward2 = agent.train(env, n_episodes)
    #agent.plot_rewards(n_episodes, reward1, reward2)
    
    #agent.plotGraph(range(n_episodes),reward1,reward2)
    from testing_functions import plot_grid, retrain
    #reward1, reward2 = retrain(env, n_episodes)

    state = 0
    nd_Actions = []
    for state in range(n_states):
        print("state = " + str(state))
        print("Up, Down, Left, Right")
        print(agent.Q1[state, :])
        print(agent.Q2[state, :])
        print("Non-dominated actions for state " + str(state))
        actions = agent.get_non_dominated_actions(state)
        print(actions)
        nd_Actions.append(actions)
        print("========================================\n")
        
    
    agent.plot_nd_actions(nd_Actions,dst_map)
   
    #trajectories = [agent.trajectories[1000]]
    #plot_grid(grid_size = env.size, trajectories = trajectories , map_values = deepst.CONCAVE_MAP)
    
    #agent.plot_nd_actions(nd_Actions,deepst.CONCAVE_MAP)
    

    
