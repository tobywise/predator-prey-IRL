import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.special import softmax
import sympy as sy
from scipy.spatial.distance import cdist
import pandas as pd
from sisyphus.mdp import ValueIteration
from sisyphus.envs._base import grid_to_adj

def get_adjacent_idx(idx):

    return [np.array(idx) + np.array(i) for i in [[0, -1], [0, 1], [-1, 0], [1, 0]]]


def get_function(eq, features):
    # https://stackoverflow.com/q/41874636
    s = sy.symbols(list(features.keys()))
    expr = sy.sympify(eq)
    f = sy.lambdify(s, expr, 'numpy')
    return f(**features)



class PredatorEnvironment(object):

    def __init__(self, name, size=(10, 15), agents=(), features=(), n_rewards=5):

        self.agents = agents
        self.size = size
        self.name = name

        # Set up array to represent environment
        self.env_array = np.zeros(size)
        self.features = features
        self.feature_arrays = dict()
        self.grid = np.arange(size[0] * size[1], dtype=int).reshape(size)

        # Add features
        for feature in features:
            # if not isinstance(feature, predators.EnvironmentFeature):
            #     raise TypeError("Feature is not of EnvironmentFeature class, is type {0}".format(type(feature)))

            # If we haven't specified an array then calculate one
            if feature.feature_array is None:
                self.feature_arrays[feature.name] = self.apply_feature(feature)
            else:
                if feature.feature_array.shape == self.size:
                    self.feature_arrays[feature.name] = feature.feature_array.astype(float)
                    # print('a')
                else:
                    raise AttributeError("Feature {0} array shape {1} is not the same shape as the environment {2}".format(feature.name, feature.feature_array.shape, self.size))

        for agent in self.agents:
            agent.intialise(self)
            agent_array = np.zeros(self.size)
            agent_array[agent.idx[1], agent.idx[0]] = 1
            self.feature_arrays[agent.name] = agent_array

        # Add rewards
        self.feature_arrays['reward'] = np.zeros(self.size)
        for r in range(n_rewards):
            self.feature_arrays['reward'][(np.random.randint(self.size[0]), np.random.randint(self.size[1]))] = 1


    def __repr__(self):
        return r'< Environment {0} | Size = {1} X {2} | Predator = {3} | {4} Features: {5}>'.format(self.name, 
                                                                                                 self.size[0], 
                                                                                                 self.size[1], 
                                                                                                 len(self.feature_arrays),
                                                                                                 'Prey, ' + ', '.join([i.name for i in self.features]))

    def apply_feature(self, feature):

        feature_array = np.zeros(self.size)

        for _ in range(feature.n_clusters):

            # Get cluster size
            start_idx = (np.random.randint(self.size[1]), np.random.randint(self.size[0]))
            cluster_size = int(np.random.normal(feature.cluster_size_mean, feature.cluster_size_sd))
            cluster_size = np.max([0, cluster_size])  # make sure this is positive or zero
            if cluster_size == 0:
                warnings.warn("Cluster size for feature is zero")
            
            # Add feature based on transition matrix
            for s in range(cluster_size):
                if s == 0:
                    idx = start_idx
                    feature_array[tuple(idx[::-1])] = 1
                else:
                    possible_transitions = np.nonzero(feature.transition_probability)
                    selected_transition = np.random.choice(np.arange(len(possible_transitions[0])), p=feature.transition_probability[possible_transitions])
                    selected_transition_idx = np.stack(possible_transitions).T[selected_transition] - np.array([1, 1])
                    idx += selected_transition_idx

                    # Indexes below zero
                    idx[idx < 0] = 0

                    # Indices above size of map
                    for i in range(2):
                        if idx[1 - i] + 1 > self.size[i]:
                            idx[1- i] = self.size[i] - 1
                    feature_array[tuple(idx[::-1])] = 1

        return feature_array

            
    def plot_environment(self, filename=None):

        # plt.imshow(self.env_array)

        f, ax = plt.subplots(dpi=150)

        # Gridlines
        ax.set_xticks(np.arange(self.size[1]) - 0.5)
        ax.set_yticks(np.arange(self.size[0]) - 0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.set_ticks_position('none') 
        ax.yaxis.set_ticks_position('none') 
        ax.grid(color='gray', linewidth=1, alpha=0.3)
        ax.set_xlim(-0.5, self.size[1] - 0.5) 
        ax.set_ylim(-0.5, self.size[0] - 0.5)

        # Features
        for i, f in enumerate(self.features):
            temp_array = self.feature_arrays[f.name].copy()
            temp_array[temp_array == 0] = np.nan
            im = ax.imshow(temp_array * (i + 1), alpha = 1 / len(self.feature_arrays), vmin=0, vmax=len(self.feature_arrays), cmap='tab10', origin='lower')

        # Rewards
        for r in np.argwhere(self.feature_arrays['reward'].T == 1):
            ax.scatter(r[0], r[1], color='tab:orange')

        # Agents
        for agent in self.agents:
            if 'redator' in agent.name:
                print(agent.idx)
                marker = 'X'
                color = 'tab:red'
            else:
                marker = '*'
                color = 'tab:blue'
            
            ax.scatter(agent.idx[0], agent.idx[1], color=color, marker=marker, s=100, zorder=100)

            for n, i in enumerate(agent.move_history[::-1]):
                ax.scatter(i[0], i[1], color=color, marker=marker, s=80, alpha=(1 - ((n + 0.8) / len(agent.move_history))) * 0.5, zorder=100)


        if filename is not None:
            plt.savefig(filename)

        plt.show()


class EnvironmentFeature(object):

    def __init__(self, feature_array=None, n_clusters=1, cluster_size_mean=10, cluster_size_sd=5, transition_probability="equal", name=''):

        if transition_probability == "equal":
            self.transition_probability = np.zeros((3, 3))
            self.transition_probability[0, 1] = 1
            self.transition_probability[1, 0] = 1
            self.transition_probability[2, 1] = 1
            self.transition_probability[1, 2] = 1
            self.transition_probability /= 4

        else:
            self.transition_probability = transition_probability

        if not np.sum(self.transition_probability) == 1:
            raise AttributeError("Sum of transition probabilities must equal 1")

        if feature_array is not None:
            self.n_clusters = 'N/A'
            self.cluster_size_mean = 'N/A'
            self.cluster_size_sd = 'N/A'
        else:
            self.n_clusters = n_clusters
            self.cluster_size_mean = cluster_size_mean
            self.cluster_size_sd = cluster_size_sd
        self.feature_array = feature_array
        self.name = name

    def __repr__(self):
        return r'< Environment feature {0} | Number of clusters = {1} | Mean cluster size = {2} | Cluster size SD = {3}>'.format(self.name, self.n_clusters, self.cluster_size_mean, self.cluster_size_sd)
    



class Agent(object):

    def __init__(self, function, strategy='function', name='', starting_position=(), gamma=0.95, policy='max', w=0.5):

        if not strategy in ['function', 'irl']:
            raise ValueError("Strategy should be one of 'function', 'irl'")
        else:
            self.strategy = strategy

        if not isinstance(function, str):
            raise AttributeError('Function value should be a string representing the function')
        
        self.function = function
        self.name = name
        self.environment = None
        self.idx = starting_position
        self.move_history = []
        self.distance_arrays = dict()
        self.solved = False

        self.solver = ValueIteration(policy=policy, gamma=gamma, w=w)

    def intialise(self, environment):
        # Things to do with environment
        self.environment = environment
        self.state_rewards = np.zeros(self.environment.size)
        
        if not len(self.idx):
            self.idx = (np.random.randint(self.environment.size[1]), np.random.randint(self.environment.size[0]))

    # def _repr_latex_(self):
    #     f = ''
    #     for n, i in enumerate(self.function):
    #         f += r'{0} \times X_{1} +'.format(i, n)
    #     f = f[:-2]
    #     return r'< Predator {0} | $y = {1}$ | Strategy = {2} >'.format('"' + self.name + '"', f, self.strategy)

    def get_state_rewards(self):

        if self.strategy == 'function':
            # for feature, feature_array in self.environment.feature_arrays.items():
            #     self.distance_arrays[feature] = self._get_distance(np.argwhere(feature_array)) 

            # Use function
            self.state_rewards = get_function(self.function, self.environment.feature_arrays)


    def update_info(self, terminal=()):

        # https://github.com/ndawlab/seqanx
        
        # sisyphus compatibility
        # self.terminal = np.stack(np.argwhere(self.state_rewards == self.state_rewards.max()))
        self.terminal = terminal
        self.start = np.ravel_multi_index((self.idx[1], self.idx[0]), self.environment.size)

        ## Define one-step transition matrix.
        self.T = grid_to_adj(self.environment.grid, self.terminal)

        ## Define rewards.
        self.get_state_rewards()

        R = 0 * np.ones_like(self.T)  # Majority transitions
        R[:, :] = self.state_rewards.flatten()    # Reward transition     
        R[self.terminal, self.terminal] = 0              # Terminal transitions
        R *= self.T
            
        ## Iteratively define MDP information.
        info = []
        for s in range(np.product(self.environment.size)):
            
            ## Observe information.
            s_prime, = np.where(~np.isnan(self.T[s]))
            r = R[s, s_prime]
            t = np.append(1, np.zeros(r.size-1))
            
            ## Iteratively append.
            for i in range(s_prime.size): 
                info.append({ "S":s, "S'":np.roll(s_prime,i), "R":np.roll(r,i), "T":t })
        
        ## Store.
        self.info = pd.DataFrame(info, columns=("S","S'","R","T"))

    def solve(self, terminal=(), max_iter=100):

        self.update_info(terminal=terminal)
        self.solver.max_iter = max_iter
        self.solver.fit(self)
        self.solved = True


    def _get_distance(self, idx_list=()):

        distance_arrays = np.zeros((len(idx_list), ) + self.environment.size)

        for n, idx in enumerate(idx_list):
            for i in range(self.environment.size[0]):
                for j in range(self.environment.size[1]):
                    distance_arrays[n, i, j] = euclidean(idx, (i, j))

        distance_array = distance_arrays.min(axis=0)
        distance_array = np.max(distance_array) - distance_array 
        distance_array = (distance_array - np.min(distance_array))/np.ptp(distance_array)
        
        return distance_array

    def move(self):

        if not self.solved:
            self.solve()
        
        self.move_history.append(self.idx)

        new_idx = np.unravel_index(self.solver.pi[1], self.environment.size)
        self.idx = (new_idx[1], new_idx[0])
        self.solved = False

    def plot_distances(self, filename='distances.svg'):

        self.get_state_rewards()

        _, ax = plt.subplots(1, len(self.distance_arrays), dpi=150, figsize=(3 * len(self.distance_arrays), 4.5))

        for n, dist in enumerate(self.distance_arrays):

            # Gridlines
            ax[n].set_xticks(np.arange(self.size[1]) - 0.5)
            ax[n].set_yticks(np.arange(self.size[0]) - 0.5)
            ax[n].set_xticklabels([])
            ax[n].set_yticklabels([])
            ax[n].xaxis.set_ticks_position('none') 
            ax[n].yaxis.set_ticks_position('none') 
            ax[n].grid(color='gray', linewidth=1, alpha=0.3)
            ax[n].set_xlim(-0.5, self.size[1] - 0.5) 
            ax[n].set_ylim(-0.5, self.size[0] - 0.5)

            im = ax[n].imshow(dist, origin='lower',)

            ax[n].set_title("Feature {0}".format(n+1), fontweight='light')
            # cbar = plt.colorbar(im, fraction=0.03, pad=0.04, label='Value')

        if filename is not None:
            plt.savefig(filename)

        plt.show()

    def plot_state_rewards(self, filename=None):

        self.get_state_rewards()

        _, ax = plt.subplots(dpi=100)

        # Gridlines
        ax.set_xticks(np.arange(self.environment.size[1]) - 0.5)
        ax.set_yticks(np.arange(self.environment.size[0]) - 0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.set_ticks_position('none') 
        ax.yaxis.set_ticks_position('none') 
        ax.grid(color='gray', linewidth=1, alpha=0.3)
        ax.set_xlim(-0.5, self.environment.size[1] - 0.5) 
        ax.set_ylim(-0.5, self.environment.size[0] - 0.5)

        im = ax.imshow(self.state_rewards, origin='lower',)
        cbar = plt.colorbar(im, fraction=0.03, pad=0.04, label='Reward')

        # function_string = ' + '.join([r'{0} \cdot X_{1}'.format(self.predator.function[i], i + 1) for i in range(len(self.predator.function))])

        ax.set_title(r'$y = {0}$'.format(self.function))

        if filename is not None:
            plt.savefig(filename)

    def plot_state_values(self, filename=None):

        if not self.solved:
            self.solve()

        _, ax = plt.subplots(dpi=100)

        # Gridlines
        ax.set_xticks(np.arange(self.environment.size[1]) - 0.5)
        ax.set_yticks(np.arange(self.environment.size[0]) - 0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.set_ticks_position('none') 
        ax.yaxis.set_ticks_position('none') 
        ax.grid(color='gray', linewidth=1, alpha=0.3)
        ax.set_xlim(-0.5, self.environment.size[1] - 0.5) 
        ax.set_ylim(-0.5, self.environment.size[0] - 0.5)

        im = ax.imshow(self.solver.V.reshape(self.environment.size), origin='lower',)
        cbar = plt.colorbar(im, fraction=0.03, pad=0.04, label='Value')

        # function_string = ' + '.join([r'{0} \cdot X_{1}'.format(self.predator.function[i], i + 1) for i in range(len(self.predator.function))])

        ax.set_title(r'$y = {0}$'.format(self.function))

        if filename is not None:
            plt.savefig(filename)

        return ax
        
    def move_to_idx(self, x, y):

        self.move_history.append(self.idx)
        self.idx = (x, y)
        self.solved = False


    def plot_policy(self, color='w', head_width=0.25, head_length=0.25):
        """Plot agent policy on grid world.
        Parameters
        ----------
        ax : matplotlib Axes
            Axes in which to draw the plot.
        pi : array
            Agent policy, i.e. ordered visitation of states.
        color : str, list
            Color(s) of arrow.
        head_width : float (default=0.25)
            Width of the arrow head.
        head_length : float (default=0.25)
            Length of the arrow head.
        Returns
        -------
        ax : matplotlib Axes
            Axes in which to draw the plot.
        """

        if not self.solved:
            self.solve()
        ax = self.plot_state_values()
        pi = self.solver.pi

        ## Error-catching.
        if isinstance(color, str):
            color = [color] * len(pi)
            
        ## Iteratively plot arrows.
        for i in range(len(pi)-1):

            ## Identify S, S' coordinates.
            y1, x1 = np.where(self.environment.grid==pi[i])
            y2, x2 = np.where(self.environment.grid==pi[i+1])

            ## Define arrow coordinates.
            x, y = int(x1), int(y1)
            dx, dy = 0.5*int(x2-x1), 0.5*int(y2-y1)
            
            ## Plot.
            ax.arrow(x, y, dx, dy, color=color[i], head_width=head_width, head_length=head_length)
            
        return ax