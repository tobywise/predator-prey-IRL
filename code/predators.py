import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.special import softmax
import sympy as sy
from scipy.spatial.distance import cdist
import pandas as pd
from sisyphus.mdp import ValueIteration
from collections import OrderedDict
import json
import re
# from sisyphus.envs._base import grid_to_adj

def offset_distance(x1,y1,x2,y2):
    ac = offset_to_cube(x1,y1)
    bc = offset_to_cube(x2,y2)
    f = cube_distance(ac, bc)
    return f

def offset_to_cube(col, row):
    x = col
    z = row - (col + (col & 1)) / 2
    y = -x-z
    f = np.array([x,y,z])
    return f

def cube_distance(a, b):
    return (np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]) + np.abs(a[2] - b[2])) / 2

def hex_dist(XA, XB):

    dist = np.zeros((len(XA), len(XB)))

    for n, i in enumerate(XA):
        for nn, j in enumerate(XB):
            dist[n, nn] = offset_distance(i[0], i[1], j[0], j[1])

    return dist

def state_to_idx(state, env):
    return np.unravel_index(state, env.size)

def grid_to_adj(grid, terminal=False, kind='square'):
    """Convert grid world to adjacency matrix.

    Modified from sisyphys
    
    Parameters
    ----------
    grid : array, shape (i,j)
        Grid world.
    terminal : array
        List of terminal states.
        
    Returns
    -------
    T : array, shape (n_states, n_states)
        Adjacency matrix of states.
        
    Notes
    -----
    The initial grid can contain any value. Grid states defined as NaNs 
    are treated as nonviable states and excluded from further processing.        
    """
    
    ## Identify coordinates of viable states.
    rr = np.array(np.where(~np.isnan(grid))).T

    ## Compute adjacency matrix.
    if kind == 'square':
        A = (cdist(rr,rr)==1).astype(int)
    elif kind == 'hex':
        A = (hex_dist(rr,rr)==1).astype(int)

    ## Define one-step transition matrix.
    T = np.where(A, 1, np.nan)
    
    ## Update terminal states.
    if np.any(terminal):
        T[terminal] = np.nan
        T[terminal,terminal] = 1
    
    return T

def get_adjacent_idx(idx):

    return [np.array(idx) + np.array(i) for i in [[0, -1], [0, 1], [-1, 0], [1, 0]]]


def get_function(eq, features):
    # https://stackoverflow.com/q/41874636
    s = sy.symbols(list(features.keys()))
    expr = sy.sympify(eq)
    f = sy.lambdify(s, expr, 'numpy')
    return f(**features)

def function_to_weights(function, features):

    weights = []

    for i in features.keys():
        w = re.search('[0-9\.]+(?=\*{0})'.format(i), function)
        if w:
            weights.append(w.group())
        else:
            weights.append(0)

    weights = np.array(weights)

    return weights


def draw_hexagons(X, outer_radius=0.1, edgecolor='#787878', cmap='Greys', ax=None, facecolor='cmap', return_coords=False, labels=False, scale=True, **kwargs):

    from matplotlib.patches import RegularPolygon
    from sklearn.preprocessing import minmax_scale

    coord = [[0,0,0],[0,1,-1],[-1,1,0],[-1,0,1],[0,-1,1],[1,-1,0],[1,0,-1]]

    # Horizontal cartesian coords
    hcoord = [c[0] for c in coord]

    # Vertical cartersian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. for c in coord]

    n_cols = X.shape[0]
    n_rows = X.shape[1]

    inner_radius = 0.86602540 * outer_radius

    y_coord = []
    x_coord = []
    
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(n_cols, n_rows * 0.33), dpi=200)

    ax.set_aspect('equal')
    
    cmap = plt.get_cmap(cmap)
    
    if scale:
        X = X.copy()
        X = (X + np.abs(np.min(X)))
        X /= np.max(X)

    coords = np.zeros((2, X.shape[0], X.shape[1]))
    
    # Get coordinates and draw hexagons
    for x in range(n_cols):
        for z in range(n_rows):
            if x % 2:
                y_coord.append(z * (inner_radius * 2))
            else:
                y_coord.append(z * (inner_radius * 2) + ((inner_radius * 2) / 2))
            x_coord.append(x * outer_radius * 1.5)
            if facecolor == 'cmap':
                if not scale:
                    cell_colour = cmap(int(X[x, z] - 1))
                else:
                    cell_colour = cmap(X[x, z])
            else:
                cell_colour = facecolor
            hex = RegularPolygon((x_coord[-1], y_coord[-1]), numVertices=6, radius=outer_radius, 
                                orientation=np.radians(30), 
                                facecolor=cell_colour, edgecolor=edgecolor, **kwargs)
            if (not scale and X[x, z] != 0) or scale:
                ax.add_patch(hex)
            if labels:
                ax.text(x_coord[-1] - inner_radius / 2, y_coord[-1], '{0}, {1}'.format(x, z), fontsize=5)
            coords[0, x, z] = x_coord[-1]
            coords[1, x, z] = y_coord[-1]

    plt.xlim(0 - outer_radius, np.max(x_coord) + outer_radius)
    plt.ylim(0 - outer_radius, np.max(y_coord) + outer_radius)
    plt.axis('off')
    
    if return_coords:
        return coords


def get_action_from_states(A, B, T, environment):

    """
    Takes a pair of adjacent states and gives the action needed to move from A to B
    """

    if not T[A, B]:
        raise ValueError("A and B are not adjacent")

    x1, y1 = np.where(environment.grid==A)
    x2, y2 = np.where(environment.grid==B)

    dx, dy = x2-x1, y2-y1

    if environment.kind == 'square':

        actions = np.arange(4)

        if dx == 0:
            if dy > 0:
                action = 0
            if dy < 0:
                action = 2
        else:
            if dx > 0:
                action = 1
            if dx < 0:
                action = 3
    
    elif environment.kind == 'hex':

        if dx == 0:
            if dy > 0:
                action = 0
            if dy < 0:
                action = 3
        else:
            if x1 % 2:
                if dx > 0 and dy == 0:
                    action = 1
                elif dx < 0 and dy == 0:
                    action = 5
                elif dx > 0 and dy < 0:
                    action = 2
                elif dx < 0 and dy < 0:
                    action = 4
            else:
                if dx > 0 and dy > 0:
                    action = 1
                elif dx < 0 and dy > 0:
                    action = 5
                elif dx > 0 and dy == 0:
                    action = 2
                elif dx < 0 and dy == 0:
                    action = 4
        
    return action

def trajectory_to_state_action(trajectory, sas):

    """
    Converts a trajectory of state IDs to an array of state-action pairs
    """

    sa_pairs = np.zeros((len(trajectory) - 1, 2))

    for n, state in enumerate(trajectory[:-1]): # Exclude the final state because there is no action taken
        sa_pairs[n, 0] = state
        sa_pairs[n, 1] = np.argmax(sas[state, :, trajectory[n+1]])

    return sa_pairs



class PredatorEnvironment(object):

    def __init__(self, name, size=(10, 15), kind='hex', agents=(), features=(), n_rewards=5, overlaps=[]):

        self.agents = agents
        self.size = size
        self.name = name
        self.kind = kind

        # Set up array to represent environment
        self.env_array = np.zeros(size)
        self.features = features
        self.feature_arrays = OrderedDict()
        self.grid = np.arange(size[0] * size[1], dtype=int).reshape(size)
        if overlaps == []:
            overlaps = [] * len(self.features)
        self.overlaps = overlaps
        self.n_states = np.product(self.size)

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
                else:
                    raise AttributeError("Feature {0} array shape {1} is not the same shape as the environment {2}".format(feature.name, feature.feature_array.shape, self.size))

        # Remove overlap 
        for i, o in enumerate(overlaps):
            for feature_o in o:
                f_keys = list(self.feature_arrays.keys())
                self.feature_arrays[f_keys[feature_o]][self.feature_arrays[f_keys[i]] > 0] = 0

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
                    selected_transition_idx = feature.transition_directions[idx[0] % 2][selected_transition]

                    idx += np.array(selected_transition_idx)

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

        if self.kind == 'square':

            f, ax = plt.subplots(dpi=150)

            # Gridlines
            ax.set_xticks(np.arange(self.size[0]) - 0.5)
            ax.set_yticks(np.arange(self.size[1]) - 0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none') 
            ax.yaxis.set_ticks_position('none') 
            ax.grid(color='gray', linewidth=1, alpha=0.3)
            ax.set_xlim(-0.5, self.size[0] - 0.5) 
            ax.set_ylim(-0.5, self.size[1] - 0.5)

            # Features
            for i, f in enumerate(self.features):
                temp_array = self.feature_arrays[f.name].copy().T
                temp_array[temp_array == 0] = np.nan
                temp_array *= (i + 1)
                temp_array = temp_array.astype(int)

                if f.name == 'Shadow':
                    im = ax.imshow(temp_array, alpha = 0.6, vmin=0, vmax=2, cmap='Greys', origin='lower')
                    
                else:
                    im = ax.imshow(temp_array, alpha = 1 / len(self.feature_arrays), vmin=0, vmax=len(self.feature_arrays), cmap='tab10', origin='lower')

            # Rewards
            for r in np.argwhere(self.feature_arrays['reward'] == 1):
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

        elif self.kind == 'hex':

            fig, ax = plt.subplots(1, figsize=(self.size[1] / 2, self.size[0] / 2), dpi=150)

            # Features
            for i, f in enumerate(self.features):
                temp_array = self.feature_arrays[f.name].copy()
                temp_array *= (i + 1)
                # temp_array = temp_array.astype(int)

                if f.name == 'Shadow':
                    hex_coords = draw_hexagons(temp_array, alpha=0.6, linewidth=0, ax=ax, return_coords=True)
                    
                else:
                    hex_coords = draw_hexagons(temp_array, alpha=1 / len(self.feature_arrays), 
                    scale=False, cmap='tab10', linewidth=0, ax=ax, return_coords=True)

            # Agents
            for agent in self.agents:
                if 'redator' in agent.name:
                    print(agent.idx)
                    marker = 'X'
                    color = 'tab:red'
                else:
                    marker = '*'
                    color = 'tab:blue'
                
                ax.scatter(hex_coords[(0,) + agent.idx[::-1]], hex_coords[(1,) + agent.idx[::-1]], 
                            color=color, marker=marker, s=100, zorder=100)

                for n, i in enumerate(agent.move_history[::-1]):
                    ax.scatter(hex_coords[(0,) + i[::-1]], hex_coords[(1,) + i[::-1]], 
                    color=color, marker=marker, s=80, alpha=(1 - ((n + 0.8) / len(agent.move_history))) * 0.5, zorder=100)


            # Gridlines
            draw_hexagons(np.ones((self.size[0], self.size[1])), ax=ax, scale=False, facecolor=(0, 0, 0, 0))

            # Rewards
            for r in np.argwhere(self.feature_arrays['reward'].T == 1):
                r = tuple(r[::-1])
                ax.scatter(hex_coords[(0, ) + r], hex_coords[(1, ) + r], color='tab:orange', alpha=1, s=100, zorder=99)

            if filename is not None:
                plt.savefig(filename)

            plt.show()

    def feature_matrix(self):

        feature_matrix = np.zeros((np.product(self.size), len(self.feature_arrays.keys())))

        for n, f in enumerate(self.feature_arrays.keys()):

            feature_matrix[:, n] = self.feature_arrays[f].flatten()

        return feature_matrix

    def to_json(self, return_string=False, fname=None):

        json_dict = dict()

        json_dict['size'] = list(self.size)

        json_dict['features'] = dict()

        for k, v in self.feature_arrays.items():
            json_dict['features'][k] = [list([int(j) for j in i]) for i in np.argwhere(v)]

    
        if fname is not None:
            with open(fname, 'w') as f:
                json.dump(json_dict, f)

        if return_string:
            json_string = json.dumps(json_dict)
            return json_string



class EnvironmentFeature(object):

    def __init__(self, feature_array=None, n_clusters=1, cluster_size_mean=10, cluster_size_sd=5, transition_probability="equal", kind='hex', name=''):

        if transition_probability == "equal":

            if kind == 'square':

                self.transition_directions = [[-1, -1], [-1, 0], [-1, +1], # Left column
                                              [0, -1], [0, 0], [0, +1], # Middle column
                                              [+1, -1], [+1, 0], [+1, +1]] # Right column   
                self.transition_probability = [self.transition_directions] * 2  # duplicate for compatibility with hex
                self.transition_probability = np.zeros(9)
                self.transition_probability[1] = 1
                self.transition_probability[3] = 1
                self.transition_probability[5] = 1
                self.transition_probability[7] = 1

                self.transition_probability /= np.sum(self.transition_probability)

            if kind == 'hex':

                self.transition_directions = [[[-1, 0], [-1, +1], [0, -1], [0, 0], [0, +1], [+1, 0], [+1, +1]], 
                                              [[-1, -1], [-1, 0], [0, -1], [0, 0], [0, +1], [+1, -1], [+1, 0]]]

                self.transition_probability = np.ones(7)
                self.transition_probability[3] = 0  # self transition

                self.transition_probability /= np.sum(self.transition_probability)

        else:
            self.transition_probability = transition_probability


        if not np.round(np.sum(self.transition_probability), 2) == 1:
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

    def __init__(self, function, strategy='function', name='', starting_position=(), gamma=0.95, policy='max', w=0.5, max_iter=100):

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
        self.sas = None

        self.solver = ValueIteration(policy=policy, gamma=gamma, w=w, max_iter=max_iter)

    def intialise(self, environment):
        # Things to do with environment
        self.environment = environment
        self.state_rewards = np.zeros(self.environment.size)
        
        if not len(self.idx):
            self.idx = (np.random.randint(self.environment.size[1]), np.random.randint(self.environment.size[0]))

    def get_state_rewards(self):

        self.state_rewards = get_function(self.function, self.environment.feature_arrays)

    def get_state_action_transitions(self):

        self.update_info()

        # Create a state X action X state array
        if self.environment.kind == 'square':
            n_actions = 4
        elif self.environment.kind == 'hex':
            n_actions = 6

        sas = np.zeros((self.T.shape[0], n_actions, self.T.shape[1]))

        # Loop over state pairs
        for i in range(self.T.shape[0]):
            for j in range(self.T.shape[1]):
                if self.T[i, j] == 1:
                    action = get_action_from_states(i, j, self.T, self.environment)
                    sas[i, action, j] = 1

        self.sas = sas


    def update_info(self, terminal=()):

        # https://github.com/ndawlab/seqanx
        
        # sisyphus compatibility
        # self.terminal = np.stack(np.argwhere(self.state_rewards == self.state_rewards.max()))
        self.terminal = terminal
        self.start = np.ravel_multi_index((self.idx[1], self.idx[0]), self.environment.size)

        ## Define one-step transition matrix.
        self.T = grid_to_adj(self.environment.grid, self.terminal, kind=self.environment.kind)

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

    def solve(self, one_step=False, terminal=()):

        if not one_step:
            self.update_info(terminal=terminal)
        else:
            adj = get_adjacent_idx(self.idx)

            adj = [np.ravel_multi_index((i[1], i[0]), self.environment.size) for i in adj]
            self.update_info(terminal=adj)

        # print("updated info")
        self.solver.fit(self)
        # print("solved")
        self.solved = True
        if self.sas is None:
            self.get_state_action_transitions()
        self.solver.sa_pairs = trajectory_to_state_action(self.solver.pi, self.sas)


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
        self.environment.feature_arrays[self.name] *= 0
        self.environment.feature_arrays[self.name][new_idx] = 1
        self.solved = False

        # Remove rewards if eaten
        if 'eward' in self.function and self.environment.feature_arrays['reward'][self.idx[1], self.idx[0]] == 1:
            print("REWARDED")
            self.environment.feature_arrays['reward'][self.idx[1], self.idx[0]] = 0

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

    def plot_state_rewards(self, filename=None, cbar=True):

        self.get_state_rewards()

        if self.environment.kind == 'square':

            _, ax = plt.subplots(dpi=100)

            # Gridlines
            ax.set_xticks(np.arange(self.environment.size[0]) - 0.5)
            ax.set_yticks(np.arange(self.environment.size[1]) - 0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none') 
            ax.yaxis.set_ticks_position('none') 
            ax.grid(color='gray', linewidth=1, alpha=0.3)
            ax.set_xlim(-0.5, self.environment.size[0] - 0.5) 
            ax.set_ylim(-0.5, self.environment.size[1] - 0.5)

            im = ax.imshow(self.state_rewards.T, origin='lower',)

            if cbar:
                cbar = plt.colorbar(im, fraction=0.03, pad=0.04, label='Reward')

            # function_string = ' + '.join([r'{0} \cdot X_{1}'.format(self.predator.function[i], i + 1) for i in range(len(self.predator.function))])
        
        elif self.environment.kind == 'hex':

            _, ax = plt.subplots(1, figsize=(self.environment.size[1] / 2, self.environment.size[0] / 2), dpi=150)

            draw_hexagons(self.state_rewards, cmap='viridis', linewidth=0, ax=ax)

            # Gridlines
            draw_hexagons(np.ones((self.environment.size[0], self.environment.size[1])), ax=ax, facecolor=(0, 0, 0, 0))

        ax.set_title(r'$y = {0}$'.format(self.function))

        if filename is not None:
            plt.savefig(filename)

    def plot_state_values(self, filename=None, ax=None, cbar=True, **kwargs):

        if not self.solved:
            self.solve()

        coords = None

        if self.environment.kind == 'square':
            print("SQUARE")
            if ax is None:
                _, ax = plt.subplots(dpi=100)

            # Gridlines
            ax.set_xticks(np.arange(self.environment.size[0]) - 0.5)
            ax.set_yticks(np.arange(self.environment.size[1]) - 0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none') 
            ax.yaxis.set_ticks_position('none') 
            ax.grid(color='gray', linewidth=1, alpha=0.3)
            ax.set_xlim(-0.5, self.environment.size[0] - 0.5) 
            ax.set_ylim(-0.5, self.environment.size[1] - 0.5)

            im = ax.imshow(self.solver.V.reshape(self.environment.size).T, origin='lower', **kwargs)

            if cbar:
                cbar = plt.colorbar(im, fraction=0.03, pad=0.04, label='Value')

        elif self.environment.kind == 'hex':
            print(ax)
            if ax is None:
                _, ax = plt.subplots(1, figsize=(self.environment.size[1] / 2, self.environment.size[0] / 2), dpi=150)

            coords = draw_hexagons(self.solver.V.reshape(self.environment.size), cmap='viridis', linewidth=0, ax=ax, return_coords=True)

            # Gridlines
            draw_hexagons(np.ones((self.environment.size[0], self.environment.size[1])), ax=ax, facecolor=(0, 0, 0, 0))
        # function_string = ' + '.join([r'{0} \cdot X_{1}'.format(self.predator.function[i], i + 1) for i in range(len(self.predator.function))])

        # ax.set_title(r'$y = {0}$'.format(self.function))

        if filename is not None:
            plt.savefig(filename)

        return ax, coords
        
    def move_to_idx(self, x, y):

        self.move_history.append(self.idx)
        self.idx = (x, y)
        self.environment.feature_arrays[self.name] *= 0
        self.environment.feature_arrays[self.name][y, x] = 1
        self.get_state_rewards()
        self.solved = False


    def plot_policy(self, color='w', head_width=0.25, head_length=0.25, ax=None, cbar=True, pi=None, **kwargs):
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

        if not self.solved and pi is None:
            self.solve()
        ax, coords = self.plot_state_values(ax=ax, cbar=cbar,  **kwargs)
        if pi is None:
            pi = self.solver.pi

        ## Error-catching.
        if isinstance(color, str):
            color = [color] * len(pi)
            
        ## Iteratively plot arrows.
        for i in range(len(pi)-1):

            ## Identify S, S' coordinates.
            if self.environment.kind == 'square':
                y1, x1 = np.where(self.environment.grid==pi[i])
                y2, x2 = np.where(self.environment.grid==pi[i+1])

                ## Define arrow coordinates.
                x, y = x1, y1

                dx, dy = 0.5*x2-x1, 0.5*y2-y1

            elif self.environment.kind == 'hex':
                x1_idx, y1_idx = np.where(self.environment.grid==pi[i])
                x1, y1 = coords[:, x1_idx, y1_idx].squeeze()
                x2_idx, y2_idx = np.where(self.environment.grid==pi[i+1])
                x2, y2 = coords[:, x2_idx, y2_idx].squeeze()

                ## Define arrow coordinates.
                x, y = x1, y1

                dx, dy = x2-x1, y2-y1

            
            ## Plot.
            ax.arrow(x, y, dx, dy, color=color[i], head_width=head_width, head_length=head_length)

            
        return ax