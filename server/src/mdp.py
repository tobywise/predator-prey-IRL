
"""
Base MDP class
Base agent class
Hexgrid to make it easy to figure out state relationships etc

Solver class? >> policy for MDP
Value iteraction class extends solver?

IRL class? >> reward function for agent

Environment class extends MDP class
Plotting (e.g. hex structure) comes from environment class

Plotting as a mixin? Base MDP class shouldn't assume a grid so plotting doesn't make sense

* Different conditions - threat close versus

Seymour paper with Ray

Jumping spiders - how many moves will it make?

Record mouse movements - i.e. moving mouse between two cells


"""
import numpy as np
from fastprogress import progress_bar
from plotting import HexPlottingMixin
from solvers import ValueIteration
import warnings
import json

def offset_distance(x1,y1,x2,y2):
    ac = offset_to_cube(x1,y1)
    bc = offset_to_cube(x2,y2)
    f = cube_distance(ac, bc)
    return f

def cube_distance(a, b):
    return (np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]) + np.abs(a[2] - b[2])) / 2

def hex_dist(XA, XB):

    dist = np.zeros((len(XA), len(XB)))

    for n, i in enumerate(XA):
        for nn, j in enumerate(XB):
            dist[n, nn] = offset_distance(i[0], i[1], j[0], j[1])

    return dist

def offset_to_cube(col, row):
    x = col
    z = row - (col + (col & 1)) / 2
    y = -x-z
    f = np.array([x,y,z])
    return f

def get_action_from_states_hex(A, B, T, coords):
    """
    Takes a pair of adjacent states on a hex grid and gives the action needed to move from A to B.

    Assumes a deterministic MDP where each state leads to another through a single action.

    Args:
        A (int): State A ID
        B (int): State B ID
        T (np.ndarray): Adjacency matrix
        coords ([type]): [description]

    Returns:
        int: Action that moves from A to B
    """

    if not T[A, B]:
        raise ValueError("A and B are not adjacent")

    x1, y1 = coords[A, :]
    x2, y2 = coords[B, :]

    dx, dy = x2-x1, y2-y1

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

def hex_adjacency(grid_coords):
    """
    Generates an adjacency matrix from an array of state coordinates. Grid must represent a hexagonal grid using
    "odd-q" coordinates

    Args:
        grid_coords (np.ndarray): State coordinate indices, shape (n_states, 2)

    Returns:
        np.ndarray: Adjacency matrix, shape (n_states, n_states)
    """
    
    grid_idx = np.array(np.where(~np.isnan(grid_coords))).T
    adjacency = (hex_dist(grid_idx,grid_idx)==1).astype(int)
    
    return adjacency

def grid_coords(state_grid):
    """
    Gets state coordinate (X, Y) indices and state IDs from an NxN grid of states

    Args:
        state_grid (np.ndarray): Grid of states, shape (x, y)

    Returns:
        list: List of state IDs
        np.ndarray: Array of state coordinate indices, shape (n_states, 2)
    """

    grid_idx = np.array(np.where(~np.isnan(state_grid))).T
    grid_state_id = list(range(grid_idx.shape[0]))

    return grid_state_id, grid_idx

def environment_from_dict(env_dict):

    if not env_dict['type'] == 'hex':
        raise NotImplementedError()

    else:
        n_features = len(env_dict['feature_names'])
        features = np.zeros((n_features, np.product(env_dict['size'])))
        for n, f in enumerate(env_dict['feature_names']):
            states = env_dict['features'][f]
            features[n, states] = 1
        newMDP = HexGridMDP(features, env_dict['size'], self_transitions=env_dict['self_transitions'])

        agent_list = []
        for agent, agent_info in env_dict['agents'].items():
            agent_list.append(Agent(agent, agent_info['reward_function'], position=agent_info['position'], solver_kwargs=agent_info['solver_kwargs']))

        newEnvironment = HexEnvironment(newMDP, agent_list)

        return newEnvironment

def environment_from_json(json_string):
    env_dict = json.loads(json_string)

    return environment_from_dict(env_dict)

class MDP():

    def __init__(self, features, sas, adjacency=None, feature_names=None):
        """
        Represents a MDP

        Args:
            adjacency (numpy.ndarray): State adjacency matrix (i.e. one-step transitions), shape (n_states, n_states)
            features (numpy.ndarray): Array of features in each state, shape (n_features, n_states)
            sas (numpy.ndarray): Array representing the probability of transitioning from state S to state S' given action A, of shape (states, actions, states)

        """
       
        # Get SAS
        if  not sas.ndim == 3 or not sas.shape[0] == sas.shape[2]:
            raise TypeError('SAS should be a numpy array of shape (states, actions, states)')

        self.sas = sas
        self.n_states = sas.shape[0]
        self.n_actions = sas.shape[1]

        # Get adjacency
        if adjacency is not None:
            if not isinstance(adjacency, np.ndarray):
                raise TypeError('Adjacency matrix should be a numpy array')
            if not adjacency.ndim == 2:
                raise AttributeError('Adjacency matrix should have two dimensions')
            if not adjacency.shape[0] == adjacency.shape[1]:
                raise AttributeError('Adjacency matrix should be square, found shape {0}'.format(adjacency.shape))
            self.adjacency = adjacency
        else:
            warnings.warn('No adjacency matrix provided, some functions may not work')
            self.adjacency = adjacency

        # Get features
        if not isinstance(features, np.ndarray):
            raise TypeError('Features should be a numpy array of shape (features, states')
        if not features.shape[1] == self.n_states:
            raise AttributeError('Feature array should contain as many states as the adjacency matrix ' +
                                'Expected {0}, found {1}'.format(self.n_states, features.shape[1]))

        self.features = features
        self.n_features = self.features.shape[0]

        # Name features
        if feature_names is None:
            self.feature_names = ['Feature_{0}'.format(i) for i in range(self.n_features)]
        elif isinstance(feature_names, list):
            if len(feature_names) == self.n_features:
                self.feature_names = feature_names
        else:
            raise TypeError('Feature names should be either list or None')

    def _trajectory_to_state_action(self, trajectory):

        """
        Converts a trajectory of state IDs to an array of state-action pairs
        """

        sa_pairs = np.zeros((len(trajectory) - 1, 2))

        for n, state in enumerate(trajectory[:-1]): # Exclude the final state because there is no action taken
            sa_pairs[n, 0] = state
            sa_pairs[n, 1] = np.argmax(self.sas[state, :, trajectory[n+1]])

        return sa_pairs
                    
    

class HexGridMDP(MDP):

    """
    Using offset coordinates ("odd-q")
    """

    def __init__(self, features, size=(10, 15), feature_names=None, self_transitions=True):
        
        self.size = size
        self.grid = np.zeros(self.size)
        self.n_actions = 6
        self.self_transitions = self_transitions
        if self_transitions:
            self.n_actions += 1
        self.n_states = np.product(self.size)

        # Get state IDs and coordinates
        self.state_ids, self.grid_coords = grid_coords(self.grid)

        # Get adjacency matrix
        self.adjacency = hex_adjacency(self.grid)

        # Loop over state pairs to get state-action-state info
        self.sas = np.zeros((self.n_states, self.n_actions, self.n_states))

        for i in range(self.n_states):
            for j in range(self.n_states):
                if self.adjacency[i, j] == 1:
                    action = get_action_from_states_hex(i, j, self.adjacency, self.grid_coords)
                    self.sas[i, action, j] = 1
            if self_transitions:
                self.sas[i, -1, i] = 1

        super().__init__(features, self.sas, self.adjacency, feature_names=feature_names)

    def get_state_coords(self, state):

        assert isinstance(state, int), 'State must be an integer'

        return self.grid_coords[state, :]

    def state_to_idx(self, state):
        return np.unravel_index(state, self.size)

    def idx_to_state(self, idx):
        return np.ravel_multi_index(idx, self.size)


class HexEnvironment(HexPlottingMixin):

    def __init__(self, mdp, agents, name=''):

        self.mdp = mdp
        self.agents = []
        self.name = name

        for a in agents:
            self.attach_agent(a)

    def attach_agent(self, agent):

        agent.mdp = self.mdp
        self.agents.append(agent)

        # Add feature layer representing agent
        agent_feature = np.zeros(self.mdp.n_states)
        agent_feature[agent.position] = 1
        self.mdp.features = np.vstack([self.mdp.features, agent_feature])
        self.mdp.n_features += 1

    def get_agent_state_values(self, agent_id):

        if not self.agents[agent_id].solver.fit_complete:
            raise AttributeError('Agent has not been fit')
        else:
            return self.agents[agent_id].solver.values_

    def get_agent_policy(self, agent_id):

        if not self.agents[agent_id].solver.fit_complete:
            raise AttributeError('Agent has not been fit')
        else:
            return self.agents[agent_id].solver.pi_

    def fit_agent(self, agent_id, **kwargs):

        self.agents[agent_id].fit(**kwargs)

    def move_agent(self, agent_id, position):

        self.agents[agent_id].position = position

        self.mdp.features[self.mdp.n_features - len(self.agents) + agent_id, :] = 0
        self.mdp.features[self.mdp.n_features - len(self.agents) + agent_id, position] = 1

    def to_dict(self, feature_names=None):

        if feature_names is None and self.mdp.feature_names is None:
            raise ValueError("No feature names provided")
        elif feature_names is None and self.mdp.feature_names is not None:
            feature_names = self.mdp.feature_names

        if not (self.mdp.n_features - len(self.agents)) == len(feature_names):
            raise AttributeError('Number of feature names should equal number of features, excluding agents')

        env_dict = dict()

        env_dict['size'] = list(self.mdp.size)
        env_dict['features'] = {}

        for f in range(self.mdp.n_features - len(self.agents)):
            env_dict['features'][feature_names[f]] = np.where(self.mdp.features[f, :])[0].astype(int).tolist()

        env_dict['type'] = 'hex'
        env_dict['self_transitions'] = self.mdp.self_transitions
        env_dict['feature_names'] = feature_names

        env_dict['agents'] = {}

        for agent in self.agents:
            env_dict['agents'][agent.name] = {}
            env_dict['agents'][agent.name]['position'] = agent.position
            env_dict['agents'][agent.name]['reward_function'] = agent.reward_function
            env_dict['agents'][agent.name]['solver_kwargs'] = agent.solver_kwargs

        return env_dict

    def to_json(self, feature_names=None, return_string=False, fname=None):

        json_dict = self.to_dict(feature_names)
    
        if fname is not None:
            with open(fname, 'w') as f:
                json.dump(json_dict, f)

        if return_string:
            json_string = json.dumps(json_dict)
            return json_string


class Agent():

    # Reward function = dict of feature names and weights

    def __init__(self, name, reward_function, position=0, solver=ValueIteration, solver_kwargs={}):

        ## MAKE IT SO THAT MDP ONLY APPEARS ONCE "ATTACHED" TO MDP THROUGH ENVIRONMENT CLASS

        # self.mdp = mdp
        self.reward_function = reward_function
        self.solver_kwargs = solver_kwargs
        self.solver = solver(**solver_kwargs)
        self.position = position
        self.startingPosition = position
        self.name = name

    def generate_trajectory(self, n_steps=5, start_state=None):

        # Assumes deterministic mdp
        if not self.solver.fit_complete:
            raise AttributeError('Solver has not been fit yet')

        if start_state is None:
            start_state = self.position

        trajectory = [start_state]

        for s in range(n_steps):
            trajectory.append(np.argmax(self.mdp.sas[trajectory[-1], self.solver.pi_[trajectory[-1]], :]))

        return trajectory

    def fit(self, **kwargs):

        self.solver.fit(self.mdp, self.reward_function, **kwargs)


