import numpy as np 
from fastprogress import progress_bar
import warnings
from scipy.optimize import minimize
from numba import jit, njit, prange

@jit(nopython=True)
def state_value_iterator(n_states, values, delta, q_values, reward, discount, sas, n_actions):

    delta_arr = np.zeros(2)

    for s in range(n_states):

        # Current estimate of V(s)
        v = values[s]

        # Values of each action taken from state s
        action_values = np.zeros(n_actions)

        # Loop over actions for this state
        for a in range(n_actions):
            # Probability of transitions given action - allows non-deterministic MDPs
            # print(s, a)
            p_sprime = sas[s, a, :]  
            # Value of each state given actions
            testtt = np.dot(p_sprime, reward + discount*values)
            action_values[a] = np.dot(p_sprime, reward + discount*values)

        q_values[s, :] = action_values

        # Update action values
        values[s] = action_values.max()

        # Update delta
        delta_arr[0] = delta
        delta_arr[1] = np.abs(v - values[s]) 
        delta = np.max(delta_arr)

    return values, delta, q_values

@jit(nopython=True)
def state_visitation_iterator(n_states, n_actions, sas, n_iter, pi_sa_, p_zero):
    # Algorithm 9.3 in Ziebart's thesis
    D_ = np.zeros(n_states)
    delta_arr = np.zeros(2)

    # Checking for convergence
    delta = np.inf
    count = 0
    # Run state visitation algorithm
    while count < n_iter:
        # print(count)
        Dprime = p_zero.copy()
        for s in range(n_states):
            for a in range(n_actions):
                for s_prime in range(n_states):
                    Dprime[s_prime] = Dprime[s_prime] + D_[s] * pi_sa_[s, a] * sas[s, a, s_prime]
        # print(np.sum(Dprime))
        # delta_arr[0] = delta
        # delta_arr[1] = np.max(np.abs(D_ - Dprime))
        # delta = np.min([delta, ])
        # delta = np.min(delta_arr)
        # print(delta)
        count += 1
        D_ = Dprime.copy()
        # if delta < tol:
        #     converged = True
    
    return D_

@njit
def solve_value_iteration(n_states, n_actions, reward_function, features, max_iter, discount, sas, tol):

    # Initialise state values at zero
    values_ = np.zeros(n_states)

    # Q values
    q_values_ = np.zeros((n_states, n_actions))

    # Get state rewards based on the supplied reward function
    reward_ = np.dot(reward_function, features)

    # Until converged   
    for i in range(max_iter):

        # Delta checks for convergence
        delta_ = 0

        values_, delta_, q_values = state_value_iterator(n_states, values_, delta_, q_values_, reward_, discount, sas, n_actions)

        if delta_ < tol:
            break
    
    return values_, q_values_


class ValueIteration():

    def __init__(self, discount=0.9, tol=1e-8, max_iter=500):

        self.discount = discount
        self.tol = tol
        self.max_iter = 500
        self.fit_complete = False
        self.v_history = []
        self._state_value_func = None


    def _get_state_values(self, mdp):

        for s in range(mdp.n_states):

            # Current estimate of V(s)
            v = self.values_[s]

            # Values of each action taken from state s
            action_values = np.zeros(mdp.n_actions)

            # Loop over actions for this state
            for a in range(mdp.n_actions):
                # Probability of transitions given action - allows non-deterministic MDPs
                # p_sprime = mdp.sas[s, a, :]  
                # Value of each state given actions
                # action_values[a] = np.sum(np.dot(p_sprime, self.reward_ + self.discount*self.values_))

                for s2 in range(mdp.n_states):
                    action_values[a] += mdp.sas[s, a, s2] * (self.reward_[s2] + self.discount * self.values_[s2])

            self.q_values_[s, :] = action_values

            # Update action values
            self.values_[s] = action_values.max()

            # Update delta
            self.delta_ = np.max([self.delta_, np.abs(v - self.values_[s])])
        self.v_history.append(self.values_)
        # print(v)
        # print('-----------\n\n')
        # print(self.values_)

    def _get_state_values_numba(self, mdp):
        
        self.values_, self.delta_, self.q_values_ = state_value_iterator(mdp.n_states, self.values_, self.delta_, self.q_values_, 
                                                                        self.reward_, self.discount, mdp.sas, mdp.n_actions)


    def _get_policy(self, mdp):

        self.pi_ = np.zeros(mdp.n_states)
        self.pi_sa_ = np.zeros((mdp.n_states, mdp.n_actions))  # Probability of choosing action A in state S given policy

        for s in range(mdp.n_states):
            # actions, states = np.where(mdp.sas[s, :, :])
            # max_action = actions[np.argmax(self.values_[states])]
            action = np.argmax(self.q_values_[s, :])
            self.pi_[s] = action
            self.pi_sa_[s, action] = 1  # Deterministic, only one action chosen

        self.pi_ = self.pi_.astype(int)

    def _solve_value_iteration(self, mdp, reward_function, show_progress=False, method='numba'):

        if method == 'python':
            # Initialise state values at zero
            self.values_ = np.zeros(mdp.n_states)

            # Q values
            self.q_values_ = np.zeros((mdp.n_states, mdp.n_actions))

            # Get state rewards based on the supplied reward function
            self.reward_ = np.dot(reward_function, mdp.features)

            # Until converged
            if show_progress:
                pb = progress_bar(range(self.max_iter))
            else:
                pb = range(self.max_iter)
            
            for i in pb:

                # Delta checks for convergence
                self.delta_ = 0

                if method == 'python':
                    self._get_state_values(mdp)
                elif method == 'numba':
                    self._get_state_values_numba(mdp)

                if show_progress:
                    pb.comment = '| Delta = {0}'.format(self.delta_)

                if self.delta_ < self.tol:
                    if show_progress:
                        print("Converged after {0} iterations".format(i))
                    break
            
            if not self.delta_ < self.tol:
                warnings.warn('Solver did not converge')
        
        elif method == 'numba':

            self.values_, self.q_values_ = solve_value_iteration(mdp.n_states, mdp.n_actions, np.array(reward_function), mdp.features, self.max_iter, self.discount, mdp.sas, self.tol)

        # Get policy from values
        self._get_policy(mdp)      
        
    def fit(self, mdp, reward_function, method='python', show_progress=False):

        self._solve_value_iteration(mdp, reward_function, show_progress=show_progress, method=method)
        self.fit_complete = True

class MaxCausalEntIRL(ValueIteration):

    # The original maximum causal entropy algorithm uses a "soft" version of value iteration
    # as it incorporates a softmax function 
    # Here we're assuming a max policy which means it's identical to value iteration
    # https://apps.dtic.mil/sti/pdfs/AD1090741.pdf

    def __init__(self, learning_rate=0.3, max_iter_irl=20, method='numba', **kwargs):

        self.learning_rate = learning_rate
        self.max_iter_irl = max_iter_irl
        self.method = method

        self.theta = None

        super().__init__(**kwargs)

    def _get_state_visitation_frequencies(self, mdp, trajectories):

        # Probability of starting in a given state
        p_zero = np.zeros(mdp.n_states)
        for t in trajectories:
            p_zero[t[0]] += 1
        p_zero /= np.sum(p_zero)

        # p_zero = np.ones(mdp.n_states) / mdp.n_states

        self.D_ = state_visitation_iterator(mdp.n_states, mdp.n_actions, mdp.sas, len(trajectories[0]), self.pi_sa_, p_zero)
        # print('D_', self.D_.sum())
        # TODO use algorithm 10.1 to get feature expectations

        # self.D_ = np.zeros(mdp.n_states)

        # # Checking for convergence
        # delta = np.inf
        # converged = False

        # # Probability of starting in a given state
        # p_zero = np.ones(mdp.n_states) / mdp.n_states

        # # Run state visitation algorithm
        # while not converged:
        #     Dprime = p_zero.copy()
        #     for s in range(mdp.n_states):
        #         for a in range(mdp.n_actions):
        #             for s_prime in range(mdp.n_states):
        #                 Dprime[s_prime] = Dprime[s_prime] + self.D_[s] * self.pi_sa_[s, a] * mdp.sas[s, a, s_prime]

        #     delta = np.min([delta, np.median(np.abs(self.D_ - Dprime))])

        #     self.D_ = Dprime.copy()
        #     if delta < self.tol:
        #         converged = True
    
    def _maxcausalent_innerloop(self, mdp, theta, trajectories):

        # Solve MDP using value iteration
        self._solve_value_iteration(mdp, theta, method=self.method)
        # Get state visitation frequencies
        self._get_state_visitation_frequencies(mdp, trajectories)
        # Get feature counts
        deltaF = (mdp.features * self.D_).sum(axis=1).astype(float)
        # deltaF /= deltaF.sum()

        return deltaF
            
    def _solve_maxcausalent(self, mdp, trajectories, reset=True, ignore_features=()):
        
        visited_states = np.zeros(mdp.n_states)

        for t in trajectories:
            for s in t:
                visited_states[s] += 1

        true_F = (mdp.features * visited_states).sum(axis=1).astype(float)
        # print(true_F)
        # true_F /= np.sum(true_F)
        true_F[ignore_features] = 0

        # Initial guess at reward function
        if reset or self.theta is None:
            self.theta = np.ones(mdp.n_features) * 0.5

        # History of error
        self.error_history = []

        # Max ent loop
        pb = progress_bar(range(self.max_iter_irl))

        for i in pb:
            
            
            deltaF = self._maxcausalent_innerloop(mdp, self.theta, trajectories)
            deltaF[ignore_features] = 0
            # print(visited_states, self.D_)
            # print(true_F, deltaF)

            error = true_F - deltaF

            # Increment reward function
            self.theta += self.learning_rate * error

            # Update progress bar
            mean_abs_error = np.mean(np.abs(error))
            pb.comment = '| Error = {0}'.format(mean_abs_error)
            self.error_history.append(mean_abs_error)
            
            if mean_abs_error < self.tol:
                print("Converged after {0} iterations".format(i))
                break

        if not np.mean(error) < self.tol:
            warnings.warn('Solver did not converge')

    def fit(self, mdp, trajectories, reset=True, ignore_features=()):

        self._solve_maxcausalent(mdp, trajectories, reset=reset, ignore_features=ignore_features)

def softmax(v):

    prob = v.exp() / (v.exp() + (1 - v).exp())

    prob


class ActionIRL(ValueIteration):

    # 1. Get value function
    # 2. Calculate likelihood of chosen action(s) based on value function (as suggested by Peter)
    # 3. Maximise this


    def __init__(self, max_iter_irl=20, method='numba', **kwargs):

        self.max_iter_irl = max_iter_irl
        self.method = method

        self.theta = None

        super().__init__(**kwargs)

        
    def get_state_action_error(self, state):
        """Gives the signed TD error associated with taking action A in state X
        Equal to the reward obtained through taking action X in state X (=reward received in state X') 
        plus the value of state X'

        Args:
            state (int): The state moved into (X')
        """

        return softmax(self.reward_[state] + self.values_[state])

    # def action_likelihood(self, mdp, theta, )

    def _solve_actionIRL(self, theta, mdp, trajectories):

        # Solve MDP using value iteration
        self._solve_value_iteration(mdp, theta, method=self.method)

        ll = 0

        for t in trajectories:
            for state in t[1:]:
                ll += self.get_state_action_error(state)

        return -ll

    
    def fit(self, mdp, trajectories, reset=True, ignore_features=()):

        res = minimize(self._solve_actionIRL, np.ones(mdp.n_features), args=(mdp, trajectories))

        self.theta = res.x


class SimpleIRL():

    def _solve_irl(self, theta, phi):
        return -np.dot(theta, phi.T).sum()

    def fit(self, mdp, trajectories):

        feature_array = np.zeros((len(trajectories), mdp.n_features))

        for n, t in enumerate(trajectories):
            feature_array[n, :] = mdp.features[:, t].sum(axis=1)
        
        self.feature_array = feature_array

        res = minimize(self._solve_irl, np.ones(mdp.n_features), feature_array)

        self.theta = res.x

def state_pair_to_action(mdp, s1, s2):

    if np.max(mdp.sas[s1, :, s2]) > 0:
        return np.argmax(mdp.sas[s1, :, s2])
    else:
        raise AttributeError("States are not adjacent")
    

def get_eyeline_features(mdp, current_state, action):

    complete = False
    states = []

    while not complete:
        states.append(current_state)
        sa_next_states = mdp.sas[current_state, action, :]
        if np.max(sa_next_states) > 0:
            current_state = np.argmax(sa_next_states)
        else:
            complete = True
    observed_features = mdp.features[:, states].sum(axis=1)
    return observed_features

def get_trajectory_eyeline_features(mdp, trajectories, normalise=True):

    feature_counts = np.zeros(mdp.n_features)

    for trajectory in trajectories:

        for n, state in enumerate(trajectory[:-1]):
            action = state_pair_to_action(mdp, state, trajectory[n+1])
            print(get_eyeline_features(mdp, state, action))
            feature_counts += get_eyeline_features(mdp, state, action)

    if normalise:
        feature_counts /= feature_counts.sum()

    return feature_counts


class SimpleActionIRL():

    def fit(self, mdp, trajectories):

        observed_feature_counts = get_trajectory_eyeline_features(mdp, trajectories)

        self.theta = observed_feature_counts

@njit
def get_actions_states(sas, current_node):

    # Get available actions from this state and the resulting next states
    actions_states = np.argwhere(sas[current_node, :, :])
    # Get actions
    actions = actions_states[:, 0] 
    # Get resulting states
    states = actions_states[:, 1]

    return actions_states, actions, states

@njit
def get_opponent_next_state(opponent_policy_method, opponent_q_values, agent_state, opponent_state, actions, states):

    if opponent_policy_method is None:
        next_state = np.random.choice(states)

    elif opponent_policy_method == 'solve':
        q_values = np.array([opponent_q_values[agent_state, opponent_state, a] for a in actions])
        next_state = states[np.argmax(q_values)]

    elif opponent_policy_method == 'precalculated':
        q_values = np.array([opponent_q_values[0, opponent_state, a] for a in actions])
        next_state = states[np.argmax(q_values)]

    return next_state

@njit
def mcts_iteration(V, N, rewards, sas, agent_start_node, opponent_start_node, n_steps, C, 
                   agent_moves=1, min_opponent_moves=2, max_opponent_moves=3, opponent_policy_method=None, 
                   caught_cost=-50, opponent_q_values=None, end_when_caught=True):

    current_node = {'agent': agent_start_node, 'opponent': opponent_start_node}
    expand = True  # This determines whether we expand or simulate
    accumulated_reward = 0  # Total reward accumulated across all states
    visited_states = []
    rewards = rewards.copy()

    player = 'agent'
    current_moves = {'agent': 0, 'opponent': 0}
    opponent_moves_this_step = 0

    for step in range(n_steps):

        # If the agent and opponent are in the same state, the agent has been caught and loses
        if current_node['agent'] == current_node['opponent']:
            accumulated_reward += caught_cost
            if end_when_caught:
                break

        # Get reward available in current state if it's the agent's turn - TODO check this is correct
        if player == 'agent':
            accumulated_reward += rewards[current_node['agent'], 0]

            # Remove reward from this state for future moves
            rewards[current_node['agent'], 0] = 0

            # Append to list of visited states 
            visited_states.append(current_node['agent'])

        # Get actions and resulting states from current node
        actions_states, actions, states = get_actions_states(sas, current_node[player])

        # Check whether we need to expand - if we haven't already expanded all possible next nodes
        if expand and np.any(N[states] == 0):

            # Identify states taht haven't been explored yet
            unexplored = states[N[states] == 0]
            # Select one of these at random
            current_node[player] = np.random.choice(unexplored)
        
            # Each step from now on will be a simulation rather than expansion
            expand = False

        # If we've not yet reached a point where we need to expand, pick next state using UCT
        elif expand:
            
            # If it's the agent's turn, use UCB
            if player == 'agent':
                # Calculate UCB (or UCT)
                ucb = (V[states] / (1 + N[states])) + C * np.sqrt((2 * np.log(N[current_node['agent']])) / (1 + N[states]))

                # Pick the node with the highest value
                current_node['agent'] = states[np.argmax(ucb)]

            # Otherwise pick according to predator policy
            else:
                current_node[player] = get_opponent_next_state(opponent_policy_method, opponent_q_values, current_node['agent'], current_node['opponent'], actions, states)


        elif not step == n_steps - 1: # Randomly select follow-up nodes for simulation
            
            # Select random node
            if player == 'agent':
                current_node[player] = np.random.choice(states)

            # Otherwise pick according to predator policy
            else:
                current_node[player] = get_opponent_next_state(opponent_policy_method, opponent_q_values, current_node['agent'], current_node['opponent'], actions, states)

        # Next step should be the other player
        if player == 'agent':
            current_moves['agent'] += 1
            if current_moves['agent'] == agent_moves:
                current_moves['agent'] = 0
                opponent_moves_this_step = np.random.randint(min_opponent_moves, max_opponent_moves)
                player = 'opponent'
        else:
            current_moves['opponent'] += 1
            if current_moves['opponent'] == opponent_moves_this_step:
                current_moves['opponent'] = 0
                player = 'agent'

    return accumulated_reward, visited_states

@njit(parallel=False)
def run_mcts(n_iter, V, N, rewards, sas, agent_start_node, opponent_start_node, 
             n_steps, C, agent_moves=1, min_opponent_moves=2, max_opponent_moves=3, opponent_policy_method=None, caught_cost=-50,
             opponent_q_values=None, end_when_caught=True):

    for i in range(n_iter):
        accumulated_reward, visited_states = mcts_iteration(V, N, rewards, sas, agent_start_node, 
                                                    opponent_start_node, n_steps, 
                                                    C, agent_moves, min_opponent_moves, max_opponent_moves, opponent_policy_method, caught_cost,
                                                    opponent_q_values, end_when_caught)
        # Backpropogate
        for v in visited_states:
            V[v] += accumulated_reward
            N[v] += 1

    return V, N

@njit
def solve_all_value_iteration(sas, predator_reward_function, features, prey_index, max_iter=None, tol=None, discount=None):

    all_q_values = np.zeros((sas.shape[0], sas.shape[0], sas.shape[1]))  # Prey idx X opponent_states X actions

    for prey_state in range(features.shape[1]):

        # Set prey feature according to the current prey location
        features[prey_index, :] = 0
        features[prey_index, prey_state] = 1

        # Do value iteration
        _, all_q_values[prey_state, ...] = solve_value_iteration(sas.shape[0], sas.shape[1], predator_reward_function, features, max_iter, discount, sas, tol)

    return all_q_values


class MCTS():

    # https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
    # https://github.com/jbradberry/mcts/blob/master/mcts/uct.py 

    def __init__(self, mdp, agents):

        self.mdp = mdp
        self.reset()
        
        if len(agents) != 2:
            raise NotImplementedError("This only works for 2 agents (prey and predator")

        self.agents = agents

    def reset(self):

        self.V = np.zeros((self.mdp.n_states))  # Value of each node
        self.N = np.zeros((self.mdp.n_states)) # Times each node visited

    def calculate_rewards(self):

        self.rewards = np.zeros((self.mdp.n_states, 2))

        for n, agent in enumerate(self.agents):
            self.rewards[:, n] = (self.mdp.features * np.array(agent.reward_function)[:, None]).sum(axis=0)
    
    def run_mcts(self, agent_start_node, opponent_start_node, n_steps, C, agent_moves=1, opponent_moves=2):

        accumulated_reward, visited_states = mcts_iteration(self.V, self.N, self.rewards, self.mdp.sas, agent_start_node, 
                                                            opponent_start_node, n_steps, 
                                                            C, agent_moves, opponent_moves)

        # Backpropogate
        self.V[visited_states] += accumulated_reward
        self.N[visited_states] += 1

    def get_action_values(self, start_node):

        actions_states, actions, states = get_actions_states(self.mdp.sas, start_node)

        action_values = self.V[states] / (1 + self.N[states])

        return actions, action_values, states


    def fit(self, agent_start_node, opponent_start_node, n_steps=20, n_iter=100, 
            C=1, agent_moves=1, min_opponent_moves=2, max_opponent_moves=3, opponent_policy_method=None, caught_cost=-50,
            opponent_q_values=None, end_when_caught=True):

        self.reset()  # Clear values

        # Get rewards for each state based on agents' reward functions
        self.calculate_rewards()  

        if opponent_policy_method == 'precalculated':
            opponent_q_values = opponent_q_values[None, :, :]
        # elif opponent_policy_method != 'solve':
        #     opponent_q_values = np.zeros((1, 1, 1))

        self.V, self.N = run_mcts(n_iter, self.V, self.N, self.rewards, self.mdp.sas, agent_start_node, 
                                                            opponent_start_node, n_steps, 
                                                            C, agent_moves, min_opponent_moves, max_opponent_moves, opponent_policy_method, caught_cost,
                                                            opponent_q_values, end_when_caught)
        
        # for i in progress_bar(range(n_iter)):
        #     self.run_mcts(agent_start_node, opponent_start_node, n_steps, 
        #                   C=C, agent_moves=agent_moves, opponent_moves=opponent_moves)

        actions, action_values, states = self.get_action_values(agent_start_node)

        return actions, action_values, states

