import numpy as np 
from fastprogress import progress_bar
import warnings
from scipy.optimize import minimize
from numba import jit

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
def state_visitation_iterator(n_states, n_actions, sas, tol, pi_sa_):

    D_ = np.zeros(n_states)
    delta_arr = np.zeros(2)

    # Checking for convergence
    delta = np.inf
    converged = False

    # Probability of starting in a given state
    p_zero = np.ones(n_states) / n_states

    # Run state visitation algorithm
    while not converged:
        Dprime = p_zero.copy()
        for s in range(n_states):
            for a in range(n_actions):
                for s_prime in range(n_states):
                    Dprime[s_prime] = Dprime[s_prime] + D_[s] * pi_sa_[s, a] * sas[s, a, s_prime]

        delta_arr[0] = delta
        delta_arr[1] = np.median(np.abs(D_ - Dprime))
        # delta = np.min([delta, ])
        delta = np.min(delta_arr)

        D_ = Dprime.copy()
        if delta < tol:
            converged = True
    
    return D_

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
                p_sprime = mdp.sas[s, a, :]  
                # Value of each state given actions
                action_values[a] = np.sum(np.dot(p_sprime, self.reward_ + self.discount*self.values_))

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

    def _solve_value_iteration(self, mdp, reward_function, show_progress=False, method='python'):

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

        # Get policy from values
        self._get_policy(mdp)      
        
    def fit(self, mdp, reward_function, method='python'):

        self._solve_value_iteration(mdp, reward_function, show_progress=True, method=method)
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

    def _get_state_visitation_frequencies(self, mdp):

        self.D_ = state_visitation_iterator(mdp.n_states, mdp.n_actions, mdp.sas, self.tol, self.pi_sa_)

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
    
    def _maxcausalent_innerloop(self, mdp, theta):

        # Solve MDP using value iteration
        self._solve_value_iteration(mdp, theta, method=self.method)
        # Get state visitation frequencies
        self._get_state_visitation_frequencies(mdp)
        # Get feature counts
        deltaF = (mdp.features * self.D_).sum(axis=1).astype(float)
        deltaF /= deltaF.sum()

        return deltaF
            
    def _solve_maxcausalent(self, mdp, trajectories, reset=True, ignore_features=()):
        
        visited_states = np.zeros(mdp.n_states)

        for t in trajectories:
            for s in t:
                visited_states[s] += 1

        true_F = (mdp.features * visited_states).sum(axis=1).astype(float)
        # print(true_F)
        true_F /= np.sum(true_F)

        # Initial guess at reward function
        if reset or self.theta is None:
            self.theta = np.ones(mdp.n_features) * 0.5

        # History of error
        self.error_history = []

        # Max ent loop
        pb = progress_bar(range(self.max_iter_irl))

        for i in pb:
            

            deltaF = self._maxcausalent_innerloop(mdp, self.theta)

            error = true_F - deltaF

            error[ignore_features] = 0

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





        