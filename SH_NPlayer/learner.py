import numpy as np
import nashpy
import copy
# Attribution Information: This code was developed off work done by tocom242242
# https://github.com/tocom242242/nash_q_learning

class NashQLearner(object):
    def __init__(self,
                 id=None,
                 epsilon=1.0,
                 gamma=0.99,
                 reward=4.0,
                 income=4.0,
                 moral=0.0,
                 rich=False,
                 strategyType='Learner',
                 ini_state="nonstate"):

        self.id = id
        self.__income = income
        self.__rich = rich
        self.__reward = reward
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = [0, 1, 2, 3, 4]
        self.__hurt = False
        self.__strategyType = strategyType
        self.__moralFactor = moral
        self.__previousContribution = 0
        self.prev_action = 0
        self.prev_state = ini_state
        self.state = ini_state
        self.reward_history = []
        self.action_history = []
        self.previous_reward = 0
        self.q, self.q_o = {}, {}
        self.n = {}

        self._check_new_state(ini_state)

    def __str__(self) :
         """ toString() """
         return  str(self.id) + ': (' + str(self.__rich) + ',' \
         + str(self.__reward) + ',' + str(self.__strategyType)  + ')'

    def get_rich(self):
        return self.__rich
    
    def get_id(self):
        return self.id

    def get_hurt(self) :
        return self.__hurt

    def get_income(self):
        return self.__income

    def set_hurt(self, pain) :
        self.__hurt = pain

    def set_previous_Contribution(self, contribution):
        self.__previousContribution = contribution

    def get_previous_Contribution(self):
        return self.__previousContribution

    def set_reward(self, new_reward):

        self.__reward = new_reward

    def get_reward(self):
        return self.__reward

    def get_strategyType(self):
        return self.__strategyType

    def get_moral_factor(self, target):
        if self.__previousContribution != 0:
            contributed = target / self.__previousContribution
        else:
            contributed = 0
        if contributed < 1 and contributed < self.__moralFactor:
            return 1 - self.__moralFactor - contributed
        else:
            return 1

    def set_previous_reward(self, reward):
        self.previous_reward = reward

    def get_previous_reward(self):
        return self.previous_reward

    def update_epsilon(self):
        self.epsilon *= self.epsilon * 0.999
        if self.epsilon < 0.01:
            self.epsilon = 0.01

    def get_pi(self):
        pi, _ = self._compute_pi(self.state)
        return pi

    def act(self, training=True):

        pi, pi_o = self._compute_pi(self.state)
        if training:
            if np.random.uniform() < self.epsilon:
                action_id = np.random.random_integers(0, len(self.actions) - 1)
            else:
                action_id = np.random.choice(np.flatnonzero(pi == pi.max()))
        else:
            action_id = np.random.choice(np.flatnonzero(pi == pi.max()))

        self.prev_action = action_id

        return action_id

    # Observe next state and learn the rewards of the state
    def observe(
            self,
            state="nonstate",
            reward=None,
            reward_o=None,
            opponent_action=0,
            learning=True):
        self.prev_state = copy.deepcopy(self.state)
        self.state = state
        self._check_new_state(state)

        if reward is not None:
            self.reward_history.append(reward)
        if learning:
            self._learn(state, reward, reward_o, opponent_action)
            self.update_epsilon()

    def _learn(self, state, reward, reward_o, opponent_action):
        pi, pi_o = self._compute_pi(state)
        nashq = self._compute_nashq(state, pi, pi_o, self.q)
        nashq_o = self._compute_nashq(state, pi_o, pi, self.q_o)

        self.q[self.prev_state][(self.prev_action, opponent_action)] = self._compute_q(
            state, reward, self.prev_action, opponent_action, self.q, nashq)
        self.q_o[self.prev_state][(opponent_action, self.prev_action)] = self._compute_q(
            state, reward_o, opponent_action, self.prev_action, self.q_o, nashq_o)

    def _compute_q(self, state, reward, own_action, opponent_action, q, nashq):
        q_old = q[self.prev_state][(own_action, opponent_action)]
        self.alpha = 1 / self.n[(state, own_action, opponent_action)]
        if self.alpha < 0.001:
            self.alpha = 0.001
        updated_q = q_old + \
                    (self.alpha * (reward + (self.gamma * nashq) - q_old))

        return updated_q

    # Compute the nash q value for that state
    def _compute_nashq(self, state, pi, pi_o, q):
        """
            compute nash q value
        """
        nashq = 0
        action1 = np.argmax(pi)
        action2 = np.argmax(pi_o)
        nashq += q[state][(action1, action2)]

        return nashq
    # Compute the equilibria value for that state
    def _compute_pi(self, state):
        q_1, q_2 = [], []
        for action1 in self.actions:
            row_q_1, row_q_2 = [], []
            for action2 in self.actions:
                joint_action = (action1, action2)
                row_q_1.append(self.q[state][joint_action])
                row_q_2.append(self.q_o[state][joint_action])

            q_1.append(row_q_1)
            q_2.append(row_q_2)

        game = nashpy.Game(q_1, q_2)
        equilibria = game.support_enumeration()
        pi_list = list(equilibria)

        pi = None
        for _pi in pi_list:
            if _pi[0].shape == (len(self.actions),) and _pi[1].shape == (
                    len(self.actions),):
                if any(
                        np.isnan(
                            _pi[0])) is False and any(
                    np.isnan(
                        _pi[1])) is False:
                    pi = _pi
                    break

        if pi is None:
            pi1 = np.repeat(
                1.0 / len(self.actions), len(self.actions))
            pi2 = np.repeat(
                1.0 / len(self.actions), len(self.actions))

            pi = (pi1, pi2)

        return pi[0], pi[1]
    # If the state is new state, add a new row to the q table
    def _check_new_state(self, state):

        if state not in self.q.keys():
            self.q[state] = {}
            self.q_o[state] = {}
            for action1 in self.actions:
                for action2 in self.actions:
                    self.q[state][(action1, action2)] = 0
                    self.q_o[state][(action1, action2)] = 0
                    self.n[(state, action1, action2)] = 1

    def reset(self, state):
        self.state = state
        self.prev_state = state
        self._check_new_state(state)
        self.reward_history = []
        self.action_history = []