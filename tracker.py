import numpy as np
import pandas as pd


class Tracker2:
    def __init__(self, means, T, store_rewards_arm=False):
        """
        :param means: means for the different arms.
        :param T: horizon.
        :param store_rewards_arm: storing the rewards for the different arms.
        """
        self.means = means.reshape(1, len(means))
        self.nb_arms = means.shape[0]
        self.T = T
        self.Sa = np.zeros(self.nb_arms)
        self.Na = np.zeros(self.nb_arms)
        self.reward = np.zeros(self.T)
        self.arm_sequence = np.empty(self.T, dtype=int)
        self.t = 0
        self.store_rewards_arm = store_rewards_arm
        if store_rewards_arm:
            self.rewards_arm = [[] for _ in range(self.nb_arms)]

    def reset(self):
        """
        Initialization of quantities of interest used for all methods
        :param T: int, time horizon
        :return: - Sa: np.array, cumulative reward for the different arms
                 - Na: np.array, number of times arm the different arms have been pulled
                 - reward: np.array, rewards
                 - arm_sequence: np.array, arm chosen at each step
        """
        self.Sa = np.zeros(self.nb_arms)
        self.Na = np.zeros(self.nb_arms)
        self.reward = np.zeros(self.T)
        self.arm_sequence = np.zeros(self.T, dtype=int)
        self.rewards_arm = [[]]*self.nb_arms
        if self.store_rewards_arm:
            self.rewards_arm = [[] for _ in range(self.nb_arms)]

    def update(self, t, arm, reward):
        """
        Update all the parameters of interest after choosing a given arm
        :param t: int, current time/round
        :param arm: int, arm chose at this round
        :param Sa:  np.array, cumulative reward array up to time t-1
        :param Na:  np.array, number of times arm has been pulled up to time t-1
        :param reward: np.array, rewards obtained with the policy up to time t-1
        :param arm_sequence: np.array, arm chose at each step up to time t-1
        """
        self.Na[arm] += 1
        self.arm_sequence[t] = arm
        self.reward[t] = reward
        self.Sa[arm] += reward
        self.t = t
        if self.store_rewards_arm:
            self.rewards_arm[arm].append(reward)

    def regret(self):
        """
        Compute the regret on a single trajectory.
        Should be launched after playing T steps.
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        return self.means.max() * np.arange(1, self.T + 1) - np.cumsum(np.array(self.means)[self.arm_sequence])


class TrackerEXP3(Tracker2):
    def __init__(self, means, T, gamma, alpha, store_rewards_arm=False):
        super(TrackerEXP3, self).__init__(means, T, store_rewards_arm)
        self.w = np.ones(self.nb_arms)
        self.p = np.ones(self.nb_arms)*1/self.nb_arms
        self.gamma = gamma
        self.alpha = alpha
        self.time_changes = [0]

    def update(self, t, arm, reward):
        sum_weights = self.w.sum()
        self.p = (1-self.gamma)/sum_weights*self.w + self.gamma/self.nb_arms
        xi_hat = np.zeros(self.nb_arms)
        xi_hat[arm] = self.gamma * reward/(self.p[arm]*self.nb_arms)
        self.w = self.w * np.exp(xi_hat) + np.exp(1)*self.alpha/self.nb_arms * sum_weights
        self.reward[t] = reward
        self.arm_sequence[t] = arm
        self.t = t

    def regret(self):
        res = np.zeros(self.T)
        n = len(self.time_changes)
        i = 0
        max_ = self.means[i].max()
        for t in range(self.T):
            if n-(i+1) > 0 and t >= self.time_changes[i+1]:
                i += 1
                max_ = self.means[i].max()
            res[t] = max_ - self.means[i][self.arm_sequence[t]]
        return np.cumsum(res)


class TrackerREXP3(Tracker2):
    def __init__(self, means, T, gamma, store_rewards_arm=True):
        super(TrackerREXP3, self).__init__(means, T, store_rewards_arm)
        self.w = np.ones(self.nb_arms)
        self.p = np.ones(self.nb_arms)*1/self.nb_arms
        self.gamma = gamma
        self.time_changes = [0]

    def update(self, t, arm, reward):
        sum_weights = self.w.sum()
        self.p = (1-self.gamma)/sum_weights*self.w + self.gamma/self.nb_arms
        xi_hat = np.zeros(self.nb_arms)
        xi_hat[arm] = self.gamma * reward/(self.p[arm]*self.nb_arms)
        self.w = self.w * np.exp(xi_hat)
        self.reward[t] = reward
        self.arm_sequence[t] = arm
        self.t = t

    def restartREXP3(self):
        self.w = np.ones(self.nb_arms)

    def regret(self):
        res = np.zeros(self.T)
        n = len(self.time_changes)
        i = 0
        max_ = self.means[i].max()
        for t in range(self.T):
            if n-(i+1) > 0 and t >= self.time_changes[i+1]:
                i += 1
                max_ = self.means[i].max()
            res[t] = max_ - self.means[i][self.arm_sequence[t]]
        return np.cumsum(res)

    def regret_old(self):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        means = np.array(pd.DataFrame(self.means, self.time_changes).reindex(np.arange(self.T)).fillna(method='ffill'))
        return np.cumsum([means[t].max() - means[t, self.arm_sequence[t]] for t in range(self.T)])


class TrackerDTS(Tracker2):
    def __init__(self, means, T, gamma, store_rewards_arm=False):
        super(TrackerDTS, self).__init__(means, T, store_rewards_arm)
        self.gamma = gamma
        self.time_changes = [0]
        self.S = np.zeros(self.nb_arms)
        self.F = np.zeros(self.nb_arms)
        self.nb_draws = np.zeros(self.nb_arms)
        self.t = 0

    def update(self, t, arm, reward):
        self.S = self.gamma*self.S
        self.F = self.gamma*self.F
        self.nb_draws = self.gamma*self.nb_draws
        self.S[arm] += reward
        self.reward[t] = reward
        self.F[arm] += 1-reward
        self.nb_draws[arm] += 1
        self.arm_sequence[t] = arm
        self.t = t
        if self.store_rewards_arm:
            self.rewards_arm[arm].append(reward)

    def regret(self):
        """
        :return:
        """
        res = np.zeros(self.T)
        n = len(self.time_changes)
        i = 0
        max_ = self.means[i].max()
        for t in range(self.T):
            if n-(i+1) > 0 and t >= self.time_changes[i+1]:
                i += 1
                max_ = self.means[i].max()
            res[t] = max_ - self.means[i][self.arm_sequence[t]]
        return np.cumsum(res)


class TrackerSWTS(Tracker2):
    def __init__(self, means, T, tau, store_rewards_arm=False):
        super(TrackerSWTS, self).__init__(means, T, store_rewards_arm)
        self.tau = tau
        self.time_changes = [0]
        self.S = np.zeros(self.nb_arms)
        self.F = np.zeros(self.nb_arms)
        self.nb_draws = np.zeros(self.nb_arms)
        self.t = 0

    def update(self, t, arm, reward):
        self.S[arm] += reward
        self.F[arm] += 1-reward
        self.nb_draws[arm] += 1
        self.arm_sequence[t] = arm
        self.t = t
        self.reward[t] = reward
        if self.store_rewards_arm:
            self.rewards_arm[arm].append(reward)
        if t >= self.tau:
            first_t = int(t - self.tau)
            first_element = int(self.arm_sequence[first_t])
            self.S[first_element] -= self.reward[first_t]
            self.F[first_element] -= (1-self.reward[first_t])
            self.nb_draws[first_element] -= 1

    def regret(self):
        res = np.zeros(self.T)
        n = len(self.time_changes)
        i = 0
        max_ = self.means[i].max()
        for t in range(self.T):
            if n-(i+1) > 0 and t >= self.time_changes[i+1]:
                i += 1
                max_ = self.means[i].max()
            res[t] = max_ - self.means[i][self.arm_sequence[t]]
        return np.cumsum(res)


class TrackerCUSUM(Tracker2):
    def __init__(self, means, T, M, eps, h, store_rewards_arm=False):
        super(TrackerCUSUM, self).__init__(means, T, store_rewards_arm)
        self.time_changes = [0]
        self.count = M * np.ones(self.nb_arms)
        self.M = M
        self.M_mean = np.zeros(self.nb_arms)
        self.t = 0
        self.g_minus = np.zeros(self.nb_arms)
        self.g_pos = np.zeros(self.nb_arms)
        self.s_minus = np.zeros(self.nb_arms)
        self.s_pos = np.zeros(self.nb_arms)
        self.eps = eps
        self.h = h

    def update(self, t, arm, reward):
        if self.count[arm] == 1:
            self.count[arm] -= 1
            self.M_mean[arm] += reward
            self.M_mean[arm] = self.M_mean[arm]/self.M
        elif self.count[arm] >= 0:
            self.count[arm] -= 1
            if self.count[arm] > 0:
                self.M_mean[arm] += reward
        self.Na[arm] += 1
        self.arm_sequence[t] = arm
        self.reward[t] = reward
        self.Sa[arm] += reward
        self.t = t
        if self.store_rewards_arm:
            self.rewards_arm[arm].append(reward)

    def reset_CD(self, arm):
        self.Sa[arm] = 0
        self.Na[arm] = 0
        self.count[arm] = self.M
        self.M_mean[arm] = 0
        self.g_minus[arm] = 0
        self.g_pos[arm] = 0
        self.s_minus[arm] = 0
        self.s_pos[arm] = 0

    def CUSUM(self, arm, reward):
        if self.count[arm] > -1:
            self.s_pos[arm] = 0
            self.s_minus[arm] = 0
        else:
            self.s_pos[arm] = reward - self.M_mean[arm] - self.eps
            self.s_minus = self.M_mean - reward - self.eps
        self.g_pos[arm] = max(0, self.g_pos[arm] + self.s_pos[arm])
        self.g_minus[arm] = max(0, self.g_minus[arm] + self.s_minus[arm])
        if max(self.g_pos[arm], self.g_minus[arm]) >= self.h:
            self.reset_CD(arm)
            return True
        return False

    def regret(self):
        res = np.zeros(self.T)
        n = len(self.time_changes)
        i = 0
        max_ = self.means[i].max()
        for t in range(self.T):
            if n-(i+1) > 0 and t >= self.time_changes[i+1]:
                i += 1
                max_ = self.means[i].max()
            res[t] = max_ - self.means[i][self.arm_sequence[t]]
        return np.cumsum(res)


class TrackerMUCB(Tracker2):
    def __init__(self, means, T, store_rewards_arm=False):
        super(TrackerMUCB, self).__init__(means, T, store_rewards_arm)
        self.time_changes = [0]
        self.t = 0

    def update(self, t, arm, reward):
        self.Na[arm] += 1
        self.arm_sequence[t] = arm
        self.reward[t] = reward
        self.Sa[arm] += reward
        self.t = t
        if self.store_rewards_arm:
            self.rewards_arm[arm].append(reward)

    def reset_CD(self):
        self.Sa = np.zeros(self.nb_arms)
        self.Na = np.zeros(self.nb_arms)

    def CD(self, arm, w, b):
        m = self.rewards_arm[arm][-int(w):]
        n = len(m)//2
        m1 = m[:n]
        m2 = m[n:]
        if np.abs(sum(m2) - sum(m1)) > b:
            return True
        else:
            return False

    def regret(self):
        res = np.zeros(self.T)
        n = len(self.time_changes)
        i = 0
        max_ = self.means[i].max()
        for t in range(self.T):
            if n-(i+1) > 0 and t >= self.time_changes[i+1]:
                i += 1
                max_ = self.means[i].max()
            res[t] = max_ - self.means[i][self.arm_sequence[t]]
        return np.cumsum(res)


class SWTracker(Tracker2):
    def __init__(self, means, T, tau, store_rewards_arm=False):
        super(SWTracker, self).__init__(means, T, store_rewards_arm)
        self.tau = tau
        self.time_changes = [0]

    def update(self, t, arm, reward):
        """
        Update all the parameters of interest after choosing the correct arm
        :param t: int, current time/round
        :param arm: int, arm chosen at this round
        :param Sa:  np.array, cumulative reward during the last tau times
        :param Na:  np.array, number of times arm during last tau times
        :param reward: np.array, rewards obtained with the policy up to time t-1
        :param arm_sequence: np.array, arm chose at each step up to time t-1
        """
        self.Na[arm] += 1
        self.arm_sequence[t] = arm
        self.reward[t] = reward
        self.Sa[arm] += reward
        self.t = t
        if self.store_rewards_arm:
            self.rewards_arm[arm].append(reward)
        if t >= self.tau:
            first_t = t - self.tau
            first_element = int(self.arm_sequence[first_t])
            self.Na[first_element] -= 1
            self.Sa[first_element] -= self.reward[first_t]

    def regret_old(self):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """

        means = np.array(pd.DataFrame(self.means, self.time_changes).reindex(np.arange(self.T)).fillna(method='ffill'))
        return np.cumsum([means[t].max() - means[t, self.arm_sequence[t]] for t in range(self.T)])

    def regret(self):
        """
        Computing the regret when a discount policy is used.
        :return:
        """
        res = np.zeros(self.T)
        n = len(self.time_changes)
        i = 0
        max_ = self.means[i].max()
        for t in range(self.T):
            if n - (i + 1) > 0:
                if t >= self.time_changes[i + 1]:
                    i += 1
                    max_ = self.means[i].max()
            res[t] = max_ - self.means[i][self.arm_sequence[t]]
        return np.cumsum(res)


class DiscountTracker(Tracker2):
    def __init__(self, means, T, gamma, store_rewards_arm=False):
        super(DiscountTracker, self).__init__(means, T, store_rewards_arm)
        self.gamma = gamma
        self.time_changes = [0]

    def update(self, t, arm, reward):
        """
        Update all the parameters of interest after choosing the correct arm
        :param t: int, current time/round
        :param arm: int, arm chosen at this round
        :param Sa:  np.array, cumulative reward during the last tau times
        :param Na:  np.array, number of times arm during last tau times
        :param reward: np.array, rewards obtained with the policy up to time t-1
        :param arm_sequence: np.array, arm chose at each step up to time t-1
        """
        # Important modification all the arms are discounted with this tracker but the update rule is different
        # for the arm selected.
        self.Na = self.gamma * self.Na
        self.Na[arm] += 1
        self.arm_sequence[t] = arm
        self.reward[t] = reward
        # Updating the Sa for all arms but different update for the arm selected
        self.Sa = self.gamma*self.Sa
        self.Sa[arm] += reward
        self.t = t
        if self.store_rewards_arm:
            self.rewards_arm[arm].append(reward)

    def regret(self):
        """
        Computing the regret when a discount policy is used.
        :return:
        """
        res = np.zeros(self.T)
        n = len(self.time_changes)
        i = 0
        max_ = self.means[i].max()
        for t in range(self.T):
            if n - (i + 1) > 0:
                if t >= self.time_changes[i + 1]:
                    i += 1
                    max_ = self.means[i].max()
            res[t] = max_ - self.means[i][self.arm_sequence[t]]
        return np.cumsum(res)

    def regret_old(self):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        means = np.array(pd.DataFrame(self.means, self.time_changes).reindex(np.arange(self.T)).fillna(method='ffill'))
        return np.cumsum([means[t].max() - means[t, self.arm_sequence[t]] for t in range(self.T)])


class TrackerLM(Tracker2):
    def __init__(self, means, T, store_rewards_arm=True):
        super(TrackerLM, self).__init__(means, T, store_rewards_arm)
        self.time_changes = [0]

    def update(self, t, arm, reward):
        pass

    def update_bis(self, t, r, arm, reward, memory_func):
        """
        Update all the parameters of interest after choosing the correct arm
        :param t: int, current time
        :param r: current round
        :param arm: int, arm chosen at this round
        :param Sa:  np.array, cumulative reward during the last tau times
        :param Na:  np.array, number of times arm during last tau times
        :param reward: np.array, rewards obtained with the policy up to time t-1
        :param arm_sequence: np.array, arm chose at each step up to time t-1
        """
        self.Na[arm] += 1
        self.arm_sequence[t] = arm
        self.reward[t] = reward
        self.Sa[arm] += reward
        self.t = t
        if self.store_rewards_arm:
            if len(self.rewards_arm[arm]) >= memory_func(r):
                self.rewards_arm[arm].pop(0)
            self.rewards_arm[arm].append(reward)

    def regret_old(self):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """

        means = np.array(pd.DataFrame(self.means, self.time_changes).reindex(np.arange(self.T)).fillna(method='ffill'))
        return np.cumsum([means[t].max() - means[t, self.arm_sequence[t]] for t in range(self.T)])

    def regret(self):
        """
        Trying to find an alternative much faster without computing a pandas dataframe
        :return:
        """
        res = np.zeros(self.T)
        n = len(self.time_changes)
        i = 0
        max_ = self.means[i].max()
        for t in range(self.T):
            if n-(i+1) > 0 and t >= self.time_changes[i+1]:
                i += 1
                max_ = self.means[i].max()
            res[t] = max_ - self.means[i][self.arm_sequence[t]]
        return np.cumsum(res)
