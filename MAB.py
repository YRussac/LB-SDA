""" Packages import """
import numpy as np
import arms
from tqdm import tqdm
from math import log
from utils import rd_argmax, get_leader, get_leader_ns, klucbBern
from tracker import SWTracker, DiscountTracker, TrackerEXP3, TrackerDTS, TrackerCUSUM, TrackerMUCB
from tracker import TrackerSWTS, TrackerREXP3, TrackerLM

mapping = {'B': arms.ArmBernoulli, 'beta': arms.ArmBeta, 'F': arms.ArmFinite, 'G': arms.ArmGaussian,
           'Exp': arms.ArmExponential, 'dirac': arms.dirac, 'TG': arms.ArmTG}


def default_exp(x):
    return np.log(x)**0.333
    # or other option return np.sqrt(np.log(x))


def default_memory(r):
    return max(10, np.log(r)**2)


def default_diversity(tau, K):
    return int((K-1)*np.log(tau)**2)


#: Default value for the tolerance for computing numerical approximations of the kl-UCB indexes.
TOLERANCE = 1e-4


class GenericMAB:
    """
    Generic class to simulate a Multi-Arm Bandit problem
    """
    def __init__(self, arms_type_start, param_start, distrib_changes):
        """
        :param arms_type_start: list of the type ['B', 'G', ...], family of each arm
        :param param_start: Parameters of each arm at the initialization step
        :param distrib_changes: dict of the form 'time of change' (int) -> [arms_type, params]
        """
        self.arms_start = arms_type_start
        self.param_start = param_start
        self.MAB = self.generate_arms(arms_type_start, param_start)
        self.nb_arms = len(self.MAB)
        self.distrib_changes = distrib_changes
        self.means = np.array([el.mean for el in self.MAB])
        self.mu_max = np.max(self.means)
        self.mc_regret = None

    def reinit_mab(self):
        self.MAB = self.generate_arms(self.arms_start, self.param_start)
        self.means = np.array([el.mean for el in self.MAB])
        self.mu_max = np.max(self.means)

    def check_restart(self, track):
        if str(int(track.t)) in self.distrib_changes.keys():
            new_mab = self.distrib_changes[str(int(track.t))]
            self.MAB = self.generate_arms(new_mab[0], new_mab[1])
            self.means = np.array([el.mean for el in self.MAB])
            self.mu_max = np.max(self.means)
            track.time_changes.append(int(track.t))
            track.means = np.concatenate([track.means.flatten(),
                                          self.means]).reshape((track.means.flatten().shape[0]//self.nb_arms+1,
                                                                self.nb_arms))

    @staticmethod
    def generate_arms(arms_type, p):
        """
        Method for generating different arms
        :param arms_type: distribution type for the different arms
        :param p: np.array or list, parameters of the probability distribution of each arm
        :return: list of class objects, list of arms
        """
        arms_list = list()
        for i, m in enumerate(arms_type):
            args = [p[i]] + [[np.random.randint(1, 312414)]]
            args = sum(args, []) if type(p[i]) == list else args
            alg = mapping[m]
            arms_list.append(alg(*args))
        return arms_list

    def MC_regret(self, method, N, T, param_dic, store_step=-1):
        """
        Implementation of Monte Carlo method to approximate the expectation of the regret
        :param method: string, method used (UCB, Thomson Sampling, etc..)
        :param N: int, number of independent Monte Carlo simulation
        :param T: int, time horizon.
        :param store_step: frequency of the steps to store. if -1 the entire trajectory is stored.
        :param param_dic: dict, parameters for the different methods, can be the value of rho for UCB model or an int
        corresponding to the number of rounds of exploration for the ExploreCommit method
        """
        mc_regret = np.zeros(T)
        store = (store_step > 0)
        if store:
            all_regret = np.zeros((np.arange(T)[::store_step].shape[0], N))
        alg = self.__getattribute__(method)
        for i in tqdm(range(N), desc='Computing ' + str(N) + ' simulations'):
            self.reinit_mab()
            tr = alg(T, **param_dic)
            regret = tr.regret()
            mc_regret += regret
            if store:
                all_regret[:, i] = regret[::store_step]
        if store:
            return mc_regret / N, all_regret
        return mc_regret / N

    def EXP3S(self, T, gamma, alpha, store_rewards_arm=True):
        """
        EXP3S algorithm from Auer et al. (2002) The non-stochastic multi-armed bandit problem
        :param T: time horizon.
        :param gamma: exploration parameter.
        Theoretical value min(1, sqrt(nb_arms*(nb_breakpoint*log(nb_arms*T) + exp)/((exp-1)*T))).
        :param alpha: theoretical value 1/T.
        :param store_rewards_arm: Storing the rewards for the different arms.
        :return:
        """
        tr = TrackerEXP3(self.means, T, gamma=gamma, alpha=alpha, store_rewards_arm=store_rewards_arm)
        for t in range(T):
            self.check_restart(tr)
            arm = np.random.choice(self.nb_arms, p=tr.p)
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def REXP3(self, T, gamma, Delta, store_rewards_arm=True):
        """
        REXP3 implementation from Besbes et al., "Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards"
        NIPS 2014.
        :param T: time horizon.
        :param gamma: proportion of exploration, theoretical value min(1, sqrt(nb_arms*log(nb_arms)/((exp-1)*Delta)).
        :param Delta: frequency of the restart, theoretical value int( (nb_arms*log(nb_arms))**(1/3)*(T/V_T)**(2/3))+1.
        :param store_rewards_arm: Storing the rewards for the different arms.
        :return:
        """
        tr = TrackerREXP3(self.means, T, gamma=gamma, store_rewards_arm=store_rewards_arm)
        j = 1
        while j <= int(T/Delta)+1:
            tau = (j-1)*Delta
            tr.restartREXP3()
            for t in range(tau+1, min(T, tau + Delta)):
                self.check_restart(tr)
                arm = np.random.choice(self.nb_arms, p=tr.p)
                reward = self.MAB[arm].sample()[0]
                tr.update(t, arm, reward)
            j += 1
        return tr

    def DTS(self, T, gamma, alpha_0=1, beta_0=1, store_rewards_arm=True):
        """
        Implementation of Discounted Thompson Sampling from Raj et al. 2017,
        "Taming non-stationary bandits: A bayesian approach".
        :param T: T: time horizon.
        :param gamma: discount factor used to compute the posterior
        No theoretical value nor practical one recommended by the author.
        :param alpha_0: Prior distribution is Beta(alpha_0, beta_0).
        :param beta_0: Prior distribution is Beta(alpha_0, beta_0).
        :param store_rewards_arm: Storing the rewards for the different arms.
        :return:
        """
        tr = TrackerDTS(self.means, T, gamma, store_rewards_arm=store_rewards_arm)
        for t in range(T):
            self.check_restart(tr)
            arm = np.argmax(np.random.beta(alpha_0 + tr.S, beta_0 + tr.F))
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def DTS_gaussian(self, T, gamma, mu_0, sigma_0, sigma, store_rewards_arm=True):
        """
        Adaptation of Discounted TS to Gaussian distributions.
        :param T: T: T: time horizon.
        :param gamma: gamma: discount factor used to compute the posterior
        :param mu_0: Prior distribution is a Gaussian distribution with mean mu_0
        :param sigma_0: Prior distribution is a Gaussian distribution with variance sigma_0**2
        :param sigma: Known variance (or upper-bound) for the different arms
        :param store_rewards_arm: Storing the rewards for the different arms.
        :return:
        """
        tr = TrackerDTS(self.means, T, gamma, store_rewards_arm=store_rewards_arm)
        for t in range(T):
            self.check_restart(tr)
            sigma_square = 1/(1/sigma_0**2 + tr.nb_draws/sigma**2)
            mu_ = sigma_square*(mu_0/sigma_0**2 + tr.S/sigma**2)
            arm = np.argmax(np.random.normal(mu_, np.sqrt(sigma_square)))
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def SW_TS(self, T, tau, store_rewards_arm=True):
        """
        Implementation of Sliding-Window Thompson Sampling from Trovo et al. (JAIR 2020)
        "Sliding-window thompson sampling for non-stationary settings".
        :param T: time horizon.
        :param tau: Length of the sliding window used to compute the posterior
        Theoretical value: tau proportional to sqrt(T/nb_breakpoints).
        :param store_rewards_arm: Storing the rewards for the different arms.
        :return:
        """
        tr = TrackerSWTS(self.means, T, tau, store_rewards_arm=store_rewards_arm)
        for t in range(T):
            self.check_restart(tr)
            arm = np.argmax(np.random.beta(1 + tr.S, 1 + tr.F))
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def SW_TS_gaussian(self, T, tau, mu_0, sigma_0, sigma, store_rewards_arm=True):
        """
        Adaptation of SW_TS to Gaussian distributions
        :param T: T: T: time horizon.
        :param tau: Length of the sliding window used to compute the posterior
        :param mu_0: Prior distribution is a Gaussian distribution with mean mu_0
        :param sigma_0: Prior distribution is a Gaussian distribution with variance sigma_0**2
        :param sigma: Known variance (or upper-bound) for the different arms
        :param store_rewards_arm: Storing the rewards for the different arms.
        :return:
        """
        tr = TrackerSWTS(self.means, T, tau, store_rewards_arm=store_rewards_arm)
        for t in range(T):
            self.check_restart(tr)
            sigma_square = 1 / (1 / sigma_0 ** 2 + tr.nb_draws / sigma ** 2)
            mu_ = sigma_square * (mu_0 / sigma_0 ** 2 + tr.S / sigma ** 2)
            arm = np.argmax(np.random.normal(mu_, np.sqrt(sigma_square)))
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def CUSUM(self, T, alpha, h, M, eps, ksi, store_rewards_arm=True):
        """
        Implementation of the CUSUM algorithm from https://arxiv.org/pdf/1711.03539.pdf Liu et al. (AAAI 2018)
        :param T: time horizon.
        :param alpha: Proportion of random draws, if alpha = 1 only random draws, if alpha = 0 only UCB.
        Recommended value alpha = sqrt(nb_breakpoints/T *log(T/nb_breakpoints)).
        :param h: threshold for the random walk for the change point detection?
        Recommended value h = log(T/nb_breakpoints).
        :param M: Assumption 1 in the paper, M is such that at least nb_arms*M time instants between two breakpoints.
        :param eps: Assumption 2 (detectability) in the paper, for all arms the minimal gap when there is a breakpoint
        is larger than 3*eps.
        :param ksi: Bonus term in the confidence bound.
        :param store_rewards_arm: Storing the rewards for the different arms.
        :return:
        """
        tr = TrackerCUSUM(self.means, T, M, eps, h, store_rewards_arm=store_rewards_arm)

        def index_func(x):
            return x.Sa / (x.Na+1e-12) + np.sqrt(ksi*np.log(np.sum(x.Na))/(x.Na+1e-12))

        for t in range(T):
            self.check_restart(tr)
            to_draw = np.where(tr.count > 0)[0]
            if len(to_draw) > 0:
                np.random.shuffle(to_draw)
                arm = to_draw[0]
            else:
                p = np.random.binomial(1, alpha)
                if p == 0:
                    arm = rd_argmax(index_func(tr))
                else:
                    arm = np.random.randint(0, self.nb_arms)
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
            tr.CUSUM(arm, reward)
        return tr

    def M_UCB(self, T, w, b, gamma, store_rewards_arm=True):
        """
        Implementation of the M-UCB algorithm from https://arxiv.org/pdf/1802.03692.pdf Cao et al. (AISTATS 2019)
        :param T: time horizon.
        :param w: Length of the sliding window for the change point detector.
        Recommended value: if delta is the minimum gap (min_max_k(gap_k))
        w = 4/delta**2* (sqrt(log(2*nb_arms*T**2)) + sqrt(log(2*nb_arm) )**2.
        :param b: Threshold for the change point detector
        Recommended value: sqrt(w/2 * log(2*nb_arms*T**2).
        :param gamma: Proportion of uniform sampling
        Recommended value: sqrt((nb_breakpoint-1)*nb_arms*(2*b + 3*sqrt(w)/(2*T)).
        :param store_rewards_arm: Storing the rewards for the different arms.
        :return:
        """
        tau = 0
        tr = TrackerMUCB(self.means, T, store_rewards_arm=store_rewards_arm)

        def index_func(x, t, tau):
            return x.Sa / (x.Na+1e-12) + np.sqrt(2*np.log(t - tau)/(x.Na+1e-12))

        for t in range(T):
            self.check_restart(tr)
            a = (t - tau) % int(self.nb_arms/gamma)
            if a < self.nb_arms:
                arm = a
            else:
                arm = rd_argmax(index_func(tr, t, tau))
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
            if tr.Na[arm] >= w:
                if tr.CD(arm, w, b):
                    tau = t
                    tr.reset_CD()
        return tr

    # INDEX POLICIES
    def Index_Policy(self, T, index_func, tau=None, start_explo=1, store_rewards_arm=True):
        """
        Implementation of UCB1 algorithm.
        :param T: int, time horizon.
        :param index_func: function which computes the index with the tracker.
        :param tau: length of the sliding window if set.
        :param start_explo: number of time to explore each arm before comparing index.
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms.
        """
        if tau is None:
            tau = T
        tr = SWTracker(self.means, T, tau=tau, store_rewards_arm=store_rewards_arm)
        for t in range(T):
            self.check_restart(tr)
            if t < self.nb_arms*start_explo:
                arm = t % self.nb_arms
            else:
                arm = rd_argmax(index_func(tr))
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
            # print('tr.t:', tr.t)
        return tr

    def Index_Policy_Discount(self, T, index_func, gamma, start_explo=1, store_rewards_arm=True):
        """
        Index policy used for the implementation of D-UCB
        """
        tr = DiscountTracker(self.means, T, gamma=gamma, store_rewards_arm=store_rewards_arm)
        for t in range(T):
            self.check_restart(tr)
            if t < self.nb_arms*start_explo:
                arm = t % self.nb_arms
            else:
                arm = rd_argmax(index_func(tr))
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def UCB1(self, T, C=1.):
        """
        Implementation of the classic UCB algorithm.
        :param T: Time Horizon.
        :param C: leverage in the upper bound (tunable param/support).
        :return:
        """
        def index_func(x):
            return x.Sa / (x.Na+1e-12) + C * np.sqrt(max(np.log(x.t), 1)/(x.Na+1e-12))
        return self.Index_Policy(T, index_func)

    def klUCB(self, T, c=1):
        """ The generic KL-UCB policy for one-parameter exponential distributions.
        - By default, it assumes Bernoulli arms.
        - Reference: [Garivier & Cappé - COLT, 2011](https://arxiv.org/pdf/1102.2490.pdf).
        """
        kl_vect = np.vectorize(klucbBern)
        def index_func(x):
            r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:
            .. math::
                \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
                U_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{c \log(t)}{N_k(t)} \right\},\\
                I_k(t) &= U_k(t).
            If rewards are in :math:`[a, b]` (default to :math:`[0, 1]`) and :math:`\mathrm{kl}(x, y)` is the Kullback-Leibler divergence between two distributions of means x and y (see :mod:`Arms.kullback`),
            and c is the parameter (default to 1).
            """
            return kl_vect(x.Sa/x.Na, c * log(x.t) / x.Na)
        return self.Index_Policy(T, index_func)

    def SW_UCB(self, T, tau, C=1.):
        """
        Implementation of Sliding Window UCB (SW-UCB) from Garivier et al. (ALT 2011)
        :param T: Time Horizon.
        :param tau: length of the sliding window, all information kept when tau = T.
        :param C: leverage in the upper bound (tunable param/support).
        :return:
        """
        def index_func(x):
            return x.Sa / (x.Na + 1e-12) + C * np.sqrt(max(min(np.log(x.t), tau), 1) / (x.Na + 1e-12))
        return self.Index_Policy(T, index_func, tau)

    def SW_klUCB(self, T, tau, c=1):
        """ Adaptation of SW_UCB to use KL divergences with no theoretical guarantees. """
        kl_vect = np.vectorize(klucbBern)
        def index_func(x):
            return kl_vect(x.Sa / x.Na, c * log(x.t) / x.Na)
        return self.Index_Policy(T, index_func, tau)

    def D_UCB(self, T, gamma, B, ksi):
        """
        Implementation of Discount UCB (D-UCB) policy from Garivier et al. (ALT 2011)
        :param T: time horizon.
        :param gamma: discount factor.
        :param B: rewards assumed to lie in [0,B].
        :param ksi: tuning of the exploration parameter.
        :return:
        """
        def index_func(x):
            return x.Sa/(x.Na+1e-12) + 2 * B * np.sqrt(ksi*np.log(np.sum(x.Na))/(x.Na + 1e-12))
        return self.Index_Policy_Discount(T, index_func, gamma)

    def D_klUCB(self, T, gamma, c=1):
        """ Adaptation of D_UCB to use KL divergences with no theoretical guarantees. """
        kl_vect = np.vectorize(klucbBern)

        def index_func(x):
            return kl_vect(x.Sa / x.Na, c * log(x.t) / x.Na)
        return self.Index_Policy_Discount(T, index_func, gamma)

    # SUB-SAMPLING POLICIES
    def LB_SDA_baseline(self, T, tau=None, explo_func=default_exp):
        """
        Implementation of LB-SDA for non-stationary environments.
        When tau = T we recover the LB-SDA from Baudry et al. (NeurIPS 2020).
        Here, the specific rules detailed in the paper Section 4 are not reported as it is the baseline
        without the diversity flag and without the definition of the leader.
        :param T: time horizon.
        :param tau: If tau=None then recover LB-SDA for stationary environments otherwise length of the sliding window.
        :param explo_func: Exploration function used.
        :return:
        """
        if tau is None:
            tau = T
        tr = SWTracker(self.means, T, tau=tau, store_rewards_arm=True)
        r, t, l = 1, 0, -1
        while t < self.nb_arms:
            self.check_restart(tr)
            arm = t
            tr.update(t, arm, self.MAB[arm].sample()[0])
            t += 1
        while t < T:
            l_prev = l
            l = get_leader(tr.Na, tr.Sa, l_prev)
            if tau == T:
                forced_explo = explo_func(r)
            else:
                forced_explo = explo_func(tau)
            t_prev = t
            indic = (tr.Na < tr.Na[l]) * (tr.Na < forced_explo) * 1.
            for j in range(self.nb_arms):
                if indic[j] == 0 and j != l and tr.Na[j] < tr.Na[l]:
                    lead_mean = np.mean(tr.rewards_arm[l][-int(tr.Na[j]):])  # sub-sample of the leader.
                    if tr.Sa[j]/tr.Na[j] >= lead_mean and t < T:
                        indic[j] = 1
            if indic.sum() == 0:
                # the leader is pulled
                self.check_restart(tr)
                tr.update(t, l, self.MAB[l].sample()[0])
                t += 1
            else:
                to_draw = np.where(indic == 1)[0]
                np.random.shuffle(to_draw)
                for i in to_draw:
                    if t < T:
                        self.check_restart(tr)
                        tr.update(t, i, self.MAB[i].sample()[0])
                        t += 1
            r += 1
        return tr

    def LB_SDA_LM(self, T, memory_func=default_memory, explo_func=default_exp):
        """
        Implementation of LB-SDA with Limited Memory.
        :param T: time horizon.
        :param memory_func: Memory function used for limiting the storage.
        :param explo_func: Exploration function used.
        :return:
        """
        tr = TrackerLM(self.means, T, store_rewards_arm=True)
        r, t, l = 1, 0, -1
        while t < self.nb_arms:
            self.check_restart(tr)
            arm = t
            tr.update_bis(t, r, arm, self.MAB[arm].sample()[0], memory_func)
            t += 1
        while t < T:
            l_prev = l
            l = get_leader(tr.Na, tr.Sa, l_prev)
            t_prev, forced_explo = t, explo_func(r)
            indic = (tr.Na < tr.Na[l]) * (tr.Na < forced_explo) * 1.
            for j in range(self.nb_arms):
                if indic[j] == 0 and j != l and tr.Na[j] < tr.Na[l]:
                    n_j = len(tr.rewards_arm[j])
                    lead_mean = np.mean(tr.rewards_arm[l][-n_j:])
                    if np.mean(tr.rewards_arm[j]) >= lead_mean and t < T:
                        indic[j] = 1
            if indic.sum() == 0:
                # the leader is pulled
                self.check_restart(tr)
                tr.update_bis(t, r, l, self.MAB[l].sample()[0], memory_func)
                t += 1
            else:
                to_draw = np.where(indic == 1)[0]
                np.random.shuffle(to_draw)
                for i in to_draw:
                    if t < T:
                        self.check_restart(tr)
                        tr.update_bis(t, r, i, self.MAB[i].sample()[0], memory_func)
                        t += 1
            r += 1
        return tr

    def LB_SDA(self, T, tau=None, explo_func=default_exp, diversity_func=default_diversity):
        """
        Implementation of SW-LB-SDA from the paper.
        Here, the specific rules detailed in the paper Section 4 are added contrarily to LB-SDA_baseline.
        :param T: time horizon.
        :param tau: ength of the sliding window.
        :param explo_func: Exploration function used.
        :param diversity_func: diversity function from the paper.
        :return:
        """
        if tau is None:
            tau = T
        tr = SWTracker(self.means, T, tau=tau, store_rewards_arm=True)
        r, t, l = 1, 0, -1
        indic = np.ones(self.nb_arms)
        last_drawn = np.ones(self.nb_arms)
        change_leader = 0
        while t < self.nb_arms:
            self.check_restart(tr)
            arm = t
            tr.update(t, arm, self.MAB[arm].sample()[0])
            t += 1
        while t < T:
            l_prev = l
            l = get_leader_ns(tr.Na, tr.Sa, l_prev, r, tau, self.nb_arms, indic)
            if l_prev != l:
                change_leader = r
            forced_explo = explo_func(tau)
            diversity_val = diversity_func(tau, self.nb_arms)

            # forced exploration part
            indic = (tr.Na < forced_explo) * 1.

            # duels part
            for j in range(self.nb_arms):
                if indic[j] == 0 and j != l:
                    if tr.Na[j] < tr.Na[l]:
                        lead_mean = np.mean(tr.rewards_arm[l][-int(tr.Na[j]):])
                    else:
                        lead_mean = np.mean(tr.rewards_arm[l][-int(tr.Na[l]):])
                    if tr.Sa[j] / tr.Na[j] >= lead_mean and t < T:
                        indic[j] = 1

            # diversity part
            if r-change_leader >= diversity_val:
                # no change of the leader for long enough
                if r-last_drawn[l] >= diversity_val:
                    # and the leader has not been pulled for long enough
                    for j in range(self.nb_arms):
                        if j != l and ((r-last_drawn[j]) >= diversity_val) and (tr.Na[j] <= np.log(tau)**2):
                            indic[j] = 1

            if indic.sum() == 0:
                # the leader is pulled
                self.check_restart(tr)
                tr.update(t, l, self.MAB[l].sample()[0])
                last_drawn[l] = r
                t += 1
            else:
                to_draw = np.where(indic == 1)[0]
                np.random.shuffle(to_draw)
                for i in to_draw:
                    if t < T:
                        self.check_restart(tr)
                        tr.update(t, i, self.MAB[i].sample()[0])
                        last_drawn[i] = r
                        t += 1
            r += 1
        return tr

    def RB_SDA(self, T, tau=None, explo_func=default_exp):
        """
        Implementation of the Random Block Subsampling strategy from Baudry et al. (NeurIPS 2020)
        :param T: time horizon.
        :param tau: length of the sliding window is set otherwise T.
        :param explo_func: Exploration function used.
        :return:
        """
        if tau is None:
            tau = T
        tr = SWTracker(self.means, T, tau, store_rewards_arm=True)
        r, t, l = 1, 0, -1
        while t < self.nb_arms:
            arm = t
            self.check_restart(tr)
            tr.update(t, arm, self.MAB[arm].sample()[0])
            t += 1
        while t < T:
            l_prev = l
            l = get_leader(tr.Na, tr.Sa, l_prev)
            t_prev, forced_explo = t, explo_func(tau)
            indic = (tr.Na < tr.Na[l]) * (tr.Na < forced_explo) * 1.
            for j in range(self.nb_arms):
                if indic[j] == 0 and j != l and tr.Na[j] < tr.Na[l]:
                    tj = np.random.randint(tr.Na[l]-tr.Na[j])
                    lead_mean = np.mean(tr.rewards_arm[l][tj: tj+int(tr.Na[j])])
                    if tr.Sa[j]/tr.Na[j] >= lead_mean and t < T:
                        indic[j] = 1
            if indic.sum() == 0:
                self.check_restart(tr)
                tr.update(t, l, self.MAB[l].sample()[0])
                t += 1
            else:
                to_draw = np.where(indic == 1)[0]
                np.random.shuffle(to_draw)
                for i in to_draw:
                    if t < T:
                        self.check_restart(tr)
                        tr.update(t, i, self.MAB[i].sample()[0])
                        t += 1
            r += 1
        return tr
