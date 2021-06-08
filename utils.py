""" Packages import """
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import bottleneck as bn
import scipy.stats as sc
from math import log, sqrt

eps = 1e-15  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]


@jit(nopython=True)
def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return np.random.choice(indices)


@jit(nopython=True)
def rd_choice(vec, size):
    return np.random.choice(vec, size=size, replace=False)


@jit(nopython=True)
def hypergeom_sample(s1, n1, n2):
    return np.random.hypergeometric(s1, n1 - s1, nsample=n2)


def rollavg_bottlneck(a, n):
    """
    :param a: array
    :param n: window of the rolling average
    :return:
    """
    return bn.move_mean(a, window=n, min_count=n)


@jit
def klBern(x, y):
    # Function extracted from the SMPBandits package from Lillian Besson https://github.com/SMPyBandits/SMPyBandits/
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))


@jit
def klucb(x, d, kl, upperbound, precision=1e-6, lowerbound=float('-inf'), max_iterations=50):
    # Function extracted from the SMPBandits package from Lillian Besson https://github.com/SMPyBandits/SMPyBandits/
    r""" The generic KL-UCB index computation.
    - ``x``: value of the cum reward,
    - ``d``: upper bound on the divergence,
    - ``kl``: the KL divergence to be used (:func:`klBern`, :func:`klGauss`, etc),
    - ``upperbound``, ``lowerbound=float('-inf')``: the known bound of the values ``x``,
    - ``precision=1e-6``: the threshold from where to stop the research,
    - ``max_iterations=50``: max number of iterations of the loop (safer to bound it to reduce time complexity).
    """
    value = max(x, lowerbound)
    u = upperbound
    _count_iteration = 0
    while _count_iteration < max_iterations and u - value > precision:
        _count_iteration += 1
        m = (value + u) * 0.5
        if kl(x, m) > d:
            u = m
        else:
            value = m
    return (value + u) * 0.5


@jit
def klucbBern(x, d, precision=1e-6):
    # Function extracted from the SMPBandits package from Lillian Besson https://github.com/SMPyBandits/SMPyBandits/
    """ KL-UCB index computation for Bernoulli distributions, using :func:`klucb`.
    """
    upperbound = min(1., klucbGauss(x, d, sig2x=0.25, precision=precision))  # variance 1/4 for [0,1] bounded distributions
    return klucb(x, d, klBern, upperbound, precision)


@jit
def klucbGauss(x, d, sig2x=0.25):
    # Function extracted from the SMPBandits package from Lillian Besson https://github.com/SMPyBandits/SMPyBandits/
    """ KL-UCB index computation for Gaussian distributions.
    """
    return x + sqrt(abs(2 * sig2x * d))


@jit(nopython=True)
def get_leader(Na, Sa, l_prev):
    """
    :param Na: np.array, number of pull of the different arms
    :param Sa: np.array, cumulative reward of the different arms
    :param l_prev: previous leader
    :return: the arm that has been pulled the most, in case of equality the arm the has the highest cumulative
    reward among the most pulled arms. If several candidates and the previous leader is among them, return the previous
    leader. Otherwise random choice among the remaining candidates.
    """
    m = np.amax(Na)
    n_argmax = np.nonzero(Na == m)[0]
    if n_argmax.shape[0] == 1:
        l = n_argmax[0]
        return l
    else:
        s_max = Sa[n_argmax].max()
        s_argmax = np.nonzero(Sa[n_argmax] == s_max)[0]
        if np.nonzero(n_argmax[s_argmax] == l_prev)[0].shape[0] > 0:
            return l_prev
    return n_argmax[np.random.choice(s_argmax)]


@jit(nopython=True)
def get_leader_ns(Na, Sa, l_prev, r, tau, K, winners):
    """
    :param Na: np.array, number of pull of the different arms
    :param Sa: np.array, cumulative reward of the different arms
    :param l_prev: previous leader
    :param r: current round
    :param tau: sliding window length
    :param K: number of arms
    :param winners: np.array, contains of 1 at position k if arm k has won its duel against l_prev
    :return: the arm that has been pulled the most, in case of equality the arm the has the highest cumulative
    reward among the most pulled arms. If several candidates and the previous leader is among them, return the previous
    leader. Otherwise random choice among the remaining candidates.
    """
    if Na[l_prev] < min(r, tau) / (2 * K):
        b_r = np.ones(K)
    else:
        b_r = winners * (Na >= min(r, tau) / K)
        b_r[l_prev] = 1
    m = np.amax(b_r * Na)
    n_argmax = np.nonzero((Na * b_r) == m)[0]
    if n_argmax.shape[0] == 1:
        l = n_argmax[0]
        return l
    else:
        s_max = (Sa * b_r)[n_argmax].max()
        s_argmax = np.nonzero((Sa * b_r)[n_argmax] == s_max)[0]
        if np.nonzero(n_argmax[s_argmax] == l_prev)[0].shape[0] > 0:
            return l_prev
    return n_argmax[np.random.choice(s_argmax)]


def get_SSMC_star_min(rewards_l, n_challenger, reshape_size):
    return (np.array(rewards_l)[:n_challenger * reshape_size].reshape(
        (reshape_size, n_challenger))).mean(axis=1).min()


def convert_tg_mean(mu, scale, step=1e-7):
    X = np.arange(0, 1, step)
    return (X * sc.norm.pdf(X, loc=mu, scale=scale)).mean() + 1 - sc.norm.cdf(1, loc=mu, scale=scale)


def traj_arms(param_start, chg_dist, T):
    nb_arms = len(param_start)
    l_index = list(chg_dist.keys())
    mean_arms = [np.zeros(T) for i in range(nb_arms)]
    idx_index = 0
    for t in range(T):
        for arm in range(nb_arms):
            if idx_index < len(l_index):
                if t >= int(l_index[idx_index]):
                    idx_index += 1
            if idx_index == 0:
                if type(param_start[arm]) == list:
                    mean_arms[arm][t] = param_start[arm][0]
                else:
                    mean_arms[arm][t] = param_start[arm]
            else:
                if type(chg_dist[l_index[idx_index - 1]][1][arm]) == list:
                    mean_arms[arm][t] = chg_dist[l_index[idx_index - 1]][1][arm][0]
                else:
                    mean_arms[arm][t] = chg_dist[l_index[idx_index - 1]][1][arm]
    return mean_arms


def plot_mean_arms(mean_arms, color_list, marker_list):
    n = len(mean_arms)
    T = len(mean_arms[0])
    for i in range(n):
        if i == 0:
            plt.plot(mean_arms[i], color=color_list[i], label='Arm ' + str(i + 1))
        else:
            plt.plot(mean_arms[i], color=color_list[i],
                     marker=marker_list[i-1], markersize=8, markevery=T//10, label='Arm ' + str(i + 1))
    plt.legend()
    plt.show()
    return 0
