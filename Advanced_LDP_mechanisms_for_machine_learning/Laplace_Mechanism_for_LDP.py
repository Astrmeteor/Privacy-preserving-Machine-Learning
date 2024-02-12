import math

import numpy as np


# The Laplace mechanism for LDP
# t_i is u_i's data record, u_i is participant
def getNoisyAns_Lap(t_i,  epsilon):
    loc = 0
    d = t_i.shape[0]
    scale = 2 * d / epsilon
    s = np.random.laplace(loc, scale, t_i.shape)
    t_star = t_i + s
    return t_star


# Duchi's mechanism for LDP in one-dimensional
def Duchi_1d(t_i, eps, t_star):
    p = (math.exp(eps) - 1) / (2 * math.exp(eps) + 2) * t_i + 0.5
    coin = np.random.binomial(1, p)
    if coin == 1:
        return t_star[1]
    else:
        return t_star[0]


# Duchi's mechanism for LDP in multi-dimensional
def Duchi_md(t_i, eps):
    d = len(t_i)
    if d % 2 != 0:
        # math.comb(n, k) "n choose k"
        C_d = pow(2, d - 1) / math.comb(d - 1, (d - 1) / 2)
    else:
        C_d = (pow(2, d - 1) + 0.5 * math.comb(d, d / 2)) / math.comb(d - 1, d / 2)

    B = C_d * (math.exp(eps) + 1) / (math.exp(eps) - 1)
    v = []
    for tmp in t_i:
        tmp_p = 0.5 + 0.5 * tmp
        tmp_q = 0.5 - 0.5 * tmp
        v.append(np.random.choice([1, -1], p=[tmp_p, tmp_q]))

    bernoulli_p = math.exp(eps) / (math.exp(eps) + 1)
    coin = np.random.binomial(1, bernoulli_p)

    t_star = np.random.choice([-B, B], len(t_i), p=[0.5, 0.5])
    # np.multiply "element-wise multiplication"
    v_times_t_star = np.multiply(v, t_star)
    sum_v_times_t_star = np.sum(v_times_t_star)

    if coin == 1:
        while sum_v_times_t_star <= 0:
            t_star = np.random.choice([-B, B], len(t_i), p=[0.5, 0.5])
            v_times_t_star = np.multiply(v, t_star)
            sum_v_times_t_star = np.sum(v_times_t_star)
    else:
        while sum_v_times_t_star > 0:
            t_star = np.random.choice([-B, B], len(t_i), p=[0.5, 0.5])
            v_times_t_star = np.multiply(v, t_star)
            sum_v_times_t_star = np.sum(v_times_t_star)

    return t_star.reshape[-1]


# Piecewise mechanism for one-dimensional
def PM_1d(t_i, eps):
    C = (math.exp(eps / 2) + 1) / (math.exp(eps / 2) - 1)
    l_t_i = (C + 1) * t_i / 2 - (C - 1) / 2
    r_t_i = l_t_i + C - 1

    x = np.random.uniform(0, 1)
    threshold = math.exp(eps / 2) / (math.exp(eps / 2) + 1)
    if x < threshold:
        t_star = np.random.uniform(l_t_i, r_t_i)
    else:
        tmp_l = np.random.uniform(-C, l_t_i)
        tmp_r = np.random.uniform(r_t_i, C)
        w = np.random.randint(2)
        t_star = (1 - w) * tmp_l + w * tmp_r

    return t_star


# Piecewise mechanism for multi-dimensional
def PM_md(t_i, eps):
    d = len(t_i)
    k = max(1, min(d, int(eps, 2.5)))
    rand_features = np.random.randint(0, d, size=k)
    res = np.zeros(t_i.shape)
    for j in rand_features:
        res[j] = (d * 1.0 / k) * PM_1d(t_i[j], eps / k)
    return res