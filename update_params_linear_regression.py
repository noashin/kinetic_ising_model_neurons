import numpy as np
from scipy.special import logit
from scipy.stats import logistic


def update_p_3(p0, N):
    return np.repeat(logit(p0), N)


def update_v_2(a, b, v_1):
    return (a ** 2 - b) ** (-1) - v_1


def update_m_2(m_1, a, v_2_new, v_1):
    return m_1 - np.multiply(a, v_2_new + v_1)


def update_p_2(v_1, v_s, m_1):
    #import ipdb; ipdb.set_trace()
    tmp_1 = 0.5 * np.log(v_1)
    tmp_2 = 0.5 * np.log(v_1 + v_s)
    tmp_3 = 0.5 * m_1 ** 2 * (1.0 / v_1 - 1.0 / (v_1 + v_s))

    if tmp_1[tmp_1 == np.inf].any() and tmp_2[tmp_2 == np.inf].any():
        return np.zeros(v_1.shape[0])

    return tmp_1 - tmp_2 + tmp_3


def calc_a(p_2_new, p_3, m_1, v_1, v_s):
    #import ipdb; ipdb.set_trace()
    tmp_1 = logistic.cdf(p_2_new + p_3) * m_1 / (v_1 + v_s)
    tmp_2 = logistic.cdf(-p_2_new - p_3) * m_1 / v_1

    return tmp_1 + tmp_2


def calc_b(p_2_new, p_3, m_1, v_1, v_s):
    import ipdb; ipdb.set_trace()
    tmp_1 = logistic.cdf(p_2_new + p_3) * (m_1 ** 2 - v_1 - v_s) / (v_1 + v_s) ** 2
    tmp_2 = logistic.cdf(- p_2_new - p_3) * (m_1 ** 2 * np.sqrt(v_1) - 1.0 / v_1)

    return tmp_1 + tmp_2


def calc_V(V_2, sigma0, XT_X):
    V_2_inv = np.linalg.inv(V_2)
    tmp_mat = V_2_inv + sigma0 ** (-2) * XT_X

    return np.linalg.pinv(tmp_mat)


def calc_m(V, V_2, m_2, sigma0, X, y):
    l_mat = np.dot(np.linalg.inv(V_2), m_2)
    r_mat = sigma0 ** (-2) * np.dot(X.transpose(), y)

    return np.dot(V, l_mat + r_mat)


def update_v_1(v_2, V):
    v_new = np.diag(V)

    return 1.0 / (1.0 / v_new - 1.0 / v_2)


def update_m_1(V, v_2, m_2, sigma0, X, y, v_1_new):
    m_new = calc_m(V, np.diag(v_2), m_2, sigma0, X, y)
    v_new = np.diag(V)

    m_v = np.multiply(m_new, 1.0 / v_new)
    m_v_2 = np.multiply(m_2, 1.0 / v_2)

    return np.multiply(m_v - m_v_2, v_1_new)

