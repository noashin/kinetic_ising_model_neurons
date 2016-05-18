import numpy as np
from scipy.special import logit


def update_p_3(p0, N):
    return np.repeat(logit(p0), N)


def update_v_2(a, b, v_1):
    return (a ** 2 - b) ** (-1) - v_1


def update_m_2(m_1, a, v_2_new, v_1):
    return m_1 - np.multiply(a, v_2_new + v_1)


def update_p_2(v_1, v_s, m_1):
    tmp_1 = 0.5 * np.log(v_1)
    tmp_2 = 0.5 * np.log(v_1 + v_s)
    tmp_3 = 0.5 * m_1 ** 2 * (1.0 / v_1 - 1.0 / (v_1 + v_s))

    return tmp_1 - tmp_2 + tmp_3


def calc_a(p_2_new, p_3, m_1, v_1, v_s):
    tmp_1 = logit(p_2_new + p_3) * m_1 / (v_1 + v_s)
    tmp_2 = logit(-p_2_new - p_3) * m_1 / v_1

    return tmp_1 + tmp_2


def calc_b(p_2_new, p_3, m_1, v_1, v_s):
    tmp_1 = logit(p_2_new, p_3)* (m_1 ** 2 - v_1 - v_s) / (v_1 + v_s) ** 2
    tmp_2 = logit(-p_2_new - p_3) * (m_1 ** 2 * np.sqrt(v_1) - 1.0 / v_1)

    return tmp_1 + tmp_2