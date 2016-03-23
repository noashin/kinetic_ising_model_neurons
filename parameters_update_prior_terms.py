import numpy as np
from scipy import stats


def update_mu(mu_old, c_1, nu_old):
    return mu_old + np.array(c_1) * np.array(nu_old)


def update_nu(nu_old, c_3):
    return nu_old - np.array(c_3) * np.array(nu_old) ** 2


def update_p_i(p_i_old, G_1, G_0):
    nominator = np.array(p_i_old) * np.array(G_1)
    denominator = np.array(p_i_old) * np.array(G_1) + np.array((1 - p_i_old)) * np.array(G_0)

    return nominator / denominator


def update_v_i(c_3, nu_old):
    return 1 / c_3 - nu_old


def update_m_i(mu_old, c_1, v_i, nu_old):
    return mu_old + np.array(c_1) * np.array(v_i + nu_old)


def update_a(p_new, p_old):
    return p_new / p_old


def update_b(p_new, p_old):
    return (1 - p_new) / (1 - p_old)


def update_s(Z, nu_old, v, c1, c3):
    exponent = 0.5 * np.array(c1) ** 2 / np.array(c3)
    tmp_term  = np.array(nu_old + v) / np.array(v)
    return Z * np.sqrt(tmp_term) * np.exp(exponent)


def calc_nu_old(nu, v):
    return np.power(1 / nu - 1 / v, - 1)


def calc_mu_old(mu, v_old, v_i, m_i):
    return mu + np.array(v_old) * (1.0 / np.array(v_i)) * np.array(mu - m_i)


def calc_Z(p_old, G1, G0):
    return np.array(p_old) * np.array(G1) + (1 - np.array(p_old)) * np.array(G0)


def calc_c1(Z, p_old, G1, mu_old, nu_old, sigma1, G0, sigma0):
    tmp_l = np.array(p_old) * np.array(G1) * np.array(-mu_old) / np.array(nu_old + sigma1 ** 2)
    tmp_r = (1 - np.array(p_old)) * np.array(G0) * np.array(-mu_old) / np.array(nu_old + sigma0 ** 2)

    return (tmp_l + tmp_r) / np.array(Z)


def calc_c2(Z, p_old, G1, mu_old, nu_old, sigma1, G0, sigma0):
    tmp_ll = np.array(mu_old) ** 2 / np.array(nu_old + np.array(sigma1) ** 2) ** 2
    tmp_rl = 1 / np.array(nu_old + np.array(sigma1) ** 2)
    tmp_l = np.array(p_old) * np.array(G1) * (tmp_ll - tmp_rl)

    tmp_lr = np.array(mu_old) ** 2 / np.array(nu_old + np.array(sigma0) ** 2) ** 2
    tmp_rr = 1 / np.array(nu_old + np.array(sigma0) ** 2)
    tmp_r = (1 - np.array(p_old)) * np.array(G0) * (tmp_lr - tmp_rr)

    return 0.5 * (tmp_l + tmp_r) / np.array(Z)


def calc_c3(c1, c2):
    return np.array(c1) ** 2 - 2 * np.array(c2)


def calc_G(mu_old, nu_old, sigma):
    return stats.norm.pdf(0.0, loc = mu_old, scale = np.sqrt(nu_old + sigma ** 2))


def calc_p_old(p, a, b):
    nominator = np.array(p) / np.array(a)
    denominator = np.array(p) / np.array(a) + (1 - np.array(p)) / np.array(b)

    return nominator / denominator
