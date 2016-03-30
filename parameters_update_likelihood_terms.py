import numpy as np
from scipy import stats


def update_mu(mu_old, alpha_i, nu_old, x_i):
    # explicit casting to np.array to make sure that * is Hadamard product
    return mu_old + alpha_i * (np.array(nu_old) * np.array(x_i))


def update_nu(nu_old, alpha_i, x_i, mu):
    ni_old_had_x_i = np.array(nu_old) * np.array(x_i)
    nominator = alpha_i * (np.dot(x_i.transpose(), mu) + alpha_i)
    denominator = np.dot(x_i.transpose(), ni_old_had_x_i) + 1

    return nu_old - (nominator / denominator) * ni_old_had_x_i * ni_old_had_x_i


def update_v_i_nu_old(nu_new, nu_old):
    # the same claculation is done to calculate v_i and v_old
    # to calculate nu_old the input should be - nu, v_i
    return np.power(1 / nu_new - 1 / nu_old, - 1)


def update_m_i(mu_old, alpha_i, v_i_new, x_i, nu_old):

    v_old_x_i = np.array(nu_old) * np.array(x_i)
    v_i_new_x_i = np.array(v_i_new) * np.array(x_i)

    return mu_old + alpha_i * v_i_new_x_i + alpha_i * v_old_x_i


def update_s_i(z, v_i_new, v_old, m_i_new, mu_old, cdf_factor):
    phi_z = stats.norm.cdf(z / cdf_factor)

    first_term = np.prod(np.sqrt((v_i_new + v_old) / v_i_new))
    second_term = np.exp(np.sum(np.power(m_i_new - mu_old, 2) / (2*(v_old + v_i_new))))

    return phi_z * first_term * second_term


def calc_mu_old(mu, nu_old, v_i, m_i):
    return mu + np.array(nu_old) * (1.0 / np.array(v_i)) * np.array(mu - m_i)


def calc_alpha_i(x_i, nu_old, z, cdf_factor):
    first_term = 1.0 / np.sqrt(np.dot(x_i.transpose(), np.array(nu_old) * np.array(x_i)) + 1)

    return first_term * stats.norm.pdf(z) / stats.norm.cdf(z / cdf_factor)


def calc_z(x_i, mu_old, nu_old):
    nominator = np.dot(x_i.transpose(), mu_old)
    denominator_2 = np.dot(x_i.transpose(), np.array(nu_old) * np.array(x_i)) + 1

    return nominator / np.sqrt(denominator_2)