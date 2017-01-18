import numpy as np


def ms(J, J_est):
    J = J.flatten()
    J_est = J_est.flatten()

    err = np.sum((J - J_est) ** 2)

    return err / J.shape[0]

def r_square(J, J_est):
    J = J.flatten()
    J_est = J_est.flatten()
    J_est_mean = J_est.mean()

    ss_res = np.sum((J - J_est) ** 2)
    ss_tot = np.sum((J_est - J_est_mean) ** 2)

    return 1.0 - ss_res / ss_tot


def corr_coef(J, J_est):
    J = J.flatten()
    J_est = J_est.flatten()

    J_mean = J.mean()
    J_est_mean = J_est.mean()

    nom = np.dot(J - J_mean, J_est - J_est_mean)
    denom = np.sum((J - J_mean) ** 2) * np.sum((J_est - J_est_mean) ** 2)

    return nom / np.sqrt(denom)


def zero_matching(J, J_est):
    J = J.flatten()
    J_est = J_est.flatten()

    zeros_J = (J > -0.05) & (J < 0.05)
    zeros_J_est = (J_est > -0.05) & (J_est < 0.05)

    diff_zero = np.logical_and(np.logical_not(np.logical_and(zeros_J, zeros_J_est)), np.logical_or(zeros_J_est, zeros_J))

    return 1.0 - 0.5 * np.sum(diff_zero) / np.sum(zeros_J)


def sign_matching(J, J_est):
    J = J.flatten()
    J_est = J_est.flatten()

    J_dot_J_est = np.multiply(J, J_est)
    nom = np.sum(J_dot_J_est < 0.0)
    denom = np.sum(J_dot_J_est != 0.0)

    return 1.0 - float(nom) / float(denom)

