import numpy as np
import scipy.stats.linregress


def r_square(J, J_est):
    J = J.flatten()
    J_est = J_est.flatten()
    J_mean = J.mean()

    ss_res = np.sum((J - J_est) ** 2)
    ss_tot = np.sum((J - J_mean) ** 2)

    return 1 - ss_res / ss_tot