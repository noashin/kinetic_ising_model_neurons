import numpy as np
from scipy import stats


def kinetic_ising_model(S, J, energy_function):
    """ Returns probabilities of S[t+1,n] being one.

    :param S: numpy.ndarray (T,N)
        Binary data where an entry is either 1 ('spike') or -1 ('silence').
    :param J: numpy.ndarray (N, N)
        Coupling matrix

    :return: numpy.ndarray (T,N)
        Probabilities that at time point t+1 neuron n fires
    """

    # compute fields
    H = compute_fields(S, J)
    # compute probabilities
    p = energy_function(H)
    # return
    return p


def compute_fields(S, J):
    """ Computes the fields for given data and couplings

    :param S: numpy.ndarray (T,N)
        Binary data where an entry is either 1 ('spike') or -1 ('silence').
    :param J: numpy.ndarray (N, N)
        Coupling matrix.

    :return: numpy.ndarray (T,N)
        Fields at time point t+1 on neuron n
    """

    # compute
    H = np.dot(J, S)
    return H


def spike_and_slab(ro, N):
    gamma = stats.bernoulli.rvs(p=ro, size=(N, N))
    normal_dist = np.random.randn(N, N)

    return gamma * normal_dist


def generate_spikes(N, T, S0, J, energy_function):
    """ Generates spike data according to kinetic Ising model

    :param J: numpy.ndarray (N, N)
        Coupling matrix.
    :param T: int
        Length of trajectory that is generated.
    :param S0: numpy.ndarray (N)
        Initial pattern that is sampling started from.

    :return: numpy.ndarray (T, N)
        Binary data where an entry is either 1 ('spike') or -1 ('silence'). First row is only ones for external fields.
    """

    # Initialize array for data
    S = np.empty((T, N))
    # Set initial spike pattern
    S[0] = S0
    # Generate random numbers
    X = np.random.rand(T-1, N)
    #X = np.random.normal(size=(T-1, N))

    # Iterate through all time points
    for i in range(1, T):
        # Compute probabilities of neuron firing
        p = kinetic_ising_model(np.array(S[i - 1, :]), J, energy_function)
        # Check if spike or not
        S[i, :] = 2 * (X[i - 1] < p) - 1
        #S[i, :] = 2*(X[i-1] + np.dot(np.array(S[i-1, :]), J) >= 0) - 1

    return S
