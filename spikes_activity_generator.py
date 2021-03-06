import numpy as np
from scipy import stats
from scipy.special import expit
from scipy.stats import multivariate_normal


def exp_cosh(H):
    beta = 1.0
    return 0.5 * np.exp(beta * H)/np.cosh(beta * H)


def gaussian(H):
    #import ipdb; ipdb.set_trace()
    a = 1

    cov = np.diag(np.repeat(a, H.shape[1]))
    return np.random.multivariate_normal(H[0], cov)


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
    # If a string was passed as the energy function use the function that is mapped to it
    string_to_func = {'exp_cosh': exp_cosh, 'gaussian': gaussian, 'logistic': expit}
    if energy_function in string_to_func.keys():
        energy_function = string_to_func[energy_function]
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
    H = np.dot(S, J)
    return H


def spike_and_slab(ro, N, bias, v_s=1.0, bias_mean=0):
    ''' This function generate spike and priors

    :param ro: sparsity
    :param N: number of neurons
    :param bias: 1 if bias is included in the model, 0 other wise
    :return:
    '''

    gamma = stats.bernoulli.rvs(p=ro, size=(N + bias, N))
    normal_dist = np.random.normal(0.0, v_s, (N + bias, N))

    if bias:
        gamma[N, :] = 1
        normal_dist[N, :] = np.random.normal(bias_mean, v_s, N)

    return gamma * normal_dist


def generate_spikes(N, T, S0, J, energy_function, bias, no_spike=-1):
    """ Generates spike data according to kinetic Ising model

    :param J: numpy.ndarray (N, N)
        Coupling matrix.
    :param T: int
        Length of trajectory that is generated.
    :param S0: numpy.ndarray (N)
        Initial pattern that is sampling started from.
    :param bias: 1 if bias is included in the model. 0 other wise.
    :param no_spike: what number should represent 'no_spike'. Default is -1.

    :return: numpy.ndarray (T, N)
        Binary data where an entry is either 1 ('spike') or -1 ('silence'). First row is only ones for external fields.
    """

    # Initialize array for data
    S = np.empty([T, N + bias])
    # Set initial spike pattern
    S[0] = S0 if no_spike == -1 else np.zeros(N + bias)
    # Last column in the activity matrix is of the bias and should be 1 at all times
    if bias:
        S[:, N] = 1
    # Generate random numbers
    X = np.random.rand(T - 1, N)
    #X = np.random.normal(size=(T-1, N))

    # Iterate through all time points
    for i in range(1, T):
        # Compute probabilities of neuron firing
        p = kinetic_ising_model(np.array([S[i - 1]]), J, energy_function)
        if energy_function == 'gaussian':
            S[i, :N] = p
        else:
            # Check if spike or not
            if no_spike == -1:
                S[i, :N] = 2 * (X[i - 1] < p) - 1
            else:
                S[i, :N] = 2 * (X[i - 1] < p) / 2.0

    return S
