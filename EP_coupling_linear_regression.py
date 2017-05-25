import time
import numpy as np
import multiprocessing as multiprocess
import click
from scipy.stats import logistic as log
from scipy.stats import norm as norm
import scipy.linalg

from spikes_activity_generator import generate_spikes, spike_and_slab
from plotting_saving import save_inference_results_to_file, get_dir_name
import update_params_linear_regression as update_params


def calc_log_evidence(m, v_2, sigma_0, X, y, m_1, v_1, m_2, v, p, p_3, p_2):
    sigma_0_inv = 1. / sigma_0
    V_2 = np.diag(v_2)
    v_2_inv = 1. / v_2
    V_2_inv = np.diag(v_2_inv)
    v_1_inv = 1. / v_1
    v_inv = 1. / v
    m_1_s_v_1 = np.multiply(m_1 ** 2, v_1_inv)
    m_2_s_v_2 = np.multiply(m_2 ** 2, v_2_inv)
    m_s_v = np.multiply(m ** 2, v_inv)

    n, d = X.shape
    alpha = scipy.linalg.det(np.identity(d) + sigma_0_inv * np.dot(V_2, np.dot(X.T, X)))

    cdf_p3 = log.cdf(p_3)
    cdf_m_p3 = log.cdf(-p_3)
    cdf_p2 = log.cdf(p_2)
    cdf_m_p2 = log.cdf(-p_2)

    c = cdf_p3 * norm.pdf(0, m_1, np.sqrt(v_1 + v)) + cdf_m_p3 * norm.pdf(0, m_1, np.sqrt(v_1))
    c[np.where(c == 0)[0]] = 0.0000000001

    log_s1 = 0.5 * (np.dot(m.T, np.dot(V_2_inv, m_2) + sigma_0_inv * np.dot(X.T, y)) -
                    n * np.log(2 * np.pi * sigma_0) - sigma_0_inv * np.dot(y.T, y) -
                    np.dot(m_2.T, np.dot(V_2_inv, m_2)) - np.log(alpha) +
                    np.sum(np.log(1. + np.multiply(v_2, v_1_inv)) + m_1_s_v_1 + m_2_s_v_2 - m_s_v))

    log_s2 = 0.5 * np.sum(2. * np.log(c) + np.log(1. + np.multiply(v_1, v_2_inv)) +
                          m_1_s_v_1 + m_2_s_v_2 - m_s_v + 2. * np.log(log.cdf(p) * cdf_m_p3 + log.cdf(-p) * cdf_p3)
                          - 2. * np.log(cdf_m_p3 * cdf_p3))

    res = log_s1 + log_s2 + 0.5 * d * np.log(2. * np.pi) + \
          0.5 * np.sum(np.log(v) + m_s_v - m_1_s_v_1 - m_2_s_v_2) + \
          np.sum(np.log(np.multiply(cdf_p2, cdf_p3) + np.multiply(cdf_m_p2, cdf_m_p3)))

    if np.isinf(res) or np.isnan(res):
        import ipdb;
        ipdb.set_trace()
    return res


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def inv_logistic(x):
    return np.log(1.0 / (1.0 - x))


def EP(activity, ro, n, v_s, sigma0, bias=0):
    """Performs EP and returns the approximated couplings

    :param activity: Network's activity
    :param ro: sparsity of the couplings
    :param n: index of the neuron being concidered
    :param v_s: the variance of the "slab" part of the network
    :param sigma0: assumed variance of the likelihood
    :return: The approximated couplings for neuron n
    """
    #import ipdb; ipdb.set_trace()
    N = activity.shape[1] #+ bias

    # Initivalization
    v_s = v_s * np.ones(N)

    if isinstance(ro, np.ndarray):
        p_3 = ro
    else:
        p_3 = update_params.update_p_3(ro, N)

    if np.isinf(p_3).any():
        import ipdb;
        ipdb.set_trace()

    p_2 = m_1 = m_2 = np.zeros(N)
    v_1 = np.inf * np.ones(N)

    # if ro is a vector this will fail in Python 3
    v_2 = ro * v_s

    # pre calculate constant matrices and vectors
    X = activity[:-1, :]
    y = activity[1:, n]

    XT_X = np.dot(X.T, X)
    XT_y = np.dot(X.T, y)

    itr = 0
    max_itr = 100000
    convergence = False

    # Repeat the updates rules until convergence
    while not convergence and itr < max_itr:
        m_1_old = m_1
        v_1_old = v_1
        V_2 = np.diag(v_2)
        V = update_params.calc_V(V_2, sigma0, XT_X)
        m = update_params.calc_m(V, V_2, m_2, sigma0, XT_y)
        v = np.diag(V)

        v_1 = update_params.update_v_1(v_2, V)
        m_1 = update_params.update_m_1(m, v, m_2, v_2, v_1)

        p_2 = update_params.update_p_2(v_1, v_s, m_1)
        a = update_params.calc_a(p_2, p_3, m_1, v_1, v_s)
        b = update_params.calc_b(p_2, p_3, m_1, v_1, v_s)

        v_2 = update_params.update_v_2(a, b, v_1)
        v_2[v_2 < 0] = 1e17
        m_2 = update_params.update_m_2(m_1, a, v_2, v_1)

        p = p_2 + p_3
        maxdiff = np.max([np.max(np.abs(m_1 / m_1_old)), np.max(np.abs(v_1 / v_1_old))])
        mindiff = np.min([np.min(np.abs(m_1 / m_1_old)), np.min(np.abs(v_1 / v_1_old))])
        convergence = maxdiff < 1.00001 and mindiff > 0.9999
        itr += 1
    log_evidence = calc_log_evidence(m, v_2, sigma0, X, y, m_1, v_1, m_2, v, p, p_3, p_2)
    # import ipdb; ipdb.set_trace()
    return {'mu': m_1, 'log_evidence': log_evidence, 'gamma': 1. / (1. + np.exp(-p))}


def EP_single_neuron(activity, ro, ns, v_s, sigma0, bias):
    results = [EP(activity, ro, n, v_s, sigma0, bias) for n in ns]
    return results


def multi_process_EP(args):
    return EP_single_neuron(*args)


def do_multiprocess(function, function_args, num_processes):
    """ processes the_args
        :param function:
        :param function_args:
        :param num_processes: how many pararell processes we want to run.
    """
    if num_processes > 1:
        pool = multiprocess.Pool(processes=num_processes)
        results_list = pool.map(function, function_args)
    else:
        results_list = [function(some_args) for some_args in function_args]
    return results_list


def generate_J_S(likelihood_function, bias, N, T, sparsity, v_s):
    """This function generates a network (couplings) and it's activity
    according to the input variables

    :param likelihood_function: function for the model's likelihood
    :param bias: 0 if there is no bias in the model, 1 if there is
    :param N: number of neurons in the network
    :param T: number of time steps for which the network is simulated
    :param sparsity: sparsity of the coupling's prior
    :param v_s: variance of the "slab" part in the coupling's prior
    :return:
    """
    if bias != 0 and bias != 1:
        raise ValueError('bias should be either 1 or 0')

    # Add a column for bias if it is part of the model
    J = spike_and_slab(sparsity, N, bias, v_s)
    J += 0.0  # to avoid negative zeros
    S0 = - np.ones(N + bias)

    if likelihood_function != 'gaussian' and likelihood_function != 'exp_cosh' and likelihood_function != 'logistic':
        raise ValueError('Unknown likelihood function')

    S = generate_spikes(N, T, S0, J, likelihood_function, bias)

    return S, J


def do_inference(S, J, N, num_processes, sparsity, v_s, sigma0, return_gamma=False, bias=0):
    # infere coupling from S
    J_est_EP = np.empty(J.shape)
    log_evidence = 0.0

    # prepare inputs for multi processing
    args_multi = []
    indices = range(N)
    inputs = [indices[i:i + N / num_processes] for i in range(0, len(indices), N / num_processes)]
    for input_indices in inputs:
        args_multi.append((S, sparsity, input_indices, v_s, sigma0, bias))

    results = do_multiprocess(multi_process_EP, args_multi, num_processes)

    for i, indices in enumerate(inputs):
        mus = [results[i][j]['mu'] for j in range(len(results[i]))]
        gammas = [results[i][j]['gamma'] for j in range(len(results[i]))]
        J_est_EP[:, indices] = np.array(mus).transpose()

        evidences_tmp = [results[i][j]['log_evidence'] for j in range(len(results[i]))]
        log_evidence += np.sum(np.array(evidences_tmp), axis=0)

    if return_gamma:
        return J_est_EP, log_evidence, gammas
    else:
        return J_est_EP, log_evidence

def EM(S, J, N, num_processes, init_ro, v_s, sigma0, bias):
    diff = 0
    ro = init_ro
    while diff < 0.9999999 or diff > 1.000001:
        J_est_EP, log_evidence, gammas = do_inference(S, J, N, num_processes, ro, v_s, sigma0, True)
        if bias:
            new_ro = np.mean(np.array(gammas)[:-1, :])
            diff = new_ro / ro[0]
            ro = np.hstack([np.repeat(new_ro, N), 1])
        else:
            new_ro = np.mean(gammas)
            diff = new_ro / ro[0]
            ro = new_ro

        print ro
    print ro
    return J_est_EP, log_evidence, ro


@click.command()
@click.option('--num_neurons',
              help='number of neurons in the network. '
                   'If a list, the inference will be done for every number of neurons.')
@click.option('--time_steps',
              help='Number of time stamps. Length of recording. '
                   'If a list, the inference will be done for every number of time steps.')
@click.option('--num_processes', type=click.INT,
              default=1)
@click.option('--likelihood_function', type=click.STRING,
              default='exp_cosh',
              help='Should be either probit or logistic')
@click.option('--sparsity', type=click.FLOAT,
              default=0.3,
              help='Set sparsity of connectivity, aka ro parameter.')
@click.option('--num_trials', type=click.INT,
              default=1,
              help='number of trials with different S ad J for given settings')
@click.option('--pprior',
              help='Set pprior for the EP. If a list the inference will be done for every pprior')
@click.option('--em',
              default=False,
              help='If True performs EM to find the most likely sparsity.')
@click.option('--bias',
              default=False,
              type=click.BOOL,
              help='Whether to include bias/external fields in the model')
def main(num_neurons, time_steps, num_processes, likelihood_function, sparsity, num_trials, pprior, em, bias):
    sigma0 = 1.0
    ppriors = [float(num) for num in pprior.split(',')]
    num_neurons = [int(num) for num in num_neurons.split(',')]
    time_steps = [int(num) for num in time_steps.split(',')]
    bias = int(bias)

    return_gamma = False
    if em:
        # import ipdb; ipdb.set_trace()
        pprior = ppriors[0]

        N = num_neurons[0]
        v_s = 1. / np.sqrt(N)
        T = time_steps[0]

        if bias:
            pprior = np.hstack([np.repeat(pprior, N), 1])

        dir_name = get_dir_name(ppriors, N, T, sparsity, likelihood_function, approx='gaussian')
        S, J = generate_J_S(likelihood_function, bias, N, T, sparsity, v_s)
        results = EM(S, J, N, num_processes, pprior, v_s, sigma0, bias)
        save_inference_results_to_file(dir_name, S, J, 0, [results[0]], likelihood_function,
                                       ppriors, results[1], [], 0)

        return

    t_stamp = time.strftime("%Y%m%d_%H%M%S") + '_'
    # sparsity_t = 1. / (1. + np.exp(-sparsity))
    for i in range(num_trials):
        for N in num_neurons:
            v_s = 1.0 / np.sqrt(N)
            for T in time_steps:
                dir_name = get_dir_name(ppriors, N, T, sparsity, likelihood_function, approx='gaussian',
                                        t_stamp=t_stamp)
                S, J = generate_J_S(likelihood_function, bias, N, T, sparsity, v_s)
                J_est_EPs = []
                log_evidences = []
                for pprior in ppriors:
                    if bias:
                        pprior = np.hstack([np.repeat(pprior, N), 1])
                    # pprior = 1. / (1. + np.exp(-pprior))
                    # print pprior
                    results = do_inference(S, J, N, num_processes, pprior, v_s, sigma0, return_gamma, bias)
                    J_est_EPs.append(results[0])
                    log_evidences.append(results[1])
                save_inference_results_to_file(dir_name, S, J, 0, [J_est_EPs], likelihood_function,
                                               ppriors, log_evidences, [], i)


if __name__ == "__main__":
    main()
