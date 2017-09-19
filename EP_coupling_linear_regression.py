import time
import numpy as np
import multiprocessing as multiprocess
import click
from scipy.stats import logistic as log
from scipy.stats import norm as norm
import scipy.linalg
from scipy.optimize import minimize, fsolve, root, brentq, ridder, newton, bisect

from spikes_activity_generator import generate_spikes, spike_and_slab
from plotting_saving import save_inference_results_to_file, get_dir_name
import update_params_linear_regression as update_params


def generate_J_S(likelihood_function, bias, N, T, sparsity, v_s, bias_mean=0):
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
    J = spike_and_slab(sparsity, N, bias, v_s, bias_mean)
    J += 0.0  # to avoid negative zeros
    S0 = - np.ones(N + bias)

    if likelihood_function != 'gaussian' and likelihood_function != 'exp_cosh' and likelihood_function != 'logistic':
        raise ValueError('Unknown likelihood function')

    S = generate_spikes(N, T, S0, J, likelihood_function, bias)

    return S, J


def calculate_C_delta(S):
    m = calculate_m(S)
    delta_s = S - m

    N = S.shape[1]
    T = S.shape[0]

    C = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = np.dot(delta_s[:, i].T, delta_s[:, j]) / T
    return C


def calculate_D_delta(S):
    m = calculate_m(S)
    delta_s = S - m

    N = S.shape[1]
    T = S.shape[0]

    C = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = np.dot(delta_s[1:, i].T, delta_s[:-1, j]) / T
    return C


def calc_log_evidence(m, v_2, sigma_0, X, y, m_1, v_1, m_2, v, p, p_3, p_2, v_s, bias):
    ## TODO: Calculate properly with the bias!!

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
    try:
        alpha = scipy.linalg.det(np.identity(d) + sigma_0_inv * np.dot(V_2, np.dot(X.T, X)))
    except ValueError:
        import ipdb;
        ipdb.set_trace()

    cdf_p3 = log.cdf(p_3)
    cdf_m_p3 = log.cdf(-p_3)
    cdf_p2 = log.cdf(p_2)
    cdf_m_p2 = log.cdf(-p_2)
    # import ipdb; ipdb.set_trace()
    c = cdf_p3 * norm.pdf(0, m_1[:d - bias], np.sqrt(v_1 + v_s)[: d - bias]) + \
        cdf_m_p3 * norm.pdf(0, m_1[: d - bias], np.sqrt(v_1)[: d - bias])
    c[np.where(c == 0)[0]] = 0.0000000001

    log_s1 = 0.5 * (np.dot(m.T, np.dot(V_2_inv, m_2) + sigma_0_inv * np.dot(X.T, y)) -
                    n * np.log(2 * np.pi * sigma_0) - sigma_0_inv * np.dot(y.T, y) -
                    np.dot(m_2.T, np.dot(V_2_inv, m_2)) - np.log(alpha) +
                    np.sum(np.log(1. + np.multiply(v_2, v_1_inv)) + m_1_s_v_1 + m_2_s_v_2 - m_s_v))
    # import ipdb; ipdb.set_trace()
    log_s2 = 0.5 * np.sum(2. * np.log(c) + np.log(1. + np.multiply(v_1, v_2_inv)[: d - bias]) +
                          m_1_s_v_1[: d - bias] + m_2_s_v_2[: d - bias] - m_s_v[: d - bias] +
                          2. * np.log(log.cdf(p) * cdf_m_p3 + log.cdf(-p) * cdf_p3)
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


def EP(activity, ro, n, v_s, sigma0, bias=0, bias_mean=0, bias_constant=0, max_itr=10000, w_i=[]):
    """Performs EP and returns the approximated couplings

    :param activity: Network's activity
    :param ro: sparsity of the couplings
    :param n: index of the neuron being concidered
    :param v_s: the variance of the "slab" part of the network
    :param sigma0: assumed variance of the likelihood
    :return: The approximated couplings for neuron n
    """
    N = activity.shape[1]

    # Initivalization. Make sure it is positive!
    v_s = v_s * np.ones(N)

    p_3 = update_params.update_p_3(ro, N - bias)

    if np.isinf(p_3).any():
        import ipdb;
        ipdb.set_trace()

    m_1 = m_2 = np.zeros(N)
    v_1 = np.inf * np.ones(N)
    v_2 = ro * v_s  # if ro is a vector this will fail in Python 3
    p_2 = np.zeros(N - bias)

    # pre calculate constant matrices and vectors

    if len(w_i):
        X = (activity[:-1, :].T * (2. * np.sqrt(w_i[:-1]))).T
        y = activity[1:, n] / (2 * np.sqrt(w_i[:-1]))
    else:
        X = activity[:-1, :]
        y = activity[1:, n]

    XT_X = np.dot(X.T, X)
    XT_y = np.dot(X.T, y - np.repeat(bias_constant, y.shape))
    itr = 0
    convergence = False
    # Repeat the updates rules until convergence
    while not convergence and itr < max_itr:
        m_1_old = m_1
        v_1_old = v_1
        m_2_old = m_2
        V_2 = np.diag(v_2)
        V = update_params.calc_V(V_2, sigma0, XT_X)
        m = update_params.calc_m(V, V_2, m_2, sigma0, XT_y)
        v = np.diag(V)

        v_1 = update_params.update_v_1(v_2, V)
        m_1 = update_params.update_m_1(m, v, m_2, v_2, v_1)

        p_2 = update_params.update_p_2(v_1[:N - bias], v_s[:N - bias], m_1[:N - bias])
        a = update_params.calc_a(p_2, p_3, m_1[:N - bias], v_1[:N - bias], v_s[:N - bias])
        b = update_params.calc_b(p_2, p_3, m_1[:N - bias], v_1[:N - bias], v_s[:N - bias])

        v_2[:N - bias] = update_params.update_v_2(a, b, v_1[:N - bias])
        v_2[v_2 < 0] = 1e17
        m_2[:N - bias] = update_params.update_m_2(m_1[:N - bias], a, v_2[:N - bias], v_1[:N - bias])

        p = p_2 + p_3

        if bias:
            v_bias = update_params.gaussian_prod_var(v_1[-1], v_s[-1])
            m_bias = update_params.gaussian_prod_mean(m_1[-1], v_1[-1], bias_mean, v_s[-1], v_bias)
            v_2[-1] = update_params.update_v_2_bias(v_bias, v_1[-1])
            m_2[-1] = update_params.update_m_2_bias(v_2[-1], v_bias, m_bias, v_1[-1], m_1[-1])

        maxdiff = np.max([np.max(np.abs(m_1 / m_1_old)), np.max(np.abs(v_1 / v_1_old)), np.max(np.abs(m_2 / m_2_old))])
        mindiff = np.min([np.min(np.abs(m_1 / m_1_old)), np.min(np.abs(v_1 / v_1_old)), np.min(np.abs(m_2 / m_2_old))])
        convergence = maxdiff < 1.00001 and mindiff > 0.9999
        itr += 1

        if np.isnan(p).any():
            import ipdb;
            ipdb.set_trace()
    # print convergence
    v_res = update_params.gaussian_prod_var(v_1, v_2)
    m_res = update_params.gaussian_prod_mean(m_1, v_1, m_2, v_2, v_res)
    log_evidence = calc_log_evidence(m, v_2, sigma0, X, y, m_1, v_1, m_2, v, p, p_3, p_2, v_s, bias)

    return {'mu': m_res, 'log_evidence': log_evidence, 'gamma': 1. / (1. + np.exp(-p))}


def maximize_bias(S, v_s, J, bias_mean, n, bias_var):
    '''T, N = S.shape
    sigma_l = (T / v_s) ** -0.5
    mu_l = (sigma_l ** 2 / v_s) * np.sum(S[1:, n] - np.dot(S[:-1], J))

    return (mu_l * bias_var + bias_mean * sigma_l ** 2) / (sigma_l ** 2 + bias_var)'''

    S_J = np.dot(S[:-1], J)

    def func(b):
        # bias_mean/bias_var - b/bias_var +
        return S[1:, n].sum() - np.tanh(S_J + b).sum()

    def func_dev(b):
        return - 1. / bias_var - (1. / np.cosh(S_J + b) ** 2).sum()

    def func_dev_dev(b):
        return 2 * (np.tanh(S_J + b) / np.cosh(S_J + b) ** 2).sum()

    init_b = bias_mean
    b = root(func, init_b).x
    return b


def get_J_tilde(n, sigma0, v_s, C_inv, D):
    '''X = activity[:-1, :]
    y = activity[1:, n]
    bias_vec = np.repeat(bias_constant, y.shape)

    per_l = np.dot(X.T, X) / sigma0
    # sigma_l = np.linalg.inv(per_l)
    per_m_l = np.dot(X.T, y - bias_vec) / sigma0

    sigma_lp = np.linalg.inv(per_l + 1. / v_s)

    m_lp = np.dot(sigma_lp, per_m_l)'''
    # without prior
    J_n = np.dot(C_inv, D[n].T)
    return J_n


def calculate_m(S):
    return np.mean(S, axis=0)


def calculate_g(m, J):
    return np.dot(J, m)


def calculate_mean_h(m, J, bias):
    g = calculate_g(m, J)
    return g + bias


def calculate_delta(J, C):
    N = J.shape[0]
    delta = np.array([J[k] * np.dot(C[k, :], J) for k in range(N)]).sum()

    return delta


def calculate_a(J, C, m, bias):
    a = -5.0
    b = 5.0
    N = 1000.0
    delta = calculate_delta(J, C)
    mean_h = calculate_mean_h(m, J, bias)
    x = np.linspace(a, b, N)
    fx = np.exp(-x ** 2 / 2) * (1 - np.tanh(x * np.sqrt(delta) + mean_h) ** 2) / np.sqrt(2 * np.pi)

    integral = np.sum(fx) * (b - a) / N
    return np.array(integral)


def EP_bias(S, ro, n, v_s, sigma0, bias, bias_mean, bias_var, C, C_inv, D, m):
    converged = False
    bias_old = bias_mean
    i = 0
    J_tilde = get_J_tilde(n, sigma0, v_s, C_inv, D)
    a_old = 1.
    while not converged:
        a_new = calculate_a(J_tilde, C, m, bias_old)
        J_new = J_tilde / a_new
        bias_new = maximize_bias(S[:, :-1], v_s, J_new, bias_mean, n, bias_var)
        # import ipdb; ipdb.set_trace()
        diff_b = np.abs(bias_new / bias_old)
        diff_a = np.abs(a_old / a_new)
        maxdiff = np.max([diff_b, diff_a])
        mindiff = np.min([diff_b, diff_a])
        # print(bias_new)
        if maxdiff < 1.001 and mindiff > 0.999:
            converged = True

        bias_old = bias_new
        a_old = a_new
        i += 1
    print(i, bias_new, maxdiff, mindiff)
    res = {}
    res['mu'] = np.hstack([J_new, bias_new])
    res['gamma'] = np.repeat(1, J_new.shape)
    res['log_evidence'] = 0
    return res


def EP_single_neuron(activity, ro, ns, v_s, sigma0, bias, bias_mean, bias_var, C, C_inv, D, m):
    if bias:
        results = [EP_bias(activity, ro, n, v_s, sigma0, bias, bias_mean, bias_var, C, C_inv, D, m) for n in ns]
    else:
        results = [EP(activity, ro, n, v_s, sigma0) for n in ns]
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


def do_inference(S, J, N, num_processes, sparsity, v_s, sigma0, return_gamma=False, bias=0, bias_mean=0, C=None,
                 C_inv=None, D=None,
                 m=None,
                 bias_var=1.0):
    # infere coupling from S
    J_est_EP = np.empty(J.shape)
    log_evidence = 0.0

    # prepare inputs for multi processing
    args_multi = []
    indices = range(N)
    inputs = [indices[i:i + N / num_processes] for i in range(0, len(indices), N / num_processes)]
    for input_indices in inputs:
        args_multi.append((S, sparsity, input_indices, v_s, sigma0, bias, bias_mean, bias_var, C, C_inv, D, m))

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


def do_inference_wrapper(x0, S, J, N, num_processes, bias):
    ro = x0[0]
    v_s = x0[1]
    sigma0 = x0[2]
    if bias:
        bias_mean = x0[3]
    else:
        bias_mean = 0

    if ro < 0.0001 or ro > 0.999 or v_s < 0.0001 or sigma0 < 0.0001:
        return np.inf

    J_est_EP, log_evidence, gammas = do_inference(S, J, N, num_processes, ro, v_s, sigma0, True, bias, bias_mean)

    return -log_evidence


def optimize_log_evidence(x0, S, J, N, num_processes, bias):
    opt_res = minimize(do_inference_wrapper, x0, args=(S, J, N, num_processes, bias), method='Nelder-Mead')
    x0_res = opt_res.x
    J_est_EP, log_evidence, gammas = do_inference(S, J, N, num_processes, x0_res[0], x0_res[1], x0_res[2], True, bias,
                                                  x0_res[3])

    print opt_res.x

    return J_est_EP, log_evidence, x0_res[0]


def EM(S, J, N, num_processes, init_ro, v_s, sigma0, bias, bias_mean):
    diff = 0
    ro = init_ro
    while diff < 0.99999 or diff > 1.0001:
        J_est_EP, log_evidence, gammas = do_inference(S, J, N, num_processes, ro, v_s, sigma0, True, bias, bias_mean)
        new_ro = np.mean(gammas)
        diff = new_ro / ro
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
@click.option('--bias_mean',
              default=0,
              type=click.FLOAT,
              help='The mean for the bias couplings')
@click.option('--optimize',
              default=False,
              type=click.BOOL,
              help='Whether to optimize the model hyper parameters or not')
def main(num_neurons, time_steps, num_processes, likelihood_function, sparsity, num_trials, pprior, em, bias, bias_mean,
         optimize):
    sigma0 = 1.0
    ppriors = [float(num) for num in pprior.split(',')]
    num_neurons = [int(num) for num in num_neurons.split(',')]
    time_steps = [int(num) for num in time_steps.split(',')]
    bias = int(bias)

    return_gamma = False

    t_stamp = time.strftime("%Y%m%d_%H%M%S") + '_'
    if em or optimize:
        pprior = ppriors[0]

        N = num_neurons[0]
        v_s = 1. / np.sqrt(N)
        T = time_steps[0]

        dir_name = get_dir_name(ppriors, N, T, sparsity, likelihood_function, approx='gaussian', t_stamp=t_stamp)
        S, J = generate_J_S(likelihood_function, bias, N, T, sparsity, v_s, bias_mean)

        if em:
            results = EM(S, J, N, num_processes, pprior, v_s, sigma0, bias, bias_mean)

        if optimize:
            v_s_init = 1.
            sigma0_init = 0.1
            bias_mean_init = 0

            x0 = [pprior, v_s_init, sigma0_init, bias_mean_init]
            results = optimize_log_evidence(x0, S, J, N, num_processes, bias)

        save_inference_results_to_file(dir_name, S, J, 0, [results[0]], likelihood_function,
                                       ppriors, results[1], [], 0)
        return

    for i in range(num_trials):
        for N in num_neurons:
            v_s = 1.0 / np.sqrt(N)
            for T in time_steps:
                dir_name = get_dir_name(ppriors, N, T, sparsity, likelihood_function, approx='gaussian',
                                        t_stamp=t_stamp)
                S, J = generate_J_S(likelihood_function, bias, N, T, sparsity, v_s, bias_mean)
                C = calculate_C_delta(S[1:, :N])
                D = calculate_D_delta(S[1:, :N])
                C_inv = np.linalg.inv(C)
                m = calculate_m(S[:, :N])
                J_est_EPs = []
                log_evidences = []
                for pprior in ppriors:
                    results = do_inference(S, J, N, num_processes, pprior, v_s, sigma0, return_gamma, bias, bias_mean,
                                           C, C_inv, D, m)
                    J_est_EPs.append(results[0])
                    log_evidences.append(results[1])
                save_inference_results_to_file(dir_name, S, J, 0, [J_est_EPs], likelihood_function,
                                               ppriors, log_evidences, [], i)


if __name__ == "__main__":
    main()
