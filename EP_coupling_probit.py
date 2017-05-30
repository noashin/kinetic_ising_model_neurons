import time

import numpy as np
from scipy import stats
from scipy.special import expit
import multiprocessing as multiprocess
import click

from spikes_activity_generator import generate_spikes, spike_and_slab
from plotting_saving import save_inference_results_to_file, get_dir_name
from mat_files_reader import get_J_S_from_mat_file
import parameters_update_prior_terms as prior_update
import parameters_update_likelihood_terms as likelihood_update


def calc_log_evidence(a, b, nu, s, mu, m, v, N, pprior):
    '''This function calculate the log evidence based on the infered model

    '''
    v_learnt = v[1:, :]
    m_learnt = m[1:, :]
    s = s[~np.isnan(s)]

    B = np.dot(mu, np.multiply(mu, nu / 1.0)) - np.sum(np.multiply(v_learnt ** (-1.0), m_learnt ** 2.0))

    log_C = np.sum(np.log(pprior * a + (1.0 - pprior) * b))

    second_term = np.log(2.0 * np.pi) * N / 2.0 + np.sum(0.5 * np.log(nu)) + B / 2 + np.sum(np.log(s))
    log_evidence = log_C + second_term

    return log_evidence if np.isfinite(log_evidence) else 0.0


def update_likelihood_terms(mu, nu, v, m, s, activity, n, cdf_factor):
    """ This function exectues the update rules for the likelihood function

    """
    T = activity.shape[0]
    N = activity.shape[1]
    for i in range(1, T):
        nu_old = likelihood_update.update_v_i_nu_old(nu, v[i, :])
        # Ignore likelihood terms for which nu_old < 0
        if sum(nu_old < 0) > 0:
            continue

        x_i = activity[i, n] * activity[i - 1, :]

        mu_old = likelihood_update.calc_mu_old(mu, nu_old, v[i, :], m[i, :])
        z = likelihood_update.calc_z(x_i, mu_old, nu_old)
        alpha_i = likelihood_update.calc_alpha_i(x_i, nu_old, z, cdf_factor)

        mu = likelihood_update.update_mu(mu_old, alpha_i, nu_old, x_i)
        nu = likelihood_update.update_nu(nu_old, alpha_i, x_i, mu)
        v[i, :] = likelihood_update.update_v_i_nu_old(nu, nu_old)

        m[i, :] = likelihood_update.update_m_i(mu_old, alpha_i, v[i, :], x_i, nu_old)
        m[i, :] = [0 if not np.isfinite(v[i, j]) else m[i, j] for j in range(N)]
        s[i] = likelihood_update.update_s_i(z, v[i, :], nu_old, m[i, :], mu_old, cdf_factor)

    return mu, nu, v, m, s


def EP(activity, n, pprior, cdf_factor, v_s, bias):
    """ This function perfoems EP and returns the infered couplings.

    :param activity: The network's activity
    :param ro: the assumed sparsity of the couplins
    :param n: index of the neuron
    :param pprior:
    :param cdf_factor:
    :param v_s: variance of the slab part of the prior distribution
    :return:
    """

    T = activity.shape[0]
    N = activity.shape[1]

    # Initivalisation
    sigma0 = 0.0
    sigma1 = v_s

    a = np.ones(N)
    b = np.ones(N)
    v = np.ones((T + 1, N)) * np.inf
    m = np.zeros((T + 1, N))
    s = np.ones(N + T + 1)

    mu = np.zeros(N)

    if bias:
        nu = pprior * sigma1 + (1 - pprior) * sigma0
        p_ = pprior.copy()
    else:
        nu = np.ones(N) * (pprior * sigma1 + (1 - pprior) * sigma0)
        p_ = np.ones(N) * pprior


    v[T, :] = nu

    mu_backup = np.copy(mu)
    nu_backup = np.copy(nu)
    p_backup = np.copy(p_)

    itr = 0
    max_itr = 300
    convergence = False

    while not convergence and itr < max_itr:
        mu, nu, v, m, s = update_likelihood_terms(mu, nu, v, m, s, activity, n, cdf_factor)

        nu_old = prior_update.calc_nu_old(nu, v[T, :])

        # we take into consideration only terms with nu_old > 0. So we want to find their positions
        nu_old_pos_positions = np.where(nu_old > 0)
        nu_old_neg_positions = np.where(nu_old <= 0)

        nu_old[nu_old_neg_positions] = nu[nu_old_neg_positions]
        mu_old = prior_update.calc_mu_old(mu, nu_old, v[T, :], m[T, :])
        mu_old[nu_old_neg_positions] = mu[nu_old_neg_positions]

        G0 = prior_update.calc_G(mu_old, nu_old, sigma0)
        G1 = prior_update.calc_G(mu_old, nu_old, sigma1)

        p_old = pprior
        z = prior_update.calc_Z(p_old, G1, G0)
        c1 = prior_update.calc_c1(z, p_old, G1, mu_old, nu_old, sigma1, G0, sigma0)
        c2 = prior_update.calc_c2(z, p_old, G1, mu_old, nu_old, sigma1, G0, sigma0)
        c3 = prior_update.calc_c3(c1, c2)

        nu[nu_old_pos_positions] = prior_update.update_nu(nu_old, c3)[nu_old_pos_positions]
        mu[nu_old_pos_positions] = prior_update.update_mu(mu_old, c1, nu_old)[nu_old_pos_positions]

        p_[nu_old_pos_positions] = prior_update.update_p_i(p_old, G1, G0)[nu_old_pos_positions]

        v[T, nu_old_pos_positions] = prior_update.update_v_i(c3, nu_old)[nu_old_pos_positions]
        m[T, nu_old_pos_positions] = prior_update.update_m_i(mu_old, c1, v[T, :], nu_old)[nu_old_pos_positions]
        a[nu_old_pos_positions] = prior_update.update_a(p_, p_old)[nu_old_pos_positions]
        b[nu_old_pos_positions] = prior_update.update_b(p_, p_old)

        s[T: T + N] = prior_update.update_s(z, nu_old, v[T], c1, c3)

        maxdiff = np.max([np.abs(nu - nu_backup), np.abs(mu - mu_backup), np.abs(p_ - p_backup)])

        convergence = maxdiff < 1e-5
        nu_backup = nu
        mu_backup = mu
        p_backup = p_

        itr = itr + 1

    log_evidence = calc_log_evidence(a, b, nu, s, mu, m, v, N, pprior)
    return {'mu': mu, 'log_evidence': log_evidence}


def EP_single_neuron(activity, ns, pprior, cdf_factor, v_s, bias):
    results = [EP(activity, n, pprior, cdf_factor, v_s, bias) for n in ns]
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


def generate_J_S(likelihood_function, bias, num_neurons, time_steps, sparsity, v_s):
    if bias != 0 and bias != 1:
        raise ValueError('bias should be either 1 or 0')

    N = num_neurons
    T = time_steps

    # Add a column for bias if it is part of the model
    J = spike_and_slab(sparsity, N, bias, v_s)
    J += 0.0
    S0 = - np.ones(N + bias)

    if likelihood_function == 'probit':
        energy_function = stats.norm.cdf
    elif likelihood_function == 'logistic':
        energy_function = expit
    elif likelihood_function == 'exp_cosh':
        energy_function = likelihood_function
    else:
        raise ValueError('Unknown likelihood function')

    S = generate_spikes(N, T, S0, J, energy_function, bias)

    cdf_factor = 1.6 if likelihood_function == 'logistic' else 1.0
    return S, J, cdf_factor


def do_inference(S, J, N, num_processes, pprior, cdf_factor, v_s=1.0, bias=0):
    # infere coupling from S
    J_est_EP = np.empty(J.shape)
    log_evidence = 0.0

    # prepare inputs for multi processing
    args_multi = []
    indices = range(N)
    inputs = [indices[i:i + N / num_processes] for i in range(0, len(indices), N / num_processes)]
    for input_indices in inputs:
        args_multi.append((S, input_indices, pprior, cdf_factor, v_s, bias))

    results = do_multiprocess(multi_process_EP, args_multi, num_processes)

    for i, indices in enumerate(inputs):
        mus = [results[i][j]['mu'] for j in range(len(results[i]))]
        J_est_EP[:, indices] = np.array(mus).transpose()

        evidences_tmp = [results[i][j]['log_evidence'] for j in range(len(results[i]))]
        log_evidence += np.sum(np.array(evidences_tmp), axis=0)

    return J_est_EP, log_evidence


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
              default='probit',
              help='Should be either probit or logistic')
@click.option('--sparsity', type=click.FLOAT,
              default=0.3,
              help='Set sparsity of connectivity, aka ro parameter.')
@click.option('--pprior',
              help='Set pprior for the EP. If a list the inference will be done for every pprior')
@click.option('--activity_mat_file', type=click.STRING,
              default='')
@click.option('--bias', type=click.INT,
              default=0,
              help='1 if bias should be included in the model, 0 otherwise')
@click.option('--num_trials', type=click.INT,
              default=1,
              help='number of trials with different S ad J for given settings')
def main(num_neurons, time_steps, num_processes, likelihood_function, sparsity, pprior,
         activity_mat_file, bias, num_trials):
    ppriors = [float(num) for num in pprior.split(',')]

    t_stamp = time.strftime("%Y%m%d_%H%M%S") + '_'
    # If a file containing S an J is supplied the read it
    if activity_mat_file:
        N, T, S, J, J_est_lasso, _, cdf_factor = get_J_S_from_mat_file(activity_mat_file, likelihood_function)
        J_est_EPs = []
        log_evidences = []
        for pprior in ppriors:
            results = do_inference(S, J, N, num_processes, pprior, cdf_factor)
            J_est_EPs.append(results[0])
            log_evidences.append(results[1])

        dir_name = get_dir_name(ppriors, N, T, sparsity, likelihood_function, t_stamp=t_stamp)
        save_inference_results_to_file(dir_name, S, J, bias, J_est_EPs, likelihood_function,
                                       ppriors, log_evidences, J_est_lasso)

    # If not generate S and J
    else:
        num_neurons = [int(num) for num in num_neurons.split(',')]
        time_steps = [int(num) for num in time_steps.split(',')]
        for i in range(num_trials):
            for N in num_neurons:
                for T in time_steps:
                    v_s = 1
                    dir_name = get_dir_name(ppriors, N, T, sparsity, likelihood_function, t_stamp=t_stamp)
                    S, J, cdf_factor = generate_J_S(likelihood_function, bias, N, T, sparsity, v_s)
                    J_est_EPs = []
                    log_evidences = []
                    for pprior in ppriors:
                        if bias:
                            pprior = np.hstack([np.repeat(pprior, N), 1])
                        results = do_inference(S, J, N, num_processes, pprior, cdf_factor, v_s, bias)
                        J_est_EPs.append(results[0])
                        log_evidences.append(results[1])
                    save_inference_results_to_file(dir_name, S, J, bias, J_est_EPs, likelihood_function,
                                                   ppriors, log_evidences, [], i)


if __name__ == "__main__":
    main()
