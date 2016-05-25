import numpy as np
import multiprocessing as multiprocess
import click

from spikes_activity_generator import generate_spikes, spike_and_slab
from plotting_saving import save_inference_results_to_file, get_dir_name
from mat_files_reader import get_J_S_from_mat_file
import update_params_linear_regression as update_params


v_inf = 100.0


def calc_log_evidence():
    # TODO:
    return 0.0


def EP(activity, ro, n, v_s, sigma0):
    '''

    :param S: Activity matrix [T, N]
    :return:
    '''
    T = activity.shape[0]
    N = activity.shape[1]

    # Initivalization
    v_s = v_s * np.ones(N)
    p_3 = update_params.update_p_3(ro, N)
    v_2_new = ro * v_s

    p_2 = m_1 = m_2 = np.zeros(N)
    v_2 = v_1 = np.inf * np.ones(N)

    X = activity[1:, :]
    XT_X = np.dot(X.T, X)
    y = activity[1:, n]

    itr = 0
    max_itr = 300
    convergence = False

    while not convergence and itr < max_itr:
        p_2_new = update_params.update_p_2(v_1, v_s, m_1)
        a = update_params.calc_a(p_2_new, p_3, m_1, v_1, v_s)
        b = update_params.calc_b(p_2_new, p_3, m_1, v_1, v_s)
        if itr > 0:
            v_2_new = update_params.update_v_2(a, b, v_1)

        # avoid negative values in v_2_new and use v_inf instead
        v_2_new[v_2_new < 0] = v_inf
        m_2_new = update_params.update_m_2(m_1, a, v_2_new, v_1)

        V_2 = np.diag(v_2)
        V = update_params.calc_V(V_2, sigma0, XT_X)

        v_1_new = update_params.update_v_1(v_2, V)
        m_1_new = update_params.update_m_1(V, v_2, m_2, sigma0, X, y, v_1_new)

        maxdiff = np.max(np.abs(m_2_new - m_2))
        convergence = maxdiff < 1e-5

        m_1 = m_1_new
        v_1 = v_1_new
        m_2 = m_2_new
        p_2 = p_2_new
        v_2 = v_2_new

        itr = itr + 1

    log_evidence =calc_log_evidence()
    return {'mu': m_2, 'log_evidence': log_evidence}


def EP_single_neuron(activity, ro, ns, pprior, cdf_factor):
    results = [EP(activity, ro, n, pprior, cdf_factor) for n in ns]
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


def generate_J_S(likelihood_function, bias, num_neurons, time_steps, sparsity):
    if bias != 0 and bias != 1:
            raise ValueError('bias should be either 1 or 0')

    N = num_neurons
    T = time_steps

    # Add a column for bias if it is part of the model
    J = spike_and_slab(sparsity, N, bias)
    J = J + 0.0
    S0 = - np.ones(N + bias)

    if likelihood_function != 'gaussian' and likelihood_function != 'exp_cosh':
        raise ValueError('Unknown likelihood function')

    S = generate_spikes(N, T, S0, J, energy_function, bias)

    cdf_factor = 1.0 if likelihood_function == 'probit' else 1.6

    return S, J, cdf_factor


def do_inference(S, J, N, num_processes, pprior, sparsity,cdf_factor):
    # infere coupling from S
    J_est_EP = np.empty(J.shape)
    log_evidence = 0.0

    #prepare inputs for multi processing
    args_multi = []
    indices = range(N)
    inputs = [indices[i:i + N / num_processes] for i in range(0, len(indices), N / num_processes)]
    for input_indices in inputs:
        args_multi.append((S, sparsity, input_indices, pprior, cdf_factor))

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

    if activity_mat_file:
        # If only
        N, T, S, J, J_est_lasso, _, cdf_factor = get_J_S_from_mat_file(activity_mat_file, likelihood_function)
        J_est_EPs = []
        log_evidences = []
        for pprior in ppriors:
            results = do_inference(S, J, N, num_processes, pprior, sparsity,cdf_factor)
            J_est_EPs.append(results[0])
            log_evidences.append(results[1])

        dir_name = get_dir_name(ppriors, N, T, sparsity, likelihood_function)
        save_inference_results_to_file(dir_name, S, J, bias, J_est_EPs, likelihood_function,
                                   ppriors, log_evidences, J_est_lasso)

    else:
        num_neurons = [int(num) for num in num_neurons.split(',')]
        time_steps = [int(num) for num in time_steps.split(',')]
        for N in num_neurons:
            for T in time_steps:
                dir_name = get_dir_name(ppriors, N, T, sparsity, likelihood_function)
                S, J, cdf_factor = generate_J_S(likelihood_function, bias, N, T, sparsity)
                for i in range(num_trials):
                    J_est_EPs = []
                    log_evidences = []
                    for pprior in ppriors:
                        results = do_inference(S, J, N, num_processes, pprior, sparsity,cdf_factor)
                        J_est_EPs.append(results[0])
                        log_evidences.append(results[1])
                    save_inference_results_to_file(dir_name, S, J, bias, J_est_EPs, likelihood_function,
                                                   ppriors, log_evidences, [], i)

if __name__ == "__main__":
    main()
