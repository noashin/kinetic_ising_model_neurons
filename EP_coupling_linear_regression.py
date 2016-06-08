import numpy as np
import multiprocessing as multiprocess
import click

from spikes_activity_generator import generate_spikes, spike_and_slab
from plotting_saving import save_inference_results_to_file, get_dir_name
import update_params_linear_regression as update_params

v_inf = 1000.0


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def inv_logistic(x):
    return np.log(1.0 / (1.0 - x))


def EP(activity, ro, n, v_s, sigma0):
    """Performs EP and returns the approximated couplings

    :param activity: Network's activity
    :param ro: sparsity of the couplings
    :param n: index of the neuron being concidered
    :param v_s: the variance of the "slab" part of the network
    :param sigma0: assumed variance of the likelihood
    :return: The approximated couplings for neuron n
    """

    N = activity.shape[1]

    # Initivalization
    v_s = v_s * np.ones(N)
    p_3 = update_params.update_p_3(ro, N)

    p_2 = m_1 = m_2 = np.zeros(N)
    v_1 = np.inf * np.ones(N)
    v_2 = ro * v_s

    # pre calculate constant matrices and vectors
    X = activity[:-1, :]
    y = activity[1:, n]

    XT_X = np.dot(X.T, X)
    XT_y = np.dot(X.T, y)

    itr = 0
    max_itr = 10000
    convergence = False

    # Repeat the updates rules until convergence
    while not convergence and itr < max_itr:
        m_1_old = m_1
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

        maxdiff = np.max(np.abs(m_1 / m_1_old))
        mindiff = np.min(np.abs(m_1 / m_1_old))
        convergence = maxdiff < 1.00001 and mindiff > 0.9999
        itr += 1

    return m_1


def EP_single_neuron(activity, ro, ns, v_s, sigma0):
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
    J += 0.0 # to avoid negative zeros
    S0 = - np.ones(N + bias)

    if likelihood_function != 'gaussian' and likelihood_function != 'exp_cosh' and likelihood_function != 'logistic':
        raise ValueError('Unknown likelihood function')

    S = generate_spikes(N, T, S0, J, likelihood_function, bias)

    return S, J


def do_inference(S, J, N, num_processes, sparsity, v_s, sigma0):
    # infere coupling from S
    J_est_EP = np.empty(J.shape)

    # prepare inputs for multi processing
    args_multi = []
    indices = range(N)
    inputs = [indices[i:i + N / num_processes] for i in range(0, len(indices), N / num_processes)]
    for input_indices in inputs:
        args_multi.append((S, sparsity, input_indices, v_s, sigma0))

    results = do_multiprocess(multi_process_EP, args_multi, num_processes)

    for i, indices in enumerate(inputs):
        mus = [results[i][j] for j in range(len(results[i]))]
        J_est_EP[:, indices] = np.array(mus).transpose()

    return J_est_EP


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
def main(num_neurons, time_steps, num_processes, likelihood_function, sparsity, num_trials):

    sigma0 = 1.0
    ppriors = ['0.0', '1.0']  # Not used, just for compatibility

    num_neurons = [int(num) for num in num_neurons.split(',')]
    time_steps = [int(num) for num in time_steps.split(',')]
    for i in range(num_trials):
        for N in num_neurons:
            v_s = 1.0 / np.sqrt(N)
            for T in time_steps:
                dir_name = get_dir_name(ppriors, N, T, sparsity, likelihood_function, approx='gaussian')
                S, J = generate_J_S(likelihood_function, 0, N, T, sparsity, v_s)
                J_est_EP = do_inference(S, J, N, num_processes, sparsity, v_s, sigma0)
                save_inference_results_to_file(dir_name, S, J, 0, [J_est_EP], likelihood_function,
                                               ppriors, [], [], i)

if __name__ == "__main__":
    main()
