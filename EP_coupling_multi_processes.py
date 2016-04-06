import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.io as sio
from scipy.special import expit
import seaborn
import multiprocessing as multiprocess
import click
import sys

from spikes_activity_generator import generate_spikes, spike_and_slab
import parameters_update_prior_terms as prior_update
import parameters_update_likelihood_terms as likelihood_update


def update_likelihood_terms(mu, nu, v, m, s, activity, n, cdf_factor):
    '''

    :param mu:
    :param nu:
    :param v:
    :param m:
    :param s:
    :param activity:
    :return:
    '''
    T = activity.shape[0]
    N = activity.shape[1]
    for i in range(1, T):
        nu_old = likelihood_update.update_v_i_nu_old(nu, v[i, :])
        # Ignore likelihood terms for which nu_old < 0
        if sum(nu_old < 0) > 0:
            continue

        x_i = activity[i, n] * activity[i - 1, :]
        #x_i = activity[i]
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


def EP(activity, ro, n, pprior, cdf_factor):
    '''

    :param S: Activity matrix [T, N]
    :return:
    '''
    T = activity.shape[0]
    N = activity.shape[1]

    # Initivalization
    sigma0 = 0.0
    sigma1 = 1.0

    a = np.ones(N)
    b = np.ones(N)
    v = np.ones((T + 1, N)) * np.inf
    m = np.zeros((T + 1, N))
    s = np.ones(N + T + 1)

    mu = np.zeros(N)
    nu = np.ones(N) * (pprior * sigma1 + (1 - pprior) * sigma0)

    p_ = np.ones(N)*pprior

    v[T, :] = nu

    mu_backup = np.copy(mu)
    nu_backup = np.copy(nu)
    p_backup = np.copy(p_)
    m_backup = np.copy(m)

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
        #p_old = prior_update.calc_p_old(p_, a, b)
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

        s[T: T+N] = prior_update.update_s(z, nu_old, v[T], c1, c3)

        maxdiff = np.max([np.abs(nu - nu_backup), np.abs(mu - mu_backup), np.abs(p_ - p_backup)])

        convergence = maxdiff < 1e-5
        nu_backup = nu
        mu_backup = mu
        p_backup = p_
        m_backup = m

        itr = itr + 1

    return mu


def EP_single_neuron(activity, ro, ns, pprior, cdf_factor):
    mus = [EP(activity, ro, n, pprior, cdf_factor) for n in ns]
    return mus


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


def get_activity_from_file(mat_file):
    ''' Given  .mat file this funciton extract the activity matrix S
    and connectivity matrix J from the file.
    The file should contain an array S and an array J.

    :param mat_file: path to the mat file
    :return: S, J
    '''

    try:
        mat_cont = sio.loadmat(mat_file)
        S = mat_cont['S']
        J = mat_cont['J']
    except IOError:
        print 'Wrong mat file name'
        sys.exit(1)
    except KeyError:
        print 'mat file does not contain S or J '
        sys.exit(1)

    return S, J


def get_params_from_file_name(mat_file):

    try:
        # assuming the type of likelihood function is the last word in the file name,
        # between the last '_' and '.mat'
        last_index = mat_file[::-1].index('_')
        f_l = len(mat_file)
        likelihood_function = mat_file[f_l - last_index: mat_file.index('.')]

        #assuming sparsity is between '_ro' and another '_'
        ro_index = mat_file.index('_ro_')
        ro_start = ro_index + len('_ro_')
        ro_str = mat_file[ro_start: f_l - last_index - 1]
        ro = float(ro_str) / 10
    except ValueError:
        print 'file name does not containt likelihood or ro as expected'
        sys.exit(1)

    return likelihood_function, ro


@click.command()
@click.option('--num_neurons', type=click.INT,
              default=10,
              help='number of neurons in the network')
@click.option('--time_steps', type=click.INT,
              default=100,
              help='Number of time stamps. Length of recording')
@click.option('--num_processes', type=click.INT,
              default=1)
@click.option('--likelihood_function', type=click.STRING,
              default='probit',
              help='Should be either probit or logistic')
@click.option('--sparsity', type=click.FLOAT,
              default=0.3,
              help='Set sparsity of connectivity, aka ro parameter.')
@click.option('--pprior', type=click.FLOAT,
              default=0.3,
              help='Set pprior for the EP.')
@click.option('--show_plot', type=click.BOOL,
              default=False)
@click.option('--activity_mat_file', type=click.STRING,
              default='')
def main(num_neurons, time_steps, num_processes, likelihood_function, sparsity, pprior, show_plot, activity_mat_file):
    # Get the spiking activity
    if activity_mat_file:
        S, J = get_activity_from_file(activity_mat_file)
        N = S.shape[1]
        T = S.shape[0]

        likelihood_function, ro = get_params_from_file_name(activity_mat_file)

    else:
        N = num_neurons
        T = time_steps
        J = spike_and_slab(sparsity, N)
        S0 = -np.ones(N)

        if likelihood_function == 'probit':
            energy_function = stats.norm.cdf
        elif likelihood_function == 'logistic':
            energy_function = expit
        else:
            raise ValueError('Unknown likelihood function')

        S = generate_spikes(N, T, S0, J, energy_function)

    cdf_factor = 1.0 if likelihood_function == 'probit' else 1.6
    # import ipdb; ipdb.set_trace()
    # infere coupling from S
    J_est_1 = np.empty(J.shape)
    args_multi = []
    indices = range(N)
    inputs = [indices[i:i + N / num_processes] for i in range(0, len(indices), N / num_processes)]

    for input_indices in inputs:
        args_multi.append((S, sparsity, input_indices, pprior, cdf_factor))

    mus = do_multiprocess(multi_process_EP, args_multi, num_processes)

    for i, indices in enumerate(inputs):
        J_est_1[indices, :] = mus[i]
    # import ipdb; ipdb.set_trace()
    # plot and compare J and J_est
    title = 'N_' + str(N) + '_T_' + str(T) + '_ro_' + str(sparsity).replace(".", "") \
            + "_pprior_" + str(pprior) + "_"+ likelihood_function
    fig = plt.figure()
    plt.plot([J.min(), J.max()], [J.min(), J.max()], 'k')
    plt.plot(J.flatten(), J_est_1.flatten(), 'o')
    plt.title(title)
    if show_plot:
        plt.show()
    fig.savefig(title + '.png')

if __name__ == "__main__":
    main()
