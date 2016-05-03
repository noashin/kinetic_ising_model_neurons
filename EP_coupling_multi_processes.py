import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.io as sio
from scipy.special import expit
import seaborn
import multiprocessing as multiprocess
import click
import sys
import os

from spikes_activity_generator import generate_spikes, spike_and_slab
from measurements import r_square, corr_coef, zero_matching, sign_matching
import parameters_update_prior_terms as prior_update
import parameters_update_likelihood_terms as likelihood_update


ros = np.arange(0.01, 1.0, 0.01)


def calc_log_evidence(a, b, nu, s, mu, m, v, N):
    '''This function calculate the log evidence based on the infered model

    :param a:
    :param b:
    :param nu:
    :param s:
    :param mu:
    :param m:
    :param v:
    :param ro:
    :param N:
    :return:
    '''
    v_learnt = v[1:, :]
    m_learnt = m[1:, :]
    s = s[~np.isnan(s)]

    B = np.dot(mu, np.multiply(mu, nu / 1)) - np.sum(np.multiply(v_learnt**(-1), m_learnt**2))

    log_C = [np.sum(np.log(ro * a + (1-ro) * b)) for ro in ros]

    second_term = np.log(2.0 * np.pi) * N / 2.0 + np.sum(0.5 * np.log(nu)) + B / 2 + np.sum(np.log(s))
    second_vec = np.repeat(second_term, len(ros))
    log_evidence = log_C + second_vec

    return log_evidence


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

        itr = itr + 1

    log_evidence = calc_log_evidence(a, b, nu, s, mu, m, v, N)
    return {'mu': mu, 'log_evidence': log_evidence}


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


def get_activity_from_file(mat_file):
    ''' Given  .mat file this funciton extract the activity matrix S
    and connectivity matrix J from the file.
    The file should contain an array S and an array J.

    :param mat_file: path to the mat file
    :return: S, J
    '''

    try:
        mat_cont = sio.loadmat(mat_file)
        J = mat_cont['J']
        J_est_lasso = mat_cont['J_est_1'] if 'J_est_1' in mat_cont.keys() else []
        # The activity genarated by the realistic model is saved as unit8 and should be converted to float.
        S = mat_cont['S'].astype(float)
        if 'realistic' in mat_file:
            S[S==255.0] = -1.0
            S = S.transpose()

    except IOError:
        print 'Wrong mat file name'
        sys.exit(1)
    except KeyError:
        print 'mat file does not contain S or J '
        sys.exit(1)
    return S, J, J_est_lasso


def get_params_from_file_name(mat_file, likelihood_function):

    try:
        # assuming the type of likelihood function is the last word in the file name,
        # between the last '_' and '.mat', or that it is before 'J_est'
        indices_ = [i for i, ltr in enumerate(mat_file) if ltr == '_']
        if 'realistic' not in mat_file:
            ending = mat_file.index('_J_est') if 'J_est' in mat_file else mat_file.index('.mat')
            likelihood_function = mat_file[indices_[7] + 1: ending]
        else:
            likelihood_function = likelihood_function

        #assuming sparsity is between '_ro' and another '_'
        ro_index = mat_file.index('_ro_')
        ro_start = ro_index + len('_ro_')
        ro_str = mat_file[ro_start: indices_[7]]
        ro = float(ro_str) / 10
    except ValueError:
        print 'file name does not containt likelihood or ro as expected'
        sys.exit(1)

    return likelihood_function, ro


def plot_results(S, J, bias, sparsity, J_est_EP, J_est_lasso, likelihood_function,
                 pprior, save_results, log_evidences, measurements, ppriors):
    N = S.shape[1] - bias
    T = S.shape[0]

    if len(ppriors) == 1:
        title = 'N_' + str(N) + '_T_' + str(T) + '_ro_' + str(sparsity).replace(".", "") \
                + "_pprior_" + str(ppriors[0]).replace('.', '') + "_" + likelihood_function
    else:
        title = 'N_' + str(N) + '_T_' + str(T) + '_ro_' + \
                str(sparsity).replace(".", "") + '_' + likelihood_function

    if list(J_est_lasso) and list(J_est_EP):
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot([J.min(), J.max()], [J.min(), J.max()], 'k')
        axarr[0].plot(J.flatten(), J_est_EP.flatten(), 'o')
        axarr[0].set_ylabel('J_est_EP')
        axarr[0].set_title(title)
        axarr[1].plot([J.min(), J.max()], [J.min(), J.max()], 'k')
        axarr[1].plot(J.flatten(), J_est_lasso.flatten() / 2.0, 'o')
        axarr[1].set_ylabel('J_est_lasso')
        axarr[1].set_xlabel('J')

    else:
        f = plt.figure()
        J_est = J_est_EP if list(J_est_EP) else J_est_lasso
        ylabel = 'J_est_EP' if list(J_est_EP) else 'J_est_lasso'
        correction = 1.0 if list(J_est_EP) else 2.0
        plt.plot([J.min(), J.max()], [J.min(), J.max()], 'k')
        plt.plot(J.flatten(), J_est.flatten() / correction, 'o')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('J')
    if save_results:
        f.savefig(os.path.join(title, 'J_vs_J_est.png'))

    plt.show()

    if list(log_evidences):
        best_ro = ros[np.argmax(log_evidences)]
        f = plt.figure()
        plt.plot(ros, log_evidences)
        plt.xlabel('ro')
        plt.ylabel('log evidence')
        plt.title('best ro: ' + str(best_ro))
        if save_results:
            f.savefig(os.path.join(title, 'log_evidence.png'))
    plt.show()

    if measurements:
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col')
        ax1.plot(ppriors, measurements['r_square'])
        ax1.set_title('r_square')
        ax2.plot(ppriors, measurements['corr_coef'])
        ax2.set_title('corr_coef')
        ax3.plot(ppriors, measurements['zero_matching'])
        ax3.set_title('zero_matching')
        ax4.plot(ppriors, measurements['sign_matching'])
        ax4.set_title('sign_matching')
        plt.show()
        if save_results:
            f.savefig(os.path.join(title, 'error_measurements.png'))

def save_results_to_file(S, J, bias, sparsity, J_est_EPs, J_est_lasso, likelihood_function, ppriors):
    N = S.shape[1] - bias
    T = S.shape[0]

    # create a new directory to save the results and the plot
    if len(ppriors) == 1:
        dir_name = 'N_' + str(N) + '_T_' + str(T) + '_ro_' + str(sparsity).replace(".", "") \
                + "_pprior_" + str(ppriors[0]).replace('.', '') + "_" + likelihood_function
    else:
        dir_name = 'N_' + str(N) + '_T_' + str(T) + '_ro_' + \
                   str(sparsity).replace(".", "") + '_' +likelihood_function
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Save simulation data to file
    if list(J_est_lasso):
        file_path = os.path.join(dir_name, 'S_J_J_est_lasso_EP.json')
        sio.savemat(file_path, {'S': S, 'J': J, 'J_est_lasso': J_est_lasso,
                                'J_est_EPs': J_est_EPs, 'ppriors': ppriors})
    else:
        file_path = os.path.join(dir_name, 'S_J_J_est_EP.json')
        sio.savemat(file_path, {'S': S, 'J': J, 'J_est_EP': J_est_EPs, 'ppriors': ppriors})


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
@click.option('--plot', type=click.BOOL,
              default=False)
@click.option('--save_results', type=click.BOOL,
              default=False)
@click.option('--activity_mat_file', type=click.STRING,
              default='')
@click.option('--bias', type=click.INT,
              default=0,
              help='1 if bias should be included in the model, 0 otherwise')
@click.option('--do_inference', type=click.BOOL,
              default=True,
              help='If false then the script will only plot J and J_est from file')
@click.option('--error_measurements', type=click.BOOL,
              default=False,
              help='f true error measurements will be taken for different ppriorI')
def main(num_neurons, time_steps, num_processes, likelihood_function, sparsity,
         pprior, plot, save_results, activity_mat_file, bias, do_inference, error_measurements):

    # Get the spiking activity
    if activity_mat_file:
        S, J, J_est_lasso = get_activity_from_file(activity_mat_file)
        N = S.shape[1]
        T = S.shape[0]

        likelihood_function, ro = get_params_from_file_name(activity_mat_file, likelihood_function)

    else:

        if bias != 0 and bias != 1:
            raise ValueError('bias should be either 1 or 0')

        N = num_neurons
        T = time_steps

        # Add a column for bias if it is part of the model
        J = spike_and_slab(sparsity, N, bias)
        S0 = - np.ones(N + bias)
        J_est_lasso = []

        if likelihood_function == 'probit':
            energy_function = stats.norm.cdf
        elif likelihood_function == 'logistic':
            energy_function = expit
        else:
            raise ValueError('Unknown likelihood function')

        S = generate_spikes(N, T, S0, J, energy_function, bias)

    cdf_factor = 1.0 if likelihood_function == 'probit' else 1.6

    if do_inference:

        J_est_EPs = []
        if error_measurements:
            ppriors = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
            measurements = {'r_square': [], 'corr_coef': [], 'zero_matching': [], 'sign_matching': []}
        else:
            ppriors = [pprior]
            measurements = []

        for pprior in ppriors:
            # infere coupling from S
            J_est_EP = np.empty(J.shape)
            log_evidences = np.empty(ros.shape[0])

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
                log_evidences += np.sum(np.array(evidences_tmp), axis=0)

            if len(ppriors) > 1:
                #calculate different error measurements
                J_est_EPs.append(J_est_EP)
                measurements['r_square'].append(r_square(J, J_est_EP))
                measurements['corr_coef'].append(corr_coef(J, J_est_EP))
                measurements['zero_matching'].append(zero_matching(J, J_est_EP))
                measurements['sign_matching'].append(sign_matching(J, J_est_EP))

    else:
        J_est_EP = []
        log_evidences = []
        ppriors = []

    if save_results:
        save_results_to_file(S, J, bias, sparsity, J_est_EPs, J_est_lasso, likelihood_function, ppriors)

    if plot:
        plot_results(S, J, bias, sparsity, J_est_EP, J_est_lasso, likelihood_function,
                     pprior, save_results, log_evidences, measurements, ppriors)


if __name__ == "__main__":
    main()
