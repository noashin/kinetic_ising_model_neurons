import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn


def get_dir_name(ppriors, N, T, sparsity, likelihood_function, approx='probit'):
    if len(ppriors) == 1:
        dir_name = approx + '_N_' + str(N) + '_T_' + str(T) + '_ro_' + str(sparsity).replace(".", "") \
                + "_pprior_" + str(ppriors[0]).replace('.', '') + "_" + likelihood_function
    else:
        dir_name = approx + '_N_' + str(N) + '_T_' + str(T) + '_ro_' + \
                   str(sparsity).replace(".", "") + '_' + likelihood_function

    # create a new directory to save the results
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def save_inference_results_to_file(dir_name, S, J, bias, J_est_EPs, likelihood_function,
                                   ppriors, log_evidences, J_est_lasso=[], i=0):

    # Save simulation data to file
    if list(J_est_lasso):
        file_path = os.path.join(dir_name, 'S_J_J_est_lasso_EP_' + str(i))
        sio.savemat(file_path, {'S': S, 'J': J, 'J_est_lasso': J_est_lasso,
                                'J_est_EPs': J_est_EPs, 'ppriors': ppriors, 'ros': ros,
                                'log_evidences': log_evidences})
    else:
        file_path = os.path.join(dir_name, 'S_J_J_est_EP_' + str(i))
        sio.savemat(file_path, {'S': S, 'J': J, 'J_est_EPs': J_est_EPs, 'ppriors': ppriors,
                                'log_evidences': log_evidences})

    return dir_name


def plot_and_save(measurements, J, J_est_lasso, J_est_EP, ppriors, log_evidences, ros,
                  plot, show_plot, dir_name):
    if plot:
        inference_fig = plot_J_J_est(J, J_est_EP, J_est_lasso, show_plot)
        inference_fig[0].savefig(os.path.join(dir_name, inference_fig[1] + '.png'))

        if measurements:
            measurements_fig = plot_error_measurements(ppriors, measurements, show_plot)
            measurements_fig[0].savefig(os.path.join(dir_name, measurements_fig[1]))

        if list(log_evidences):
            log_evidence_fig = plot_log_evidence(ros, log_evidences, show_plot)
            log_evidence_fig[0].savefig(os.path.join(dir_name, log_evidence_fig[1]))


def plot_J_J_est(J, J_est_EP, J_est_lasso, show_plot):
    """This functions plots a scatter plot of the infered couplings (J_est)
    versus the real coupling (J).
    If both J_est_EP and J_est_lasso are supplied then 2 subplots will be plotted.

    :param J: The real coupling
    :param J_est_EP: couplings infered by EP
    :param J_est_lasso: couplings infered by LASSO
    :param show_plot: if True the plot will be shown
    :return: the figure object and a title for future saving
    """

    title = 'J_vs_J_est'

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

    if show_plot:
        plt.show()

    return f, title


def plot_log_evidence(ros, log_evidences, show_plot):
    best_ro = ros[np.argmax(log_evidences)]
    f = plt.figure()
    plt.plot(ros, log_evidences)
    plt.xlabel('ro')
    plt.ylabel('log evidence')
    plt.title('best ro: ' + str(best_ro))

    if show_plot:
        plt.show()

    return f, 'log_evidences.png'


def plot_error_measurements(ppriors, measurements, show_plot):

    ppriors = np.array(ppriors)
    indices = np.argsort(ppriors)
    ppriors = ppriors[indices]
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col')
    ax1.plot(ppriors, np.array(measurements['r_square'])[indices])
    ax1.set_title('r_square')
    ax2.plot(ppriors, np.array(measurements['corr_coef'])[indices])
    ax2.set_title('corr_coef')
    ax3.plot(ppriors, np.array(measurements['zero_matching'])[indices])
    ax3.set_title('zero_matching')
    ax4.plot(ppriors, np.array(measurements['sign_matching'])[indices])
    ax4.set_title('sign_matching')

    if show_plot:
        plt.show()

    return f, 'error_measurements.png'
