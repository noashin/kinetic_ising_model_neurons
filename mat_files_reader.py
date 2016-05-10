import sys
import os

import scipy.io as sio


def get_J_S_from_mat_file(activity_mat_file, likelihood_function):
    S, J, J_est_lasso = get_activity_from_file(activity_mat_file)
    N = S.shape[1]
    T = S.shape[0]

    likelihood_function, ro = get_params_from_file_name(activity_mat_file, likelihood_function)
    cdf_factor = 1.0 if likelihood_function == 'probit' else 1.6

    return N, T, S, J, J_est_lasso, cdf_factor


def get_activity_from_file(mat_file_path):
    ''' Given  .mat file this funciton extract the activity matrix S
    and connectivity matrix J from the file.
    The file should contain an array S and an array J.

    :param mat_file: path to the mat file
    :return: S, J
    '''

    try:
        mat_cont = sio.loadmat(mat_file_path)
        J = mat_cont['J']
        J_est_lasso = mat_cont['J_est_1'] if 'J_est_1' in mat_cont.keys() else []
        # The activity genarated by the realistic model is saved as unit8 and should be converted to float.
        S = mat_cont['S'].astype(float)
        if 'realistic' in mat_file_path:
            S[S == 255.0] = -1.0
            S = S.transpose()

    except IOError:
        print 'Wrong mat file name'
        sys.exit(1)
    except KeyError:
        print 'mat file does not contain S or J '
        sys.exit(1)
    return S, J, J_est_lasso


def get_params_from_file_name(mat_file_path, likelihood_function):

    try:
        mat_file_name = os.path.basename(mat_file_path)
        # assuming the type of likelihood function is the last word in the file name,
        # between the last '_' and '.mat', or that it is before 'J_est'
        indices_ = [i for i, ltr in enumerate(mat_file_name) if ltr == '_']
        if 'realistic' not in mat_file_name:
            ending = mat_file_name.index('_J_est') if 'J_est' in mat_file_name else mat_file_name.index('.mat')
            likelihood_function = mat_file_name[indices_[7] + 1: ending]
        else:
            likelihood_function = likelihood_function

        #assuming sparsity is between '_ro' and another '_'
        ro_index = mat_file_name.index('_ro_')
        ro_start = ro_index + len('_ro_')
        ro_str = mat_file_name[ro_start: indices_[7]]
        ro = float(ro_str) / 10
    except ValueError:
        print 'file name does not containt likelihood or ro as expected'
        sys.exit(1)

    return likelihood_function, ro
