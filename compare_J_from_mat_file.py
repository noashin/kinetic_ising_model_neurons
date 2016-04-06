import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn
import click
import sys
from sklearn.linear_model import Lasso


@click.command()
@click.option('--show_plot', type=click.BOOL,
              default=False)
@click.option('--mat_file', type=click.STRING,
              default='')
def main(show_plot, mat_file):
    try:
        mat_cont = sio.loadmat(mat_file)
        J = mat_cont['J']
        J_est_1 = mat_cont['J_est_1']
        J_est_2 = mat_cont['J_est_2']

    except IOError:
        print 'Wrong mat file name'
        sys.exit(1)
    except KeyError:
        print 'mat file does not contain S or J '
        sys.exit(1)
    slope, intercept, r_value, p_value, std_err = linregress(J.flatten(), J_est_1.flatten())
    print slope, intercept, r_value, p_value, std_err
    # plot and compare J and J_est
    title = mat_file[:mat_file.index('.')] + '_LASSO'
    fig = plt.figure()
    plt.plot([J.min(), J.max()], [J.min(), J.max()], 'k')
    plt.plot(J.flatten(), J_est_1.flatten() / 2.0, 'o')
    plt.title(title)
    if show_plot:
        plt.show()
    #fig.savefig(title + '.png')

if __name__ == "__main__":
    main()
