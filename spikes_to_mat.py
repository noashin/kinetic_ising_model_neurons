import click
import numpy as np
from scipy import stats
import scipy.io as sio
from scipy.special import expit

from spikes_activity_generator import generate_spikes, spike_and_slab

@click.command()
@click.option('--num_neurons', type=click.INT,
              default=10,
              help='number of neurons in the network')
@click.option('--time_steps', type=click.INT,
              default=100,
              help='Number of time stamps. Length of recording')
@click.option('--likelihood_function', type=click.STRING,
              default='probit',
              help='Should be either probit or logistic')
@click.option('--sparsity', type=click.FLOAT,
              default=0.3,
              help='Set sparsity of connectivity, aka ro parameter.')
def main(num_neurons, time_steps, likelihood_function, sparsity):
    # Get the spiking activity
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

    file_name = 'spikes_connectivity_N_' + str(N) + '_T_' + str(T) + '_ro_' + str(sparsity).replace(".", "") \
                + "_"+ likelihood_function + '.mat'

    sio.savemat(file_name, {'S': S, 'J': J})

if __name__ == "__main__":
    main()