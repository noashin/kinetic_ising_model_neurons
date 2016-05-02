import numpy as np


def test_update_mu():
    from parameters_update_likelihood_terms import update_mu
    mu_old = np.array([1.0, 2.0, 3.0])
    alpha_i = 2.0
    v_old = np.array([5.0, 4.0, 2.0])
    x_i = np.array([2.0, 0.0, 1.0])

    expected_result = np.array([21.0, 2.0, 7.0])

    np.testing.assert_array_equal(expected_result, update_mu(mu_old, alpha_i, v_old, x_i))


def test_update_nu():
    from parameters_update_likelihood_terms import update_nu

    nu_old = np.array([1.0, 2.0, 3.0])
    alpah_i = 2.0
    x_i = np.array([3.0, 1.0, 2.0])
    mu_new = np.array([4.0, 2.0, 1.0])

    result = update_nu(nu_old, alpah_i, x_i, mu_new)
    expected_result = np.array([-25.0 / 2.0, -4.0, -51.0])

    np.testing.assert_array_equal(expected_result, result)


def test_update_v_i_nu_old():
    from parameters_update_likelihood_terms import update_v_i_nu_old

    nu_old = np.array([1.0, 0.5, 0.25])
    nu_new = np.array([0.125, 1.0, 0.1])

    result = update_v_i_nu_old(nu_new, nu_old)
    expected_result = np.array([1.0 / 7.0, -1.0, 1.0 / 6.0])

    np.testing.assert_array_equal(result, expected_result)


def test_update_m_i():
    from parameters_update_likelihood_terms import update_m_i
    mu_old = np.array([1.0, 2.0, 3.0])
    alpha_i = 2.0
    v_i_new = np.array([2.0, 5.0, 1.0])
    x_i = np.array([4.0, 2.0, 3.0])
    v_old = np.array([3.0, 1.0, 2.0])

    result = update_m_i(mu_old, alpha_i, v_i_new , x_i, v_old)
    expected_result = np.array([41.0, 26.0, 21.0])

    np.testing.assert_array_equal(result, expected_result)


def test_update_s_i():
    from parameters_update_likelihood_terms import update_s_i
    from scipy import stats

    z = 1
    v_i_new = np.array([1.0, 4.0, 2.0])
    v_old = np.array([2.0, 1.0, 3.0])
    m_i_new = np.array([3.0, 5.0, 4.0])
    mu_old = np.array([4.0, 1.0, 2.0])

    result = update_s_i(z, v_i_new, v_old, m_i_new, mu_old, 1.0)
    expected_result = stats.norm.cdf(z) * np.sqrt(3) * 5 * np.exp(1.0 / 6.0 + 8.0 / 5.0 + 2.0 / 5.0) / (2.0*np.sqrt(2.0))

    np.testing.assert_almost_equal(result, expected_result)


def test_calc_mu_old():
    from parameters_update_likelihood_terms import calc_mu_old

    mu = np.array([1.0, 2.0, 3.0])
    v_old = np.array([2.0, 3.0, 4.0])
    v_i = np.array([3.0, 1.0, 2.0])
    m_i = np.array([2.0, 4.0, 1.0])

    result = calc_mu_old(mu, v_old, v_i, m_i)
    expected_result = np.array([1 / 3.0, -4.0, 7.0])

    np.testing.assert_almost_equal(result, expected_result)


def test_calc_alpha_i():
    from parameters_update_likelihood_terms import calc_alpha_i
    from scipy import stats

    x_i = np.array([1.0, 2.0, 3.0])
    v_old = np.array([2.0, 1.0, 0.0])
    z = 1.0

    result = calc_alpha_i(x_i, v_old, z, 1.0)
    expected_result = (1.0 / np.sqrt(7.0)) * stats.norm.pdf(z) / stats.norm.cdf(z)

    np.testing.assert_array_equal(result, expected_result)


def test_calc_z():
    from parameters_update_likelihood_terms import calc_z

    x_i = np.array([1.0, 2.0, 3.0])
    mu_old = np.array([4.0, 5.0, 1.0])
    v_old = np.array([2.0, 3.0, 4.0])

    result = calc_z(x_i, mu_old, v_old)
    expected_result = 17.0 / np.sqrt(51.0)

    np.testing.assert_array_equal(result, expected_result)