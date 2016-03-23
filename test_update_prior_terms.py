import numpy as np


def test_update_mu_i():
    from parameters_update_prior_terms import update_mu

    mu_i_old = 3.0
    c_1 = 2.0
    v_i_old = 4.0
    result = update_mu(mu_i_old, c_1, v_i_old)
    expected_result = 11.0
    assert result == expected_result

    mu_i_old = np.array([1.0, 2.0, 3.0])
    c_1 = np.array([1.0, 2.0, 2.0])
    v_i_old = np.array([2.0, 4.0, 1.0])
    result = update_mu(mu_i_old, c_1, v_i_old)
    expected_result = np.array([3.0, 10.0, 5.0])
    np.testing.assert_array_equal(result, expected_result)


def test_update_nu():
    from parameters_update_prior_terms import update_nu
    nu_old = np.array([2.0, 4.0, 1.0])
    c_3 = np.array([1.0, 2.0, 2.0])
    result = update_nu(nu_old, c_3)
    expected_result = np.array([-2.0, -28.0, -1.0])
    np.testing.assert_array_equal(result, expected_result)


def test_update_p_i():
    from parameters_update_prior_terms import update_p_i

    p_i_old = np.array([1.0, 2.0, 3.0])
    G_1 = np.array([2.0, 2.0, 1.0])
    G_0 = np.array([1.0, 3.0, 2.0])
    result = update_p_i(p_i_old, G_1, G_0)
    expected_result = np.array([1.0, 4.0, -3.0])
    np.testing.assert_array_equal(result, expected_result)


def test_update_v_i():
    from parameters_update_prior_terms import update_v_i

    c_3 = np.array([1.0, 2.0, 3.0])
    nu_old = np.array([2.0, 4.0, 2.0 / 3.0])
    result = update_v_i(c_3, nu_old)
    expected_result = np.array([-1.0, -3.5, -1.0 / 3.0])
    np.testing.assert_array_equal(result, expected_result)


def test_update_m_i():
    from parameters_update_prior_terms import update_m_i

    mu_old = np.array([1.0, 2.0, 3.0])
    v_i = np.array([2.0, 3.0, 1.0])
    c_1 = np.array([0.1, 0.2, 0.5])
    nu_old = np.array([0.0, 1.0, 2.0])
    result = update_m_i(mu_old, c_1, v_i, nu_old)
    expected_result = np.array([1.2, 2.8, 4.5])
    np.testing.assert_array_equal(result, expected_result)


def test_update_a():
    from parameters_update_prior_terms import update_a

    p_new = np.array([1.0, 1.0, 1.0])
    p_old = np.array([10.0, 2.0, 4.0])
    result = update_a(p_new, p_old)
    expected_result = np.array([0.1, 0.5, 0.25])
    np.testing.assert_array_equal(result, expected_result)


def test_update_b():
    from parameters_update_prior_terms import update_b

    p_new = np.array([0.0, 0.0, 0.0])
    p_old = np.array([-9.0, -1.0, -3.0])
    result = update_b(p_new, p_old)
    expected_result = np.array([0.1, 0.5, 0.25])
    np.testing.assert_array_equal(result, expected_result)


def test_update_s():
    from parameters_update_prior_terms import update_s

    Z = np.array([1.0, 2.0, 3.0])
    nu_old = np.array([2.0, 1.0, 3.0])
    v = np.array([1.0, 1.0, 1.0])
    c1 = np.array([4.0, 2.0, 5.0])
    c3 = np.array([2.0, 1.0, 1.0])
    result = update_s(Z, nu_old, v, c1, c3)
    expected_result = np.array([np.sqrt(3) * np.exp(4), 2 * np.sqrt(2) * np.exp(2), 6 * np.exp(12.5)])
    np.testing.assert_array_equal(result, expected_result)


def test_calc_Z():
    from parameters_update_prior_terms import calc_Z

    p_old = np.array([1.0, 2.0, 2.0])
    G1 = np.array([2.0, 1.0, 1.0])
    G0 = np.array([3.0, 2.0, 1.0])
    result = calc_Z(p_old, G1, G0)
    expected_result = np.array([2.0, 0.0, 1.0])
    np.testing.assert_array_equal(result, expected_result)


def test_calc_c1():
    from parameters_update_prior_terms import calc_c1

    Z = np.array([0.5, 0.25, 1.0])
    p_old = np.array([1.0, 2.0, 3.0])
    G1 = np.array([2.0, 2.0, 4.0])
    mu_old = np.array([3.0, 1.0, 1.0])
    sigma1 = np.array([1.0, 2.0, 1.0])
    nu_old = np.array([2.0, 2.0, 2.0])
    G0 = np.array([3.0, 1.0, 2.0])
    sigma0 = np.array([2.0, 2.0, 1.0])

    result = calc_c1(Z, p_old, G1, mu_old, nu_old, sigma1, G0, sigma0)
    expected_result = np.array([-4.0, -2.0, -8.0 / 3.0])
    np.testing.assert_almost_equal(result, expected_result)


def test_calc_c2():
    from parameters_update_prior_terms import calc_c2

    Z = np.array([0.5, 0.25, 1.0])
    p_old = np.array([1.0, 2.0, 3.0])
    G1 = np.array([2.0, 2.0, 4.0])
    mu_old = np.array([3.0, 1.0, 1.0])
    sigma1 = np.array([1.0, 2.0, 1.0])
    nu_old = np.array([2.0, 2.0, 2.0])
    G0 = np.array([3.0, 1.0, 2.0])
    sigma0 = np.array([2.0, 2.0, 1.0])

    result = calc_c2(Z, p_old, G1, mu_old, nu_old, sigma1, G0, sigma0)
    expected_result = np.array([4.0 / 3.0, -5.0 / 6.0, -8.0 / 9.0])
    np.testing.assert_almost_equal(result, expected_result)


def test_calc_c3():
    from parameters_update_prior_terms import calc_c3

    c1 = np.array([1.0, 2.0, 3.0])
    c2 = np.array([2.0, 3.0, 1.0])
    result = calc_c3(c1, c2)
    expected_result = np.array([-3.0, -2.0, 7.0])
    np.testing.assert_array_equal(result, expected_result)


def test_calc_p_old():
    from parameters_update_prior_terms import calc_p_old

    p = np.array([1.0, 2.0, 3.0])
    a = np.array([2.0, 3.0, 1.0])
    b = np.array([3.0, 2.0, 1.0])
    result = calc_p_old(p, a, b)
    expected_result = np.array([1.0, 4.0, 3.0])
    np.testing.assert_almost_equal(result, expected_result)
