import numpy as np


def test_update_p_3():
    from scipy.special import logit
    from update_params_linear_regression import update_p_3

    N = 4
    p0 = 0.3

    res = update_p_3(p0, N)
    expected_res = np.array([logit(p0), logit(p0), logit(p0), logit(p0)])

    np.testing.assert_array_equal(res, expected_res)


def test_update_v_2():
    from update_params_linear_regression import update_v_2

    a = np.array([1.0, 2.0, 2.0])
    b = np.array([0.0, 1.0, 0.0])
    v_1 = np.array([0.0, 2.0, 1.0])

    res = update_v_2(a, b, v_1)
    expected_res = np.array([1.0, - 5.0 / 3.0, -0.75])

    np.testing.assert_array_equal(res, expected_res)


def test_update_m_2():
    from update_params_linear_regression import update_m_2

    m_1 = np.array([1.0, 2.0, 3.0])
    a = np.array([2.0, 3.0, 4.0])
    v_2_new = np.array([3.0, 2.0, 4.0])
    v_1 = np.array([3.0, 1.0, 1.0])

    res = update_m_2(m_1, a, v_2_new, v_1)
    expected_res = np.array([-11, -7, -17])

    np.testing.assert_array_equal(res, expected_res)


def test_update_p_2():
    from update_params_linear_regression import update_p_2

    v_1 = np.array([1.0, 2.0, 3.0])
    v_s = np.array([0.0, 1.0, 1.0])
    m_1 = np.array([1.0, 0.0, 2.0])

    res = update_p_2(v_1, v_s, m_1)
    expected_res = np.array([0,
                             0.5 * (np.log(2) - np.log(3)),
                             0.5 * (np.log(3) - np.log(4)) + 1.0 / 6.0])

    np.testing.assert_array_almost_equal(res, expected_res)


def test_calc_a():
    from scipy.stats import logistic
    from update_params_linear_regression import calc_a

    p_2_new = np.array([1.0, 2.0, 3.0])
    p_3 = np.array([2.0, 1.0, 1.0])
    m_1 = np.array([2.0, 4.0, 3.0])
    v_1 = np.array([1.0, 1.0, 1.0])
    v_s = np.array([3.0, 1.0, 2.0])

    res = calc_a(p_2_new, p_3, m_1, v_1, v_s)
    expected_res = np.array([0.5, 2.0, 1.0]) * logistic.cdf(np.array([3.0, 3.0, 4.0])) + \
                   np.array([2.0, 4.0, 3.0]) * logistic.cdf(np.array([-3.0, -3.0, -4.0]))

    np.testing.assert_array_almost_equal(res, expected_res)


def test_calc_b():
    from scipy.stats import logistic
    from update_params_linear_regression import calc_b

    p_2_new = np.array([1.0, 2.0, 3.0])
    p_3 = np.array([2.0, 1.0, 1.0])
    m_1 = np.array([2.0, 4.0, 3.0])
    v_1 = np.array([1.0, 1.0, 1.0])
    v_s = np.array([3.0, 1.0, 2.0])

    res = calc_b(p_2_new, p_3, m_1, v_1, v_s)
    expected_res = np.array([0.0, 3.5, 2.0 / 3.0]) * logistic.cdf(np.array([3.0, 3.0, 4.0])) + \
                   np.array([3.0, 15.0, 8.0]) * logistic.cdf(np.array([-3.0, -3.0, -4.0]))

    np.testing.assert_array_almost_equal(res, expected_res)


def test_calc_V():
    from update_params_linear_regression import calc_V

    sigma0 = 2.0
    V_2 = np.diag(np.array([1.0, 2.0, 4.0]))
    XT_X = np.array([[1.0, 2.0, 2.0], [1.0, 2.0, 1.0], [4.0, 4.0, 4.0]])

    res = calc_V(V_2, sigma0, XT_X)
    expected_res = np.array([[1.18518519, -0.14814815, -0.44444444],
                            [-0.07407407,  1.25925926, -0.22222222],
                            [-0.88888889, -0.88888889,  1.33333333]])

    np.testing.assert_array_almost_equal(res, expected_res)


def test_calc_m():
    from update_params_linear_regression import calc_m

    V_2 = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    V = np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0], [2.0, 0.0, 1.0]])
    X = np.array([[1.0, 2.0, 1.0], [2.0, 3.0, 1.0]])
    y = np.array([1.0, -1.0])
    XT_y = np.dot(X.T, y)
    m_2 = np.array([1.0, 4.0, 3.0])
    sigma0 = 0.5

    res = calc_m(V, V_2, m_2, sigma0, XT_y)
    expected_res = np.array([-5.0, -1.0, -3.0])

    np.testing.assert_array_equal(res, expected_res)


def test_update_v_1():
    from update_params_linear_regression import update_v_1

    v_2 = np.array([1.0, 2.0, 4.0])
    V = np.array([[1.18518519, -0.14814815, -0.44444444],
                [-0.07407407,  1.25925926, -0.22222222],
                [-0.88888889, -0.88888889,  1.33333333]])

    res = update_v_1(v_2, V)
    expected_res = np.array([-6.39999986,  3.40000001,  1.99999999])

    np.testing.assert_array_almost_equal(res, expected_res)


def tes_update_m_1():
    from update_params_linear_regression import update_m_1

    v_2 = np.array([1.0, 2.0, 1.0])
    V = np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0], [2.0, 0.0, 1.0]])
    X = np.array([[1.0, 2.0, 1.0], [2.0, 3.0, 1.0]])
    y = np.array([1.0, -1.0])
    m_2 = np.array([1.0, 4.0, 3.0])
    sigma0 = 0.5
    v_1_new = np.array([-1.0 / 6.0, 2.0, 1.0 / 3.0])

    res = update_m_1(V, v_2, m_2, sigma0, X, y, v_1_new)
    expected_res = np.array([1.0, - 5.0, - 2.0])

    np.testing.assert_array_equal(res, expected_res)