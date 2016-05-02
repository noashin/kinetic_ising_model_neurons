import numpy as np
import scipy.stats


def test_r_square():
    from measurements import r_square

    b = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    a = np.array([[1.0, 2.0, 4.0], [1.0, 2.0, 3.0]])

    assert r_square(a,a) == 1.0
    assert r_square(a, b) == - 0.5


def test_corr_coef():
    from measurements import corr_coef

    a = np.array([[1.0, 6.0, -5.0, 0.0], [8.0, 1.0, 1.0, 0.0]])
    b = np.array([[1.0, 2.0, 0.0, -7.0], [-1.0, 5.0, 0.0, 1.0]])

    assert corr_coef(a, a) == 1.0
    assert corr_coef(a, b) == scipy.stats.pearsonr(a.flatten(), b.flatten())[0]


def test_zero_matching():
    from measurements import zero_matching

    a = np.array([[1.0, 0.0, 2.0, 0.0], [3.0, 2.0, 8.0, 8.0]])
    b = np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 7.0, 0.0, 0.0]])

    assert zero_matching(a, a) == 1.0
    assert zero_matching(a, b) == 0.0


def test_sign_matching():
    from measurements import sign_matching

    a = np.array([[1.0, 0.0, 2.0, -1.0], [-2.0, 3.0, 0.0, 0.0]])
    b = np.array([[-2.0, 1.0, 2.0, 0.0], [2.0, 1.0, 1.0, 1.0]])

    assert sign_matching(a, a) == 1.0
    assert sign_matching(a, b) == 0.5