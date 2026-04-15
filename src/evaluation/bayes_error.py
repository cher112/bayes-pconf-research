"""
Instance-free Bayes error estimation.

Implements the estimator from Ishida et al. (ICLR 2023, Oral):

    R* = E_{x~p(x)}[min{eta(x), 1-eta(x)}]
       = 1/2 - 1/2 * E[|2*eta(x) - 1|]

where eta(x) = P(y=1|x) is the class-posterior probability. When we have
n calibrated soft labels eta_hat_i, we estimate R* by the sample mean:

    R*_hat = 1/n * sum_i min{eta_hat_i, 1 - eta_hat_i}

This estimator is:
- Unbiased under clean soft labels
- Instance-free: does not need the input features x
- Dimension-free: convergence rate does not depend on d
- Consistent with rate O_p(1/sqrt(n))

References:
    Ishida, T., Yamane, I., Charoenphakdee, N., Niu, G., Sugiyama, M. (2023).
    Is the performance of my deep network too good to be true? A direct
    approach to estimating the Bayes error in binary classification. ICLR.
    https://openreview.net/forum?id=FZdJQgy05rz
"""
import numpy as np


def estimate_rstar_binary(eta_hat):
    """
    Instance-free Bayes error estimator for binary classification.

    Args:
        eta_hat: (N,) calibrated class-posterior estimates in [0, 1]

    Returns:
        rstar: scalar Bayes error estimate in [0, 0.5]
    """
    eta_hat = np.asarray(eta_hat)
    return float(np.mean(np.minimum(eta_hat, 1.0 - eta_hat)))


def estimate_rstar_equivalent(eta_hat):
    """
    Alternative form: R* = 1/2 - 1/2 * E[|2*eta - 1|].

    Mathematically equivalent to estimate_rstar_binary; used in the
    original ICLR 2023 paper for its geometric interpretation.
    """
    eta_hat = np.asarray(eta_hat)
    return float(0.5 - 0.5 * np.mean(np.abs(2 * eta_hat - 1)))


def compute_ceiling(rstar):
    """Bayes ceiling = 1 - R*: maximum achievable accuracy."""
    return 1.0 - rstar


def estimate_rstar_pconf(r, class_prior):
    """
    Bayes error estimator from positive-confidence data only.

    From Theorem 4.1 of Ishida et al. (2023):
        R*_Pconf = pi_+ * (1 - E_+[max(0, 2 - 1/r(x))])

    where r(x) = P(y=+1|x) is the positive confidence on positive samples,
    and pi_+ is the class prior P(y=+1).

    Args:
        r: (N_+,) positive confidences for positive samples
        class_prior: scalar pi_+ = P(y=+1)

    Returns:
        rstar: scalar Bayes error estimate
    """
    r = np.asarray(r)
    term = np.maximum(0.0, 2.0 - 1.0 / r)
    return float(class_prior * (1.0 - np.mean(term)))
