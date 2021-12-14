import numpy as np

TOLERANCE = 1e-7


def h_PI(mdp, h=1, tolerance=TOLERANCE):
    if mdp.discount_factor < 0 or mdp.discount_factor >= 1:
        raise ValueError("h_PI() will not converge unless discount_factor ∈ [0, 1[")
    Vs = [mdp.V]
    i = 0
    while True:
        policy = mdp.utils.get_h_greedy_policy(Vs[-1], h)
        V = mdp.utils.policy_evaluation(policy)
        Vs.append(V)
        i += 1
        if np.allclose(Vs[-1], Vs[-2], tolerance): break
    return policy, V


def NC_hm_PI(mdp, h = 1, m = 1, tolerance = TOLERANCE, max_calls=None):
    Vs = [mdp.V]
    while True:
        policy, V = mdp.utils.get_NC_hm_greedy_policy(Vs[-1], h , m)
        Vs.append(V)
        if np.allclose(Vs[-1], Vs[-2], tolerance): break
        if max_calls is not None and mdp.n_calls >= max_calls: return policy, V
    return policy, V

def hm_PI(mdp, h = 1, m = 1, tolerance = TOLERANCE, max_calls=None):
    Vs = [mdp.V]
    while True:
        policy, V = mdp.utils.get_hm_greedy_policy(Vs[-1], h , m)
        Vs.append(V)
        if np.allclose(Vs[-1], Vs[-2], tolerance): break
        if max_calls is not None and mdp.n_calls >= max_calls: return policy, V
    return policy, V

def NC_h_lambda_PI(mdp, h=1, lda=1, error_range=None, tolerance=TOLERANCE, max_calls=None):
    Vs = [mdp.V]
    i = 0
    while True:
        policy, V = mdp.utils.get_NC_lambda_greedy_policy(Vs[-1], lda=lda, h=h, error_range=error_range)
        i += 1
        Vs.append(V)
        if np.allclose(Vs[-1], Vs[-2], tolerance): break
        if max_calls is not None and mdp.n_calls >= max_calls: return policy, V
    return policy, V


def h_lambda_PI(mdp, h=1, lda=1, error_range=None, tolerance=TOLERANCE, max_calls=None):
    Vs = [mdp.V]
    i = 0
    while True:
        policy, V = mdp.utils.get_lambda_greedy_policy(Vs[-1], lda=lda, h=h, error_range=error_range)
        Vs.append(V)
        i += 1
        if np.allclose(Vs[-1], Vs[-2], tolerance): break
        if max_calls is not None and mdp.n_calls >= max_calls: return policy, V
    return policy, V
