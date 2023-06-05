import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import math

def _cal_pearson(x, y):
    v = pearsonr(x, y)[0]
    if np.isnan(v):
        return 0
    return v

def _cal_spearman(x, y):
    v = spearmanr(x, y)[0]
    if np.isnan(v):
        return 0
    return v

def _cal_kendall(x, y):
    v = kendalltau(x, y)[0]
    if np.isnan(v):
        return 0
    return v

def correlation_summ(refs, cands, corr_func):
    """
    summary level correlation
    refs: [num_system, num_summ] array
    cands: [num_system, num_summ] array
    corr_func: correlation_function
    """
    corr = 0
    assert refs.shape == cands.shape
    for i in range(refs.shape[1]):
        corr += corr_func(refs[:, i], cands[:, i])
    return corr / refs.shape[1]


def correlation_summ_values(refs, cands, corr_func):
    """
    summary level correlation, return the correlation for each summary
    refs: [num_system, num_summ] array
    cands: [num_system, num_summ] array
    corr_func: correlation_function
    """
    corr = 0
    assert refs.shape == cands.shape
    results = []
    for i in range(refs.shape[1]):
        _corr = corr_func(refs[:, i], cands[:, i])
        corr += _corr
        results.append(_corr)
    return corr / refs.shape[1], np.array(results)


def correlation_system(refs, cands, corr_func):
    """
    system level correlation
    refs: [num_system, num_summ] array
    cands: [num_system, num_summ] array
    corr_func: correlation_function
    """
    assert refs.shape == cands.shape
    ref = refs.mean(axis=1)
    cand = cands.mean(axis=1)
    return corr_func(ref, cand)

def modified_kendall_tau_system(humans, cands, pairs):
    """
    modified kendall tau, system level
    for each pair of systems, calculate the difference between human and system
    humans: {system_name: [num_summ]}
    cands: {system_name: [num_summ]}
    pairs: [(system_name, system_name)], pairs of systems
    """
    same, different = 0, 0
    for x in pairs:
        sys1, sys2 = x[0][0], x[1][0]
        if (humans[sys1].mean() - humans[sys2].mean()) * (cands[sys1].mean() - cands[sys2].mean()) > 0:
            same += 1
        else:
            different += 1
    return (same - different) / (same + different)

def modified_kendall_tau_summary(humans, cands, pairs):
    """
    modified kendall tau, summary level
    for each pair of systems, calculate the difference between human and system
    humans: {system_name: [num_summ]}
    cands: {system_name: [num_summ]}
    pairs: [(system_name, system_name)], pairs of systems
    """
    score = 0
    num = list(humans.values())[0].shape[0]
    for i in range(num):
        same, different = 0, 0
        tie1, tie2 = 0, 0
        for x in pairs:
            sys1, sys2 = x[0][0], x[1][0]
            if humans[sys1][i] == humans[sys2][i]:
                if cands[sys1][i] != cands[sys2][i]:
                    tie1 += 1
            elif cands[sys1][i] == cands[sys2][i]:
                if humans[sys1][i] != humans[sys2][i]:
                    tie2 += 1
            elif (humans[sys1][i] - humans[sys2][i]) * (cands[sys1][i] - cands[sys2][i]) > 0:
                same += 1
            else:
                different += 1
        score += (same - different) / (math.sqrt((same + different + tie1) * (same + different + tie2)) + 1e-10)
    return score / num