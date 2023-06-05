import numpy as np
from multiprocessing import Pool
import numpy as np
from correlation import _cal_kendall, _cal_spearman, _cal_pearson, correlation_system, correlation_summ_values

def bootstraping(fn, refs, sys1, sys2, corr_func, sample_size=-1, repetitions=1e5, num_workers=32):
    # perform bootstraping test, multi process
    with Pool(processes=num_workers) as pool:
        results = [pool.apply_async(fn, args=(refs, sys1, sys2, corr_func, sample_size, repetitions / num_workers)) for _ in range(num_workers)]
        cnts = [res.get() for res in results]
    return sum(cnts) / repetitions

### bootstrap tests for correlation significance values ###

## bootstraping significance values, system level
def _bootstraping_system(refs, sys1, sys2, corr_func, sample_size=-1, repetitions=1e5):
    # perform bootstraping test, single process
    np.random.seed()
    if sample_size < 0:
        sample_size = refs.shape[1]
    cnt = 0
    for i in range(int(repetitions)):
        idx = np.random.choice(refs.shape[1], sample_size)  # sample examples
        refs_sample = refs[:, idx]
        sys1_sample = sys1[:, idx]
        sys2_sample = sys2[:, idx]
        corr1 = correlation_system(refs_sample, sys1_sample, corr_func)
        corr2 = correlation_system(refs_sample, sys2_sample, corr_func)
        now_delta = corr1 - corr2 # calculate delta (difference)
        if now_delta < 0:
            cnt += 1
    return cnt

def bootstrap_system(refs, sys1, sys2, corr_func, num_workers=1, verbose=False):
    """
    refs: [num_system, num_summ] array
    sys1, sys2: [num_system, num_summ] array
    corr_func: name of correlation function
    num_workers: number of processes
    """
    if corr_func == "pearson":
        corr_func = _cal_pearson
    elif corr_func == "spearman":
        corr_func = _cal_spearman
    elif corr_func == "kendall":
        corr_func = _cal_kendall
    else:
        raise NotImplementedError
    corr_sys1 = correlation_system(refs, sys1, corr_func)
    corr_sys2 = correlation_system(refs, sys2, corr_func)   
    diff = corr_sys1 - corr_sys2
    if verbose:
        print("sys1 correlation:", corr_sys1)
        print("sys2 correlation:", corr_sys2)
        print("correlation difference:", diff)
    if diff < 0:
        sys1, sys2 = sys2, sys1
    p = bootstraping(_bootstraping_system, refs, sys1, sys2, corr_func, num_workers=num_workers)
    if verbose:
        print("p-value:", p)
    return p, diff


## bootstraping significance values, summary level
def _bootstraping_summ_test(data, sample_size=-1, repetitions=1e5):
    # perform bootstraping test, single process
    np.random.seed()
    if sample_size < 0:
        sample_size = data.shape[0]
    cnt = 0
    for i in range(int(repetitions)):
        idx = np.random.choice(data.shape[0], sample_size)  # sample examples
        samples = data[idx]
        now_delta = samples.mean()  # calculate delta (difference)
        if now_delta < 0:
            cnt += 1
    return cnt

def _bootstraping_summ(diff, sample_size=-1, repetitions=1e5, num_workers=32):
    # perform bootstraping test, multi process
    with Pool(processes=num_workers) as pool:
        results = [pool.apply_async(_bootstraping_summ_test, args=(diff, sample_size, repetitions / num_workers)) for _ in range(num_workers)]
        cnts = [res.get() for res in results]
    return sum(cnts) / repetitions


def bootstrap_summ(refs, sys1, sys2, corr_func, num_workers, verbose=False):
    """
    refs: [num_system, num_summ] array
    sys1, sys2: [num_system, num_summ] array
    corr_func: name of correlation function
    num_workers: number of processes
    """
    if corr_func == "pearson":
        corr_func = _cal_pearson
    elif corr_func == "spearman":
        corr_func = _cal_spearman
    elif corr_func == "kendall":
        corr_func = _cal_kendall
    else:
        raise NotImplementedError
    corr_sys1, values1 = correlation_summ_values(refs, sys1, corr_func)
    corr_sys2, values2 = correlation_summ_values(refs, sys2, corr_func)   
    diff = corr_sys1 - corr_sys2
    if verbose:
        print("sys1 correlation:", corr_sys1)
        print("sys2 correlation:", corr_sys2)
        print("correlation difference:", diff)
    if diff < 0:
        sys1, sys2 = sys2, sys1
        values1, values2 = values2, values1
    diffs = values1 - values2
    # print(diffs)
    p = _bootstraping_summ(diffs, num_workers=num_workers)
    if verbose:
        print("p-value:", p)
    return p, diff

### permutation tests for correlation significance values ###

## permutation test, system level
def _permutation_system(refs, sys1, sys2, corr_func, sample_size=-1, repetitions=1e5):
    # perform permutation test, single process
    np.random.seed()
    if sample_size < 0:
        sample_size = refs.shape[1]
    cnt = 0
    corr_sys1 = correlation_system(refs, sys1, corr_func)
    corr_sys2 = correlation_system(refs, sys2, corr_func)   
    delta = corr_sys1 - corr_sys2
    for i in range(int(repetitions)):
        idx = np.random.random(refs.shape[1]) < 0.5
        sys1_sample = np.copy(sys1)
        sys2_sample = np.copy(sys2)
        sys2_sample[:, idx] = sys1[:, idx]
        sys1_sample[:, idx] = sys2[:, idx]
        corr1 = correlation_system(refs, sys1_sample, corr_func)
        corr2 = correlation_system(refs, sys2_sample, corr_func)
        now_delta = corr1 - corr2  # calculate delta (difference)
        if now_delta > delta:
            cnt += 1
    return cnt

def permutation_system(refs, sys1, sys2, corr_func, num_workers, verbose=False):
    """
    refs: [num_system, num_summ] array
    sys1, sys2: [num_system, num_summ] array
    corr_func: name of correlation function
    num_workers: number of processes
    """
    if corr_func == "pearson":
        corr_func = _cal_pearson
    elif corr_func == "spearman":
        corr_func = _cal_spearman
    elif corr_func == "kendall":
        corr_func = _cal_kendall
    else:
        raise NotImplementedError
    corr_sys1 = correlation_system(refs, sys1, corr_func)
    corr_sys2 = correlation_system(refs, sys2, corr_func)   
    diff = corr_sys1 - corr_sys2
    if verbose:
        print("sys1 correlation:", corr_sys1)
        print("sys2 correlation:", corr_sys2)
        print("correlation difference:", diff)
    if diff < 0:
        sys1, sys2 = sys2, sys1
    p = bootstraping(_permutation_system, refs, sys1, sys2, corr_func, num_workers=num_workers)
    if verbose:
        print("p-value:", p)
    return p, diff


## bootstraping significance values, summary level
def _permutation_summ_test(corr_sys1, corr_sys2, sample_size=-1, repetitions=1e5):
    # perform bootstraping test, single process
    np.random.seed()
    if sample_size < 0:
        sample_size = corr_sys1.shape[0]
    cnt = 0
    delta = (corr_sys1 - corr_sys2).mean()
    for i in range(int(repetitions)):
        idx = np.random.random(sample_size) < 0.5
        sys1_sample = np.copy(corr_sys1)
        sys2_sample = np.copy(corr_sys2)
        sys2_sample[idx] = corr_sys1[idx]
        sys1_sample[idx] = corr_sys2[idx]
        now_delta = (sys1_sample - sys2_sample).mean() # calculate delta (difference)
        if now_delta > delta:
            cnt += 1
    return cnt

def _permutation_summ(x, y, sample_size=-1, repetitions=1e5, num_workers=32):
    # perform bootstraping test, multi process
    with Pool(processes=num_workers) as pool:
        results = [pool.apply_async(_permutation_summ_test, args=(x, y, sample_size, repetitions / num_workers)) for _ in range(num_workers)]
        cnts = [res.get() for res in results]
    return sum(cnts) / repetitions


def permutation_summ(refs, sys1, sys2, corr_func, num_workers, verbose=False):
    """
    refs: [num_system, num_summ] array
    sys1, sys2: [num_system, num_summ] array
    corr_func: name of correlation function
    num_workers: number of processes
    """
    if corr_func == "pearson":
        corr_func = _cal_pearson
    elif corr_func == "spearman":
        corr_func = _cal_spearman
    elif corr_func == "kendall":
        corr_func = _cal_kendall
    corr_sys1, values1 = correlation_summ_values(refs, sys1, corr_func)
    corr_sys2, values2 = correlation_summ_values(refs, sys2, corr_func)   
    diff = corr_sys1 - corr_sys2
    if verbose:
        print("sys1 correlation:", corr_sys1)
        print("sys2 correlation:", corr_sys2)
        print("correlation difference:", diff)
    if diff < 0:
        sys1, sys2 = sys2, sys1
        values1, values2 = values2, values1
        corr_sys1, corr_sys2 = corr_sys2, corr_sys1
    p = _permutation_summ(values1, values2, num_workers=num_workers)
    if verbose:
        print("p-value:", p)
    return p, diff

### confidence interval ###

## bootstraping confidence intervals
def _confidence_system(refs, cands, corr_func, sample_size=-1, repetitions=1e5):
    # perform bootstraping test, single process
    np.random.seed()
    if sample_size < 0:
        sample_size = refs.shape[1]
    results = []
    for i in range(int(repetitions)):
        idx = np.random.choice(refs.shape[1], sample_size)  # sample examples
        refs_sample = refs[:, idx]
        cands_sample = cands[:, idx]
        corr = correlation_system(refs_sample, cands_sample, corr_func)
        results.append(corr)
    return results


def _confidence_summ(refs, cands, corr_func, sample_size=-1, repetitions=1e5):
    # perform bootstraping test, single process
    np.random.seed()
    if sample_size < 0:
        sample_size = refs.shape[1]
    results = []
    _, values = correlation_summ_values(refs, cands, corr_func)
    for i in range(int(repetitions)):
        idx = np.random.choice(refs.shape[1], sample_size)  # sample examples
        corr = values[idx].mean()
        results.append(corr)
    return results


def confidence_interval(level, refs, cands, corr_func, sample_size=-1, repetitions=1e5, num_workers=1):
    """
    calculate confidence interval (2.5%, 97.5%) for system level or summary level
    level: "system" or "summ"
    refs: [num_system, num_summ] array
    cands: [num_system, num_summ] array
    corr_func: name of correlation function
    num_workers: number of processes
    """
    if corr_func == "pearson":
        corr_func = _cal_pearson
    elif corr_func == "spearman":
        corr_func = _cal_spearman
    elif corr_func == "kendall":
        corr_func = _cal_kendall
    else:
        raise NotImplementedError
    if level == "system":
        fn = _confidence_system
    elif level == "summ":
        fn = _confidence_summ
    else:
        raise NotImplementedError
    with Pool(processes=num_workers) as pool:
        results = [pool.apply_async(fn, args=(refs, cands, corr_func, sample_size, repetitions / num_workers)) for _ in range(num_workers)]
        results = [res.get() for res in results]
    _results = []
    for res in results:
        _results.extend(res)
    results = _results
    # compute confidence interval
    results = np.array(results)
    head = np.percentile(results, 2.5)
    tail = np.percentile(results, 97.5)
    print("confidence interval:", head, tail)
    return head, tail