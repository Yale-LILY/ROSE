import numpy as np
from multiprocessing import Pool
import numpy as np
from itertools import combinations
from tabulate import tabulate

def _bootstraping(data, sample_size=-1, repetitions=1e5):
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


def bootstraping(diff, sample_size=-1, repetitions=1e5, num_workers=32):
    # perform bootstraping test, multi process
    with Pool(processes=num_workers) as pool:
        results = [pool.apply_async(_bootstraping, args=(diff, sample_size, repetitions / num_workers)) for _ in range(num_workers)]
        cnts = [res.get() for res in results]
    return sum(cnts) / repetitions


def power_analysis(a, b, sample_nums=None, trial_num=1000, num_workers=32, verbose=False):
    """
    perform power analysis w.r.t. the difference between two systems
    a: [num_samples]
    b: [num_samples]
    sample_nums: list of sample sizes
    trial_num: number of trials
    num_workers: number of processes
    """
    if sample_nums is None:
        sample_nums = [10, 50, 100, 200, 300, 500, 700, 1000, 10000]
    if a.mean() > b.mean():
        sys1 = a
        sys2 = b
    else:
        sys1 = b
        sys2 = a
    result = dict()
    score_diff = sys1 - sys2
    print("score difference", score_diff.mean())
    for num in sample_nums:
        # loop through different sample sizes
        cnt = 0
        for i in range(trial_num):
            if i % 10 == 0 and verbose:
                print(f"sample num {num}: trail num {i}")
            idx = np.random.choice(score_diff.shape[0], num)  # sample num examples
            p = bootstraping(score_diff[idx]) # run bootstraping test
            if p < 0.05:  # get significant result
                cnt += 1
        result[num] = cnt / trial_num
    if verbose:
        print("powers")
        for x in result:
            print(f"{x}: {result[x]}")
    return result


def power_analysis_dataset(system_scores, sample_nums=None, trial_num=1000, num_workers=32, verbose=False):
    """
    perform power analysis over all system pairs
    system_scores: {system_name: [num_samples]}
    the results are grouped into five bins according to the difference between the two systems
    """
    if sample_nums is None:
        sample_nums = [10, 50, 100, 200, 300, 500, 700, 1000, 10000]
    systems = system_scores.keys()
    systems = sorted(list(systems))
    all_system_pairs = list(combinations(systems, 2))
    result = []
    for x in all_system_pairs:
        system1, system2 = x[0], x[1]
        print("{} vs {}".format(system1, system2))
        system1_scores = np.array(system_scores[system1])
        system2_scores = np.array(system_scores[system2])
        power = power_analysis(system1_scores, system2_scores, num_workers=num_workers, sample_nums=sample_nums, trial_num=trial_num, verbose=verbose)
        result.append({
            "system1": system1,
            "system2": system2,
            "system1_score": system1_scores.mean(),
            "system2_score": system2_scores.mean(),
            "score_diff": abs(system1_scores.mean() - system2_scores.mean()),
            "power": power
        })
    # calculate the average power over all system pairs into six bins
    diffs = [x["score_diff"] for x in result]
    diffs = np.array(diffs)
    # split the system pairs into five bins
    pencentile = np.percentile(diffs, [0, 20, 40, 60, 80, 100])
    bins = []
    for i in range(len(pencentile) - 1):
        bins.append([x for x in result if x["score_diff"] >= pencentile[i] and x["score_diff"] < pencentile[i+1]])
    outputs = [["sample_num"] + [str(x) for x in range(len(bins))]]
    # calculate the average power over each bin
    for x in sample_nums:
        new_row = [x]
        for i in range(len(bins)):
            power = np.mean([y["power"][x] for y in bins[i]])
            new_row.append(f"{power:.4f}")
        outputs.append(new_row)
    print(tabulate(outputs, headers="firstrow", tablefmt="github"))


            
