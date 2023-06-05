from stat_test import bootstrap_system, bootstrap_summ, permutation_system, permutation_summ, confidence_interval
from correlation import correlation_system, correlation_summ, _cal_kendall
from power_analysis import power_analysis, power_analysis_dataset
from datasets import load_dataset
import json
import numpy as np


def example_significance():
    """
    example of performing significance test w.r.t. 
    two different automatic metrics (ROUGE-1 and ROUGE-2) on CNNDM test set
    """
    # load human annotations
    cnndm_test = load_dataset("Salesforce/rose", "cnndm_test")["data"]
    systems = cnndm_test[0]["annotations"].keys()
    systems = sorted(list(systems))
    acu_scores = np.array([[x["annotations"][system]["acu"] for x in cnndm_test] for system in systems])
    # load automatic metrics
    with open("./computed_metrics/cnndm_test.jsonl") as f:
        metric_scores = [json.loads(line) for line in f]
    # get ROUGE-1 and ROUGE-2 recall scores
    rouge1r = np.array([[x["metric_scores"]["rouge1r"][system] for x in metric_scores] for system in systems])
    rouge2r = np.array([[x["metric_scores"]["rouge2r"][system] for x in metric_scores] for system in systems])

    # compute kendall tau correlation
    ## system level
    rouge1r_corr = correlation_system(acu_scores, rouge1r, _cal_kendall)
    rouge2r_corr = correlation_system(acu_scores, rouge2r, _cal_kendall)
    print("System level correlation: ROUGE-1: {:.4f}, ROUGE-2: {:.4f}".format(rouge1r_corr, rouge2r_corr))
    ## summary level
    rouge1r_corr_summ = correlation_summ(acu_scores, rouge1r, _cal_kendall)
    rouge2r_corr_summ = correlation_summ(acu_scores, rouge2r, _cal_kendall)
    print("Summary level correlation: ROUGE-1: {:.4f}, ROUGE-2: {:.4f}".format(rouge1r_corr_summ, rouge2r_corr_summ))

    # compute significance w.r.t. the performance difference between ROUGE-1 and ROUGE-2
    ## bootstrap test
    ### system level
    significance, diff = bootstrap_system(acu_scores, rouge1r, rouge2r, "kendall", num_workers=16)
    print("System level bootstrap test, correlation difference: {:.4f}, p-value: {:.4f}".format(diff, significance))
    ### summary level
    significance, diff = bootstrap_summ(acu_scores, rouge1r, rouge2r, "kendall", num_workers=16)
    print("Summary level bootstrap test, correlation difference: {:.4f}, p-value: {:.4f}".format(diff, significance))

    ## permutation test
    ### system level
    significance, diff = permutation_system(acu_scores, rouge1r, rouge2r, "kendall", num_workers=16)
    print("System level permutation test, correlation difference: {:.4f}, p-value: {:.4f}".format(diff, significance))
    ### summary level
    significance, diff = permutation_summ(acu_scores, rouge1r, rouge2r, "kendall", num_workers=16)
    print("Summary level permutation test, correlation difference: {:.4f}, p-value: {:.4f}".format(diff, significance))

    
def example_confidence_interval():
    """
    example of computing confidence interval of the metric correlations
    """
    # load human annotations
    cnndm_test = load_dataset("Salesforce/rose", "cnndm_test")["data"]
    systems = cnndm_test[0]["annotations"].keys()
    systems = sorted(list(systems))
    acu_scores = np.array([[x["annotations"][system]["acu"] for x in cnndm_test] for system in systems])
    # load automatic metrics
    with open("./computed_metrics/cnndm_test.jsonl") as f:
        metric_scores = [json.loads(line) for line in f]
    # get ROUGE-1 recall score
    rouge1r = np.array([[x["metric_scores"]["rouge1r"][system] for x in metric_scores] for system in systems])
   
    # compute kendall tau correlation
    ## system level
    rouge1r_corr = correlation_system(acu_scores, rouge1r, _cal_kendall)
    low, high = confidence_interval("system", acu_scores, rouge1r, "kendall", num_workers=16)
    print("System level correlation: ROUGE-1: {:.4f}, confidence interval: [{:.4f}, {:.4f}]".format(rouge1r_corr, low, high))
    ## summary level
    rouge1r_corr_summ = correlation_summ(acu_scores, rouge1r, _cal_kendall)
    low, high = confidence_interval("summ", acu_scores, rouge1r, "kendall", num_workers=16)
    print("Summary level correlation: ROUGE-1: {:.4f}, confidence interval: [{:.4f}, {:.4f}]".format(rouge1r_corr_summ, low, high))


def example_power_analysis():
    """
    example of performing power analysis w.r.t.
    the difference between two systems
    """
    # load human annotations
    cnndm_test = load_dataset("Salesforce/rose", "cnndm_test")["data"]
    # compute the power of the difference between BART and Brio
    bart_acu_scores = np.array([x["annotations"]["bart"]["acu"] for x in cnndm_test])
    brio_acu_scores = np.array([x["annotations"]["brio"]["acu"] for x in cnndm_test])
    print("BART vs Brio")
    power_analysis(bart_acu_scores, brio_acu_scores, num_workers=16, verbose=True)
    # compute the power over all system pairs
    systems = cnndm_test[0]["annotations"].keys()
    system_scores = {system: np.array([x["annotations"][system]["acu"] for x in cnndm_test]) for system in systems}
    power_analysis_dataset(system_scores, num_workers=16, verbose=True)

    

if __name__ == "__main__":
    example_significance()
    example_confidence_interval()
    example_power_analysis()
