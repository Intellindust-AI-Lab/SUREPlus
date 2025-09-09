import pandas as pd
import os
import numpy as np
from pprint import pprint
from copy import deepcopy
import utils.metrics as  metrics


def get_curve(known, novel, method=None):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1
        
    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95, threshold

def cal_ood_metrics(known, novel, method=None):

    """
    Note that the convention here is that ID samples should be labelled '1' and OoD samples should be labelled '0'
    Computes standard OoD-detection metrics: mtypes = ['FPR' (FPR @ TPR 95), 'AUROC', 'DTERR', 'AUIN', 'AUOUT']
    """

    tp, fp, fpr_at_tpr95, threshold = get_curve(known, novel, method)
    # get_energy_curve
    results = dict()
    
    mtype = 'Threshold'
    results[mtype] = threshold

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    return results

def compute_oscr(x1, x2, pred, labels):

    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    return OSCR


def get_metric(id_labels, id_class_pred, id_ood_pred, ood_ood_pred, num_to_avg, logger=None):
    ood_aurocs, ood_fprs, accs = [], [], []
    fail_aurocs, fail_fprs, fail_aurcs = [], [], []
    
    for _ in range(num_to_avg):
        ood_results = get_single_metric(
            deepcopy(id_labels), deepcopy(id_class_pred), deepcopy(id_ood_pred), ood_ood_pred)
        ood_aurocs.append(ood_results['AUROC']); ood_fprs.append(ood_results['FPR']); accs.append(ood_results['ACC']);
        # fail_aurocs.append(failure_results['AUROC']); fail_fprs.append(failure_results['FPR']); fail_aurcs.append(failure_results['AURC']);
    ood_auroc = np.mean(ood_aurocs); ood_fpr = np.mean(ood_fprs); acc = np.mean(accs);
    # fail_auroc = np.mean(fail_aurocs); fail_fpr = np.mean(fail_fprs); fail_aurc = np.mean(fail_aurcs);
    
    logger.info('AUROC | FPR95 | ACC')
    if num_to_avg >= 5:
        logger.info('{:.2f} | {:.2f} | {:.2f}'.format(
            100*ood_auroc, 100*ood_fpr,  100*acc))
        logger.info('{:.2f} | {:.2f} | {:.2f} '.format(
            100*np.std(ood_aurocs), 100*np.std(ood_fprs),  100*np.std(accs)))
    else:
        logger.info('{:.2f} | {:.2f} | {:.2f}'.format(
            100*ood_auroc, 100*ood_fpr,  100*acc))
        
    return ood_fpr, ood_auroc, acc

        
def get_single_metric(id_labels, id_class_pred, id_ood_pred, ood_ood_pred):

    # get failure prediction metrics
    # correctness = (np.array(id_class_pred) == id_labels).astype(int)
    # aurc, eaurc = metrics.calc_aurc_eaurc(id_ood_pred, correctness)
    # auroc, aupr_success, aupr_err, fpr_in_tpr_95 = metrics.calc_fpr_aupr(id_ood_pred, correctness)
    # failure_results = {'AURC' : aurc, 'AUROC' : auroc, 'FPR' : fpr_in_tpr_95}

    # get ood metrics
    ood_results = cal_ood_metrics(id_ood_pred, ood_ood_pred)         
    ood_results['ACC'] = (id_labels == id_class_pred).mean()

    return ood_results


