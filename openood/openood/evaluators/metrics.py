import numpy as np
from sklearn import metrics
import torch

def compute_all_metrics(conf, label, pred):
    np.set_printoptions(precision=3)
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall)

    accuracy = acc(pred, label)

    results = [fpr, auroc, aupr_in, aupr_out, accuracy]

    return results


# accuracy
def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc


# fpr_recall
def fpr_recall(conf, label, tpr):
    gt = np.ones_like(label)
    gt[label == -1] = 0

    fpr_list, tpr_list, threshold_list = metrics.roc_curve(gt, conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr)]
    thresh = threshold_list[np.argmax(tpr_list >= tpr)]
    return fpr, thresh


# auc
def auc_and_fpr_recall(conf, label, tpr_th):
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(1 - ood_indicator, conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(ood_indicator, -conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr


# ccr_fpr
def ccr_fpr(conf, fpr, pred, label):
    ind_conf = conf[label != -1]
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    ood_conf = conf[label == -1]

    num_ind = len(ind_conf)
    num_ood = len(ood_conf)

    fp_num = int(np.ceil(fpr * num_ood))
    thresh = np.sort(ood_conf)[-fp_num]
    num_tp = np.sum((ind_conf > thresh) * (ind_pred == ind_label))
    ccr = num_tp / num_ind

    return ccr


def detection(ind_confidences,
              ood_confidences,
              n_iter=100000,
              return_data=False):
    # calculate the minimum detection error
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / n_iter

    best_error = 1.0
    best_delta = None
    all_thresholds = []
    all_errors = []
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        detection_error = (tpr + error2) / 2.0

        if return_data:
            all_thresholds.append(delta)
            all_errors.append(detection_error)

        if detection_error < best_error:
            best_error = np.minimum(best_error, detection_error)
            best_delta = delta

    if return_data:
        return best_error, best_delta, all_errors, all_thresholds
    else:
        return best_error, best_delta


def normalize_ood(id_ood_scores, ood_ood_scores):
    """Min-Max normalize OOD scores across ID+OOD samples."""
    all_ood = np.concatenate([id_ood_scores, ood_ood_scores])
    mn, mx = all_ood.min(), all_ood.max()
    denom = (mx - mn) + 1e-12
    return (id_ood_scores - mn) / denom, (ood_ood_scores - mn) / denom


def f1_on_selection(all_labels, all_preds):
    """
    Compute precision, TP, F1 for selected samples.
    Positive: "accepted & correct ID" samples.

    Returns:
        precision, tp, None
    """
    if len(all_labels) == 0:
        return 0.0, 0, None

    valid = all_labels != -1  # ID samples only
    correct = all_preds == all_labels
    tp = np.sum(valid & correct)  # Accepted & correct ID
    fp = len(all_labels) - tp  # Accepted but not TP (OOD or wrong ID)

    precision = tp / (tp + fp + 1e-12)
    return precision, tp, None


def compute_aurc(id_pred, id_conf, id_gt):
    """
    Compute AURC (Area Under the Risk-Coverage Curve)

    Args:
        id_pred: ndarray or tensor, model predicted classes (N,)
        id_conf: ndarray or tensor, predicted confidence scores (N,)
        id_gt:   ndarray or tensor, ground truth labels (N,)

    Returns:
        aurc: float
    """
    # Convert to numpy arrays if inputs are tensors
    if not isinstance(id_pred, np.ndarray):
        id_pred = id_pred.cpu().numpy()
    if not isinstance(id_conf, np.ndarray):
        id_conf = id_conf.cpu().numpy()
    if not isinstance(id_gt, np.ndarray):
        id_gt = id_gt.cpu().numpy()

    # Sort samples by confidence in descending order
    sorted_idx = np.argsort(-id_conf)
    sorted_pred = id_pred[sorted_idx]
    sorted_gt = id_gt[sorted_idx]

    # Determine whether each prediction is correct
    correct = (sorted_pred == sorted_gt).astype(np.int32)
    risk_list = []
    coverage_list = []

    n = len(correct)
    cum_error = 0
    for i in range(1, n + 1):
        coverage = i / n
        cum_error += (1 - correct[i - 1])  # accumulate errors
        risk = cum_error / i  # risk = errors / selected samples
        coverage_list.append(coverage)
        risk_list.append(risk)

    # Compute AURC using the trapezoidal rule
    aurc = np.trapz(risk_list, coverage_list)

    return aurc


def area_auc(x, y):
    """
    Compute Area Under Curve using trapezoidal rule,
    but when there are duplicate x values, only the minimal y is kept.

    Parameters
    ----------
    x : array-like
        coverage values (should be between 0~1)
    y : array-like
        risk values (should be between 0~1)

    Returns
    -------
    float
        area under the risk-coverage curve
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2:
        return 0.0

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    unique_x = []
    unique_y = []
    current_x = None
    current_min_y = None
    for xi, yi in zip(x, y):
        if current_x is None or xi != current_x:
            if current_x is not None:
                unique_x.append(current_x)
                unique_y.append(current_min_y)
            current_x = xi
            current_min_y = yi
        else:
            current_min_y = min(current_min_y, yi)
    unique_x.append(current_x)
    unique_y.append(current_min_y)

    unique_x = np.array(unique_x)
    unique_y = np.array(unique_y)

    area = np.trapz(unique_y, unique_x)
    return float(area)

def reduce_to_min_risk_curve(coverage_array, risk_array, num_bins=1000):
    """
    Reduce coverage-risk points by filling empty bins with the next available risk (right fill).
    Ensures dual AURC <= OOD AURC.

    Args:
        coverage_array: np.array, sorted coverage values [0,1]
        risk_array: np.array, corresponding risk values
        num_bins: number of equal-width bins
    Returns:
        bin_centers: np.array, bin centers
        filled_risks: np.array, risk values per bin after filling
    """
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    filled_risks = np.zeros_like(bin_centers)

    N = len(coverage_array)
    j = 0  # pointer for coverage_array

    # Go through each bin
    for i in range(len(bin_centers)):
        # Collect all coverage points in this bin
        bin_mask = (coverage_array >= bins[i]) & (coverage_array < bins[i+1])
        bin_risks = risk_array[bin_mask]

        if len(bin_risks) > 0:
            filled_risks[i] = bin_risks.min()  # take min risk within bin
        else:
            # fill from the next non-empty bin to the right
            k = i + 1
            while k < len(bin_centers) and not ((coverage_array >= bins[k]) & (coverage_array < bins[k+1])).any():
                k += 1
            if k < len(bin_centers):
                next_bin_mask = (coverage_array >= bins[k]) & (coverage_array < bins[k+1])
                next_bin_risks = risk_array[next_bin_mask]
                filled_risks[i] = next_bin_risks.min()
            else:
                # If no bin to the right, fallback to max risk
                filled_risks[i] = 1.0

    return bin_centers, filled_risks




def compute_f1_and_aurc(
    id_labels, id_preds, id_msp, id_ood,
    ood_msp, ood_ood, aurc_bins=100, num_bins=1000, eps=1e-12,
    device='cuda', batch_size_ood=200, batch_size_n=2000
):
    """
    GPU batched version of compute_f1_and_aurc
    Avoids OOM by splitting both over OOD thresholds and over samples N.
    Returns:
        max_f1_ood: best F1 when only OOD threshold is varied
        max_f1: best F1 when both MSP & OOD thresholds vary
        dual_aurc: area under dual risk-coverage curve
        aurc_ood: area under OOD-only risk-coverage curve
    """

    # Move to GPU
    id_labels = torch.tensor(id_labels, dtype=torch.int64, device=device)
    id_preds = torch.tensor(id_preds, dtype=torch.int64, device=device)
    id_msp = torch.tensor(id_msp, dtype=torch.float32, device=device)
    id_ood = torch.tensor(id_ood, dtype=torch.float32, device=device)
    ood_msp = torch.tensor(ood_msp, dtype=torch.float32, device=device)
    ood_ood = torch.tensor(ood_ood, dtype=torch.float32, device=device)

    total_id = len(id_labels)

    # Normalize OOD scores to [0,1]
    all_ood = torch.cat([id_ood, ood_ood])
    mn, mx = all_ood.min(), all_ood.max()
    denom = mx - mn + eps
    id_ood_n = (id_ood - mn) / denom
    ood_ood_n = (ood_ood - mn) / denom

    # Combine ID + OOD
    all_labels = torch.cat([id_labels, -1 * torch.ones_like(ood_msp, dtype=torch.int64)])
    all_preds = torch.cat([id_preds, -1 * torch.ones_like(ood_msp, dtype=torch.int64)])
    all_ood_n = torch.cat([id_ood_n, ood_ood_n])
    all_msp = torch.cat([id_msp, ood_msp])

    # -----------------------------
    # 1. Single-threshold F1 (OOD only)
    # -----------------------------
    th_ood = torch.linspace(0.0, 1.0, num_bins+1, device=device)
    max_f1_ood = 0.0
    coverage_list_ood = []
    sa_list_ood = []

    for t_ in th_ood:
        sel = all_ood_n >= t_
        sel_labels = all_labels[sel]
        sel_preds = all_preds[sel]

        coverage = (sel_labels != -1).sum().float() / total_id
        correct_id = ((sel_labels != -1) & (sel_labels == sel_preds)).sum().float()
        sa = correct_id / (sel.sum().float() + eps)
        coverage_list_ood.append(coverage.item())
        sa_list_ood.append(sa.item())

        tp = correct_id
        fp = ((sel_labels != -1).sum() - tp) + (sel_labels == -1).sum()
        precision = tp / (tp + fp + eps)
        recall = tp / (total_id + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        max_f1_ood = max(max_f1_ood, f1.item())

    coverage_array_ood = np.array(coverage_list_ood)
    sa_array_ood = np.array(sa_list_ood)
    cov_bins_ood, min_risks_ood = reduce_to_min_risk_curve(
        coverage_array_ood, 1 - sa_array_ood, aurc_bins
    )
    aurc_ood = float(np.trapz(min_risks_ood, cov_bins_ood))

    # -----------------------------
    # 2. Dual-threshold MSP + OOD (double batched)
    # -----------------------------
    th_ood = torch.linspace(0.0, 1.0, num_bins + 1, device=device)
    th_msp = torch.linspace(0.0, 1.0, num_bins + 1, device=device)

    N = all_labels.shape[0]
    B_ood = th_ood.shape[0]
    B_msp = th_msp.shape[0]

    # labels
    is_id = (all_labels != -1).float()
    correct_id = ((all_labels == all_preds) & (all_labels != -1)).float()

    all_f1, all_sa, all_cov = [], [], []

    for i in range(0, B_ood, batch_size_ood):
        th_ood_batch = th_ood[i:i+batch_size_ood]  # [B_ood_chunk]

        # Precompute OOD mask chunk
        ood_mask = (all_ood_n[:, None] >= th_ood_batch[None, :]).float()  # [N, B_ood_chunk]

        # Loop over N in mini-batches
        tp_chunk = torch.zeros((len(th_ood_batch), B_msp), device=device)
        total_sel_chunk = torch.zeros_like(tp_chunk)
        fp_chunk = torch.zeros_like(tp_chunk)
        sa_chunk = torch.zeros_like(tp_chunk)
        cov_chunk = torch.zeros_like(tp_chunk)

        for j in range(0, N, batch_size_n):
            idx = slice(j, j+batch_size_n)
            ood_mask_n = ood_mask[idx]               # [n, B_ood_chunk]
            msp_mask_n = (all_msp[idx, None] >= th_msp[None, :]).float()  # [n, B_msp]

            # [n, B_ood_chunk, B_msp]
            sel = ood_mask_n[:, :, None] * msp_mask_n[:, None, :]

            is_id_n = is_id[idx][:, None, None]
            correct_id_n = correct_id[idx][:, None, None]
            labels_n = all_labels[idx][:, None, None]

            tp_chunk += (correct_id_n * sel).sum(0)
            total_sel_chunk += sel.sum(0)
            fp_chunk += ((is_id_n * sel).sum(0) - (correct_id_n * sel).sum(0)) \
                        + (((labels_n == -1).float() * sel).sum(0))
            sa_chunk += (correct_id_n * sel).sum(0)
            cov_chunk += (is_id_n * sel).sum(0)

            del sel, is_id_n, correct_id_n, labels_n
            torch.cuda.empty_cache()

        precision = tp_chunk / (tp_chunk + fp_chunk + eps)
        recall = tp_chunk / (total_id + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        sa = sa_chunk / (total_sel_chunk + eps)
        coverage = cov_chunk / total_id

        all_f1.append(f1.detach().cpu())
        all_sa.append(sa.detach().cpu())
        all_cov.append(coverage.detach().cpu())

    f1 = torch.cat(all_f1, dim=0).numpy()
    sa = torch.cat(all_sa, dim=0).numpy()
    coverage = torch.cat(all_cov, dim=0).numpy()

    max_f1 = f1.max().item()
    coverage_array_dual = coverage.flatten()
    sa_array_dual = sa.flatten()
    risk_array_dual = 1 - sa_array_dual
    
    # Sort by coverage
    sort_idx = np.argsort(coverage_array_dual)
    coverage_array_dual_sorted = coverage_array_dual[sort_idx]
    risk_array_dual_sorted = risk_array_dual[sort_idx]
    cov_bins_dual, min_risks_dual = reduce_to_min_risk_curve(
        coverage_array_dual_sorted, risk_array_dual_sorted, aurc_bins
    )
    ds_aurc = float(np.trapz(min_risks_dual, cov_bins_dual))
    
    
    # #  ood threshold = 0
    # sa_array_dual_aligned = sa[0,:]  
    # coverage_array_dual_aligned = coverage[0, :]
    # cov_bins_dual, min_risks_dual = reduce_to_min_risk_curve(
    #     coverage_array_dual_aligned, 1 - sa_array_dual_aligned
    # )
    # dual_aurc_aligned = float(np.trapz(min_risks_dual, cov_bins_dual))

    return float(max_f1_ood), float(aurc_ood), float(max_f1), float(ds_aurc)