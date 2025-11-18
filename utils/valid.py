import numpy as np
import torch

# ----------------------------------------------------------------------
# Calc AURC & EAURC (vectorized)
# ----------------------------------------------------------------------
def calc_aurc_eaurc(confidence, correctness):
    """
    confidence: shape (N,)   float
    correctness: shape (N,)  0/1 values
    """

    # Sort by confidence descending
    idx = np.argsort(-confidence)
    sorted_correct = correctness[idx]

    # Coverage: 1..N / N
    n = len(correctness)
    coverage = (np.arange(1, n + 1) / n)

    # Risk = (number of errors) / k
    error = 1 - sorted_correct
    cum_error = np.cumsum(error)
    risk = cum_error / np.arange(1, n + 1)

    # AURC = integral approx
    aurc = np.mean(risk)

    # Optimal risk area
    final_risk = risk[-1]
    optimal_risk_area = final_risk + (1 - final_risk) * np.log(1 - final_risk)
    eaurc = aurc - optimal_risk_area

    return aurc, eaurc

@torch.no_grad()
def validation(loader, net):
    net.eval()

    val_log = {'conf': [], 'correct': []}

    for data in loader:
        image = data[0].cuda(non_blocking=True)
        target = data[1].cuda(non_blocking=True)

        output = net(image)
        softmax = torch.softmax(output, dim=1)
        conf, pred = softmax.max(1)  # conf: max softmax

        correct = pred.eq(target).float()

        val_log['conf'].append(conf.cpu().numpy())
        val_log['correct'].append(correct.cpu().numpy())

    # concat
    conf = np.concatenate(val_log['conf'])
    correct = np.concatenate(val_log['correct'])

    # accuracy
    acc = correct.mean() * 100.0

    # AURC & EAURC
    aurc, eaurc = calc_aurc_eaurc(conf, correct)

    return {
        'Acc.': acc,
        'AURC': aurc * 1000,   
        'EAURC': eaurc * 1000,
    }
