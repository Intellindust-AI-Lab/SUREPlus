import torch
import numpy as np 

# Calc coverage, risk
def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list

# Calc aurc, eaurc
def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    return aurc, eaurc

# AURC, EAURC
def calc_aurc_eaurc(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x:x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

@torch.no_grad()
def validation(loader, net):
    net.eval()

    val_log = {'softmax': [], 'correct': [], 'pred': [], 'target': []}

    for data in loader:
        image, target = data[0].cuda(), data[1].cuda()
        output = net(image)
        softmax = torch.softmax(output, dim=1)
        conf, pred_cls = softmax.max(1)  

        correct = pred_cls.eq(target).float()

        val_log['correct'].append(correct.cpu().numpy())
        val_log['softmax'].append(conf.cpu().numpy())  
        val_log['pred'].append(pred_cls.cpu().numpy())
        val_log['target'].append(target.cpu().numpy())

    for key in val_log:
        val_log[key] = np.concatenate(val_log[key])

    acc = 100. * val_log['correct'].mean()

    aurc = calc_aurc_eaurc(
        val_log['softmax'],  
        val_log['correct']
    )


    # log
    res = {
        'Acc.': acc,
        'AURC': aurc*1000,
    }
    

    return res

