import torch
import utils.metrics
import numpy as np 

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

    aurc = utils.metrics.compute_aurc(
        val_log['pred'],
        val_log['softmax'],  
        val_log['target']
    )


    # log
    res = {
        'Acc.': acc,
        'AURC': aurc*1000,
    }
    

    return res

