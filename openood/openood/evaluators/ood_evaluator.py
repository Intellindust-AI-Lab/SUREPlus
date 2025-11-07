import csv
import os
from typing import Dict, List
from pathlib import Path
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from openood.postprocessors import BasePostprocessor, MSPPostprocessor
from openood.utils import Config

from .base_evaluator import BaseEvaluator
from .metrics import compute_all_metrics, compute_f1_and_aurc, compute_aurc


class OODEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        """OOD Evaluator.

        Args:
            config (Config): Config file from
        """
        super(OODEvaluator, self).__init__(config)
        self.id_pred = None
        self.id_conf = None
        self.id_gt = None
        self.msp_eval = BasePostprocessor(Config)
        self.id_aurc = 1000

    def eval_ood(self,
                 net: nn.Module,
                 id_data_loaders: Dict[str, DataLoader],
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor: BasePostprocessor,
                 fsood: bool = False):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        assert 'test' in id_data_loaders, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name

        if self.config.postprocessor.APS_mode:
            assert 'val' in id_data_loaders
            assert 'val' in ood_data_loaders
            self.hyperparam_search(net, id_data_loaders['val'],
                                   ood_data_loaders['val'], postprocessor)

        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        
        # ood score of id samples
        id_pred, id_ood_conf, id_gt = postprocessor.inference(
            net, id_data_loaders['test'])
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_ood_conf, id_gt, dataset_name)
            
        # id confidence of id samples
        p = Path(self.config.output_dir)
        ce_path = p.parent.parent / "msp" / p.name / "scores" / f"{dataset_name}.npz"
        ce_pred = np.load(ce_path)
        id_pred, id_id_conf, id_gt = ce_pred['pred'], ce_pred['conf'], ce_pred['label']
        
        # get aurc
        self.id_aurc = compute_aurc(id_pred, id_ood_conf, id_gt)
        print("ID AURC:{:.2f}".format(self.id_aurc*1000))
        
        

        if fsood:
            # load csid data and compute confidence
            for dataset_name, csid_dl in ood_data_loaders['csid'].items():
                print(f'Performing inference on {dataset_name} dataset...',
                      flush=True)
                csid_pred, csid_conf, csid_gt = postprocessor.inference(
                    net, csid_dl)
                if self.config.recorder.save_scores:
                    self._save_scores(csid_pred, csid_conf, csid_gt,
                                      dataset_name)
                id_pred = np.concatenate([id_pred, csid_pred])
                id_conf = np.concatenate([id_conf, csid_conf])
                id_gt = np.concatenate([id_gt, csid_gt])
        
        

        # load nearood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_ood_conf, id_gt, id_id_conf],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='nearood')
        
        

        # load farood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_ood_conf, id_gt, id_id_conf],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='farood')

    def _eval_ood(self,
                  net: nn.Module,
                  id_list: List[np.ndarray],
                  ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                  postprocessor: BasePostprocessor,
                  ood_split: str = 'nearood'):
        print(f'Processing {ood_split}...', flush=True)
        [id_pred, id_ood_conf, id_gt, id_id_conf] = id_list
        metrics_list = []
        for dataset_name, ood_dl in ood_data_loaders[ood_split].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            # ood score of ood samples
            ood_pred, ood_ood_conf, ood_gt = postprocessor.inference(net, ood_dl)
            if self.config.recorder.save_scores:
                self._save_scores(ood_pred, ood_ood_conf, ood_gt, dataset_name)
            # id confidence of ood samples
            # Load MSP score from ce output
            p = Path(self.config.output_dir)
            ce_path = p.parent.parent / "msp" / p.name / "scores" / f"{dataset_name}.npz"
            ce_pred = np.load(ce_path)
            _, ood_id_conf, _ = ce_pred['pred'], ce_pred['conf'], ce_pred['label']
            
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_ood_conf, ood_ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            
    

            print(f'Computing metrics on {dataset_name} dataset...')

            ood_metrics = compute_all_metrics(conf, label, pred)
            # New Metrics
            ood_f1, ood_aurc, ds_f1, ds_aurc = compute_f1_and_aurc(id_gt, id_pred, id_id_conf, id_ood_conf,
                                              ood_id_conf, ood_ood_conf)
            ood_metrics.append(ood_f1)
            ood_metrics.append(ood_aurc)
            ood_metrics.append(ds_f1)
            ood_metrics.append(ds_aurc)
            ood_metrics.append(self.id_aurc)
            
            metrics_list.append(ood_metrics)
            if self.config.recorder.save_csv:
                self._save_csv(ood_metrics, dataset_name=dataset_name)

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name=ood_split)

    def eval_ood_val(self, net: nn.Module, id_data_loaders: Dict[str,
                                                                 DataLoader],
                     ood_data_loaders: Dict[str, DataLoader],
                     postprocessor: BasePostprocessor):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        assert 'val' in id_data_loaders
        assert 'val' in ood_data_loaders
        if self.config.postprocessor.APS_mode:
            val_auroc = self.hyperparam_search(net, id_data_loaders['val'],
                                               ood_data_loaders['val'],
                                               postprocessor)
        else:
            id_pred, id_conf, id_gt = postprocessor.inference(
                net, id_data_loaders['val'])
            ood_pred, ood_conf, ood_gt = postprocessor.inference(
                net, ood_data_loaders['val'])
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            val_auroc = ood_metrics[1]
        return {'auroc': 100 * val_auroc}

    def _save_csv(self, metrics, dataset_name):
        """
        save to ood.csv, include dataset_name, ACC, AURC, AUROC, DS-F1, DS-AURC
        """
        [fpr, auroc, aupr_in, aupr_out, accuracy, ood_f1, ood_aurc, ds_f1, ds_aurc, aurc] = metrics

        write_content = {
            'dataset': dataset_name,
            'ACC': '{:.2f}'.format(100 * accuracy),
            'ID AURC': '{:.2f}'.format(1000 * aurc),
            'AUROC': '{:.2f}'.format(100 * auroc),
            'FPR' : '{:.2f}'.format(100 * fpr),
            'OOD F1': '{:.2f}'.format(100 * ood_f1),
            'OOD AURC': '{:.2f}'.format(1000 * ood_aurc),
            'DS-F1': '{:.2f}'.format(100 * ds_f1),
            'DS-AURC': '{:.2f}'.format(1000 * ds_aurc),
        }

        fieldnames = list(write_content.keys())

        print('ACC: {:.2f} | AURC: {:.2f} | AUROC: {:.2f} | F1: {:.2f} | OOD-AURC: {:.2f} | DS-F1: {:.2f} | DS-AURC: {:.2f}'.format(
            100 * accuracy, 1000 * aurc, 100 * auroc, 100 * ood_f1, 1000*ood_aurc,  100 * ds_f1, 1000 * ds_aurc))
        print(u'\u2500' * 70, flush=True)

        csv_path = os.path.join(self.config.output_dir, 'ood.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)


    def _save_scores(self, pred, conf, gt, save_name):
        save_dir = os.path.join(self.config.output_dir, 'scores')
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name),
                 pred=pred,
                 conf=conf,
                 label=gt)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1,
                 fsood: bool = False,
                 csid_data_loaders: DataLoader = None):
        """Returns the accuracy score of the labels and predictions.

        :return: float
        """
        if type(net) is dict:
            net['backbone'].eval()
        else:
            net.eval()
        self.id_pred, self.id_conf, self.id_gt = postprocessor.inference(
            net, data_loader)

        if fsood:
            assert csid_data_loaders is not None
            for dataset_name, csid_dl in csid_data_loaders.items():
                csid_pred, csid_conf, csid_gt = postprocessor.inference(
                    net, csid_dl)
                self.id_pred = np.concatenate([self.id_pred, csid_pred])
                self.id_conf = np.concatenate([self.id_conf, csid_conf])
                self.id_gt = np.concatenate([self.id_gt, csid_gt])

        metrics = {}
        metrics['acc'] = sum(self.id_pred == self.id_gt) / len(self.id_pred)
        metrics['epoch_idx'] = epoch_idx
        return metrics

    def report(self, test_metrics):
        print('Completed!', flush=True)

    def hyperparam_search(
        self,
        net: nn.Module,
        id_data_loader,
        ood_data_loader,
        postprocessor: BasePostprocessor,
    ):
        print('Starting automatic parameter search...')
        aps_dict = {}
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0
        for name in postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1
        for name in hyperparam_names:
            hyperparam_list.append(postprocessor.args_dict[name])
        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)
        for hyperparam in hyperparam_combination:
            postprocessor.set_hyperparam(hyperparam)
            id_pred, id_conf, id_gt = postprocessor.inference(
                net, id_data_loader)
            ood_pred, ood_conf, ood_gt = postprocessor.inference(
                net, ood_data_loader)
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            index = hyperparam_combination.index(hyperparam)
            aps_dict[index] = ood_metrics[1]
            print('Hyperparam:{}, auroc:{}'.format(hyperparam,
                                                   aps_dict[index]))
            if ood_metrics[1] > max_auroc:
                max_auroc = ood_metrics[1]
        for key in aps_dict.keys():
            if aps_dict[key] == max_auroc:
                postprocessor.set_hyperparam(hyperparam_combination[key])
        print('Final hyperparam: {}'.format(postprocessor.get_hyperparam()))
        return max_auroc

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results
