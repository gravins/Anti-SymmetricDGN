import torch

import os
import ray
import dill
import tqdm
import datetime
import numpy as np
import pandas as pd
from typing import Union
from conf import CONFIGS
from typing import Optional
from utils import get_dataset
from train import train_and_eval
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from utils.io import load, dump, join, create_if_not_exist

def split(y, eval_size=0.1, n_splits=1, seed=9):
    """
    :param y: the target variable for supervised learning problems 
    :param eval_size: evaluation set size, type: float or int. Default 0.1
    :param n_splits: number of re-shuffling and splitting iterations
    :param seed: seed used for random number generator
    :return two lists containing the fold's index list
    """

    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=eval_size, random_state=seed)
                
    tr_folds, eval_folds = [], []
    for tr_ids, eval_ids in splitter.split([0]*len(y), y):
        tr_folds.append(torch.LongTensor(tr_ids))
        eval_folds.append(torch.LongTensor(eval_ids))

    return tr_folds, eval_folds

def generate_seed(num_seeds, seeds_path, seed, num_trials=100):
    if os.path.isfile(seeds_path):
        config_seeds = load(seeds_path)
    else:
        rng = np.random.default_rng(seed)
        config_seeds = set(rng.integers(2 ** 32 - 1, size = num_seeds).tolist())
        i = 0
        while len(config_seeds) < num_seeds and i < num_trials:
            config_seeds.union(rng.integers(2 ** 32 - 1, size = num_seeds - len(config_seeds)).tolist())
            i += 1

        if len(config_seeds) < num_seeds:
            print(f'After {num_trials} trials we were not able to compute {num_seeds} seeds between 0 and 2**31-1')

        dump(list(config_seeds), seeds_path)
    
    return list(config_seeds)


def model_selection(model_name: str,
                    data_name: str,
                    metric: str,
                    early_stopping_patience: Optional[int] = None,
                    epochs: int = 1000,
                    data_dir: str = '.',
                    exp_dir: str = '.',
                    device: str = torch.device('cpu')):
    """
    Perform a model selection phase through standard validation or k-fold model selection.
    All the results are saved into a DataFrame and the best configuration is returned.

    Parameters
    ----------
    model_name : str
        The model name
    data_name : str
        The name of the dataset
    metric : str
        The name of the optimized metric
    early_stopping_patience : int, optional
        Number of epochs with no improvement after which training will be stopped, by default None
    epochs : int, optional
        The number of epochs, by default 1000
    data_dir : str, optional
        The data directory, by default the current directory
    exp_dir : str, optional
        The saving directory, by default the current directory
    device : torch.device, optional
        The torch device for the experiment, i.e., cpu or cuda; by default torch.device('cpu')
    seed : int, optional
        the experimental seed, by default 9
    """

    assert ray.is_initialized() == True, "Ray is not initialized"
    data_dir = os.path.abspath(data_dir) # ray wants absolute paths
    exp_dir = os.path.abspath(exp_dir)

    assert not os.path.exists(join(exp_dir, 'validation_results.csv')), 'The file validation_results.csv already exists.'
    
    if os.path.exists(join(data_dir, data_name)):
        # Download data once for all configurations
        data, num_features, num_classes = get_dataset(root=data_dir, name=data_name)
        del data

    # Create the checkpoint directory
    checkpoint_dir = join(exp_dir, 'checkpoints')
    create_if_not_exist(checkpoint_dir)

    config_fun, model = CONFIGS[model_name]  #  config_fun, _ = CONFIGS[model_name]
    ray_ids = []
    ids_to_configs = {}
    print_params = True
    
    for conf_id, conf in enumerate(config_fun(num_features, num_classes)):
        conf.update({
            'exp':{'conf_id': conf_id,
                    'epochs': epochs,
                    'patience': early_stopping_patience}
        })
        conf['model'].update({
            'input_dim': num_features,
            'output_dim': num_classes
        })
        if print_params:
            print_params = False
            from prettytable import PrettyTable

            def count_parameters(model):
                table = PrettyTable(["Modules", "Parameters"])
                total_params = 0
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad: 
                        continue
                    param = parameter.numel()
                    table.add_row([name, param])
                    total_params+=param
                print(table)
                print(f"Total Trainable Params: {total_params}")
                return total_params

            print(conf)
            count_parameters(model(**conf['model']))
        checkpoint_path = join(checkpoint_dir, f'conf_id_{conf_id}')
        ray_ids.append(
            train_and_eval.remote(model, conf, data_dir, data_name,
                                metric=metric, 
                                early_stopping_patience=early_stopping_patience,
                                device=device,
                                mode="Validation", path_save_best=checkpoint_path)
        )
        ids_to_configs[ray_ids[-1]] = conf

    # Wait and collect results
    df = []
    result_per_split = []
    best_score = None
    tqdm_ids = tqdm.tqdm(ray_ids)
    for id_ in tqdm_ids:
        res = ray.get(id_)

        result_per_split.append(res)

        conf = ids_to_configs[id_]
        result = {'ray_id': id_}
        for key_name, values in conf.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    result[f'{key_name}_{k}'] = v
            else:
                result[key_name] = values

        for suffix, m in [('tr', 'Training'), ('vl', 'Validation'), ('ts', 'Test')]:
            tmp = defaultdict(list)
            for i in range(len(res)):
                for k,v in res[i][m].items():
                    tmp[k].append(v)
            aus = {}
            for k,v in tmp.items():
                if 'time' in k:
                    v = [vv.total_seconds() for vv in v]
                aus[f'{suffix} {k} mean'] = np.mean(v) if not 'confusion_matrix' in k else np.mean(v, axis=0)
                aus[f'{suffix} {k} std'] = np.std(v) if not 'confusion_matrix' in k else np.std(v, axis=0)
            result.update(aus)
        tmp = [r['best_epoch'] for r in res]
        result['epoch mean'] = np.mean(tmp)
        result['epoch std'] = np.std(tmp)

        if best_score is None or result[f'vl {metric} mean'] > best_score:
                best_score = result[f'vl {metric} mean']
                tqdm_ids.set_postfix(best_test_acc = result[f'ts {metric} mean'],
                             best_val_acc = result[f'vl {metric} mean'],
                             best_train_acc = result[f'tr {metric} mean']) 
        
        df.append(result)

    df = pd.DataFrame(df).sort_values(f'vl {metric} mean', ascending=False, ignore_index=True)
    df.to_csv(join(exp_dir, 'validation_results.csv'), index=False)
    dill.dump(result_per_split, open(join(exp_dir, 'result_per_split.p'), 'wb'))

    # Return best configuration
    best_conf = ids_to_configs[df.loc[0,'ray_id']]
    cols = [c for c in df.columns if 'tr ' in c or 'vl ' in c or 'ts ' in c]
    best_conf['Results'] = df.loc[0, cols]
    
    return best_conf
