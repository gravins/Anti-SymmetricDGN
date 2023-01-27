import torch

import os
import ray
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
                    num_runs: int = 1,
                    standard_split: bool = True, # If present then the standard splits of the data are used.
                    num_splits: Optional[int] = False,
                    data_dir: str = '.',
                    exp_dir: str = '.',
                    num_trials: int = 100,
                    device: str = torch.device('cpu'), 
                    seed: int = 9):
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
    num_runs : int, optional
        The number of runs with different initializations, by default 1
    standard_split : bool, optional
        If true then the standard splits of the data are used, by default True
    num_splits : int, optional
        The number of random train/validation/test splits, by default None
    data_dir : str, optional
        The data directory, by default the current directory
    exp_dir : str, optional
        The saving directory, by default the current directory
    num_trials: int, optional
        The number of trials to generate num_runs and/or num_splits seeds, default 100
    device : torch.device, optional
        The torch device for the experiment, i.e., cpu or cuda; by default torch.device('cpu')
    seed : int, optional
        the experimental seed, by default 9
    """

    assert standard_split or num_splits, "num_splits and standard_splits cannot be set together"
    assert ray.is_initialized() == True, "Ray is not initialized"
    data_dir = os.path.abspath(data_dir) # ray wants absolute paths
    exp_dir = os.path.abspath(exp_dir)

    assert not os.path.exists(join(exp_dir, 'validation_results.xlsx')), 'The file validation_results.xlsx already exists.'
    
    if os.path.exists(join(data_dir, data_name)):
        # Download data once for all configurations
        data, num_features, num_classes = get_dataset(root=data_dir, name=data_name)
        del data

    split_path = join(data_dir, data_name)
    rng = np.random.default_rng(seed)
    data_seeds = [None]
    if num_splits:
        data_seeds_path = join(split_path, f'data_seeds_{num_splits}.json')
        data_seeds = generate_seed(num_splits, data_seeds_path, rng.integers(2**16), num_trials)

    config_seeds = [seed]
    if num_runs > 1:
        config_seeds_path = join(split_path, f'conf_seeds_{num_runs}.json')
        config_seeds = generate_seed(num_runs, config_seeds_path, rng.integers(2**16), num_trials)

    # Create the checkpoint directory
    checkpoint_dir = join(exp_dir, 'checkpoints')
    create_if_not_exist(checkpoint_dir)

    config_fun, model = CONFIGS[model_name]
    ray_ids = []
    ids_to_configs = {}
    print_params = True
    for data_seed in tqdm.tqdm(data_seeds):
        for conf_id, conf in enumerate(config_fun(num_features, num_classes)):
            for conf_seed in config_seeds:
                conf.update({
                    'exp':{'conf_id': conf_id,
                           'data_seed': data_seed,
                           'seed': conf_seed,
                           'epochs': epochs,
                           'num_runs': num_runs,
                           'num_splits': num_splits,
                           'patience': early_stopping_patience,
                           'standard_split': standard_split}
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
                checkpoint_path = join(checkpoint_dir, f'conf_id_{conf_id}_seed_{conf_seed}_data_seed_{data_seed}.pth')
                ray_ids.append(
                    train_and_eval.remote(model, conf, data_dir, data_name,
                                          metric=metric, 
                                          early_stopping_patience=early_stopping_patience,
                                          device=device,
                                          mode="Validation", path_save_best=checkpoint_path)
                )
                ids_to_configs[ray_ids[-1]] = conf

    df = []
    # Wait and collect results
    for id_ in tqdm.tqdm(ray_ids):
        res = ray.get(id_)

        conf = ids_to_configs[id_]
        result = {'ray_id': id_}
        for key_name, values in conf.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    result[f'{key_name}_{k}'] = v
            else:
                result[key_name] = values

        result.update({f'tr {k}': v for k,v in res['best_score']['Training'].items()})
        result.update({f'vl {k}': v for k,v in res['best_score']['Validation'].items()})
        result['epoch'] = res['best_score']['epoch']
        
        df.append(result)

    df = pd.DataFrame(df)

    best_configs = {}
    # Group by each TR/VL/TS split
    for data_seed, datagroup_df in df.groupby('exp_data_seed'):
        mod_sel_df = defaultdict(list)
        # Group by conf id
        for conf_id, confgroup_df in datagroup_df.groupby('exp_conf_id'):
            # Compute the mean over the multiple runs
            for col_name, values in confgroup_df.items():
                if 'exp_' in col_name or 'optim_' in col_name or 'model_' in col_name or col_name == 'ray_id':
                    if 'seed' not in col_name:
                        mod_sel_df[col_name].append(values[values.index[0]])
                    continue

                if 'confusion_matrix' in col_name:
                    mean = np.mean(confgroup_df[col_name].tolist(), axis=0).tolist()
                    std = np.std(confgroup_df[col_name].tolist(), axis=0).tolist()
                else:            
                    mean = confgroup_df[col_name].mean()
                    std = confgroup_df[col_name].std()
                
                mod_sel_df[f'{col_name} mean'].append(mean)
                mod_sel_df[f'{col_name} std'].append(std)

        # Save the results
        mod_sel_df = pd.DataFrame(mod_sel_df).sort_values(f'vl {metric} mean', ascending=False, ignore_index=True)
        if os.path.exists(join(exp_dir, 'validation_results.xlsx')):
            with pd.ExcelWriter(join(exp_dir, 'validation_results.xlsx'), mode='a') as writer:  
                mod_sel_df.to_excel(writer, sheet_name=f'data_seed={data_seed}', index=False)
                writer.save()
        else:
            mod_sel_df.to_excel(join(exp_dir, 'validation_results.xlsx'), 
                                sheet_name=f'data_seed={data_seed}', index=False)

        # Return best configuration
        best = ids_to_configs[mod_sel_df.loc[0,'ray_id']]
        best['exp']['seed'] = config_seeds

        cols = [k for k in mod_sel_df.columns if metric in k or 'loss' in k] 
        best['Model Selection'] = mod_sel_df.loc[0, cols].to_dict()

        best_configs[data_seed] = best


    best_conf_path = join(exp_dir, 'best_conf.json')
    dump(best_configs, best_conf_path)
    
    return best_configs
