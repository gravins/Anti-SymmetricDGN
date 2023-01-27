import os
import torch

import ray
import time
import argparse
import datetime
import numpy as np
from conf import CONFIGS
from train import eval_test
from utils import set_seed, SCORE, DATA
from model_selection import model_selection
from utils.io import dump, load, create_if_not_exist, join

# Ignore warnings
from sklearn.exceptions import UndefinedMetricWarning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


ray.init() # local ray initialization

print('Settings:')
print('\tKMP_SETTING:', os.environ.get('KMP_SETTING'))
print('\tOMP_NUM_THREADS:', os.environ.get('OMP_NUM_THREADS'))
print('\tKMP_BLOCKTIME:', os.environ.get('KMP_BLOCKTIME'))
print('\tMALLOC_CONF:', os.environ.get('MALLOC_CONF'))
print('\tLD_PRELOAD:', os.environ.get('LD_PRELOAD'))
print()

if __name__ == "__main__":
    t0 = time.time()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  

    parser.add_argument('--data_name', 
                        help='The name of the dataset to load.',
                        default='PubMed',
                        choices=DATA.keys())
    parser.add_argument('--model_name',
                        help='The model name.',
                        default='GraphAntiSymmetricNN',
                        choices=CONFIGS.keys())
    parser.add_argument('--mode', 
                        help='The experiment to run.',
                        default='Validation',
                        choices=['Validation', 'Test'])
    parser.add_argument('--standard_split',
                        help='If present then the standard splits of the data are used.',
                        action='store_true')
    parser.add_argument('--vl_runs',
                        help='The number of random initializaiton per configuration',
                        default=5,
                        type=int)
    parser.add_argument('--data_splits',
                        help='The number of random train/validation/test splits',
                        default=5,
                        type=int)
    parser.add_argument('--epochs', help='The number of epochs.', default=10000, type=int)
    parser.add_argument('--early_stopping', 
                        help='Training stops if the selected metric does not improve for X epochs',
                        type=int,
                        default=50)
    parser.add_argument('--save_dir', help='The saving directory.', default='.')
    parser.add_argument('--conf_path', 
                        help=('The path of the best configuration to test. It needs to be set when '
                              'mode is Test'),
                        required=False,
                        type=str)
    parser.add_argument('--metric',
                        help='The matric that will be optimized.', 
                        default='accuracy',
                        choices= SCORE.keys(),
                        type = str)
    parser.add_argument('--seed', help='The seed of the experiment.', default=9, type = int)
    args = parser.parse_args()
    
    print(args)

    set_seed(args.seed)

    assert args.epochs >= 1, 'The number of epochs should be greather than 0'
    assert not args.mode == 'Test' or args.conf_path, 'The best configuration path needs to be set when mode is Test'
    args.save_dir = os.path.abspath(args.save_dir)

    exp_dir = join(args.save_dir, args.data_name, args.model_name)
    create_if_not_exist(exp_dir)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))

    if args.mode == 'Validation':
        # Run model selection
        conf = model_selection(model_name = args.model_name,
                               data_name = args.data_name,
                               metric = args.metric,
                               epochs = args.epochs,
                               early_stopping_patience = args.early_stopping,
                               num_runs = args.vl_runs,
                               standard_split = args.standard_split,
                               num_splits=args.data_splits,
                               data_dir = args.save_dir, 
                               exp_dir = exp_dir,
                               device = device, 
                               seed = args.seed)
                               
    # Load checkpoints and evaluate on test set
    best_conf_path = (join(exp_dir, 'best_conf.json') if args.conf_path is None
                      else args.conf_path)
    
    print(f'Loading best conf from {best_conf_path}...')
    best_conf = load(best_conf_path)

    paths_ = join(exp_dir, 'checkpoints')
    ckpts_paths = []
    for data_seed in best_conf.keys():
        conf_id = best_conf[data_seed]['exp']['conf_id']

        ckpts_paths += [os.path.abspath(join(paths_, f))
                        for f in os.listdir(paths_)
                        if f'conf_id_{conf_id}_seed_' in f and f'data_seed_{data_seed}.pth' in f]

    ray_id = eval_test.remote(data_dir = os.path.abspath(args.save_dir), # ray wants absolute paths
                              data_name = args.data_name, 
                              metric = args.metric,
                              checkpoints_paths = ckpts_paths, 
                              best_conf = best_conf,
                              model_name =  args.model_name,
                              device = device)

    ts_score = ray.get(ray_id)

    for k in ts_score.keys():
        if 'test' not in k:
            best_conf[k]['Risk Assessment'] = ts_score[k]

    best_conf['Average Risk Assessment'] = {k: v 
                                            for k, v in ts_score.items()
                                            if not isinstance(v, dict)}
    dump(best_conf, best_conf_path)

    num_features, num_classes = best_conf.get('model', None), best_conf.get('model', None) 
    if num_features is not None:
        num_features = num_features.get('input_dim', None)
        num_classes = num_classes.get('output_dim', None) 

    configurations = ([c for c in CONFIGS[args.model_name][0](num_features, num_classes)] 
                      if args.mode == 'Validation' 
                      else None)

    dump({'epochs': args.epochs,
          'model_structure': best_conf,
          'model_selection': configurations}, join(exp_dir, args.model_name + '_settings.json'))

    elapsed = time.time() - t0
    print(datetime.timedelta(seconds=elapsed))
