import os
import torch

from conf import CONFIGS
import ray
import time
import argparse
import datetime
from utils import DATA, SCORE
from model_selection import model_selection
from utils.io import create_if_not_exist, join

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
                        default='Squirrel',
                        choices=DATA.keys())
    parser.add_argument('--metric',
                        help='The matric that will be optimized.', 
                        default='accuracy',
                        choices= SCORE.keys(),
                        type = str)
    parser.add_argument('--model_name',
                        help='The model name.',
                        default='GraphAntiSymmetricNN',
                        choices=CONFIGS.keys())
    parser.add_argument('--epochs', help='The number of epochs.', default=1500, type=int)
    parser.add_argument('--early_stopping', 
                        help='Training stops if the selected metric does not improve for X epochs',
                        type=int,
                        default=100)
    parser.add_argument('--save_dir', help='The saving directory.', default='.')
    args = parser.parse_args()
    
    print(args)
    assert args.epochs >= 1, 'The number of epochs should be greather than 0'
    args.save_dir = os.path.abspath(args.save_dir)

    p = join(args.save_dir, args.data_name)
    create_if_not_exist(p)    
    exp_dir = join(p, args.model_name)
    create_if_not_exist(exp_dir)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))

    # Run model selection
    best_conf_res = model_selection(model_name = args.model_name,
                            data_name = args.data_name,
                            metric = args.metric,
                            early_stopping_patience = args.early_stopping,
                            epochs = args.epochs,
                            data_dir = args.save_dir,
                            exp_dir = exp_dir,
                            device = device)

    print(best_conf_res)
    elapsed = time.time() - t0
    print(datetime.timedelta(seconds=elapsed))
