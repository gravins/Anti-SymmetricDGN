import torch
import ray
import time
import datetime
import numpy as np
from conf import CONFIGS
from utils import scoring, get_dataset


def train(model, optimizer, data, train_mask, criterion):
    model.train()
    # Reset gradients from previous step
    model.zero_grad()

    # Perform a forward pass
    preds = model.forward(data)

    # Perform a backward pass to calculate the gradients
    loss = criterion(preds[train_mask], data.y[train_mask])
    loss.backward()

    # Update parameters
    optimizer.step()


def evaluate(model, data, eval_mask, criterion, return_true_values=False):
    t0 = time.time()
    model.eval()
    y_true, y_preds, y_preds_confidence = [], [], []
    with torch.no_grad():
        # Perform the forward pass
        preds = model.forward(data)[eval_mask]
        y_true = data.y[eval_mask]

        loss = criterion(preds, y_true)
               
        y_preds = preds.argmax(dim=1).cpu().tolist()
        y_true = y_true.cpu().tolist()
        if return_true_values:
            preds = (torch.sigmoid(preds) > 0.5).float()
            y_preds_confidence = preds.cpu().tolist()

    scores = {'loss': loss.cpu().item(),
              'time': datetime.timedelta(seconds=time.time() - t0)}

    # Compute scores
    scores.update(scoring(y_true, y_preds))
    true_values = (y_true, y_preds, y_preds_confidence) if return_true_values else None
    return scores, true_values


@ray.remote(num_cpus=1, num_gpus=1/10)
def train_and_eval(model, config, 
                   data_dir, data_name,
                   metric='accuracy', device="cpu", mode="Validation", early_stopping_patience=None, 
                   path_save_best=None, #eg, 'best_epoch_model.pth'
                   verbose=False):
    total_time = time.time()

    # Load dataset
    data, _, _ = get_dataset(root=data_dir, name=data_name, seed=config['exp']['data_seed'])
    tr_mask = data.train_mask
    eval_mask = data.val_mask

    epochs = config['exp']['epochs']

    #print('ip_addr:', get_node_ip_address(), 'train', config)
    print('train', config)
    
    seed = config['exp']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    model = model(**config['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optim']['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)

    max_score = -1
    best_epoch = 0
    best_score = None
    for e in range(epochs):
        t0 = time.time()
        data = data.to(device)
        train(model, optimizer, data, tr_mask, criterion)
        
        # Evaluate the model on the training set
        train_scores, _ = evaluate(model, data, tr_mask, criterion)
        tr_time = datetime.timedelta(seconds=time.time() - t0)
        
        # Evaluate the model on the evaluation set
        eval_scores, _ = evaluate(model, data, eval_mask, criterion)

        # Record all statistics from this epoch
        train_scores['time'] = tr_time
        h = {'epoch': e + 1,
             'Training': train_scores,
             mode: eval_scores}

        if eval_scores[metric] >= max_score:
            max_score = eval_scores[metric]
            best_epoch = e
            best_score = h

            # Save model with highest evaluation score
            if path_save_best is not None:
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, path_save_best)

        if verbose:
            print(f'Epochs: {e}, '
                  f'TR Loss: {train_scores["loss"]}'
                  f'VL Loss:{eval_scores["loss"]}'
                  f'TR Acc: {train_scores["accuracy"]}'
                  f'VL Acc:{eval_scores["accuracy"]}')

        if (early_stopping_patience is not None) and (e - best_epoch > early_stopping_patience):
            break

    m = 'VL' if mode == 'Validation' else 'TS'
    print(config, f" - Total training took {datetime.timedelta(seconds=time.time()-total_time)} (hh:mm:ss)",
          f'Training stopped after {e} epochs',
          f'*** Best Epoch: {best_epoch}, '
          f'TR Loss: {best_score["Training"]["loss"]}'
          f'{m} Loss:{best_score[mode]["loss"]}'
          f'TR Acc: {best_score["Training"]["accuracy"]}'
          f'{m} Acc:{best_score[mode]["accuracy"]}')

    return {'best_score': best_score, 'best_epoch': best_epoch}


@ray.remote(num_cpus=1, num_gpus=1)
def eval_test(data_dir, data_name, metric, 
              best_conf, model_name, checkpoints_paths, device):
    
    avg_ts_score = {}
    m, l = [], []
    for data_seed in best_conf.keys():
        # Load dataset
        data, num_features, num_classes = get_dataset(root=data_dir, name=data_name, seed=int(data_seed))
        ts_mask = data.test_mask

        paths = [p for p in checkpoints_paths if f'data_seed_{data_seed}' in p] # this contains the different initializations of the best conf for a particular TR/VL/TS split 

        ts_score = {'seed': [], 'loss':[], metric:[], 'conf_mat':[]}
        for ckpt_path in paths:
            model = CONFIGS[model_name][1]
            model = model(**best_conf[data_seed]['model'])

            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device)
            scores, _ = evaluate(model = model, 
                                data = data.to(device),
                                eval_mask = ts_mask,
                                criterion = torch.nn.CrossEntropyLoss())
            seed = ckpt_path.split('_seed_')[-1].replace('.pth', '')
            ts_score['seed'].append(seed)
            ts_score[metric].append(scores[metric])
            ts_score['loss'].append(scores['loss'])
            ts_score['conf_mat'].append(str(scores['confusion_matrix'].tolist()))
            print(f'data seed {data_seed}, seed {seed}, confusion matrix {scores["confusion_matrix"]}')

        avg_ts_score[data_seed] = ts_score
        m.append(np.mean(ts_score[metric]).item())
        l.append(np.mean(ts_score['loss']).item())

    avg_ts_score[f'avg test {metric}'] = np.mean(m).item()
    avg_ts_score[f'std test {metric}'] = np.std(m).item()
    avg_ts_score['avg test loss'] = np.mean(l).item()
    avg_ts_score['std test loss'] = np.std(l).item()

    return avg_ts_score
