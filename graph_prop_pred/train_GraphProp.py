import torch
import ray
import time
import random
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from utils import get_dataset
from torch_geometric.nn import global_add_pool
from utils.pna_dataset import GRAPH_LVL_TASKS, NODE_LVL_TASKS
from torch_scatter import scatter
import pdb

def train(model, optimizer, dataloader, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_train_MSE = 0
    for iter, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)

        out = model(batch)
        loss = criterion(out, batch.y, batch.batch)
        
        loss.backward()
        optimizer.step()

        loss_ = loss.detach().item()
        epoch_loss += loss_
        epoch_train_MSE += loss_
    epoch_loss /= (iter + 1)
    epoch_train_MSE /= (iter + 1)

    return epoch_loss, np.log10(epoch_train_MSE), optimizer



def evaluate(model, criterion, dataloader, device):
    model.eval()
    epoch_test_loss = 0
    epoch_test_MSE = 0
    with torch.no_grad():
        for iter, batch in enumerate(dataloader):
            batch = batch.to(device)

            out = model(batch)
            loss = criterion(out, batch.y, batch.batch)
            
            loss_ = loss.detach().item()
            epoch_test_loss += loss_
            epoch_test_MSE += loss_
        epoch_test_loss /= (iter + 1)
        epoch_test_MSE /= (iter + 1)
        
    return epoch_test_loss, np.log10(epoch_test_MSE)


@ray.remote(num_cpus=1, num_gpus=1/10)
def train_val_pipeline_GraphProp(model_class, 
                           config, 
                           data_dir,
                           data_name,
                           device="cpu",
                           early_stopping_patience=None, 
                           path_save_best=None, #eg, 'best_epoch_model.pth'
                           verbose=False):

    print('train', config)

    # Load dataset
    data_train, data_valid, data_test, _, _ = get_dataset(root=data_dir, name=data_name, task=config['exp']['task'])
    
    results = []
    seeds = config['exp']['seeds']
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        model = model_class(**config['model'])
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=config['optim']['lr'], 
                                    weight_decay=config['optim']['weight_decay'])

        node_level = config['exp']['task'] in NODE_LVL_TASKS
        assert node_level or config['exp']['task'] in GRAPH_LVL_TASKS
        def single_loss(pred, label, batch):
            # for node-level
            if node_level:
                nodes_in_graph = scatter(torch.ones(batch.shape[0]).to(device), batch).unsqueeze(1).to(device)
                #nodes_in_graph = torch.tensor([[(batch == i).sum()] for i in range(max(batch)+1)]).to(device)
                nodes_loss = (pred - label.reshape(label.shape[0], 1)) ** 2

                # Implementing global add pool of the node losses, reference here
                # https://github.com/cvignac/SMP/blob/62161485150f4544ba1255c4fcd39398fe2ca18d/multi_task_utils/util.py#L99
                error = global_add_pool(nodes_loss, batch) / nodes_in_graph #average_nodes
                loss = torch.mean(error)
                return loss
            
            # for graph-level
            loss = torch.mean((pred - label.reshape(label.shape[0], 1)) ** 2)
            return loss
        
        criterion = single_loss
        model.to(device)

        t0 = time.time()
        per_epoch_time = []
        
        train_loader = DataLoader(data_train, batch_size=config['exp']['batch_size'], shuffle=True)
        val_loader = DataLoader(data_valid, batch_size=config['exp']['batch_size'], shuffle=False)
        test_loader = DataLoader(data_test, batch_size=config['exp']['batch_size'], shuffle=False)
    
        epoch_train_losses, epoch_val_losses, epoch_test_losses = [], [], []
        epoch_train_scores, epoch_val_scores, epoch_test_scores = [], [] , []
    
        epochs = config['exp']['epochs']
        best_score = None
        best_epoch = 0
  
        for epoch in range(epochs):
                start = time.time()
                
                epoch_train_loss, epoch_train_score, optimizer = train(model, optimizer, train_loader, criterion, device)
                epoch_val_loss, epoch_val_score = evaluate(model, criterion, val_loader, device)
                per_epoch_time.append(time.time()-start)

                epoch_test_loss, epoch_test_score = evaluate(model, criterion, test_loader, device)
            
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_test_losses.append(epoch_test_loss)
                epoch_train_scores.append(epoch_train_score)
                epoch_val_scores.append(epoch_val_score)
                epoch_test_scores.append(epoch_test_score)

                # Record all statistics from this epoch
                if best_score is None or epoch_val_score <= best_score:
                    best_score = epoch_val_score
                    best_epoch = epoch

                    # Save model with highest evaluation score
                    if path_save_best is not None:
                        assert path_save_best[-4:] == ".pth", f'path_save_best should terminate with ".pth", received {path_save_best}'
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            #'scheduler': scheduler
                            }, path_save_best.replace(".pth", f"_seed_{seed}.pth"))

                if verbose:
                    print(f'Epochs: {epoch}, '
                            f'TR loss: {epoch_train_loss}, '
                            f'VL loss: {epoch_val_loss}, '
                            f'TR score: {epoch_train_score}, '
                            f'VL score: {epoch_val_score}, '
                            f'TEST score: {epoch_test_score}, '
                            f'lr: {optimizer.param_groups[0]["lr"]}')

                if (early_stopping_patience is not None) and (epoch - best_epoch > early_stopping_patience):
                    print(config, f'-- seed: {seed}', f': early-stopped at epoch {epoch}')
                    break

                if epoch % 100 == 0:
                    print(np.mean(per_epoch_time), f'at epoch {epoch}')
    
        results.append({
            'best_train_loss': epoch_train_losses[best_epoch],
            'best_val_loss': epoch_val_losses[best_epoch],
            'best_test_loss': epoch_test_losses[best_epoch],
            'best_train_score': epoch_train_scores[best_epoch],
            'best_val_score': epoch_val_scores[best_epoch],
            'best_test_score': epoch_test_scores[best_epoch],
            'convergence time (epochs)': epoch,
            'total time taken': time.time() - t0,
            'avg time per epoch': np.mean(per_epoch_time),
            'model_params': sum(p.numel() for p in model.parameters())
        })

        del model, optimizer

    avg = {}
    for k in results[0].keys():
        avg[f'avg {k}'] = np.mean([r[k] for r in results])
        avg[f'std {k}'] = np.std([r[k] for r in results])

    return {'avg_res': avg, 'single_res': results}
