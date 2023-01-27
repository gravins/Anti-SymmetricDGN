from os import remove
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.utils import subgraph
from ogb.nodeproppred import PygNodePropPredDataset
from collections import defaultdict
from typing import Tuple, Optional

def set_seed(seed):
    import torch
    import random
    import numpy as np

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scoring(y_true, y_pred, labels=None):
    s = {}
    for k in SCORE:
        s[k] = (SCORE[k](y_true, y_pred, average='macro') if k != 'accuracy'
                else SCORE[k](y_true, y_pred))

    s["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels)
    return s

def get_dataset(root:str, name:str, seed:Optional[int]=None) -> Tuple:
    assert name in DATA, f'Dataset name should be one of {DATA.keys()}, not {name}'
    data, num_features, num_classes = DATA[name][1](root, name)

    if seed is not None:
        import torch
        import random
        random.seed(seed)

        classes_idx = defaultdict(list)
        for i, c in enumerate(data.y):
            classes_idx[c.item()].append(i)
        
        # As in Pitfalls of Graph Neural Network Evaluation, Shchur et al. remove classes which 
        # have less than 51 nodes
        remove_labels = []
        remove_idx = []
        for k, v in classes_idx.items():
            if len(v) <= 50:
                remove_labels.append((k, len(v)))
                remove_idx += v

        if len(remove_labels) > 0:
            # If a label does not have >50 nodes then remove the class from the dataset, 
            # (all nodes in that class included), and update the label vector
            keep_idx = list(set(range(data.y.shape[-1])) - set(remove_idx))
            edge_index, edge_attr = subgraph(subset=torch.LongTensor(keep_idx), 
                                             edge_index=data.edge_index,
                                             relabel_nodes=True) 
            data.edge_index = edge_index
            data.x = data.x[keep_idx]
            data.y = data.y[keep_idx]

            unique_labels = torch.unique(data.y)
            mapping = {int(l):i for i, l in enumerate(unique_labels)}
            data.y = torch.tensor([mapping[int(y)] for y in data.y], dtype=data.y.dtype)

            print(f'Removed classes: {remove_labels}')
            num_classes = num_classes - len(remove_labels)

            classes_idx = defaultdict(list)
            for i, c in enumerate(data.y):
                classes_idx[c.item()].append(i)

        train_idx = []
        valid_idx = []
        test_idx = []
        for v in classes_idx.values():
            random.shuffle(v)
            train_idx += v[:20]
            valid_idx += v[20:50]
            test_idx += v[50:]

        data.train_mask = train_idx
        data.val_mask = valid_idx
        data.test_mask = test_idx

    return data, num_features, num_classes
    

def ogb_data_(root:str, name:str) -> Tuple:
    data = PygNodePropPredDataset(root=root, name=name)
    num_classes, num_features = data.num_classes, data[0].x.shape[1]
    split_idx = data.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    data = data[0]
    if len(data.y.size()) > 1:
        data.y = data.y.squeeze()
    data.train_mask = train_idx 
    data.val_mask = valid_idx 
    data.test_mask = test_idx 
    return data, num_features, num_classes

def pyg_data_(root:str, name:str) -> Tuple:
    data_getter = DATA[name][0]
    data = data_getter(root=root, name=name)
    num_features = data.num_features
    num_classes = data.num_classes
    data = data[0]

    return data, num_features, num_classes


DATA = {
    'PubMed' : (Planetoid, pyg_data_),
    'CS' : (Coauthor, pyg_data_),
    'Physics' : (Coauthor, pyg_data_),
    'Computers' : (Amazon, pyg_data_),
    'Photo' : (Amazon, pyg_data_),
    }

SCORE = {
 "f1_score": f1_score,
 "recall": recall_score,
 "accuracy": accuracy_score
}
