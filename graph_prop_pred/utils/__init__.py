from torch_geometric.datasets import GNNBenchmarkDataset
from .pna_dataset import transform, GraphPropDataset, TASKS

def get_dataset(root: str, name: str, task=None):

    if name == 'GraphProp':
        assert task is None or task in TASKS
        
        data_train = GraphPropDataset(root, split='train', task=task, pre_transform = transform) 
        data_valid = GraphPropDataset(root, split='val', task=task, pre_transform = transform) 
        data_test = GraphPropDataset(root, split='test', task=task, pre_transform = transform)
        num_features = data_train[0]['x'].size(1)
        num_classes = 1 # regression task 
    else:
        data_train = GNNBenchmarkDataset(root=root, name=name, split='train')
        data_valid = GNNBenchmarkDataset(root=root, name=name, split='val')
        data_test = GNNBenchmarkDataset(root=root, name=name, split='test') 

        num_features = data_train[0].pos.size(1)
        num_classes = data_train.num_classes
    return data_train, data_valid, data_test, num_features, num_classes



DATA = ['GraphProp'] 
