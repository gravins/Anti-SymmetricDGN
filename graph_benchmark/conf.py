from models import *

def config_antisymmetric_dgn(num_features, num_classes, gcn_norm=False):
    for lr in [1e-2, 1e-3, 1e-4]:
        for hidden in [128, 64, 32]:
            for l in [30, 20, 10, 5, 3, 2, 1]:
                for e in [1., 1e-1, 1e-2, 1e-3, 1e-4]: # the step size
                    for g in [1., 1e-1, 1e-2, 1e-3, 1e-4]: 
                        yield {
                            'model': {
                                'input_dim': num_features,
                                'output_dim': num_classes,
                                'hidden_dim': hidden,
                                'num_layers': l,
                                'epsilon': e,
                                'gamma': g,
                                'activ_fun': 'tanh',
                                'gcn_norm': gcn_norm
                            },
                            'optim': {
                                'lr': lr
                            }
                        }


def config_dgn(num_features, num_classes, conv):
    for num_layers in [30, 20, 10, 5, 3, 2, 1]:
        for lr in [1e-2, 1e-3, 1e-4]:
            for hidden in [128,64,32]:
                    yield {
                        'model': {
                            'input_dim': num_features,
                            'output_dim': num_classes,
                            'hidden_dim': hidden,
                            'num_layers': num_layers,
                            'conv_layer': conv,
                        },
                        'optim': {
                            'lr': lr
                        }
                    }


def config_ODE(num_features, num_classes):
    for h in [128,64,32]:
        for num_layers in [30, 20, 10, 5, 3, 2, 1]:
            for lr in [1e-2, 1e-3, 1e-4]:
                for e in [1., 1e-1, 1e-2, 1e-3, 1e-4]:
                    yield {
                        'model': {
                            'input_dim': num_features,
                            'output_dim': num_classes,
                            'hidden_dim': h,
                            'epsilon': e,
                            'iterations':num_layers,
                            'cached': True
                        },
                        'optim': {
                            'lr': lr,
                        }
                    }


def config_gcn2(num_features, num_classes, conv):
    for lr in [1e-2, 1e-3, 1e-4]:
        for h in [128,64,32]:
            for num_layers in [30, 20, 10, 5, 3, 2, 1]:
                for alpha in [1., 1e-1, 1e-2]:
                    yield {
                        'model': {
                            'input_dim': num_features,
                            'output_dim': num_classes,
                            'hidden_dim': h,
                            'num_layers': num_layers,
                            'conv_layer': conv,
                            'alpha': alpha
                        },
                        'optim': {
                            'lr': lr
                        }
                    }


c0 = lambda num_features, num_classes: config_dgn(num_features, num_classes, 'GINConv')
c1 = lambda num_features, num_classes: config_dgn(num_features, num_classes, 'GCNConv')
c2 = lambda num_features, num_classes: config_dgn(num_features, num_classes, 'SAGEConv')
c3 = lambda num_features, num_classes: config_dgn(num_features, num_classes, 'GATConv')
c4 = lambda num_features, num_classes: config_antisymmetric_dgn(num_features, num_classes, gcn_norm=True)
c5 = lambda num_features, num_classes: config_antisymmetric_dgn(num_features, num_classes)
c6 = lambda num_features, num_classes: config_ODE(num_features, num_classes)
c7 = lambda num_features, num_classes: config_gcn2(num_features, num_classes, 'GCN2Conv')
CONFIGS = {
    'GIN': (c0, DGN),
    'GCN': (c1, DGN),
    'SAGE': (c2, DGN),
    'GAT': (c3, DGN),
    'GraphAntiSymmetricNN_weight_sharing_gcnnorm': (c4, GraphAntiSymmetricNN),
    'GraphAntiSymmetricNN_weight_sharing': (c5, GraphAntiSymmetricNN),
    'DGC': (c6, DGC),
    'GRAND': (c6, GRAND),
    'GCN2': (c7, DGN)
 }

