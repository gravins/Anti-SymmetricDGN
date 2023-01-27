from models import *


def config_antisymmetric_dgn_GraphProp(num_features, num_classes, gcn_norm=False):
    for h in [30, 20, 10]:
        for l in [20, 10, 5, 1]:
            for e in [1., 1e-1, 1e-2, 1e-3]:
                    for g in [1., 1e-1, 1e-2, 1e-3]:
                        yield {
                            'model': {
                                'input_dim': num_features,
                                'output_dim': num_classes,
                                'hidden_dim': h,
                                'num_layers': l,
                                'epsilon': e,
                                'gamma': g,
                                'activ_fun': 'tanh',
                                'gcn_norm': gcn_norm                                
                            },
                            'optim': {
                                'lr': 0.003,
                                'weight_decay': 1e-6
                            }
                        }


def config_dgn_GraphProp(num_features, num_classes, conv):
    for h in [30, 20, 10]:
        for num_layers in [20, 10, 5, 1]:
                yield {
                    'model': {
                        'input_dim': num_features,
                        'output_dim': num_classes,
                        'hidden_dim': h,
                        'num_layers': num_layers,
                        'conv_layer': conv
                    },
                    'optim': {
                        'lr': 0.003,
                        'weight_decay': 1e-6
                    }
                }


def config_gcn2_GraphProp(num_features, num_classes, conv):
    for h in [30, 20, 10]:
        for num_layers in [20, 10, 5, 1]:
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
                        'lr': 0.003,
                        'weight_decay': 1e-6
                    }
                }


def config_ODE_GraphProp(num_features, num_classes):
    for h in [30, 20, 10]:
        for num_layers in [20, 10, 5, 1]:
            for e in [1., 1e-1, 1e-2, 1e-3]:
                conf = {
                    'model': {
                        'input_dim': num_features,
                        'output_dim': num_classes,
                        'hidden_dim': h,
                        'epsilon': e,
                        'iterations':num_layers,
                        'cached': False
                    },
                    'optim': {
                        'lr': 0.003,
                        'weight_decay': 1e-6
                    }
                }
                yield conf


c0 = lambda num_features, num_classes: config_dgn_GraphProp(num_features, num_classes, 'GINConv')
c1 = lambda num_features, num_classes: config_dgn_GraphProp(num_features, num_classes, 'GCNConv')
c2 = lambda num_features, num_classes: config_dgn_GraphProp(num_features, num_classes, 'SAGEConv')
c3 = lambda num_features, num_classes: config_dgn_GraphProp(num_features, num_classes, 'GATConv')
c4 = lambda num_features, num_classes: config_antisymmetric_dgn_GraphProp(num_features, num_classes, gcn_norm=True)
c5 = lambda num_features, num_classes: config_antisymmetric_dgn_GraphProp(num_features, num_classes)
c6 = lambda num_features, num_classes: config_ODE_GraphProp(num_features, num_classes)
c7 = lambda num_features, num_classes: config_gcn2_GraphProp(num_features, num_classes, 'GCN2Conv')

CONFIGS = {
    'GIN_GraphProp': (c0, DGN_GraphProp),
    'GCN_GraphProp': (c1, DGN_GraphProp),
    'SAGE_GraphProp': (c2, DGN_GraphProp),
    'GAT_GraphProp': (c3, DGN_GraphProp),
    'GraphAntiSymmetricNN_weight_sharing_gcnnorm_GraphProp': (c4, GraphAntiSymmetricNN_GraphProp),
    'GraphAntiSymmetricNN_weight_sharing_GraphProp': (c5, GraphAntiSymmetricNN_GraphProp),
    'DGC_GraphProp': (c6, DGC_GraphProp),
    'GRAND_GraphProp': (c6, GRAND_GraphProp),
    'GCN2_GraphProp': (c7, DGN_GraphProp)
}


