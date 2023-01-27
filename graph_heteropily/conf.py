from models import GraphAntiSymmetricNN

def config_antisymmetric_dgn(num_features, num_classes, gcn_norm=False):
    for lr in [1e-2, 1e-3]:
        for wd in [1e-2, 1e-3, 1e-4]:
            for hidden in [80, 64, 32]: #, 16, 8]:
                for l in [64, 32, 16, 8, 4, 2]:
                    for e in [1., 1e-1, 1e-2]: #, 1e-3, 1e-4]: # the step size
                        for g in [1e-1, 1e-2]: #,1., 1e-3, 1e-4]: 
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
                                    'lr': lr,
                                    'wd': wd
                                }
                            }

c4 = lambda num_features, num_classes: config_antisymmetric_dgn(num_features, num_classes, gcn_norm=True)
c5 = lambda num_features, num_classes: config_antisymmetric_dgn(num_features, num_classes, gcn_norm=False)

CONFIGS = {
    'GraphAntiSymmetricNN_weight_sharing_gcnnorm': (c4, GraphAntiSymmetricNN),
    'GraphAntiSymmetricNN_weight_sharing': (c5, GraphAntiSymmetricNN)
 }

