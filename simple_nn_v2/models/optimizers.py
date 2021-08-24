import torch.optim

def _initialize_optimizer(inputs, model):
    optim_type = inputs['neural_network']['method']
    lr=inputs['neural_network']['learning_rate']
    regularization = float(inputs['neural_network']['regularization'])

    optimizer = {
        'Adadelta': torch.optim.Adadelta,
        'Adagrad': torch.optim.Adagrad,
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        #'SparseAdam': torch.optim.SparseAdam,
        'Adamax': torch.optim.Adamax,
        #'NAdam': torch.optim.NAdam,
        #'RAdam': torch.optim.RAdam,
        'ASGD': torch.optim.ASGD,
        'SGD': torch.optim.SGD,
#        'LBFGS': torch.optim.LBFGS,
        'RMSprop': torch.optim.RMSprop,
        'Rprop': torch.optim.Rprop,
    }
    default = {'lr': lr, 'weight_decay': regularization}
    kwarg = {
        'Adadelta': default,
        'Adagrad': default,
        'Adam': default,
        'AdamW': default,
        #'SparseAdam': {'lr':lr},
        'Adamax': default,
        #'NAdam': default,
        #'RAdam': default,
        'ASGD': default,
        'SGD': default,
#        'LBFGS': {'lr':lr},
        'RMSprop': default,
        'Rprop': {'lr':lr},
    }
    return optimizer[optim_type](model.parameters(), **kwarg[optim_type])
