import torch.optim

def _initialize_optimizer(inputs, model, user=None):
    optim_type  = inputs['neural_network']['optimizer']['method']
    optim_params =inputs['neural_network']['optimizer']['params'] if inputs['neural_network']['optimizer']['params'] else {}
    lr=inputs['neural_network']['learning_rate']
    regularization = float(inputs['neural_network']['l2_regularization'])

    optimizer = {
        'Adadelta': torch.optim.Adadelta,
        'Adagrad': torch.optim.Adagrad,
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'Adamax': torch.optim.Adamax,
        'ASGD': torch.optim.ASGD,
        'SGD': torch.optim.SGD,
        'RMSprop': torch.optim.RMSprop,
        'Rprop': torch.optim.Rprop,
        'user': user
    }
    default = {'lr': lr, 'weight_decay': regularization}

    optim_params.update(default)
    if 'betas' in optim_params.keys():
        optim_params['betas'] = tuple(float(b_range) for b_range in optim_params['betas'].split())

    rprop_params = optim_params.copy()
    if 'weight_decay' in rprop_params.keys():
        del rprop_params['weight_decay']

    kwarg = {
        'Adadelta'  : optim_params,
        'Adagrad'   : optim_params,
        'Adam'      : optim_params,
        'AdamW'     : optim_params,
        'Adamax'    : optim_params,
        'ASGD'      : optim_params,
        'SGD'       : optim_params,
        'RMSprop'   : optim_params,
        'Rprop'     : rprop_params,
        'user'      : optim_params,
    }
    return optimizer[optim_type](model.parameters(), **kwarg[optim_type])
