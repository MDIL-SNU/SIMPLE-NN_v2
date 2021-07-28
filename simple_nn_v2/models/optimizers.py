import torch.optim

def _initialize_optimizer(inputs, model):
    optim_type = inputs['neural_network']['method']
    lr=inputs['neural_network']['learning_rate']
    regularization = float(inputs['neural_network']['regularization'])

    optimizer = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD
    }
    kwarg = {
        'Adam': {'lr': lr, 'weight_decay': regularization},
        'SGD': {'lr': lr, 'weight_decay': regularization},
    }
    return optimizer[optim_type](model.parameters(), **kwarg[optim_type])
