import torch.nn.init as init

def _define_gain(logfile, acti_fn, init_name, init_params):
    if init_name in ['xavier uniform', 'xavier normal', 'orthogonal']:
        #Nonlinear functions that gain defined
        if acti_fn in ['sigmoid', 'tanh', 'relu', 'selu']:
            init_params['gain'] = init.calculate_gain(acti_fn)
        else:
            if init_params['gain'] is None:
                logfile.write("gain must be defined for using '{}' initializer with '{}' activation function".format(init_name, acti_fn))

def _get_implemented_initializer_list():
    implemented_init = ['xavier uniform', 'xavier normal', 'normal', 'constant', 'kaiming normal', 'he normal', 'kaiming uniform', 'he uniform', 'orthogonal']
    return implemented_init

def _get_initializer_and_kwarg(init_name, init_params):
    weight_initializer = {
        'xavier uniform': init.xavier_uniform_,
        'xavier normal': init.xavier_uniform_,
        'normal': init.normal_,
        'constant': init.constant_,
        'kaiming normal': init.kaiming_normal_,
        'kaiming uniform': init.kaiming_uniform_,
        'he normal': init.kaiming_normal_,
        'he uniform': init.kaiming_uniform_,
        'orthogonal': init.orthogonal_
    }
    kwarg = {
        'xavier uniform': {'gain': init_params['gain']},
        'xavier normal': {'gain': init_params['gain']},
        'normal': {'mean': init_params['mean'], 'std': init_params['std']},
        'constant': {'var': init_params['var']},
        'kaiming normal': {'mode': 'fan_out', 'nonlinearity': 'relu'},
        'kaiming uniform': {'mode': 'fan_out', 'nonlinearity': 'relu'},
        'he normal': {'mode': 'fan_out', 'nonlinearity': 'relu'},
        'he unifrom': {'mode': 'fan_out', 'nonlinearity': 'relu'},
        'orthogonal': {'gain': init_params['gain']}
    }

    return weight_initializer[init_name], kwarg[init_name]
