from torch.nn import Linear, init
import torch


implemented_initializer = ['xavier uniform', 'xavier normal', 'normal', 'constant', \
        'kaiming normal', 'he normal', 'kaiming uniform', 'he uniform', 'orthogonal']

def _set_optim_inputs(inputs, logfile):
    init_name = inputs['neural_network']['weight_initializer']['type']
    if inputs['neural_network']['weight_initializer']['params'] is None:
        init_params = None
    else:
        init_params = inputs['neural_network']['weight_initializer']['params']
        acti_fn = inputs['neural_network']['acti_func']

        if init_params['gain'] is None\
            and init_name in ['xavier uniform', 'xavier normal', 'orthogonal']:
            #Nonlinear functions that gain defined
            if acti_fn in ['sigmoid', 'tanh', 'relu', 'selu']:
                init_params['gain'] = init.calculate_gain(acti_fn)
            else:
                init_params['gain'] = None

    return init_name, init_params

def _get_implemented_initializer_list():
    implemented_init = ['xavier uniform', 'xavier normal', 'normal', 'constant', 'kaiming normal', 'he normal', 'kaiming uniform', 'he uniform', 'orthogonal']
    return implemented_init

def _get_initializer_and_kwarg(init_name, init_params):
    weight_initializer = {
        'xavier uniform' : init.xavier_uniform_,
        'xavier normal'  : init.xavier_normal_,
        'normal'         : init.normal_,
        'constant'       : init.constant_,
        'kaiming normal' : init.kaiming_normal_,
        'kaiming uniform': init.kaiming_uniform_,
        'he normal'      : init.kaiming_normal_,
        'he uniform'     : init.kaiming_uniform_,
        'orthogonal'     : init.orthogonal_,
        'sparse'         : init.sparse_
    }
    #If not kwarg in input use defalt value
    kwarg = {
        'xavier uniform' : {'gain': init_params['gain'] if init_params['gain'] else 1.0},
        'xavier normal'  : {'gain': init_params['gain'] if init_params['gain'] else 1.0},
        'normal'         : {'mean': init_params['mean'] if init_params['mean'] else 0.0,\
                            'std': init_params['std'] if init_params['std'] else 1.0},
        'constant'       : {'val': init_params['val']},
        'kaiming normal' : {'mode': init_params['mode'], 'nonlinearity': init_params['nonlinearity']},
        'kaiming uniform': {'mode': init_params['mode'], 'nonlinearity': init_params['nonlinearity']},
        'he normal'      : {'mode': init_params['mode'] if init_params['mode'] else 'fan_in',\
                            'nonlinearity': init_params['nonlinearity'] if init_params['nonlinearity'] else 'relu'},
        'he uniform'     : {'mode': init_params['mode'] if init_params['mode'] else 'fan_in',\
                            'nonlinearity': init_params['nonlinearity'] if init_params['nonlinearity'] else 'relu'},
        'orthogonal'     : {'gain': init_params['gain'] if init_params['gain'] else 1.0},
        'sparse'         : {'sparsity': init_params['sparsity'] if init_params['sparsity'] else 0.1,\
                            'std': init_params['std'] if init_params['std'] else 0.01},
    }

    return weight_initializer[init_name], kwarg[init_name]


def _initialize_weights(inputs, logfile, model):
    init_name, init_params = _set_optim_inputs(inputs, logfile)
    #implemented_initializer = _get_implemented_initializer_list()
    try:
        if init_name is None:
            weight_log = ""
        elif init_name not in implemented_initializer:
            weight_log = f"Warning : {init_name} weight initializer infomation is not implemented\n".format()
        else:
            weight_initializer, kwarg = _get_initializer_and_kwarg(init_name, init_params)
            for lin in model.lin:
                if isinstance(lin, Linear):
                    weight_initializer(lin.weight, **kwarg)
                    tmp_bias = torch.zeros([1, lin.bias.size(0)])
                    weight_initializer(tmp_bias, **kwarg)
                    lin.bias.data = tmp_bias[0]
            weight_log = ""
    except:
        import sys
        print(sys.exc_info())
        weight_log = "During weight initialization error occured. Default Initializer used\n"

    return weight_log

