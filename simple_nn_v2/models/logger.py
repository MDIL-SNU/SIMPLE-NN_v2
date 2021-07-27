from simple_nn_v2.models import run
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':6.4e', sqrt=False):
        self.name = name
        self.fmt = fmt
        self.sqrt = sqrt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        if self.sqrt:
            self.sqrt_val = self.val ** 0.5
            self.sqrt_avg = self.avg ** 0.5

class TimeMeter(object):
    """Computes total time elapsed"""
    def __init__(self, name, fmt=':6.4e'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
    
    def update(self, val):
        self.val += val

#Show avg rmse
def _show_avg_rmse(inputs, logfile, epoch, lr, total_time, train_progress_dict, valid_progress_dict=None):
    log = '-' * 94
    print(log)
    logfile.write(log+'\n')

    log=''
    log += 'Epoch {0:>7d}'.format(epoch)
    
    log += _formatting_avg_rmse('e_err', 'E RMSE', train_progress_dict, valid_progress_dict)
    if inputs['neural_network']['use_force']:
        log += _formatting_avg_rmse('f_err', 'F RMSE', train_progress_dict, valid_progress_dict)
    if inputs['neural_network']['use_stress']:
        log += _formatting_avg_rmse('s_err', 'S RMSE', train_progress_dict, valid_progress_dict)
    log += ' learning_rate: {0:.4e}\n'.format(lr)
    
    if valid_progress_dict != None:
        log += 'Data load(T V): {0:.4e} {1:.4e} s/epoch'.format(train_progress_dict['data_time'].val, valid_progress_dict['data_time'].val)
        log += ' Total time(T V): {0:.4e} {1:.4e} s/epoch'.format(train_progress_dict['batch_time'].val, valid_progress_dict['batch_time'].val)
    else:
        log += 'Data load: {0:.4e} s/epoch'.format(train_progress_dict['data_time'].val)
        log += ' Total load: {0:.4e} s/epoch'.format(train_progress_dict['batch_time'].val)
    log += ' Elapsed: {0:.4e} s'.format(total_time)
    print(log)
    logfile.write(log+'\n')

    log = '-' * 94
    print(log)
    logfile.write(log+'\n')

def _formatting_avg_rmse(key, title, t_progress_dict, v_progress_dict):
    if v_progress_dict != None:
        log = ' {}(T V) {:.4e} {:.4e}'.format(title, t_progress_dict[key].sqrt_avg, v_progress_dict[key].sqrt_avg)
    else:
        log = ' {}(T V) {:.4e} {:>10}'.format(title, t_progress_dict[key].sqrt_avg, '-')
    return log

#Show structure rmse
def _show_structure_rmse(inputs, logfile, labeled_train_loader, labeled_valid_loader, model, device, optimizer=None, criterion=None, atomic_e=False):
    dtype = torch.get_default_dtype()
    non_block = False if (device == torch.device('cpu')) else True

    logfile.write('structural breakdown:\n')
    log = '  {:<20}'.format('label')
    log += '   E_RMSE(T)   E_RMSE(V)'
    if inputs['neural_network']['use_force']:
        log += '   F_RMSE(T)   F_RMSE(V)'
    if inputs['neural_network']['use_stress']:
        log += '   S_RMSE(T)   S_RMSE(V)'
    logfile.write(log+'\n')

    keys = labeled_train_loader.keys()
    if labeled_valid_loader:
        for key in labeled_valid_loader.keys():
            if key not in keys:
                keys.append(labeled_valid_loader[key])

    for key in keys:
        valid_progress_dict = None
        log = ''
        train_epoch_result = run.progress_epoch(inputs, labeled_train_loader[key], model, optimizer, criterion, 0, dtype, device, non_block, valid=True, atomic_e=atomic_e)
        if labeled_valid_loader and labeled_valid_loader[key] is not None:
            valid_epoch_result = run.progress_epoch(inputs, labeled_valid_loader[key], model, optimizer, criterion, 0, dtype, device, non_block, valid=True, atomic_e=atomic_e)
        else:
            valid_epoch_result = None

        log += '  {0:20}'.format(key)
        log += _formatting_structure_rmse('e_err', train_epoch_result, valid_epoch_result)
        if inputs['neural_network']['use_force']:
            log += _formatting_structure_rmse('f_err', train_epoch_result, valid_epoch_result)
        if inputs['neural_network']['use_stress']:
            log += _formatting_structure_rmse('s_err', train_epoch_result, valid_epoch_result)
        print(log)
        logfile.write(log+'\n')

    log = '-' * 94
    print(log)
    logfile.write(log+'\n')

def _formatting_structure_rmse(key, t_progress_dict, v_progress_dict):
    t_val = '{:>10}'.format('-') if t_progress_dict == None else '{:.4e}'.format(t_progress_dict[key].sqrt_avg)
    v_val = '{:>10}'.format('-') if v_progress_dict == None else '{:.4e}'.format(v_progress_dict[key].sqrt_avg)
    log = '  {}  {}'.format(t_val, v_val)

    return log
