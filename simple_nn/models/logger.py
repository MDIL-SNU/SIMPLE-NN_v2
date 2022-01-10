class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':6.4e', sqrt=False):
        self.name = name
        self.fmt  = fmt
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
        self.fmt  = fmt
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val += val

def _init_meters(struct_labels, use_force, use_stress, atomic_e):
    ## Setting LOG with progress meter
    batch_time = TimeMeter('time', ':6.3f')
    data_time  = TimeMeter('data', ':6.3f')
    total_time = TimeMeter('total time', ':8.4e')
    losses = AverageMeter('loss', ':8.4e')
    e_err = dict()
    for label in struct_labels:
        e_err[label] = AverageMeter('E err', ':6.4e', sqrt=True)
    tot_e_err = AverageMeter('E err', ':6.4e', sqrt=True)
    progress_dict = {'batch_time': batch_time, 'data_time': data_time, 'losses': losses, 'e_err': e_err, 'tot_e_err': tot_e_err, 'total_time': total_time, 'struct_labels': struct_labels}

    if use_force and not atomic_e:
        f_err = dict()
        for label in struct_labels:
            f_err[label] = AverageMeter('F err', ':6.4e', sqrt=True)
        progress_dict['f_err'] = f_err
        progress_dict['tot_f_err'] = AverageMeter('F err', ':6.4e', sqrt=True)
    if use_stress and not atomic_e:
        s_err = dict()
        for label in struct_labels:
            s_err[label] = AverageMeter('S err', ':6.4e', sqrt=True)
        progress_dict['s_err'] = s_err
        progress_dict['tot_s_err'] = AverageMeter('S err', ':6.4e', sqrt=True)

    return progress_dict

# Show avg rmse
def _show_avg_rmse(logfile, epoch, lr, total_time, train_progress_dict, valid_progress_dict=None):
    logfile.write("-" * 88 + '\n')

    log = "Epoch {0:>7d}".format(epoch)
    log += _formatting_avg_rmse('e_err', 'E RMSE', train_progress_dict, valid_progress_dict)
    if 'f_err' in train_progress_dict.keys():
        log += _formatting_avg_rmse('f_err', 'F RMSE', train_progress_dict, valid_progress_dict)
    if 's_err' in train_progress_dict.keys():
        log += _formatting_avg_rmse('s_err', 'S RMSE', train_progress_dict, valid_progress_dict)
    log += " learning_rate: {0:.4e}\n".format(lr)

    if valid_progress_dict != None:
        log += "Data load(T V): {0:.4e} {1:.4e} s/epoch".format(train_progress_dict['data_time'].val, valid_progress_dict['data_time'].val)
        log += " Total time(T V): {0:.4e} {1:.4e} s/epoch".format(train_progress_dict['batch_time'].val, valid_progress_dict['batch_time'].val)
    else:
        log += "Data load: {0:.4e} s/epoch".format(train_progress_dict['data_time'].val)
        log += " Total load: {0:.4e} s/epoch".format(train_progress_dict['batch_time'].val)
    log += " Elapsed: {0:.4e} s".format(total_time)
    logfile.write(log + '\n')

    logfile.write("-" * 88 + '\n')

def _formatting_avg_rmse(key, title, t_progress_dict, v_progress_dict):
    # calc average rmse
    t_sum = 0
    t_count = 0
    for label in t_progress_dict['struct_labels']:
        t_sum   += t_progress_dict[key][label].sum
        t_count += t_progress_dict[key][label].count
    t_rmse = (t_sum / t_count) ** 0.5

    if v_progress_dict:
        v_sum = 0
        v_count = 0
        for label in v_progress_dict['struct_labels']:
            v_sum   += v_progress_dict[key][label].sum
            v_count += v_progress_dict[key][label].count
        v_rmse = (v_sum / v_count) ** 0.5

    if v_progress_dict != None:
        log = " {}(T V) {:.4e} {:.4e}".format(title, t_rmse, v_rmse)
    else:
        log = " {}(T V) {:.4e} {:>10}".format(title, t_rmse, '-')

    return log

# Show structure rmse
def _show_structure_rmse(logfile, train_epoch_result, valid_epoch_result):
    logfile.write("Structure breakdown:\n")
    log = "  {:<14}".format('label')
    log += "   E_RMSE(T)   E_RMSE(V)"

    if 'f_err' in train_epoch_result.keys():
        log += "   F_RMSE(T)   F_RMSE(V)"
    if 's_err' in train_epoch_result.keys():
        log += "   S_RMSE(T)   S_RMSE(V)"
    logfile.write(log+'\n')

    for label in train_epoch_result['struct_labels']:
        log = "  {: <14}".format(label)
        log += _formatting_structure_rmse('e_err', label, train_epoch_result, valid_epoch_result)

        if 'f_err' in train_epoch_result.keys():
            log += _formatting_structure_rmse('f_err', label, train_epoch_result, valid_epoch_result)
        if 's_err' in train_epoch_result.keys():
            log += _formatting_structure_rmse('s_err', label, train_epoch_result, valid_epoch_result)
        logfile.write(log+'\n')

    log = "-" * 88
    logfile.write(log+'\n')

def _formatting_structure_rmse(key, label, t_progress_dict, v_progress_dict):
    t_val = "{:>10}".format('-') if t_progress_dict[key][label].sum == 0 else "{:.4e}".format(t_progress_dict[key][label].sqrt_avg)
    if v_progress_dict:
        v_val = "{:>10}".format('-') if v_progress_dict[key][label].sum == 0 else "{:.4e}".format(v_progress_dict[key][label].sqrt_avg)
        log = "  {}  {}".format(t_val, v_val)
    else:
        log = "  {}".format(t_val)

    return log
