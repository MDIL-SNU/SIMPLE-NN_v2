
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
    
    #Update values using value, batch_number 
    #And calculate average, sqrt of batch value
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        #self.sum += val
        self.count += n
        self.avg = self.sum / self.count

        if self.sqrt:
            self.sqrt_val = self.val ** 0.5
            self.sqrt_avg = self.avg ** 0.5


    #Show average value : average value
    def show_avg(self):
        if self.sqrt: #sqrt need
            #fmtstr = '{name} ( {sqrt_avg' + self.fmt + '} )'
            fmtstr = '{name}  {sqrt_avg' + self.fmt + '} '
        else:
            #fmtstr = '{name} ( {avg' + self.fmt + '} )'
            fmtstr = '{name}  {avg' + self.fmt + '} '
        return fmtstr.format(**self.__dict__)

    #Show sqrt value & average value : batch_value ( average_value )
    def __str__(self):
        if self.sqrt: #sqrt need
            fmtstr = '{name} {sqrt_val' + self.fmt + '} ( {sqrt_avg' + self.fmt + '} )'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ( {avg' + self.fmt + '} )'
        return fmtstr.format(**self.__dict__)

class StructureMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':6.4e', sqrt=False):
        self.name = name
        self.fmt = fmt
        self.sqrt = sqrt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sqrt_avg = 0
        self.sum = 0
        self.count = 0
    
    #Update values using value, batch_number 
    #And calculate average, sqrt of batch value
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        if self.sqrt:
            self.sqrt_avg = self.avg ** 0.5

    #Show sqrt value & average value : batch_value ( average_value )
    def __str__(self):
        #fmtstr = '{0}  ( {1' + self.fmt + '} )'
        fmtstr = '{0}   {1' + self.fmt + '} '
        if self.sqrt: #sqrt need
            fmtstr = fmtstr.format(self.name, self.sqrt_avg)
        else:
            fmtstr = fmtstr.format(self.name, self.avg)
        return fmtstr


class TimeMeter(object):
    """Computes total time elapsed"""
    def __init__(self, name, fmt=':6.4e'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
    
    #Update values using value, batch_number 
    #And calculate average, sqrt of batch value
    def update(self, val):
        self.val += val

    def show_avg(self):
        return self.__str__()
     
    #Show sqrt value & average value : batch_value ( average_value )
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} '
        return fmtstr.format(**self.__dict__)



#Show logfile & print information
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", suffix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.suffix = suffix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += [self.suffix]
        print('\t'.join(entries), flush=True)

    def display_epoch(self):
        entries = [self.prefix]
        entries += [meter.show_avg() for meter in self.meters]
        entries += [self.suffix]
        print('\t'.join(entries), flush=True)

    def test(self):
        entries = [self.prefix]
        entries += [meter.show_avg() for meter in self.meters]
        entries += [self.suffix]
        print('\t'.join(entries), flush=True)
        return '\t'.join(entries)+'\n'  

    def log(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += [self.suffix]
        return '\t'.join(entries)+'\n'

    def log_epoch(self):
        entries = [self.prefix]
        entries += [meter.show_avg() for meter in self.meters]
        entries += [self.suffix]
        return '\t'.join(entries)+'\n'

    def string(self):
        entries = [self.prefix]
        entries += [str(meter) for meter in self.meters]
        entries += [self.suffix]
        return ' '.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
