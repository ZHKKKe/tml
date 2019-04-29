import sys


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {
            name + postfix: meter.val
            for name, meter in self.meters.items()
        }

    def averages(self, postfix='/avg'):
        return {
            name + postfix: meter.avg
            for name, meter in self.meters.items()
        }

    def sums(self, postfix='/sum'):
        return {
            name + postfix: meter.sum
            for name, meter in self.meters.items()
        }

    def counts(self, postfix='/count'):
        return {
            name + postfix: meter.count
            for name, meter in self.meters.items()
        }


class AverageMeter:
    """ Computes and stores the average and current value"""

    def __init__(self):
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

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format)

