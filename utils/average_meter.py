class AverageMeter(object):
    def __init__(self, name, lambda_=0.9):
        self.name = name
        self.lambda_ = lambda_
        self.current = None
        self.val = None
        self.step = 0
        self.reset()

    def reset(self):
        self.val = None

    def update(self, value, step):
        # exponential moving average
        self.current = value
        self.step = step
        if self.val is None:
            self.val = value
        else:
            self.val = self.lambda_ * self.val + (1 - self.lambda_) * value

    def __repr__(self):
        fmtstr = f"{self.name}:{self.step:d} loss = {self.val:.3f} (current = {self.current:.3f})"
        return fmtstr
