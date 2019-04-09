# This code is from https://github.com/bckenstler/CLR

from keras.callbacks import *

class OneCycleLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).

    ref: http://arxiv.org/abs/1803.09820

    - Example
        ```python
            clr = OneCycleLR(num_iters=10000, max_lr=0.003)
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    - Arguments
        num_iters: number of iterations of the whole training process.
        max_lr: upper bound of lr in the cycle.
        moms: upper and lower bounds of momentum in the cycle
        div_factor: init_lr = max_lr / div_factor.
        sep_ratio: ratio of iterations to segment the cycle.
        final_div: final_lr = max_lr / final_div.
    """

    def __init__(self, num_iters, max_lr=0.006, moms=(.95, .85),
                 div_factor=25, sep_ratio=0.3, final_div=None):
        super(OneCycleLR, self).__init__()

        self.max_lr = max_lr
        self.init_lr = max_lr/div_factor
        if final_div is None: final_div = div_factor*1e4
        self.final_lr = max_lr / final_div
        self.moms = moms

        self.up_iteration = int(num_iters * sep_ratio)
        self.down_iteration = num_iters - self.up_iteration

        self.curr_iter = 0
        self.history = {}

    def _annealing_cos(self, start, end, pct):
        cos_out = np.cos(np.pi * pct) + 1
        return end + (start-end)/2 * cos_out

    def step(self):
        if self.curr_iter <= self.up_iteration:
            pct = self.curr_iter / self.up_iteration
            curr_lr = self._annealing_cos(self.init_lr, self.max_lr, pct)
            curr_mom = self._annealing_cos(self.moms[0], self.moms[1], pct)
        else:
            pct = (self.curr_iter-self.up_iteration) / self.down_iteration
            curr_lr = self._annealing_cos(self.max_lr, self.final_lr, pct)
            curr_mom = self._annealing_cos(self.moms[1], self.moms[0], pct)
        return curr_lr, curr_mom

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.curr_iter == 0:
            K.set_value(self.model.optimizer.lr, self.init_lr)
            K.set_value(self.model.optimizer.momentum, self.moms[0])
        else:
            curr_lr, curr_mom = self.step()
            K.set_value(self.model.optimizer.lr, curr_lr)
            K.set_value(self.model.optimizer.momentum, curr_mom)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.curr_iter += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.curr_iter)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        curr_lr, curr_mom = self.step()
        K.set_value(self.model.optimizer.lr, curr_lr)
        K.set_value(self.model.optimizer.momentum, curr_mom)


