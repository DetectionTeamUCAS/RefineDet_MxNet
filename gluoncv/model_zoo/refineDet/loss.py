# -*- coding: utf-8 -*-

from gluoncv.loss import SSDMultiBoxLoss
from gluoncv.loss import _as_list
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like
from mxnet import nd

class RefineDetMultiBoxLoss(SSDMultiBoxLoss):

    def __init__(self, negative_mining_ratio=3, rho=1.0, lambd=1.0, ignore_label=-1, **kwargs):

        super(RefineDetMultiBoxLoss, self).__init__(negative_mining_ratio, rho, lambd, **kwargs)
        self._ignore_label = ignore_label

    def forward(self, cls_pred, box_pred, cls_target, box_target, invalid):
        """Compute loss in entire batch across devices."""
        # require results across different devices at this time
        cls_pred, box_pred, cls_target, box_target, invalid = [_as_list(x) \
            for x in (cls_pred, box_pred, cls_target, box_target, invalid)]
        # cross device reduction to obtain positive samples in entire batch
        num_pos = []
        for cp, bp, ct, bt in zip(*[cls_pred, box_pred, cls_target, box_target]):
            # cp (b, N, num_cls+1); bp (b, N, 4); ct (b, N); bt (b, N, 4)
            pos_samples = (ct > 0)
            num_pos.append(pos_samples.sum())
        num_pos_all = sum([p.asscalar() for p in num_pos])
        if num_pos_all < 1:
            # no positive samples found, return dummy losses
            return nd.zeros((1,)), nd.zeros((1,)), nd.zeros((1,))

        # compute element-wise cross entropy loss and sort, then perform negative mining
        cls_losses = []
        box_losses = []
        sum_losses = []
        for cp, bp, ct, bt, inval in zip(*[cls_pred, box_pred, cls_target, box_target, invalid]):
            # cp (b, N, num_cls+1); bp (b, N, 4); ct (b, N); bt (b, N, 4); inval (b, N)

            pred = nd.log_softmax(cp, axis=-1)  # (b, N, cls_num+1)
            pos = ct > 0  # (b, N)
            cls_loss = -nd.pick(pred, ct, axis=-1, keepdims=False)  # (b, N)
            # to ignored the classified well anchors.
            cls_loss = nd.where(inval, nd.zeros_like(cls_loss), cls_loss)

            rank = (cls_loss * (pos - 1)).argsort(axis=1).argsort(axis=1)  # get the response id in the sorted loss.
            hard_negative = rank < (pos.sum(axis=1) * self._negative_mining_ratio).expand_dims(-1)  # (b, N)
            # mask out if not positive or negative
            cls_loss = nd.where((pos + hard_negative) > 0, cls_loss, nd.zeros_like(cls_loss))
            cls_losses.append(nd.sum(cls_loss, axis=0, exclude=True) / num_pos_all)

            bp = _reshape_like(nd, bp, bt)
            box_loss = nd.abs(bp - bt)
            box_loss = nd.where(box_loss > self._rho, box_loss - 0.5 * self._rho,
                                (0.5 / self._rho) * nd.square(box_loss))
            # box loss only apply to positive samples
            box_loss = box_loss * pos.expand_dims(axis=-1)
            box_losses.append(nd.sum(box_loss, axis=0, exclude=True) / num_pos_all)
            sum_losses.append(cls_losses[-1] + self._lambd * box_losses[-1])

        return sum_losses, cls_losses, box_losses
