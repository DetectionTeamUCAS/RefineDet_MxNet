"""SSD training target generator."""
from __future__ import absolute_import

from mxnet import nd
from mxnet.gluon import Block
from ...nn.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher
from ...nn.sampler import OHEMSampler, NaiveSampler
from ...nn.coder import MultiClassEncoder, NormalizedBoxCenterEncoder
from ...nn.bbox import BBoxCenterToCorner
from mxnet import autograd

class ODMTargetGenerator(Block):
    """Training targets generator for RefineDet.
    We only compute the targets for each refined anchor.
    Do not implement the hard negative mining here.
    Parameters
    ----------
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    stds : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.
    """
    def __init__(self, iou_thresh=0.5,
                 stds=(0.1, 0.1, 0.2, 0.2), **kwargs):
        super(ODMTargetGenerator, self).__init__(**kwargs)
        self._matcher = CompositeMatcher([BipartiteMatcher(), MaximumMatcher(iou_thresh)])

        self._sampler = NaiveSampler()  # implement the hard negative mining in the Loss Block
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder(stds=stds)
        # self._center_to_corner = BBoxCenterToCorner(split=False)

    # pylint: disable=arguments-differ
    def forward(self, refined_anchors, targets, num_objects):
        """Generate training targets.
        Parameters
        ----------
        refined_anchors. corner boxes. i.e. (xmin, ymin, xmax, ymax). (B, N, 4)
        targets: shape is (B, P, 5). (xmin, ymin, xmax, ymax, label)
        num_objects: shape is (B, ). the num of objects in each img.
        """
        cls_targets = []
        box_targets = []
        box_masks = []
        with autograd.pause():
            for refined_anchor, target, num_object in zip(refined_anchors, targets, num_objects):
                # shape is (N, 4), (P, 5), scalar
                target = nd.slice_axis(target, axis=0, begin=0, end=num_object[0].asscalar())  # (M, 5)
                gt_id = nd.slice_axis(target, axis=1, begin=-1, end=None).reshape((1, -1))  # (M, 1) -> (1, M)
                gt_box = nd.slice_axis(target, axis=1, begin=0, end=-1).reshape((1, -1, 4))  # (M, 4) -> (1, M, 4)

                # ious (N, 1, M) --> (1, N, M)
                ious = nd.transpose(nd.contrib.box_iou(refined_anchor, gt_box), (1, 0, 2))
                matches = self._matcher(ious)  # matched_object: 0<= val<= M-1, not-matched is -1. shape: (1, N)
                samples = self._sampler(matches)  # object is +1,  bg is -1. ignored is 0. (1, N)

                cls_target = self._cls_encoder(samples, matches, gt_id)  # (1, N).
                # cls_targets: >1 for objects(fg); 0 for bg; -1 for ignored;

                refined_anchor = nd.expand_dims(refined_anchor, axis=0)  # (N, 4) --> (1, N, 4)
                box_target, box_mask = self._box_encoder(samples, matches, refined_anchor, gt_box)  # (1, N, 4)

                cls_targets.append(cls_target)
                box_targets.append(box_target)
                box_masks.append(box_mask)
            cls_targets = nd.concat(*cls_targets, dim=0)  # (B, N)
            box_targets = nd.concat(*box_targets, dim=0)  # (B, N, 4)
            box_masks = nd.concat(*box_masks, dim=0)  # (B, N, 4). positive box are 1.0 others are 0.0.

            # cls_targets: >1 for objects(fg); 0 for bg; -1 for ignored;
            return cls_targets, box_targets, box_masks



