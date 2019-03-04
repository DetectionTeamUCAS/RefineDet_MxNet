# pylint: disable=unused-import
"""Anchor box generator for S3FD: Single Shot Scale-invariant Face Detector"""
from __future__ import absolute_import

import numpy as np
from mxnet import gluon
from gluoncv.model_zoo.rpn.anchor import RPNAnchorGenerator
from gluoncv.model_zoo.ssd.anchor import SSDAnchorGenerator


class S3FDAnchorGenerator(SSDAnchorGenerator):
    def __init__(self, index, im_size, sizes, ratios, step, alloc_size=(128, 128),
                 offsets=(0.5, 0.5), clip=False, **kwargs):
        assert step * 4 == sizes[0], "step*4 not equal to anchor size"
        super(S3FDAnchorGenerator, self).__init__(index, im_size, sizes, ratios, step,
                                                  alloc_size=alloc_size, offsets=offsets,
                                                  clip=clip,**kwargs)

    @property
    def num_depth(self):
        """Number of anchors at each pixel."""
        return len(self._ratios)

    def _generate_anchors(self, sizes, ratios, step, alloc_size, offsets):
        """Generate anchors for once. Anchors are stored with (center_x, center_y, w, h) format.
           Diff from SSD anchors which have two sizes in a location, S3FD anchors only include 1 size.
        """
        anchors = []
        for i in range(alloc_size[0]):
            for j in range(alloc_size[1]):
                cy = (i + offsets[0]) * step
                cx = (j + offsets[1]) * step
                # ratio = ratios[0], size = size_min or sqrt(size_min * size_max)
                r = ratios[0]
                anchors.append([cx, cy, sizes[0], sizes[0]])  # one size
                # anchors.append([cx, cy, sizes[1], sizes[1]])
                # size = sizes[0], ratio = ...
                for r in ratios[1:]:
                    sr = np.sqrt(r)
                    w = sizes[0] * sr
                    h = sizes[0] / sr
                    anchors.append([cx, cy, w, h])
        return np.array(anchors).reshape(1, 1, alloc_size[0], alloc_size[1], -1)