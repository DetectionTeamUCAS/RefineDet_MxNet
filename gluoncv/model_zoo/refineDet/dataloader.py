# -*- coding: utf-8 -*-
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.data.transforms import experimental
import numpy as np
import mxnet as mx
from gluoncv.data.transforms import bbox as tbbox
from gluoncv.data.transforms import image as timage

class RefineDetDefaultTrainTransform(SSDDefaultTrainTransform):
    def __init__(self, width, height, anchors=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2),
                 **kwargs):
        super(RefineDetDefaultTrainTransform, self).__init__(width, height, anchors, mean, std, iou_thresh, box_norm,
                                                             **kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # random color jittering
        img = experimental.image.random_color_distort(src)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
            bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = img, label

        # random cropping
        h, w, _ = img.shape
        bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = timage.imresize(img, self._width, self._height, interp=interp)
        bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype(img.dtype)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        anchor_cls_targets, anchor_box_targets, _ = self._target_generator(
            self._anchors, None, gt_bboxes, gt_ids)
        anchor_cls_targets = mx.nd.where(anchor_cls_targets > 0,
                                         mx.nd.ones_like(anchor_cls_targets),
                                         anchor_cls_targets)
        # positive anchor is 1, negative anchor is 0 and ignored is -1.
        return img, anchor_cls_targets[0], anchor_box_targets[0], bbox[:, :5].astype(img.dtype)



