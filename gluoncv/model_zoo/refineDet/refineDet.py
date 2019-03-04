"""Single-Shot Refinement Neural Network for Object Detection."""
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock
from ...nn.feature import FeatureExpander
from ..ssd.anchor import SSDAnchorGenerator
from .anchor import S3FDAnchorGenerator
from ...nn.predictor import ConvPredictor
from ...nn.coder import MultiPerClassDecoder, NormalizedBoxCenterDecoder
from .vgg_atrous_fusion import vgg16_atrous_320
from ...data import VOCDetection
from .target import ODMTargetGenerator

__all__ = ['RefineDet', 'get_refineDet',
           'refineDet_320_vgg16_atrous_voc',]


class RefineDet(HybridBlock):
    """Single-Shot Refinement Neural Network for Object Detection.

    Parameters
    ----------
    network : string or None
        Name of the base network, if `None` is used, will instantiate the
        base network from `features` directly instead of composing.
    base_size : int
        Base input size, it is speficied so SSD can support dynamic input shapes.
    features : list of str or mxnet.gluon.HybridBlock
        Intermediate features to be extracted or a network with multi-output.
        If `network` is `None`, `features` is expected to be a multi-output network.
    num_filters : list of int
        Number of channels for the appended layers, ignored if `network`is `None`.
    sizes : iterable fo float
        Sizes of anchor boxes, this should be a list of floats, in incremental order.
        The length of `sizes` must be len(layers) + 1. For example, a two stage SSD
        model can have ``sizes = [30, 60, 90]``, and it converts to `[30, 60]` and
        `[60, 90]` for the two stages, respectively. For more details, please refer
        to original paper.
    ratios : iterable of list
        Aspect ratios of anchors in each output layer. Its length must be equals
        to the number of SSD output layers.
    steps : list of int
        Step size of anchor boxes in each output layer.
    classes : iterable of str
        Names of all categories.
    use_1x1_transition : bool
        Whether to use 1x1 convolution as transition layer between attached layers,
        it is effective reducing model capacity.
    use_bn : bool
        Whether to use BatchNorm layer after each attached convolutional layer.
    reduce_ratio : float
        Channel reduce ratio (0, 1) of the transition layer.
    min_depth : int
        Minimum channels for the transition layers.
    global_pool : bool
        Whether to attach a global average pooling layer as the last output layer.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    stds : tuple of float, default is (0.1, 0.1, 0.2, 0.2)
        Std values to be divided/multiplied to box encoded values.
    nms_thresh : float, default is 0.45.
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    anchor_alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `anchor_alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, scalar, etc.
    ctx : mx.Context
        Network context.

    """
    def __init__(self, network, base_size, features, num_filters, sizes, ratios,
                 steps, classes, anchor_method='ssd', filter_neg=True, use_1x1_transition=True, use_bn=True,
                 reduce_ratio=1.0, min_depth=128, global_pool=False, pretrained=False,
                 stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.45, nms_topk=400, post_nms=100,
                 anchor_alloc_size=128, negative_anchor_thresh=0.99, ctx=mx.cpu(), **kwargs):
        super(RefineDet, self).__init__(**kwargs)
        if network is None:  # for vgg
            num_layers = len(ratios)  # [[1, 2.0, 0.5]] * 4
        else:  # for resnet
            num_layers = len(features) + len(num_filters) + int(global_pool)
        assert len(sizes) == num_layers + 1
        sizes = list(zip(sizes[:-1], sizes[1:]))
        assert isinstance(ratios, list), "Must provide ratios as list or list of list"
        if not isinstance(ratios[0], (tuple, list)):
            ratios = ratios * num_layers  # propagate to all layers if use same ratio
        assert num_layers == len(sizes) == len(ratios), \
            "Mismatched (number of layers) vs (sizes) vs (ratios): {}, {}, {}".format(
                num_layers, len(sizes), len(ratios))
        assert num_layers > 0, "refineDet require at least one layer, suggest multiple."
        self._num_layers = num_layers
        self.classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self.neg_anchor_thresh = negative_anchor_thresh
        self.filter_neg_anchor = filter_neg
        self._odm_target_generator = {ODMTargetGenerator(stds=stds)}  # reGenerate targets for refined_anchors

        with self.name_scope():
            if network is None:
                # use fine-grained manually designed block as features
                self.features = features(pretrained=pretrained, ctx=ctx)
            else:
                self.features = FeatureExpander(
                    network=network, outputs=features, num_filters=num_filters,
                    use_1x1_transition=use_1x1_transition,
                    use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
                    global_pool=global_pool, pretrained=pretrained, ctx=ctx)

            # anchor refine module
            self.anchor_cls_predictors = nn.HybridSequential(prefix="anchorcls_")
            self.anchor_box_predictors = nn.HybridSequential(prefix="anchorbox_")

            # object detection module
            self.object_cls_predictors = nn.HybridSequential(prefix="objcls_")
            self.object_box_predictors = nn.HybridSequential(prefix="objbox_")

            #
            self.anchor_generators = nn.HybridSequential(prefix="anchor_gen_")
            im_size = (base_size, base_size)
            asz = anchor_alloc_size
            for i, s, r, step in zip(range(num_layers), sizes, ratios, steps):
                if anchor_method.strip() == 'ssd':
                    anchor_generator = SSDAnchorGenerator(i, im_size, s, r, step, (asz, asz))
                elif anchor_method.strip() == 's3fd':
                    anchor_generator = S3FDAnchorGenerator(i, im_size, s, r, step, (asz, asz))
                else:
                    raise NotImplementedError("do not implement other anchor methods yet")

                self.anchor_generators.add(anchor_generator)
                asz = max(asz//2, 16)
                num_anchors = anchor_generator.num_depth

                # ARM: anchor_refine_modules
                self.anchor_cls_predictors.add(ConvPredictor(num_anchors * 2))
                self.anchor_box_predictors.add(ConvPredictor(num_anchors * 4))

                # ODM: object_detection_modules
                self.object_cls_predictors.add(ConvPredictor(num_anchors * (len(self.classes) + 1)))
                self.object_box_predictors.add(ConvPredictor(num_anchors * 4))
            self.anchor_box_decoder = NormalizedBoxCenterDecoder(stds)

            self.bbox_decoder = NormalizedBoxCenterDecoder(stds, convert_anchor=True)
            self.cls_decoder = MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)

    @property
    def num_classes(self):
        """Return number of foreground classes.

        Returns
        -------
        int
            Number of foreground classes

        """
        return len(self.classes)

    @property
    def odm_target_generator(self):
        """Returns stored target generator

        Returns
        -------
        mxnet.gluon.HybridBlock
            The RCNN target generator

        """
        return list(self._odm_target_generator)[0]

    def set_nms(self, nms_thresh=0.45, nms_topk=400, post_nms=100):
        """Set non-maximum suppression parameters.

        Parameters
        ----------
        nms_thresh : float, default is 0.45.
            Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that every Detection
             result is used in NMS.
        post_nms : int, default is 100
            Only return top `post_nms` detection results, the rest is discarded. The number is
            based on COCO dataset which has maximum 100 objects per image. You can adjust this
            number if expecting more objects. You can use -1 to return all detections.

        Returns
        -------
        None

        """
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        features = self.features(x)

        # ----------------------------------------------anchor_refinement----------------------------------------------
        anchor_cls_preds = [F.flatten(F.transpose(acp(feat), (0, 2, 3, 1)))
                            for feat, acp in zip(features["ARM_features"], self.anchor_cls_predictors)]
        anchor_box_preds = [F.flatten(F.transpose(abp(feat), (0, 2, 3, 1)))
                            for feat, abp in zip(features["ARM_features"], self.anchor_box_predictors)]

        anchors = [F.reshape(ag(feat), shape=(1, -1))
                   for feat, ag in zip(features["ARM_features"], self.anchor_generators)]

        anchors = F.concat(*anchors, dim=1).reshape((1, -1, 4))  # (1, N, 4)

        anchor_cls_preds = F.concat(*anchor_cls_preds, dim=1).reshape((0, -1, 2))  # (B, N, 2)
        anchor_box_preds = F.concat(*anchor_box_preds, dim=1).reshape((0, -1, 4))  # (B, N, 4)
        negative_anchor_scores = F.slice_axis(F.softmax(F.stop_gradient(anchor_cls_preds), axis=-1),
                                              begin=0, end=1, axis=-1)  # score of negative anchor, shape is (B, N, 1)
        if self.filter_neg_anchor:
            invalid = negative_anchor_scores > self.neg_anchor_thresh
        else:
            invalid = F.zeros_like(negative_anchor_scores)  # all the anchors are valid

        # refined_anchors: [x1, y1, x2, y2]  || anchors: [x, y, w, h]. refined_anchor shape: (B, N, 4)
        refined_anchors = self.anchor_box_decoder(anchor_box_preds, anchors)

        # -----------------------------------------------object_detection----------------------------------------------

        obj_cls_preds = [F.flatten(F.transpose(ocp(feat), (0, 2, 3, 1)))
                         for feat, ocp in zip(features["ODM_features"], self.object_cls_predictors)]
        obj_box_preds = [F.flatten(F.transpose(obp(feat), (0, 2, 3, 1)))
                         for feat, obp in zip(features["ODM_features"], self.object_box_predictors)]

        obj_cls_preds = F.concat(*obj_cls_preds, dim=1).reshape((0, -1, self.num_classes + 1))  # (B, N, cls_num + 1)
        obj_box_preds = F.concat(*obj_box_preds, dim=1).reshape((0, -1, 4))  # (B, N, 4)

        # -------------------------------------------------------------------------------------------------------------

        if autograd.is_training():
            return [anchors, anchor_cls_preds, anchor_box_preds, refined_anchors, obj_cls_preds, obj_box_preds, invalid]
            # shape info: (1, N. 4). (B, N, 2), (B, N, 4), (B, N, 4), (B, N, num_cls+1), (B, N, 4), (B, N, 1)

        obj_bboxes = self.bbox_decoder(obj_box_preds, refined_anchors)  # [x1, y1, x2, y2], (B, N, 4)
        obj_cls_ids, obj_scores = self.cls_decoder(F.softmax(obj_cls_preds, axis=-1))
        # cls_ids and scores shape: (B, N, cls_num)

        # ignore negative anchors
        box_invalid = F.repeat(invalid, axis=-1, repeats=4)
        score_invalid = F.repeat(invalid, axis=-1, repeats=self.num_classes)
        id_invalid = F.repeat(invalid, axis=-1, repeats=self.num_classes)

        obj_cls_ids = F.where(id_invalid, F.ones_like(obj_cls_ids) * -1, obj_cls_ids)
        obj_bboxes = F.where(box_invalid, F.ones_like(obj_bboxes) * -1, obj_bboxes)
        obj_scores = F.where(score_invalid, F.zeros_like(obj_scores), obj_scores)
        # # #

        results = []
        for i in range(self.num_classes):
            cls_id = obj_cls_ids.slice_axis(axis=-1, begin=i, end=i + 1)
            score = obj_scores.slice_axis(axis=-1, begin=i, end=i + 1)
            per_result = F.concat(*[cls_id, score, obj_bboxes], dim=-1)
            results.append(per_result)

        result = F.concat(*results, dim=1)

        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=2, begin=2, end=6)

        return ids, scores, bboxes



def get_refineDet(name, base_size, features, filters, sizes, ratios, steps, classes,
            dataset, pretrained=False, pretrained_base=True, ctx=mx.cpu(),
            root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """Get refineDet models.

    Parameters
    ----------
    name : str or None
        Model name, if `None` is used, you must specify `features` to be a `HybridBlock`.
    base_size : int
        Base image size for training, this is fixed once training is assigned.
        A fixed base size still allows you to have variable input size during test.
    features : iterable of str or `HybridBlock`
        List of network internal output names, in order to specify which layers are
        used for predicting bbox values.
        If `name` is `None`, `features` must be a `HybridBlock` which generate mutliple
        outputs for prediction.
    filters : iterable of float or None
        List of convolution layer channels which is going to be appended to the base
        network feature extractor. If `name` is `None`, this is ignored.
    sizes : iterable fo float
        Sizes of anchor boxes, this should be a list of floats, in incremental order.
        The length of `sizes` must be len(layers) + 1. For example, a two stage SSD
        model can have ``sizes = [30, 60, 90]``, and it converts to `[30, 60]` and
        `[60, 90]` for the two stages, respectively. For more details, please refer
        to original paper.
    ratios : iterable of list
        Aspect ratios of anchors in each output layer. Its length must be equals
        to the number of SSD output layers.
    steps : list of int
        Step size of anchor boxes in each output layer.
    classes : iterable of str
        Names of categories.
    dataset : str
        Name of dataset. This is used to identify model name because models trained on
        differnet datasets are going to be very different.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    pretrained_base = False if pretrained else pretrained_base
    base_name = None if callable(features) else name
    net = RefineDet(base_name, base_size, features, filters, sizes, ratios, steps, filter_neg=True,
                    pretrained=pretrained_base, classes=classes, ctx=ctx, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('refinedet', str(base_size), name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx)
    return net


def refineDet_320_vgg16_atrous_voc(pretrained=False, pretrained_base=True, **kwargs):
    """refineDet architecture with VGG16 atrous 320x320 base network for Pascal VOC.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A refineDet detection network.
    """
    classes = VOCDetection.CLASSES
    net = get_refineDet('vgg16_atrous', 320, features=vgg16_atrous_320, filters=None,
                  sizes=[32, 64, 128, 256, 320],
                  ratios=[[1, 2, 0.5]] * 4,
                  steps=[8, 16, 32, 64],
                  classes=classes, dataset='voc', pretrained=pretrained,
                  pretrained_base=pretrained_base, **kwargs)
    return net

