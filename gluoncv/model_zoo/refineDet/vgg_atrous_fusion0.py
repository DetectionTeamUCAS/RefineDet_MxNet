# pylint: disable=arguments-differ
"""VGG atrous network for object detection. support fusion layers
   2018.12.04
"""
from __future__ import division
import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.initializer import Xavier

__all__ = ['VGGAtrousExtractor', 'get_vgg_atrous_extractor', 'vgg16_atrous_300',
           'vgg16_atrous_512']


class Normalize(gluon.HybridBlock):
    """Normalize layer described in https://arxiv.org/abs/1512.02325.

    Parameters
    ----------
    n_channel : int
        Number of channels of input.
    initial : float
        Initial value for the rescaling factor.
    eps : float
        Small value to avoid division by zero.

    """
    def __init__(self, n_channel, initial=1, eps=1e-5):
        super(Normalize, self).__init__()
        self.eps = eps
        with self.name_scope():
            self.scale = self.params.get('normalize_scale', shape=(1, n_channel, 1, 1),
                                         init=mx.init.Constant(initial))

    def hybrid_forward(self, F, x, scale):
        x = F.L2Normalization(x, mode='channel', eps=self.eps)
        return F.broadcast_mul(x, scale)

class VGGAtrousBase(gluon.HybridBlock):
    """VGG Atrous multi layer base network. You must inherit from it to define
    how the features are computed.

    Parameters
    ----------
    layers : list of int
        Number of layer for vgg base network.
    filters : list of int
        Number of convolution filters for each layer.
    batch_norm : bool, default is False
        If `True`, will use BatchNorm layers.

    """
    def __init__(self, layers, filters, batch_norm=False, **kwargs):
        super(VGGAtrousBase, self).__init__(**kwargs)
        assert len(layers) == len(filters)
        self.init = {
            'weight_initializer': Xavier(
                rnd_type='gaussian', factor_type='out', magnitude=2),
            'bias_initializer': 'zeros'
        }
        with self.name_scope():
            # we use pre-trained weights from caffe, initial scale must change
            init_scale = mx.nd.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) * 255
            self.init_scale = self.params.get_constant('init_scale', init_scale)
            self.stages = nn.HybridSequential()
            for l, f in zip(layers, filters):
                stage = nn.HybridSequential(prefix='')
                with stage.name_scope():
                    for _ in range(l):
                        stage.add(nn.Conv2D(f, kernel_size=3, padding=1, **self.init))
                        if batch_norm:
                            stage.add(nn.BatchNorm())
                        stage.add(nn.Activation('relu'))
                self.stages.add(stage)
            # self.stages have [stage1, stage2, stage3, stage4, stage5] now
            # the stride are   [1     , 2     , 4     , 8     , 16] before max pool.

            # use dilated convolution instead of dense layers
            stage = nn.HybridSequential(prefix='dilated_')
            with stage.name_scope():
                stage.add(nn.Conv2D(1024, kernel_size=3, padding=6, dilation=6, **self.init))
                if batch_norm:
                    stage.add(nn.BatchNorm())
                stage.add(nn.Activation('relu'))
                stage.add(nn.Conv2D(1024, kernel_size=1, **self.init))
                if batch_norm:
                    stage.add(nn.BatchNorm())
                stage.add(nn.Activation('relu'))
            self.stages.add(stage)
            # self.stages ==>[stage1, ..., stage5, stage6(the dense layers in vgg)].
            # the stride of stage6 is 32.

            # normalize layer for 4-th and 5-th stage
            self.norm4 = Normalize(filters[3], 10)
            self.norm5 = Normalize(filters[4], 8)

    def hybrid_forward(self, F, x, init_scale):
        raise NotImplementedError

class VGGAtrousExtractor(VGGAtrousBase):
    """VGG Atrous multi layer feature extractor which produces multiple output
    feauture maps.

    Parameters
    ----------
    layers : list of int
        Number of layer for vgg base network.
    filters : list of int
        Number of convolution filters for each layer.
    extras : list of list
        Extra layers configurations.
    batch_norm : bool
        If `True`, will use BatchNorm layers.

    """
    def __init__(self, layers, filters, extras, batch_norm=False, **kwargs):
        super(VGGAtrousExtractor, self).__init__(layers, filters, batch_norm, **kwargs)
        with self.name_scope():
            self.extras = nn.HybridSequential()
            for i, config in enumerate(extras):
                extra = nn.HybridSequential(prefix='extra%d_'%(i))
                with extra.name_scope():
                    for f, k, s, p in config:
                        extra.add(nn.Conv2D(f, k, s, p, **self.init))
                        if batch_norm:
                            extra.add(nn.BatchNorm())
                        extra.add(nn.Activation('relu'))
                self.extras.add(extra)

            self.transitions = nn.HybridSequential()
            for _ in range(4):
                # since it has 4 levels, 4 transition modules are needed.
                transition = nn.HybridSequential(prefix="transition%d_" % _)
                # [conv3_1_(bn)relu, conv3_1_(bn), conv3_1_(bn)relu]

                for i in range(3):
                    sub_transitions = nn.HybridSequential()
                    sub_transitions.add(nn.Conv2D(256, kernel_size=(3, 3,), strides=(1, 1), padding=1, **self.init))
                    if batch_norm:
                        sub_transitions.add(nn.BatchNorm())
                    if i != 1:
                        # the second sub_block do not have "relu".
                        sub_transitions.add(nn.Activation('relu'))
                    transition.add(sub_transitions)
                self.transitions.add(transition)

            # upsample module
            self.upsamples = nn.HybridSequential()
            for _ in range(3):
                upsample = nn.HybridSequential(prefix="upsample%d_" % _)
                upsample.add(nn.Conv2DTranspose(channels=256, kernel_size=(4, 4), strides=(2, 2), **self.init))
                if batch_norm:
                    upsample.add(nn.BatchNorm())
                self.upsamples.add(upsample)

    def hybrid_forward(self, F, x, init_scale):
        x = F.broadcast_mul(x, init_scale)
        assert len(self.stages) == 6
        # outputs = []
        outputs = {"ARM_features": [],
                   "ODM_features": []}
        for stage in self.stages[:3]:
            x = stage(x)
            x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2),
                          pooling_convention='full')
        x = self.stages[3](x)
        norm4 = self.norm4(x)  # norm for conv4_3
        outputs["ARM_features"].append(norm4)  # conv4_3
        x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2),
                      pooling_convention='full')

        x = self.stages[4](x)
        norm5 = self.norm5(x)  # norm for conv5_3
        outputs["ARM_features"].append(norm5)  # conv5_3
        x = F.Pooling(x, pool_type='max', kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      pooling_convention='full')

        x = self.stages[5](x)
        outputs["ARM_features"].append(x)  # conv_fc7
        for extra in self.extras:
            x = extra(x)
            outputs["ARM_features"].append(x)
        # outputs["ARM_features] ==>[conv4_3, conv5_3, conv_fc7, conv6_2]
        # strid are ==>             [8      , 16     , 32      , 64     ]

        # ---------------------------------------------- features fusion ----------------------------------------------
        # transitions module
        # [conv3_1_(bn)relu, conv3_1_(bn), conv3_1_(bn)relu]
        outputs["ODM_features"] = [None] * 4

        # for conv6_2. do not upsample
        tmp = F.Activation(self.transitions[-1][:2](outputs['ARM_features'][-1]),
                           act_type='relu')
        outputs["ODM_features"][-1] = self.transitions[-1][2](tmp)

        levels = len(outputs["ARM_features"])  # levels = 4
        for i in range(levels - 1, 0, -1):
            fet_deep, fet_shallow = outputs["ODM_features"][i], outputs["ARM_features"][i-1]
            upsampled_fet = self.upsamples[i-1](fet_deep)
            transition_fet = self.transitions[i-1][:2](fet_shallow)
            upsampled_fet = F.slice_like(upsampled_fet, transition_fet*0, axes=(2, 3))  # [b, C, H, W]
            tmp = F.Activation(upsampled_fet + transition_fet, act_type='relu')
            outputs["ODM_features"][i-1] = self.transitions[i-1][2](tmp)
        return outputs

vgg_spec = {
    11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
    13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
    16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
    19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])
}

extra_spec = {
    300: [((256, 1, 1, 0), (512, 3, 2, 1))],  # only two extra conv layers
    512: [((256, 1, 1, 0), (512, 3, 2, 1))],
}

def get_vgg_atrous_extractor(num_layers, im_size, pretrained=False, ctx=mx.cpu(),
                             root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """Get VGG atrous feature extractor networks.

    Parameters
    ----------
    num_layers : int
        VGG types, can be 11,13,16,19.
    im_size : int
        VGG detection input size, can be 300, 512.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : mx.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    mxnet.gluon.HybridBlock
        The returned network.

    """
    layers, filters = vgg_spec[num_layers]
    extras = extra_spec[im_size]
    net = VGGAtrousExtractor(layers, filters, extras, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        batch_norm_suffix = '_bn' if kwargs.get('batch_norm') else ''
        net.initialize(ctx=ctx)
        net.load_parameters(get_model_file('vgg%d_atrous%s' % (num_layers, batch_norm_suffix),
                                           tag=pretrained, root=root), ctx=ctx, allow_missing=True)
    return net

def vgg16_atrous_300(**kwargs):
    """Get VGG atrous 16 layer 300 in_size feature extractor networks."""
    return get_vgg_atrous_extractor(16, 300, **kwargs)

def vgg16_atrous_512(**kwargs):
    """Get VGG atrous 16 layer 512 in_size feature extractor networks."""
    return get_vgg_atrous_extractor(16, 512, **kwargs)
