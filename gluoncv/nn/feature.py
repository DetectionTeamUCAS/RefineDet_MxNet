# pylint: disable=abstract-method
"""Feature extraction blocks.
Feature or Multi-Feature extraction is a key component in object detection.
Class predictor/Box predictor are usually applied on feature layer(s).
A good feature extraction mechanism is critical to performance.
"""
from __future__ import absolute_import
import mxnet as mx
from mxnet.symbol import Symbol
from mxnet.gluon import HybridBlock, SymbolBlock
from mxnet.base import string_types

def _parse_network(network, outputs, inputs, pretrained, ctx):
    """Parse network with specified outputs and other arguments.

    Parameters
    ----------
    network : str or HybridBlock or Symbol
        Logic chain: load from gluoncv.model_zoo if network is string.
        Convert to Symbol if network is HybridBlock
    outputs : str or iterable of str
        The name of layers to be extracted as features.
    inputs : iterable of str
        The name of input datas.
    pretrained : bool
        Use pretrained parameters as in gluon.model_zoo
    ctx : Context
        The context, e.g. mxnet.cpu(), mxnet.gpu(0).

    Returns
    -------
    inputs : list of Symbol
        Network input Symbols, usually ['data']
    outputs : list of Symbol
        Network output Symbols, usually as features
    params : ParameterDict
        Network parameters.
    """
    inputs = list(inputs) if isinstance(inputs, tuple) else inputs
    for i, inp in enumerate(inputs):
        if isinstance(inp, string_types):
            inputs[i] = mx.sym.var(inp)
        assert isinstance(inputs[i], Symbol), "Network expects inputs are Symbols."
    if len(inputs) == 1:
        inputs = inputs[0]
    else:
        inputs = mx.sym.Group(inputs)
    params = None
    prefix = ''
    if isinstance(network, string_types):
        from ..model_zoo import get_model
        network = get_model(network, pretrained=pretrained, ctx=ctx)
    if isinstance(network, HybridBlock):
        params = network.collect_params()
        prefix = network._prefix
        network = network(inputs)
    assert isinstance(network, Symbol), \
        "FeatureExtractor requires the network argument to be either " \
        "str, HybridBlock or Symbol, but got %s"%type(network)

    if isinstance(outputs, string_types):
        outputs = [outputs]
    assert len(outputs) > 0, "At least one outputs must be specified."
    outputs = [out if out.endswith('_output') else out + '_output' for out in outputs]
    outputs = [network.get_internals()[prefix + out] for out in outputs]
    return inputs, outputs, params


class FeatureExtractor(SymbolBlock):
    """Feature extractor.

    Parameters
    ----------
    network : str or HybridBlock or Symbol
        Logic chain: load from gluoncv.model_zoo if network is string.
        Convert to Symbol if network is HybridBlock
    outputs : str or list of str
        The name of layers to be extracted as features
    inputs : list of str or list of Symbol
        The inputs of network.
    pretrained : bool
        Use pretrained parameters as in gluon.model_zoo
    ctx : Context
        The context, e.g. mxnet.cpu(), mxnet.gpu(0).
    """
    def __init__(self, network, outputs, inputs=('data',), pretrained=False, ctx=mx.cpu()):
        inputs, outputs, params = _parse_network(network, outputs, inputs, pretrained, ctx)
        super(FeatureExtractor, self).__init__(outputs, inputs, params=params)


class FeatureExpander(SymbolBlock):
    """Feature extractor with additional layers to append.
    This is very common in vision networks where extra branches are attched to
    backbone network.

    Parameters
    ----------
    network : str or HybridBlock or Symbol
        Logic chain: load from gluoncv.model_zoo if network is string.
        Convert to Symbol if network is HybridBlock.
    outputs : str or list of str
        The name of layers to be extracted as features
    num_filters : list of int
        Number of filters to be appended.
    use_1x1_transition : bool
        Whether to use 1x1 convolution between attached layers. It is effective
        reducing network size.
    use_bn : bool
        Whether to use BatchNorm between attached layers.
    reduce_ratio : float
        Channel reduction ratio of the transition layers.
    min_depth : int
        Minimum channel number of transition layers.
    global_pool : bool
        Whether to use global pooling as the last layer.
    pretrained : bool
        Use pretrained parameters as in gluon.model_zoo if `True`.
    ctx : Context
        The context, e.g. mxnet.cpu(), mxnet.gpu(0).
    inputs : list of str
        Name of input variables to the network.

    """
    def __init__(self, network, outputs, num_filters, use_1x1_transition=True,
                 use_bn=True, reduce_ratio=1.0, min_depth=128, global_pool=False,
                 pretrained=False, ctx=mx.cpu(), inputs=('data',)):
        inputs, outputs, params = _parse_network(network, outputs, inputs, pretrained, ctx)
        # append more layers
        y = outputs[-1]
        weight_init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)
        for i, f in enumerate(num_filters):
            if use_1x1_transition:
                num_trans = max(min_depth, int(round(f * reduce_ratio)))
                y = mx.sym.Convolution(
                    y, num_filter=num_trans, kernel=(1, 1), no_bias=use_bn,
                    name='expand_trans_conv{}'.format(i), attr={'__init__': weight_init})
                if use_bn:
                    y = mx.sym.BatchNorm(y, name='expand_trans_bn{}'.format(i))
                y = mx.sym.Activation(y, act_type='relu', name='expand_trans_relu{}'.format(i))
            y = mx.sym.Convolution(
                y, num_filter=f, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                no_bias=use_bn, name='expand_conv{}'.format(i), attr={'__init__': weight_init})
            if use_bn:
                y = mx.sym.BatchNorm(y, name='expand_bn{}'.format(i))
            y = mx.sym.Activation(y, act_type='relu', name='expand_reu{}'.format(i))
            outputs.append(y)
        if global_pool:
            outputs.append(mx.sym.Pooling(y, pool_type='avg', global_pool=True, kernel=(1, 1)))
        super(FeatureExpander, self).__init__(outputs, inputs, params)
