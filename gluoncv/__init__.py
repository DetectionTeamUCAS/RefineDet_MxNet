# coding: utf-8
# pylint: disable=wrong-import-position
"""GluonCV: a deep learning vision toolkit powered by Gluon."""
from __future__ import absolute_import

# mxnet version check
mx_version = '1.3.0'
try:
    import mxnet as mx
    from distutils.version import LooseVersion
    if LooseVersion(mx.__version__) < LooseVersion(mx_version):
        msg = (
            "Legacy mxnet=={} detected, some new modules may not work properly. "
            "mxnet>={} is required. You can use pip to upgrade mxnet "
            "`pip install mxnet/mxnet-cu90 --pre --upgrade`").format(mx.__version__, mx_version)
        raise ImportError(msg)
except ImportError:
    raise ImportError(
        "Unable to import dependency mxnet. "
        "A quick tip is to install via `pip install mxnet/mxnet-cu90 --pre`. "
        "please refer to https://gluon-cv.mxnet.io/#installation for details.")

__version__ = '0.4.0'

from . import data
from . import model_zoo
from . import nn
from . import utils
from . import loss
