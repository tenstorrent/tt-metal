import copy
import torch
import tt_lib as ttl
from contextlib import AbstractContextManager
from loguru import logger
from functools import wraps
from tt_lib.fallback_ops import fallback_ops
from tt_models.utility_functions import (
    run_conv_on_device_wrapper,
    is_conv_supported_on_device,
)


class UseDeviceConcat:
    READY = True


class UseDeviceConv:
    READY = True

def disable_concat(fn):
    @wraps(fn)
    def __wrapper__(*args,**kwargs):
        with DisableDeviceConvAndConcat(use_concat=False,use_conv=True) as _:
            values = fn(*args,**kwargs)
        return values
    return __wrapper__

def disable_conv(fn):
    @wraps(fn)
    def __wrapper__(*args,**kwargs):
        with DisableDeviceConvAndConcat(use_concat=True,use_conv=False) as _:
            values = fn(*args,**kwargs)
        return values
    return __wrapper__

def disable_conv_and_concat(fn):
    @wraps(fn)
    def __wrapper__(*args,**kwargs):
        with DisableDeviceConvAndConcat() as _:
            values = fn(*args,**kwargs)
        return values
    return __wrapper__

class DisableDeviceConvAndConcat(AbstractContextManager):
    """useful for testing"""

    def __init__(self,use_conv=False,use_concat=False):
        self.state = [UseDeviceConcat.READY, UseDeviceConv.READY]
        UseDeviceConv.READY = use_conv
        UseDeviceConcat.READY = use_concat
        logger.debug('Disabled Device Conv, and Concat operators.')


    def __exit__(self, exc_type, exc_value, traceback):
        UseDeviceConcat.READY = self.state[0]
        UseDeviceConv.READY = self.state[1]
        logger.debug('Restored Device Conv, and Concat operators.')
        del self.state


def parse_conv2d_interface(
    conv_weight=None, conv_bias=None, in_channels=-1, out_channels=-1, **kwargs
):
    # conv_weight, conv_bias, in_channels, out_channels, **kwargs
    # conv_weight, conv_bias, in_channels, out_channels, kernel_size, stride, padding
    if "biases" in kwargs:
        bias = kwargs["biases"]
    elif "bias" in kwargs:
        bias = kwargs["bias"]
    else:
        bias = conv_bias
    if "weights" in kwargs:
        weights = kwargs["weights"]
    else:
        weights = conv_weight
    kernel_size = kwargs["kernel_size"]  # required
    stride = kwargs.get("stride", (1, 1))
    padding = kwargs.get("padding", (0, 0))
    dilation = kwargs.get("dilation", 1)
    groups = kwargs.get("groups", 1)

    if isinstance(kernel_size, (list, tuple)):
        kernel_x, kernel_y = kernel_size
    else:
        kernel_x = kernel_size
        kernel_y = kernel_size

    if isinstance(padding, (list, tuple)):
        padding_x, padding_y = padding
    else:
        padding_x = padding
        padding_y = padding

    if isinstance(stride, (list, tuple)):
        stride_x, stride_y = stride
    else:
        stride_x = stride
        stride_y = stride

    # in run_conv_on_device_wrapper format
    params = [
        out_channels,
        in_channels,
        kernel_x,
        kernel_y,
        stride_x,
        stride_y,
        padding_x,
        padding_y,
        dilation,
        groups,
    ]
    return weights, bias, params


def Conv2d(*args, **kwargs):
    if UseDeviceConv.READY:
        conv1_weight, conv1_bias, conv1_params = parse_conv2d_interface(*args, **kwargs)
        device = ttl.device.GetDefaultDevice()
        return run_conv_on_device_wrapper(
            conv1_weight.reshape(-1).tolist(),
            conv1_params,
            device,
            conv1_bias.tolist() if conv1_bias is not None else None,
            channel_transpose=True,
        )

    return fallback_ops.Conv2d(*args, **kwargs)


def concat(*args, **kwargs):
    device = ttl.device.GetDefaultDevice()
    if UseDeviceConcat.READY:
        dim = kwargs.get("dim", 0)


        #force move tensor to the Device.
        #breakpoint()
        _args = copy.copy(args[0])
        for idx,_ in enumerate(args[0]):
            if not isinstance(_,ttl.tensor.Tensor):
                #cannot convert torch tensor to device tensor
                _args[idx] = ttl.tensor.Tensor(_.reshape(-1).tolist(),_.shape,ttl.tensor.Layout.ROW_MAJOR).to(device) #,ttl.tensor.TILE)
                #raise ValueError("all tensors need to be on device for concat")
            _args[idx] = _
            assert isinstance(_args[idx],ttl.tensor.Tensor)
            if _args[idx].storage_type() != ttl.tensor.StorageType.DEVICE:
                _args[idx] = _args[idx].to(device)

        return ttl.tensor.concat(*_args, dim)

    return fallback_ops.concat(*args, **kwargs)
