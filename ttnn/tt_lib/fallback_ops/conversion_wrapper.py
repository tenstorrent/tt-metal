# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tt_lib import tensor as ttl_tensor
import ttnn
import torch
from functools import wraps
from loguru import logger
from contextlib import contextmanager


# Log only once to not pollute output
def check_log_pytorch_warning(arg):
    if not getattr(check_log_pytorch_warning, "_pytorch_warning_logged", False) and torch.is_tensor(arg):
        logger.warning(
            "Pytorch tensor was passed as input to fallback op instead of TT tensor. This is currently supported to improve perf but support for this will be deprecated."
        )
        check_log_pytorch_warning._pytorch_warning_logged = True


@contextmanager
def custom_tensor_print_handler(tensor_cls):
    def custom_tt_tensor_to_str_fn(tensor):
        # We just report that this was a tt tensor and its shape as detailed information is already reported in other columns
        return f"tt_lib.tensor.Tensor({'_'.join(map(str, tensor.get_legacy_shape()))})"

    def custom_pt_tensor_to_str_fn(tensor):
        return f"torch.Tensor({'|'.join(['_'.join(map(str, tensor.shape)), str(tensor.layout), str(tensor.dtype), str(tensor.device)])})"

    # Save original methods
    tensor_str_og = tensor_cls.__str__
    tensor_repr_og = tensor_cls.__repr__
    if tensor_cls == ttl_tensor.Tensor:
        custom_tensor_to_str_fn = custom_tt_tensor_to_str_fn
    elif tensor_cls == torch.Tensor:
        custom_tensor_to_str_fn = custom_pt_tensor_to_str_fn
    else:
        assert False, f"No custom tensor str fn found for class {tensor_cls}"
    # Replace methods
    tensor_cls.__str__ = custom_tensor_to_str_fn
    tensor_cls.__repr__ = custom_tensor_to_str_fn
    try:
        yield None
    finally:
        # Restore methods
        tensor_cls.__str__ = tensor_str_og
        tensor_cls.__repr__ = tensor_repr_og


def convert_tt_tensor_to_pt_tensor(tt_tensor, output_format):
    # Update output_format with format of first encountered arg
    if (
        output_format.get("device", None) is None
        and tt_tensor.storage_type() == ttl_tensor.StorageType.DEVICE
        and output_format["on_device"]
    ):
        output_format["device"] = tt_tensor.device()

    tt_tensor = tt_tensor.cpu()
    tt_tensor = tt_tensor.to(ttl_tensor.Layout.ROW_MAJOR)
    torch_tensor = tt_tensor.to_torch()
    # Required as not all torch ops/layers support bfloat16
    if torch_tensor.dtype == torch.bfloat16:
        torch_tensor = torch_tensor.float()
    return torch_tensor


def convert_pt_tensor_to_tt_tensor(pt_tensor, output_format):
    output_shape = pt_tensor.shape
    if len(output_shape) < 4:
        output_shape = [1] * (4 - len(output_shape)) + output_shape

    # Required as tt ops don't currently support float
    if pt_tensor.dtype == torch.float32:
        pt_tensor = pt_tensor.bfloat16()
    tt_tensor = ttl_tensor.Tensor(pt_tensor.reshape(output_shape))

    if output_format["layout"] == ttl_tensor.Layout.TILE:
        if (
            tt_tensor.get_legacy_shape()[2] % 32 == 0 and tt_tensor.get_legacy_shape()[3] % 32 == 0
        ):  # Restore tile layout only if legal or else leave as RM
            tt_tensor = tt_tensor.to(ttl_tensor.Layout.TILE)
    else:
        assert output_format["layout"] == ttl_tensor.Layout.ROW_MAJOR

    if output_format["on_device"]:
        assert "device" in output_format
        assert isinstance(output_format["device"], ttnn.Device)
        if (
            tt_tensor.get_layout() == ttl_tensor.Layout.TILE
            or tt_tensor.get_layout() == ttl_tensor.Layout.ROW_MAJOR
            and tt_tensor.get_legacy_shape()[3] % 2 == 0
        ):
            tt_tensor = tt_tensor.to(output_format["device"])
    return tt_tensor


def convert_tt_tensors_to_pt_tensors(args, output_format):
    check_log_pytorch_warning(args)
    if isinstance(args, ttl_tensor.Tensor):
        return convert_tt_tensor_to_pt_tensor(args, output_format)
    elif isinstance(args, dict):
        outputs = {}
        for key, value in args.items():
            outputs[key] = convert_tt_tensors_to_pt_tensors(value, output_format)
        return outputs
    elif isinstance(args, (list, tuple)):
        outputs = []
        for arg in args:
            outputs.append(convert_tt_tensors_to_pt_tensors(arg, output_format))
        return outputs
    else:
        check_log_pytorch_warning(args)
        return args


def convert_pt_tensors_to_tt_tensors(args, output_format):
    if isinstance(args, torch.Tensor):
        return convert_pt_tensor_to_tt_tensor(args, output_format)
    elif isinstance(args, dict):
        outputs = []
        for key, value in args.items():
            outputs[key] = convert_pt_tensors_to_tt_tensors(value, output_format)
        return outputs
    elif isinstance(args, (list, tuple)):
        outputs = []
        for arg in args:
            outputs.append(convert_pt_tensors_to_tt_tensors(arg, output_format))
        return outputs
    else:
        return args


def convert_tt_tensors_wrapper(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        output_format = {}
        if "output_on_device" in kwargs:
            output_format["on_device"] = kwargs["output_on_device"]
        else:
            output_format["on_device"] = True
        if "output_layout" in kwargs:
            output_format["layout"] = kwargs["output_layout"]
        else:
            output_format["layout"] = ttl_tensor.Layout.TILE

        new_args = convert_tt_tensors_to_pt_tensors(args, output_format)
        new_kwargs = convert_tt_tensors_to_pt_tensors(kwargs, output_format)

        # Set default output format
        if output_format.get("device", None) is None and output_format["on_device"]:
            output_format["device"] = ttnn.GetDefaultDevice()

        outputs = func(*new_args, **new_kwargs)

        # Convert pt tensors in outputs to tt tensors
        new_outputs = convert_pt_tensors_to_tt_tensors(outputs, output_format)

        return new_outputs

    return wrap
