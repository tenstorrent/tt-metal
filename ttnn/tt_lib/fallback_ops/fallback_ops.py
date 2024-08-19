# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from typing import List, Tuple, Union, Optional
from .conversion_wrapper import convert_tt_tensors_wrapper
from tt_lib import tensor as ttl_tensor

# python 3.10 has types.EllipsisType
EllipsisType = type(Ellipsis)


@convert_tt_tensors_wrapper
def full(size: List[int], fill_value: float) -> ttl_tensor.Tensor:
    """
    Creates a ``tt_lib.tensor.Tensor`` of shape ``size`` filled with ``fill_value`` value.

    +------------+-----------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                             | Data type   | Valid range     | Required |
    +============+=========================================+=============+=================+==========+
    | size       | Shape of output tensor                  | List[int]   | list of 4 ints  | Yes      |
    +------------+-----------------------------------------+-------------+-----------------+----------+
    | fill_value | Value with which to fill output tensor  | float       |                 | Yes      |
    +------------+-----------------------------------------+-------------+-----------------+----------+
    """
    return torch.full(size, fill_value, dtype=torch.float32)


@convert_tt_tensors_wrapper
def tensor_slice(input: ttl_tensor.Tensor, slices: List[Union[slice, EllipsisType]]) -> ttl_tensor.Tensor:
    """
    Creates a ``tt_lib.tensor.Tensor`` from ``input`` using ``slices``.
    To use ``...``, pass in ``...`` or ``Ellipsis``.
    To use ``:``, pass in ``slice(None)``.

    +------------+------------------------------------------+-----------------------+-----------------+----------+
    | Argument   | Description                              | Data type             | Valid range     | Required |
    +============+==========================================+=======================+=================+==========+
    | input      | Input tensor                             | Tensor                |                 | Yes      |
    +------------+------------------------------------------+-----------------------+-----------------+----------+
    | slices     | List of slices to slice the input tensor | List[slice, Ellipsis] |                 | Yes      |
    +------------+------------------------------------------+-----------------------+-----------------+----------+
    """
    return input[slices]


@convert_tt_tensors_wrapper
def reshape(
    input: ttl_tensor.Tensor,
    N: int,
    C: int,
    H: int,
    W: int,
    output_layout: Optional[ttl_tensor.Layout] = ttl_tensor.Layout.TILE,
    output_on_device: Optional[bool] = True,
) -> ttl_tensor.Tensor:
    """
    Returns a new ``tt_lib.tensor.Tensor`` with the same data and number of elements as ``input``, but with the specified shape ``[N, C, H, W]``.

    +------------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument         | Description                                   | Data type   | Valid range     | Required |
    +==================+===============================================+=============+=================+==========+
    | input            | Input tensor                                  | Tensor      |                 | Yes      |
    +------------------+-----------------------------------------------+-------------+-----------------+----------+
    | N                | Size of the first dimension of output tensor  | int         |                 | Yes      |
    +------------------+-----------------------------------------------+-------------+-----------------+----------+
    | C                | Size of the second dimension of output tensor | int         |                 | Yes      |
    +------------------+-----------------------------------------------+-------------+-----------------+----------+
    | H                | Size of the third dimension of output tensor  | int         |                 | Yes      |
    +------------------+-----------------------------------------------+-------------+-----------------+----------+
    | W                | Size of the fourth dimension of output tensor | int         |                 | Yes      |
    +------------------+-----------------------------------------------+-------------+-----------------+----------+
    | output_layout    | Output layout                                 | Layout      | default is TILE | No       |
    +------------------+-----------------------------------------------+-------------+-----------------+----------+
    | output_on_device | Output on device                              | bool        | default is True | No       |
    +------------------+-----------------------------------------------+-------------+-----------------+----------+
    """
    return torch.reshape(input, (N, C, H, W))


@convert_tt_tensors_wrapper
def permute(
    input: ttl_tensor.Tensor,
    dims: Tuple[int],
    output_layout: Optional[ttl_tensor.Layout] = ttl_tensor.Layout.TILE,
    output_on_device: Optional[bool] = True,
) -> ttl_tensor.Tensor:
    """
    Returns a new ``tt_lib.tensor.Tensor`` with the same data and number of elements as ``input``, but with the specified shape ``[N, C, H, W]``.

    +------------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument         | Description                                   | Data type   | Valid range     | Required |
    +==================+===============================================+=============+=================+==========+
    | input            | Input tensor                                  | Tensor      |                 | Yes      |
    +------------------+-----------------------------------------------+-------------+-----------------+----------+
    | dims             | Desired ordering of dimensions                | Tuple of int|                 | Yes      |
    +------------------+-----------------------------------------------+-------------+-----------------+----------+
    | output_layout    | Output layout                                 | Layout      | default is TILE | No       |
    +------------------+-----------------------------------------------+-------------+-----------------+----------+
    | output_on_device | Output on device                              | bool        | default is True | No       |
    +------------------+-----------------------------------------------+-------------+-----------------+----------+
    """
    return torch.permute(input, dims)


@convert_tt_tensors_wrapper
def chunk(input: ttl_tensor.Tensor, chunks: int, dim: int = 0) -> List[ttl_tensor.Tensor]:
    """
    Attempts to split a ``tt_lib.tensor.Tensor`` into the specified number of chunks. Each chunk is a new copy of part of the input tensor.

    If the tensor size along the given dimension ``dim`` is divisible by ``chunks``, all returned chunks will be the same size.

    If the tensor size along the given dimension ``dim`` is not divisible by ``chunks``, all returned chunks will be the same size, except the last one. If such division is not possible, this function may return fewer than the specified number of chunks.

    +------------+--------------------------------------------+-------------+---------------------------+----------+
    | Argument   | Description                                | Data type   | Valid range               | Required |
    +============+============================================+=============+===========================+==========+
    | input      | Input tensor                               | Tensor      |                           | Yes      |
    +------------+--------------------------------------------+-------------+---------------------------+----------+
    | chunks     | Number of chunks to return                 | int         |                           | Yes      |
    +------------+--------------------------------------------+-------------+---------------------------+----------+
    | dim        | Dimension along which to split the tensor  | int         | 0, 1, 2, 3 (default is 0) | No       |
    +------------+--------------------------------------------+-------------+---------------------------+----------+

    """
    return torch.chunk(input, chunks, dim)


@convert_tt_tensors_wrapper
def conv2d(
    input: ttl_tensor.Tensor,
    weight: ttl_tensor.Tensor,
    bias: Optional[ttl_tensor.Tensor] = None,
    stride: Union[int, Tuple] = 1,
    padding: Union[int, str, Tuple] = 0,
    dilation: Union[int, Tuple] = 1,
    groups: int = 1,
) -> ttl_tensor.Tensor:
    """
    Applies a 2D convolution over an input image composed of several input planes.

    +------------+-----------------------------------------------------------------------+-----------------------------+------------------------------------+----------+
    | Argument   | Description                                                           | Data type                   | Valid range                        | Required |
    +============+=======================================================================+=============================+====================================+==========+
    | input      | Input tensor                                                          | Tensor                      |                                    | Yes      |
    +------------+-----------------------------------------------------------------------+-----------------------------+------------------------------------+----------+
    | weight     | Weights tensor                                                        | Tensor                      |                                    | Yes      |
    +------------+-----------------------------------------------------------------------+-----------------------------+------------------------------------+----------+
    | bias       | Bias tensor                                                           | Tensor                      |                                    | No       |
    +------------+-----------------------------------------------------------------------+-----------------------------+------------------------------------+----------+
    | strides    | Stride of the convolution                                             | int or tuple[int] (size 2)  | positive ints (default value is 1) | No       |
    +------------+-----------------------------------------------------------------------+-----------------------------+------------------------------------+----------+
    | padding    | Padding added to all four sides of the input                          | int or tuple[int] (size 2)  | positive ints (default value is 0) | No       |
    |            |                                                                       |                             |                                    |          |
    |            |                                                                       | or string                   | for string `valid` or `same`       |          |
    +------------+-----------------------------------------------------------------------+-----------------------------+------------------------------------+----------+
    | dilation   | Spacing between kernel elements                                       | int or (int, int)           | positive ints (default value is 1) | No       |
    +------------+-----------------------------------------------------------------------+-----------------------------+------------------------------------+----------+
    | groups     | Number of blocked connections from input channels to output channels  | int                         | positive ints (default value is 1) | No       |
    +------------+-----------------------------------------------------------------------+-----------------------------+------------------------------------+----------+

    """
    if bias is not None:
        bias = torch.reshape(bias, (bias.shape[-1],))
    return torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)


@convert_tt_tensors_wrapper
def group_norm(
    input: ttl_tensor.Tensor,
    num_groups: int,
    weight: Optional[ttl_tensor.Tensor] = None,
    bias: Optional[ttl_tensor.Tensor] = None,
    eps: float = 1e-05,
) -> ttl_tensor.Tensor:
    r"""
    Applies Group Normalization over a mini-batch of inputs as described in the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`_.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    +------------+----------------------------------------------------------+-----------+--------------------------------------------------------------+----------+
    | Argument   | Description                                              | Data type | Valid range                                                  | Required |
    +============+==========================================================+===========+==============================================================+==========+
    | input      | Input tensor                                             | Tensor    |                                                              | Yes      |
    +------------+----------------------------------------------------------+-----------+--------------------------------------------------------------+----------+
    | num_groups | Number of groups to separate the input channels into     | int       | int, such that number of channels in input is divisble by it | Yes      |
    +------------+----------------------------------------------------------+-----------+--------------------------------------------------------------+----------+
    | weight     | Weight tensor :math:`\gamma`                             | Tensor    |                                                              | No       |
    +------------+----------------------------------------------------------+-----------+--------------------------------------------------------------+----------+
    | bias       | Bias tensor :math:`\beta`                                | Tensor    |                                                              | No       |
    +------------+----------------------------------------------------------+-----------+--------------------------------------------------------------+----------+
    | eps        | A value added to the denominator for numerical stability | float     | default value is 1e-05                                       | No       |
    +------------+----------------------------------------------------------+-----------+--------------------------------------------------------------+----------+
    """
    if weight is not None:
        weight = weight.reshape(input.shape[1])
    if bias is not None:
        bias = bias.reshape(input.shape[1])
    return torch.nn.functional.group_norm(
        input,
        num_groups,
        weight,
        bias,
        eps,
    )


@convert_tt_tensors_wrapper
def layer_norm(
    input: ttl_tensor.Tensor,
    normalized_shape: Union[int, List[int]],
    weight: Optional[ttl_tensor.Tensor] = None,
    bias: Optional[ttl_tensor.Tensor] = None,
    eps: float = 1e-05,
) -> ttl_tensor.Tensor:
    r"""
    Applies Layer Normalization over a mini-batch of inputs as described in the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \text{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} * \gamma + \beta

    +------------------+----------------------------------------------------------+------------------+-------------------------+----------+
    | Argument         | Description                                              | Data type        | Valid range             | Required |
    +==================+==========================================================+==================+=========================+==========+
    | input            | Input tensor                                             | Tensor           |                         | Yes      |
    +------------------+----------------------------------------------------------+------------------+-------------------------+----------+
    | normalized_shape | Shape over which to normalize                            | int or List[int] |                         | Yes      |
    +------------------+----------------------------------------------------------+------------------+-------------------------+----------+
    | weight           | Weight tensor :math:`\gamma`                             | Tensor           |                         | No       |
    +------------------+----------------------------------------------------------+------------------+-------------------------+----------+
    | bias             | Bias tensor :math:`\beta`                                | Tensor           |                         | No       |
    +------------------+----------------------------------------------------------+------------------+-------------------------+----------+
    | eps              | A value added to the denominator for numerical stability | float            | default value is 1e-05  | No       |
    +------------------+----------------------------------------------------------+------------------+-------------------------+----------+
    """
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]

    if weight is not None:
        assert list(weight.shape[-len(normalized_shape) :]) == list(normalized_shape)
        weight = weight.reshape(normalized_shape)

    if bias is not None:
        assert list(bias.shape[-len(normalized_shape) :]) == list(normalized_shape)
        bias = bias.reshape(normalized_shape)

    return torch.nn.functional.layer_norm(
        input,
        normalized_shape,
        weight,
        bias,
        eps,
    )


@convert_tt_tensors_wrapper
def pad(
    input: ttl_tensor.Tensor,
    pad: Tuple[int],
    mode: str = "constant",
    value: Optional[int] = None,
    output_layout: Optional[ttl_tensor.Layout] = ttl_tensor.Layout.TILE,
    output_on_device: Optional[bool] = True,
) -> ttl_tensor.Tensor:
    r"""
    Pads tensor.

    ``pad`` determines how much padding to add.

    Values in ``pad`` specify padding starting from the last dimension of input tensor ``input`` and moving forward.

    ``pad`` is and m-elements tuple, where m/2 is less of equal to  input dimensions and m is even.

    +------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    | Argument         | Description                                               | Data type        | Valid range                                                               | Required |
    +==================+===========================================================+==================+===========================================================================+==========+
    | input            | Input tensor                                              | Tensor           |                                                                           | Yes      |
    +------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    | pad              | The padding size by which to pad some dimensions of input | Tuple[int]       |                                                                           | Yes      |
    +------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    | mode             | Padding mode                                              | string           | `constant`, `reflect`, `replicate`, or `circular` (default is `constant`) | No       |
    +------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    | value            | Fill value for `constant` padding                         | int              | default is 0                                                              | No       |
    +------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    | output_layout    | Output layout                                             | Layout           | default is TILE                                                           | No       |
    +------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    | output_on_device | Output on device                                          | bool             | default is True                                                           | No       |
    +------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    """
    return torch.nn.functional.pad(input, pad, mode, value)


@convert_tt_tensors_wrapper
def interpolate(
    input: ttl_tensor.Tensor,
    size: Optional[Union[int, Tuple[int]]] = None,
    scale_factor: Optional[Union[float, Tuple[float]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> ttl_tensor.Tensor:
    r"""
    Down/up samples the input to either the given size or the given scale_factor

    The algorithm used for interpolation is determined by mode.

    +------------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    | Argument               | Description                                               | Data type        | Valid range                                                               | Required |
    +========================+===========================================================+==================+===========================================================================+==========+
    | input                  | Input tensor                                              | Tensor           |                                                                           | Yes      |
    +------------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    | size                   | Output spatial size                                       | Tuple[int]       | default is None                                                           | No       |
    +------------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    | scale_factor           | Multiplier for spatial size                               | Tuple[float]     | default is None                                                           | No       |
    +------------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    | mode                   | algorithm used for upsampling                             | string           | `nearest`, `linear`, `bilinear`, `bicubic`, `trilinear`,                  | No       |
    |                        |                                                           |                  | `area`, or `nearest-exact` (default is `nearest`)                         |          |
    +------------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    | align_corners          | Whether to align center or corner points of corner pixels | bool             | default is None                                                           | No       |
    +------------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    | recompute_scale_factor | Recompute the scale_factor for use in interpolation       | bool             | default is None                                                           | No       |
    +------------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    | antialias              | Flag to apply anti-aliasing                               | bool             | default is False                                                          | No       |
    +------------------------+-----------------------------------------------------------+------------------+---------------------------------------------------------------------------+----------+
    """
    return torch.nn.functional.interpolate(
        input,
        size,
        scale_factor,
        mode,
        align_corners,
        recompute_scale_factor,
        antialias,
    )


@convert_tt_tensors_wrapper
def repeat(input: ttl_tensor.Tensor, sizes: List[int]) -> ttl_tensor.Tensor:
    r"""
    Returns the input tensor ``input`` repeated along the specified dims.

    +------------------+-------------------------------------------------------------+------------------+--------------+----------+
    | Argument         | Description                                                 | Data type        | Valid range  | Required |
    +==================+=============================================================+==================+==============+==========+
    | input            | Input tensor                                                | Tensor           |              | Yes      |
    +------------------+-------------------------------------------------------------+------------------+--------------+----------+
    | sizes            | The number of times to repeat the tensor along each dim     | int              |              | Yes      |
    +------------------+-------------------------------------------------------------+------------------+--------------+----------+
    """
    return input.repeat(sizes)


@convert_tt_tensors_wrapper
def repeat_interleave(
    input: ttl_tensor.Tensor,
    repeats: Union[ttl_tensor.Tensor, int],
    dim: Optional[int] = None,
    *,
    output_size: Optional[int] = None,
) -> ttl_tensor.Tensor:
    r"""
    Returns a tensor with repeated elements of input tensor ``input``.

    +------------------+-------------------------------------------------------------+------------------+--------------+----------+
    | Argument         | Description                                                 | Data type        | Valid range  | Required |
    +==================+=============================================================+==================+==============+==========+
    | input            | Input tensor                                                | Tensor           |              | Yes      |
    +------------------+-------------------------------------------------------------+------------------+--------------+----------+
    | repeats          | The number of repetitions for each element                  | Tensor or int    |              | Yes      |
    +------------------+-------------------------------------------------------------+------------------+--------------+----------+
    | dim              | The dimension along which to repeat values                  | int              |              | No       |
    +------------------+-------------------------------------------------------------+------------------+--------------+----------+
    | output_size      | Total output size for the given axis ( e.g. sum of repeats) | int              |              | No       |
    +------------------+-------------------------------------------------------------+------------------+--------------+----------+
    """
    return torch.repeat_interleave(input, repeats, dim, output_size=output_size)


@convert_tt_tensors_wrapper
def concat(tensors: List[ttl_tensor.Tensor], dim: int = 0) -> ttl_tensor.Tensor:
    r"""
    Concatenates input tensors in list ``tensors`` on provided dimension ``dim``.

    All tensors must either have the same shape (except in the concatenating dimension) or be empty.

    +------------------+-------------------------------------------+------------------+------------------------------+----------+
    | Argument         | Description                               | Data type        | Valid range                  | Required |
    +==================+===========================================+==================+==============================+==========+
    | tensors          | Input tensors                             | List[Tensor]     |                              | Yes      |
    +------------------+-------------------------------------------+------------------+------------------------------+----------+
    | dim              | The dimension along which to concatenate  | int              | 0, 1, 2, or 3 (default is 0) | No       |
    +------------------+-------------------------------------------+------------------+------------------------------+----------+
    """
    return torch.concat(tensors, dim)


@convert_tt_tensors_wrapper
def silu(input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
    r"""
    Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    +------------------+-------------------------------------------+------------------+------------------------------+----------+
    | Argument         | Description                               | Data type        | Valid range                  | Required |
    +==================+===========================================+==================+==============================+==========+
    | input            | Input tensor                              | Tensor           |                              | Yes      |
    +------------------+-------------------------------------------+------------------+------------------------------+----------+
    """
    return torch.nn.functional.silu(input)


@convert_tt_tensors_wrapper
def softmax(input: ttl_tensor.Tensor, dim: Optional[int] = None) -> ttl_tensor.Tensor:
    r"""
    Applies a softmax function to input tensor ``input``.

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    It is applied to all slices along dim, and will re-scale them so that the elements lie in the range `[0, 1]` and sum to 1.

    +------------------+--------------------------------------------------+------------------+------------------------------+----------+
    | Argument         | Description                                      | Data type        | Valid range                  | Required |
    +==================+==================================================+==================+==============================+==========+
    | input            | Input tensor                                     | Tensor           |                              | Yes      |
    +------------------+--------------------------------------------------+------------------+------------------------------+----------+
    | dim              | A dimension along which Softmax will be computed | int              | 0, 1, 2, or 3                | No       |
    +------------------+--------------------------------------------------+------------------+------------------------------+----------+
    """
    return torch.nn.functional.softmax(input, dim)


class Conv2d(torch.nn.Module):
    r"""
    Applies a 2D convolution over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)

    where :math:`\star` is a valid 2D cross-correlation operator, :math:`N` is batch size, :math:`C` denotes the number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is width in pixels.

    +------------------+----------------------------------------------------------------------+-------------------+------------------------------------------------+----------+
    | Argument         | Description                                                          | Data type         | Valid range                                    | Required |
    +==================+======================================================================+===================+================================================+==========+
    | weights          | Weights tensor                                                       | Tensor            |                                                | Yes      |
    +------------------+----------------------------------------------------------------------+-------------------+------------------------------------------------+----------+
    | biases           | Bias tensor                                                          | Tensor            |                                                | Yes      |
    +------------------+----------------------------------------------------------------------+-------------------+------------------------------------------------+----------+
    | in_channels      | Number of channels in the input image                                | int               |                                                | Yes      |
    +------------------+----------------------------------------------------------------------+-------------------+------------------------------------------------+----------+
    | out_channels     | Number of channels produced by the convolution                       | int               |                                                | Yes      |
    +------------------+----------------------------------------------------------------------+-------------------+------------------------------------------------+----------+
    | kernel_size      | Size of the convolving kernel                                        | int or Tuple[int] |                                                | Yes      |
    +------------------+----------------------------------------------------------------------+-------------------+------------------------------------------------+----------+
    | stride           | Stride of the convolution                                            | int or Tuple[int] | default is 1                                   | No       |
    +------------------+----------------------------------------------------------------------+-------------------+------------------------------------------------+----------+
    | padding          | Padding added to all four sides of the input                         | int or Tuple[int] | default is 0                                   | No       |
    |                  |                                                                      |                   |                                                |          |
    |                  |                                                                      | or string         | ‘valid’ or ‘same’                              |          |
    +------------------+----------------------------------------------------------------------+-------------------+------------------------------------------------+----------+
    | dilation         | Spacing between kernel elements                                      | int or Tuple[int] | default is 1                                   | No       |
    +------------------+----------------------------------------------------------------------+-------------------+------------------------------------------------+----------+
    | groups           | Number of blocked connections from input channels to output channels | int               | default is 1                                   | No       |
    +------------------+----------------------------------------------------------------------+-------------------+------------------------------------------------+----------+
    | bias             | If `True`, adds a learnable bias to the output                       | bool              | default is `True`                              | No       |
    +------------------+----------------------------------------------------------------------+-------------------+------------------------------------------------+----------+
    | padding_mode     | Padding mode                                                         | string            | `zeros`, `reflect`, `replicate`, or `circular` | No       |
    |                  |                                                                      |                   |                                                |          |
    |                  |                                                                      |                   | default is `zeros`                             |          |
    +------------------+----------------------------------------------------------------------+-------------------+------------------------------------------------+----------+
    """

    @convert_tt_tensors_wrapper
    def __init__(
        self,
        weights: ttl_tensor.Tensor,
        biases: Union[ttl_tensor.Tensor, None],
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        padding: Union[int, str, Tuple] = 0,
        dilation: Union[int, Tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.pt_fallback = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.pt_fallback.weight = torch.nn.Parameter(weights)
        self.pt_fallback.bias = (
            torch.nn.Parameter(biases.reshape((biases.shape[-1],))) if biases is not None else biases
        )

    @convert_tt_tensors_wrapper
    def forward(self, input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
        return self.pt_fallback(input)


class BatchNorm2d(torch.nn.Module):
    r"""
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    +---------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | Argument            | Description                                                          | Data type         | Valid range       | Required |
    +=====================+======================================================================+===================+===================+==========+
    | weights             | Weights tensor                                                       | Tensor            |                   | Yes      |
    +---------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | biases              | Bias tensor                                                          | Tensor            |                   | Yes      |
    +---------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | running_mean        | Tracked Running Mean tensor                                          | Tensor            |                   | Yes      |
    +---------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | running_var         | Tracked Running Variances tensor                                     | Tensor            |                   | Yes      |
    +---------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | num_batches_tracked | Number of Batches Tracked tensor                                     | Tensor            |                   | Yes      |
    +---------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | num_features        | C from an expected input of size (N, C, H, W)                        | int               |                   | Yes      |
    +---------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | eps                 | A value added to the denominator for numerical stability             | float             | default is 1e-05  | No       |
    +---------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | momentum            | The value used for the running_mean and running_var computation.     | float/None        | default is 0.1    | No       |
    +---------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | affine              | Controls initialization of weights and biases                        | bool              | default is `True` | No       |
    +---------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | track_running_stats | Whether to track the running mean and variance                       | bool              | default is `True` | No       |
    +---------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    """

    @convert_tt_tensors_wrapper
    def __init__(
        self,
        weights: ttl_tensor.Tensor,
        biases: ttl_tensor.Tensor,
        running_mean: ttl_tensor.Tensor,
        running_var: ttl_tensor.Tensor,
        num_batches_tracked: ttl_tensor.Tensor,
        num_features: int,
        eps: float = 1e-05,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        weights = weights.reshape(
            num_features,
        )
        biases = biases.reshape(
            num_features,
        )
        running_mean = running_mean.reshape(num_features)
        running_var = running_var.reshape(num_features)
        num_batches_tracked = torch.tensor(num_batches_tracked.item())

        self.pt_fallback = torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pt_fallback.weight = torch.nn.Parameter(weights)
        self.pt_fallback.bias = torch.nn.Parameter(biases)
        self.pt_fallback.running_mean = running_mean
        self.pt_fallback.running_var = running_var
        self.pt_fallback.num_batches_tracked = num_batches_tracked

    @convert_tt_tensors_wrapper
    def forward(self, input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
        return self.pt_fallback(input)


class GroupNorm(torch.nn.Module):
    r"""
    Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    ``affine`` is a boolean value that when set to `True`, this module has lernable per-channel affine parameters initialized to ones (for weights) and zeros (for biases).

    +------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | Argument         | Description                                                          | Data type         | Valid range       | Required |
    +==================+======================================================================+===================+===================+==========+
    | weights          | Weights tensor                                                       | Tensor            |                   | Yes      |
    +------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | biases           | Bias tensor                                                          | Tensor            |                   | Yes      |
    +------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | num_groups       | Number of groups to separate the channels into                       | int               |                   | Yes      |
    +------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | num_channels     | Number of channels expected in input                                 | int               |                   | Yes      |
    +------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | eps              | A value added to the denominator for numerical stability             | float             | default is 1e-05  | No       |
    +------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | affine           | Controls initialization of weights and biases                        | bool              | default is `True` | No       |
    +------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    """

    @convert_tt_tensors_wrapper
    def __init__(
        self,
        weights: ttl_tensor.Tensor,
        biases: ttl_tensor.Tensor,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
    ):
        super().__init__()
        weights = weights.reshape(
            num_channels,
        )
        biases = biases.reshape(
            num_channels,
        )
        self.pt_fallback = torch.nn.GroupNorm(num_groups, num_channels, eps, affine)
        self.pt_fallback.weight = torch.nn.Parameter(weights)
        self.pt_fallback.bias = torch.nn.Parameter(biases)

    @convert_tt_tensors_wrapper
    def forward(self, input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
        return self.pt_fallback(input)


class LayerNorm(torch.nn.Module):
    r"""
    Applies Layer Normalization over a mini-batch of inputs as described in the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \text{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} * \gamma + \beta

    ``elementwise_affine`` is a boolean value that when set to `True`, this module has learnable per-element affine parameters initialized to ones (for weights) and zeros (for biases).

    +--------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | Argument           | Description                                                          | Data type         | Valid range       | Required |
    +====================+======================================================================+===================+===================+==========+
    | weights            | Weights tensor                                                       | Tensor            |                   | Yes      |
    +--------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | biases             | Bias tensor                                                          | Tensor            |                   | Yes      |
    +--------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | normalized_shape   | Shape over which to normalize                                        | int or List[int]  |                   | Yes      |
    +--------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | eps                | A value added to the denominator for numerical stability             | float             | default is 1e-05  | No       |
    +--------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    | elementwise_affine | Controls initialization of weights and biases                        | bool              | default is `True` | No       |
    +--------------------+----------------------------------------------------------------------+-------------------+-------------------+----------+
    """

    @convert_tt_tensors_wrapper
    def __init__(
        self,
        weights: ttl_tensor.Tensor,
        biases: ttl_tensor.Tensor,
        normalized_shape: Union[int, List[int]],
        eps: float = 1e-05,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]

        assert list(weights.shape[-len(normalized_shape) :]) == list(normalized_shape)
        assert list(biases.shape[-len(normalized_shape) :]) == list(normalized_shape)
        self.pt_fallback = torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
        self.pt_fallback.weight = torch.nn.Parameter(weights.reshape(normalized_shape))
        self.pt_fallback.bias = torch.nn.Parameter(biases.reshape(normalized_shape))

    @convert_tt_tensors_wrapper
    def forward(self, input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
        return self.pt_fallback(input)


class MaxPool2d(torch.nn.Module):
    r"""
    Applies a 2D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    +--------------------+----------------------------------------------------------------------+-------------------+------------------------+----------+
    | Argument           | Description                                                          | Data type         | Valid range            | Required |
    +====================+======================================================================+===================+========================+==========+
    | kernel_size        | The size of the window to take a max over                            | int or Tuple[int] |                        | Yes      |
    +--------------------+----------------------------------------------------------------------+-------------------+------------------------+----------+
    | stride             | The stride of the window. Default value is kernel_size               | int or Tuple[int] | default is kernel_size | No       |
    +--------------------+----------------------------------------------------------------------+-------------------+------------------------+----------+
    | padding            | Implicit negative infinity padding to be added on both sides         | int or Tuple[int] | default is 0           | No       |
    +--------------------+----------------------------------------------------------------------+-------------------+------------------------+----------+
    | dilation           | A parameter that controls the stride of elements in the window       | int or Tuple[int] | default is 1           | No       |
    +--------------------+----------------------------------------------------------------------+-------------------+------------------------+----------+
    | return_indices     | If True, will return the max indices along with the outputs.         | bool              | default is `False`     | No       |
    +--------------------+----------------------------------------------------------------------+-------------------+------------------------+----------+
    | ceil_mode          | If True, will use ceil instead of floor to compute the output shape  | bool              | default is `False`     | No       |
    +--------------------+----------------------------------------------------------------------+-------------------+------------------------+----------+
    """

    @convert_tt_tensors_wrapper
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        channels_last=False,
        reshape_2d=False,
    ):
        super().__init__()
        self.pt_fallback = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.channels_last = channels_last
        self.reshape_2d = reshape_2d

    @convert_tt_tensors_wrapper
    def forward(self, input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
        output = input
        if self.channels_last:
            output = torch.permute(output, (0, 3, 1, 2))
        output = self.pt_fallback(output)
        if self.channels_last:
            output = torch.permute(output, (0, 2, 3, 1))
        if self.reshape_2d:
            output = torch.reshape(output, (1, 1, output.shape[0] * output.shape[1] * output.shape[2], output.shape[3]))
        return output


class AdaptiveAvgPool2d(torch.nn.Module):
    r"""
    Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    +--------------------+----------------------------------------------------------------------+-------------------+----------------------+----------+
    | Argument           | Description                                                          | Data type         | Valid range          | Required |
    +====================+======================================================================+===================+======================+==========+
    | output_size        |  The target output size of the image                                 | int               | int/None or tuple    |          |
    |                    |                                                                      |                   | of int/None (size 2) | yes      |
    +--------------------+----------------------------------------------------------------------+-------------------+----------------------+----------+
    """

    @convert_tt_tensors_wrapper
    def __init__(
        self,
        output_size: Union[int, None, Tuple[Optional[int], Optional[int]]],
        channels_last=False,
    ):
        super().__init__()
        self.pt_fallback = torch.nn.AdaptiveAvgPool2d(output_size)
        self.channels_last = channels_last

    @convert_tt_tensors_wrapper
    def forward(self, input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
        output = input
        if self.channels_last:
            output = torch.permute(output, (0, 3, 1, 2))
        output = self.pt_fallback(output)
        if self.channels_last:
            output = torch.permute(output, (0, 2, 3, 1))
        return output


@convert_tt_tensors_wrapper
def ceil(input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
    """
    Returns a new tensor with the ceil of the elements of ``input``, the smallest integer greater than or equal to each element.

    +------------+-----------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                             | Data type   | Valid range     | Required |
    +============+=========================================+=============+=================+==========+
    | input      | Input tensor for ceil                   | Tensor      |                 | Yes      |
    +------------+-----------------------------------------+-------------+-----------------+----------+
    """
    return torch.ceil(input)


@convert_tt_tensors_wrapper
def floor(input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
    """
    Returns a new tensor with the floor of the elements of ``input``, the largest integer less than or equal to each element.

    +------------+-----------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                             | Data type   | Valid range     | Required |
    +============+=========================================+=============+=================+==========+
    | input      | Input tensor for floor                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------+-------------+-----------------+----------+
    """
    return torch.floor(input)


@convert_tt_tensors_wrapper
def trunc(input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
    """
    Returns a new tensor with the truncated integer values of the elements of ``input``.

    +------------+-----------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                             | Data type   | Valid range     | Required |
    +============+=========================================+=============+=================+==========+
    | input      | Input tensor for trunc                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------+-------------+-----------------+----------+
    """
    return torch.trunc(input)


@convert_tt_tensors_wrapper
def unary_fmod(input: ttl_tensor.Tensor, other: float) -> ttl_tensor.Tensor:
    """
    Applies mod operations and the result has the same sign as the dividend ``input`` and
    its absolute value is less than that of ``other``.

    +------------+-----------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                             | Data type   | Valid range     | Required |
    +============+=========================================+=============+=================+==========+
    | input      | Input tensor                            | Tensor      |                 | Yes      |
    +------------+-----------------------------------------+-------------+-----------------+----------+
    | Other      | Scalar                                  | float       |                 | Yes      |
    +------------+-----------------------------------------+-------------+-----------------+----------+
    """
    return torch.fmod(input, other)


@convert_tt_tensors_wrapper
def binary_fmod(input: ttl_tensor.Tensor, other: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
    """
    Applies mod operations and the result has the same sign as the dividend ``input`` and
    its absolute value is less than that of ``other``.

    +------------+-----------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                             | Data type   | Valid range     | Required |
    +============+=========================================+=============+=================+==========+
    | input      | First tensor                            | Tensor      |                 | Yes      |
    +------------+-----------------------------------------+-------------+-----------------+----------+
    | Other      | Second tensor                           | Tensor      |                 | Yes      |
    +------------+-----------------------------------------+-------------+-----------------+----------+
    """
    return torch.fmod(input, other)


@convert_tt_tensors_wrapper
def bitwise_not(input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
    """
    Computes the bitwise NOT of the given ``input`` tensor.

    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                                   | Data type   | Valid range     | Required |
    +============+===============================================+=============+=================+==========+
    | input      | Input tensor                                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+

    """
    return torch.bitwise_not(input)


@convert_tt_tensors_wrapper
def unary_bitwise_or(input: ttl_tensor.Tensor, other: int) -> ttl_tensor.Tensor:
    """
    Computes the bitwise OR of ``input`` and ``other``.

    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                                   | Data type   | Valid range     | Required |
    +============+===============================================+=============+=================+==========+
    | input      | Input tensor                                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | other      | Immediate value                               | int         |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    """
    return torch.bitwise_or(input, other)


@convert_tt_tensors_wrapper
def unary_bitwise_and(input: ttl_tensor.Tensor, other: int) -> ttl_tensor.Tensor:
    """
    Computes the bitwise AND of ``input`` and ``other``.

    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                                   | Data type   | Valid range     | Required |
    +============+===============================================+=============+=================+==========+
    | input      | Input tensor                                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | other      | Immediate value                               | int         |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    """
    return torch.bitwise_and(input, other)


@convert_tt_tensors_wrapper
def unary_bitwise_xor(input: ttl_tensor.Tensor, other: int) -> ttl_tensor.Tensor:
    """
    Computes the bitwise XOR of ``input`` and ``other``.

    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                                   | Data type   | Valid range     | Required |
    +============+===============================================+=============+=================+==========+
    | input      | Input tensor                                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | other      | Immediate value                               | int         |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    """
    return torch.bitwise_xor(input, other)


@convert_tt_tensors_wrapper
def binary_bitwise_or(input: ttl_tensor.Tensor, other: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
    """
    Computes the bitwise OR of ``input`` and ``other``.

    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                                   | Data type   | Valid range     | Required |
    +============+===============================================+=============+=================+==========+
    | input      | First tensor                                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | other      | Second tensor                                 | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    """
    return torch.bitwise_or(input, other)


@convert_tt_tensors_wrapper
def binary_bitwise_and(input: ttl_tensor.Tensor, other: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
    """
    Computes the bitwise AND of ``input`` and ``other``.

    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                                   | Data type   | Valid range     | Required |
    +============+===============================================+=============+=================+==========+
    | input      | First tensor                                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | other      | Second Tensor                                 | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    """
    return torch.bitwise_and(input, other)


@convert_tt_tensors_wrapper
def binary_bitwise_xor(input: ttl_tensor.Tensor, other: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
    """
    Computes the bitwise XOR of ``input`` and ``other``.

    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                                   | Data type   | Valid range     | Required |
    +============+===============================================+=============+=================+==========+
    | input      | First tensor                                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | other      | Second tensor                                 | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    """
    return torch.bitwise_xor(input, other)


@convert_tt_tensors_wrapper
def unary_bitwise_right_shift(input: ttl_tensor.Tensor, other: int) -> ttl_tensor.Tensor:
    """
    Computes the right arithmetic shift of ``input`` by ``other`` bits. The input tensor must be of integral type.
    In any case, if the value of the right operand is negative or is greater or equal to the number of bits in the
    promoted left operand, the behavior is undefined.

    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                                   | Data type   | Valid range     | Required |
    +============+===============================================+=============+=================+==========+
    | input      | Input tensor                                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | other      | Immediate value                               | int         |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    """
    return torch.bitwise_right_shift(input, other)


@convert_tt_tensors_wrapper
def unary_bitwise_left_shift(input: ttl_tensor.Tensor, other: int) -> ttl_tensor.Tensor:
    """
    Computes the left arithmetic shift of ``input`` by ``other`` bits. The input tensor must be of integral type.

    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                                   | Data type   | Valid range     | Required |
    +============+===============================================+=============+=================+==========+
    | input      | Input tensor                                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | other      | Immediate value                               | int         |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    """
    return torch.bitwise_left_shift(input, other)


@convert_tt_tensors_wrapper
def binary_bitwise_right_shift(input: ttl_tensor.Tensor, other: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
    """
    Computes the right arithmetic shift of ``input`` by ``other`` bits. The input tensor must be of integral type.
    In any case, if the value of the right operand is negative or is greater or equal to the number of bits in the
    promoted left operand, the behavior is undefined.

    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                                   | Data type   | Valid range     | Required |
    +============+===============================================+=============+=================+==========+
    | input      | First tensor                                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | other      | Second tensor                                 | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    """
    return torch.bitwise_right_shift(input, other)


@convert_tt_tensors_wrapper
def binary_bitwise_left_shift(input: ttl_tensor.Tensor, other: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
    """
    Computes the left arithmetic shift of ``input`` by ``other`` bits. The input tensor must be of integral type.

    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                                   | Data type   | Valid range     | Required |
    +============+===============================================+=============+=================+==========+
    | input      | First tensor                                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | other      | Second tensor                                 | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    """
    return torch.bitwise_left_shift(input, other)


@convert_tt_tensors_wrapper
def torch_argmax(input: ttl_tensor.Tensor, dim: int = None, keepdim: bool = False) -> ttl_tensor.Tensor:
    """
    Returns the indices of the maximum values of a tensor along a dimension.

    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                                   | Data type   | Valid range     | Required |
    +============+===============================================+=============+=================+==========+
    | input      | Input tensor                                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | dim        | Dimension along which to compute the argmax   | int         |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | keepdim    | Whether to retain the dimensionality of input | bool        |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    """
    return torch.argmax(input, dim=dim, keepdim=keepdim)


@convert_tt_tensors_wrapper
def torch_argmin(input: ttl_tensor.Tensor, dim: int, keepdim: bool) -> ttl_tensor.Tensor:
    """
    Returns the indices of the minimum values of a tensor along a dimension.

    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | Argument   | Description                                   | Data type   | Valid range     | Required |
    +============+===============================================+=============+=================+==========+
    | input      | Input tensor                                  | Tensor      |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | dim        | Dimension along which to compute the argmin   | int         |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    | keepdim    | Whether to retain the dimensionality of input | bool        |                 | Yes      |
    +------------+-----------------------------------------------+-------------+-----------------+----------+
    """
    return torch.argmin(input, dim=dim, keepdim=keepdim)
