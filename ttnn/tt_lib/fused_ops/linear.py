# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional
from .. import tensor

from loguru import logger


def Linear(in_features: int, out_features: int, weight: List[Union[int, float]], bias, device):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be the weight as a tilized list of values.
    """
    logger.warning("Linear will be deprecated soon, please use linear in models/helper_funcs")
    assert weight.get_layout() == tensor.Layout.TILE
    weight = weight.to(device)

    if bias is None:
        bias = None
    else:
        assert bias.get_layout() == tensor.Layout.TILE
        bias = bias.to(device)

    def linear_(activation):
        weight_T = tensor.transpose(weight, -2, -1)
        output = tensor.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = tensor.bcast(output, bias, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H)
            return output_plus_bias

        return output

    return linear_
