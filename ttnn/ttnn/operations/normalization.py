# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import ttnn


def _golden_function(input_tensor: ttnn.Tensor, dim: Optional[int] = None, **_):
    import torch

    dim = dim or -1

    return torch.nn.Softmax(dim)(input_tensor)


ttnn.attach_golden_function(
    ttnn.softmax,
    golden_function=_golden_function,
)

ttnn.attach_golden_function(
    ttnn.softmax_in_place,
    golden_function=_golden_function,
)


def _golden_function(input_tensor: ttnn.Tensor, scalar: float, attention_mask=None, **_):
    import torch

    input_tensor = input_tensor.float()
    input_tensor = input_tensor * scalar
    if attention_mask is not None:
        input_tensor = input_tensor + attention_mask
    return torch.softmax(input_tensor, dim=-1)


ttnn.attach_golden_function(
    ttnn.scale_mask_softmax_in_place,
    golden_function=_golden_function,
)

ttnn.attach_golden_function(
    ttnn.scale_mask_softmax,
    golden_function=_golden_function,
)

ttnn.attach_golden_function(
    ttnn.scale_causal_mask_hw_dims_softmax_in_place,
    golden_function=_golden_function,
)


SoftmaxProgramConfig = ttnn._ttnn.operations.normalization.SoftmaxProgramConfig
SoftmaxDefaultProgramConfig = ttnn._ttnn.operations.normalization.SoftmaxDefaultProgramConfig
SoftmaxShardedMultiCoreProgramConfig = ttnn._ttnn.operations.normalization.SoftmaxShardedMultiCoreProgramConfig

__all__ = []
