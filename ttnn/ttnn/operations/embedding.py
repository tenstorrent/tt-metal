# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def _golden_function(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, **_):
    import torch

    output_tensor = torch.nn.functional.embedding(input_tensor, weight)
    return output_tensor


embedding = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.embedding.embedding)


__all__ = []
