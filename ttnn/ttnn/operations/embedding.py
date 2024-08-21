# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def _golden_function(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, **_):
    import torch

    output_tensor = torch.nn.functional.embedding(input_tensor, weight)
    return output_tensor


ttnn.attach_golden_function(ttnn.embedding, golden_function=_golden_function)

EmbeddingsType = ttnn._ttnn.operations.embedding.EmbeddingsType

__all__ = []
