# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


from loguru import logger

import ttnn


@ttnn.register_python_operation(name="ttnn.pearson_correlation_coefficient")
def pearson_correlation_coefficient(expected, actual):
    import torch
    import numpy as np

    if isinstance(actual, ttnn.Tensor):
        actual = ttnn.to_torch(actual)

    if not isinstance(expected, torch.Tensor):
        raise ValueError("Expected tensor is not a torch.Tensor")

    expected = torch.Tensor(expected)
    actual = torch.Tensor(actual)

    if expected.dtype != actual.dtype:
        actual = actual.type(expected.dtype)

    if torch.all(torch.isnan(expected)) and torch.all(torch.isnan(actual)):
        logger.warning("Both tensors are 'nan'")
        return 1.0

    if torch.all(torch.isnan(expected)) or torch.all(torch.isnan(actual)):
        logger.error("One tensor is all nan, the other is not.")
        return 0.0

    # Test if either is completely zero — also a constant (zero-variance) case.
    # Fall back to allclose so that zero-vs-small-constant within tolerance passes.
    if torch.any(expected.bool()) != torch.any(actual.bool()):
        logger.warning("One tensor is all zero. PCC undefined; falling back to allclose.")
        return float(torch.allclose(expected, actual, rtol=1e-05, atol=1e-04))

    # For now, mask all infs and nans so that we check the rest... TODO
    expected = expected.clone()
    expected[
        torch.logical_or(
            torch.isnan(expected),
            torch.logical_or(torch.isinf(expected), torch.isneginf(expected)),
        )
    ] = 0
    actual = actual.clone()
    actual[
        torch.logical_or(
            torch.isnan(actual),
            torch.logical_or(torch.isinf(actual), torch.isneginf(actual)),
        )
    ] = 0

    if torch.equal(expected, actual):
        return 1.0

    if expected.dtype == torch.bfloat16:
        expected = expected.type(torch.float32)
        actual = actual.type(torch.float32)

    # If either tensor is constant (zero std dev), PCC is undefined. Fall back to allclose
    # rather than returning a misleading 1.0 from the corrcoef diagonal.
    if torch.max(expected) == torch.min(expected) or torch.max(actual) == torch.min(actual):
        logger.warning("One or both tensors are constant (zero std dev). PCC undefined; falling back to allclose.")
        return float(torch.allclose(expected, actual, rtol=1e-05, atol=1e-04))

    output = np.ma.corrcoef(
        np.ma.masked_invalid(torch.squeeze(expected).detach().numpy()).flatten(),
        np.ma.masked_invalid(torch.squeeze(actual).detach().numpy()).flatten(),
    )[0, 1]

    if isinstance(output, np.ma.core.MaskedConstant) or np.isnan(float(output)):
        logger.warning("PCC returned NaN/masked. Falling back to allclose.")
        return float(torch.allclose(expected, actual, rtol=1e-05, atol=1e-04))

    return float(output)


__all__ = []
