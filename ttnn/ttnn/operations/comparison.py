# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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

    if expected.is_complex() and actual.is_complex():
        expected = torch.view_as_real(expected.clone())
        actual = torch.view_as_real(actual.clone())

    if not (expected.is_floating_point() or actual.is_floating_point()):
        expected = expected.to(torch.float)
        actual = actual.to(torch.float)

    if expected.dtype != actual.dtype:
        actual = actual.type(expected.dtype)

    if torch.all(torch.isnan(expected)) and torch.all(torch.isnan(actual)):
        logger.warning("Both tensors are 'nan'")
        return 1.0

    if torch.all(torch.isnan(expected)) or torch.all(torch.isnan(actual)):
        logger.error("One tensor is all nan, the other is not.")
        return 0.0

    # Test if either is completely zero
    if torch.any(expected.bool()) != torch.any(actual.bool()):
        logger.error("One tensor is all zero")
        return 0.0

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

    # If one tensor is constant, perform allclose comparison with default tolerances
    if torch.max(expected) == torch.min(expected) or torch.max(actual) == torch.min(actual):
        logger.warning(
            "One tensor is constant. Performing allclose comparison with default tolerances instead of pearson correlation coefficient."
        )
        return float(torch.allclose(expected, actual))

    output = np.ma.corrcoef(
        np.ma.masked_invalid(torch.squeeze(expected).detach().numpy()).flatten(),
        np.ma.masked_invalid(torch.squeeze(actual).detach().numpy()).flatten(),
    )

    # Remove correlation coefficient with self (typically always 1.0)
    mask = np.ones(output.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    output = np.min(output[mask])

    if isinstance(output, np.ma.core.MaskedConstant):
        return 1.0

    return output


__all__ = []
