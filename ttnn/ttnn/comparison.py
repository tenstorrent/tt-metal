# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC and ULP comparisons used by the operation decorator."""

from __future__ import annotations

import math

from loguru import logger


def _comp_nonfinite(golden, calculated):
    """
    Returns True if tensors contain the same non-finite values (nan, inf, -inf) at the same positions. Also returns True if all elements are finite.
    Returns False if non-finite values differ between both tensors.
    """

    import torch

    # torch.equal(['nan'], ['nan']] => False
    # For this reason, we check for nan and inf separately
    if torch.not_equal(torch.isnan(golden), torch.isnan(calculated)).any():
        return False

    golden_inf_mask = torch.isinf(golden)
    calculated_inf_mask = torch.isinf(calculated)

    if torch.not_equal(golden_inf_mask, calculated_inf_mask).any():
        return False

    golden_inf = golden[golden_inf_mask]
    calculated_inf = calculated[calculated_inf_mask]
    return torch.equal(golden_inf, calculated_inf)


def comp_pcc(golden, calculated, pcc=0.99, rtol=1e-05, atol=1e-04):
    import torch

    golden = torch.Tensor(golden)
    calculated = torch.Tensor(calculated)

    # PCC is undefined for a constant tensor -- every single-element tensor included -- so the
    # two checks below fall back to allclose. The default rtol is float32-grade; a 16-bit float
    # carries eps = 2^-7 (bfloat16) or 2^-10 (float16), so a result one ULP from the golden fails
    # it and is reported as PCC 0.0. Widen only the RELATIVE tolerance to a few ULP of the coarser
    # 16-bit float on EITHER side -- most callers pass a float32 torch golden against a bfloat16
    # device result, and the cast below would otherwise hide the 16-bit side -- never below what
    # the caller asked for. atol stays the caller's: an epsilon is a relative precision, and an
    # absolute floor of that size would accept wrong small-magnitude results. FP8 and integer
    # dtypes keep the caller's tolerances unchanged.
    fallback_rtol = rtol
    sixteen_bit_eps = [
        torch.finfo(dtype).eps for dtype in (golden.dtype, calculated.dtype) if dtype in (torch.bfloat16, torch.float16)
    ]
    if sixteen_bit_eps:
        fallback_rtol = max(rtol, 4 * max(sixteen_bit_eps))

    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        logger.warning("Both tensors are 'nan'")
        return True, 1.0

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        logger.error("One tensor is all nan, the other is not.")
        return False, 0.0

    # Test if either is completely zero — but a zero tensor is also a constant tensor,
    # so fall back to allclose instead of a hard 0.0: zero-vs-small-constant may be
    # within the caller's tolerances.
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        logger.warning("One tensor is all zero. PCC undefined; falling back to allclose.")
        result = torch.allclose(golden, calculated, rtol=fallback_rtol, atol=atol)
        return result, float(result)

    golden = torch.squeeze(golden).flatten()
    calculated = torch.squeeze(calculated).flatten()

    # For now, mask all infs and nans (to zero) so that we check the rest... TODO
    # Skip this for integer types which don't have NaN/Inf values.
    if golden.dtype.is_floating_point:
        # FP8 doesn't support isfinite/nan_to_num and bfloat16 products lose precision,
        # so correlate these in float32.
        if golden.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16):
            golden = golden.to(torch.float32)
            calculated = calculated.to(torch.float32)

        # Zero out NaN/Inf, preserving the historical PCC values. nan_to_num allocates a
        # full-size copy of each tensor, so only do it when invalid values are actually
        # present; on the common all-finite path the tensors stay as views and no copy is
        # made (this short-circuit is what keeps peak memory near 1x of one input).
        if not bool((torch.isfinite(golden) & torch.isfinite(calculated)).all()):
            golden = torch.nan_to_num(golden, nan=0.0, posinf=0.0, neginf=0.0)
            calculated = torch.nan_to_num(calculated, nan=0.0, posinf=0.0, neginf=0.0)

    if torch.equal(golden, calculated):
        return True, 1.0

    # Integer tensors must be correlated in floating point (centering/products would
    # otherwise truncate/overflow). float32 keeps the working set small.
    if not golden.dtype.is_floating_point:
        golden = golden.to(torch.float32)
        calculated = calculated.to(torch.float32)

    # Pearson r with float64 *accumulation* (dtype= on the reductions) over the float32
    # data: no float64 copy of either tensor is materialized, so peak memory stays near
    # 1x of one input on large tensors while matching a full-float64 correlation to
    # |Δ|<1e-9 across the high-PCC (>=0.999) range.
    n = golden.numel()
    g_centered = golden - (golden.sum(dtype=torch.float64) / n).to(golden.dtype)
    c_centered = calculated - (calculated.sum(dtype=torch.float64) / n).to(calculated.dtype)
    cov = (g_centered * c_centered).sum(dtype=torch.float64)
    g_sq_sum = g_centered.pow(2).sum(dtype=torch.float64)
    c_sq_sum = c_centered.pow(2).sum(dtype=torch.float64)
    denom = torch.sqrt(g_sq_sum * c_sq_sum)
    # pow/sum stay in float32 before the reduction; large-magnitude tensors (e.g. ldexp)
    # can overflow to inf here even though float64 accumulation would be finite.
    if not math.isfinite(denom.item()) or not math.isfinite(cov.item()):
        g_centered64 = g_centered.to(torch.float64)
        c_centered64 = c_centered.to(torch.float64)
        cov = (g_centered64 * c_centered64).sum()
        denom = torch.sqrt(g_centered64.pow(2).sum() * c_centered64.pow(2).sum())
    cal_pcc = (cov / denom).item()

    # Zero variance -> denom == 0 -> cal_pcc is nan: PCC is undefined for constant tensors.
    # Fall back to allclose rather than returning a misleading 1.0.
    if math.isnan(cal_pcc):
        logger.warning("PCC is NaN (zero variance / constant tensor). Falling back to allclose check.")
        result = torch.allclose(golden, calculated, rtol=fallback_rtol, atol=atol)
        return result, float(result)

    return cal_pcc >= pcc, cal_pcc


def ulp(x: ttnn.Tensor | torch.Tensor) -> ttnn.Tensor | torch.Tensor:
    "Return Unit of Least Precision for each element of a given tensor"

    import torch

    import ttnn

    received_ttnn_input = False
    if isinstance(x, ttnn.Tensor):
        x = ttnn.to_torch(x)
        received_ttnn_input = True

    # Notes:
    # - This should be identical to the definition of ULP by Goldberg
    #   "What every computer scientist should know about floating-point arithmetic"
    #   https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
    # - We use torch.abs(x) to ensure symmetry ULP(-x) == ULP(x)
    # - For x powers of 2, x + ULP(x) is not closest number but second closest (previous number is 2x closer)
    #   However, this avoids rounding-to-nearest-tie-to-even issues on addition (i.e. x + ULP(x) != x)
    abs_x = torch.abs(x)
    next = torch.nextafter(
        abs_x, torch.tensor(math.inf, dtype=x.dtype)
    )  # 1 ULP ~ Difference between two consecutive floating point numbers
    ulp_value = next - abs_x

    # Special case: if abs_x == torch.finfo(x.dtype).max, then next == math.inf, which leads to ULP(x) == inf rather than finite number
    # We fix this problem by manually calculating ULP at max value, and masking tensor when input == max
    dtype_max = torch.finfo(x.dtype).max
    max_epsilon = dtype_max - torch.nextafter(
        torch.tensor(dtype_max, dtype=x.dtype), torch.tensor(-math.inf, dtype=x.dtype)
    )
    ulp_value = torch.where(abs_x == dtype_max, max_epsilon, ulp_value)

    if received_ttnn_input:  # Ensures that type(input) == type(output)
        ulp_value = ttnn.from_torch(ulp_value)

    return ulp_value


def comp_ulp(golden, calculated, ulp_threshold, allow_nonfinite=False):
    """
    Compute absolute error between two tensors in Units of Least Precision (ULP)
    """

    import torch

    # If both tensors are empty, then we can return True
    if torch.numel(golden) == 0 and torch.numel(calculated) == 0:
        return True, "Both tensors are empty"

    if not allow_nonfinite and not torch.all(torch.isfinite(calculated)):
        return False, "Calculated tensor contains non-finite values"

    if not _comp_nonfinite(golden, calculated):
        return False, "Tensors are not finite at the same positions"
    # nonfinite elements can interfere with ULP error calculation
    # To avoid this, replace nan, +inf, -inf with 0
    # (we have already checked that both tensors have the same nonfinite elements)
    mask_finite = ~torch.isfinite(golden)
    golden = golden.clone()
    calculated = calculated.clone()
    golden[mask_finite] = 0
    calculated[mask_finite] = 0

    # ULP is measured according to the golden tensor
    # In most cases, data type of golden tensor should be the same as calculated tensor.
    # However, in some cases, we may want to measure < 1 ULP differences, which requires golden tensor
    # to have higher precision than calculated tensor.
    # If we passed golden tensor to ulp() as is, we would get ULP of higher precision.
    # e.g. ulp of float32 rather bfloat16 calculation, which would give us a wrong value.
    ulp_value = ulp(golden.type(calculated.dtype))

    if golden.dtype != calculated.dtype:  # Note: assumes that golden has higher precision than calculated tensor
        calculated = calculated.type(golden.dtype)
        ulp_value = ulp_value.type(golden.dtype)  # Convert ULP to higher precision (for sub-1 ULP measurements)

    ulp_tensor = torch.abs(calculated - golden) / ulp_value
    ulp_delta = torch.max(ulp_tensor)
    within_threshold = ulp_delta <= ulp_threshold
    message = f"Max ULP Delta: {ulp_delta}"
    if not within_threshold:
        ulp_index = torch.argmax(ulp_tensor)
        ulp_index_tuple = tuple(int(idx) for idx in torch.unravel_index(ulp_index, golden.shape))
        message += (
            f" @ {list(ulp_index_tuple)} = "
            f"|calculated {calculated[ulp_index_tuple]} - golden {golden[ulp_index_tuple]}| "
            f"/ ULP(golden) {ulp_value[ulp_index_tuple]}"
        )
    return (within_threshold, message)
