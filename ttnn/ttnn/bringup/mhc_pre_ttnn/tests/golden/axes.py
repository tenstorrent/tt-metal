# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Observe-only runtime axis tagging for mhc_pre.

Runtime inverse of run_mhc_pre: classify_call reconstructs the registry
cell from the real op call, observe() records it (verify_supported /
dashboard chips), and observed() wraps the op so golden AND translated
tests tag uniformly. INPUT_TAGGERS are imported from the op (single source
of truth) — never re-declared here. The op still owns the support gate
(validate raises SupportRefusal); this module only witnesses the call.
"""

from __future__ import annotations

import warnings

import ttnn  # noqa: F401  (kept for parity with sibling axes modules)

from eval import metrics_plugin
from eval.feature_matrix import apply_input_taggers
from ttnn.bringup.mhc_pre_ttnn import (  # type: ignore
    INPUT_TAGGERS,
    default_compute_kernel_config,
)

# The C++ op (ttnn.bringup.mhc_pre); the Python module supplies the registry declarations and the default config.
_raw = ttnn.bringup.mhc_pre


def classify_call(
    input_tensor,
    proj_weight,
    proj_bias,
    *,
    scale,
    sinkhorn_iters=20,
    eps=1e-6,
    norm_eps=1e-6,
    compute_kernel_config=None,
    **_,
):
    """Reconstruct the registry axes cell from a real mhc_pre call.

    Mirrors the op signature. fp32_dest_acc_en is read through the op's own
    default-config factory (never a hardcoded default here).
    """
    cfg = compute_kernel_config or default_compute_kernel_config()
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "weight_dtype": proj_weight.dtype,
        "fp32_dest_acc_en": bool(getattr(cfg, "fp32_dest_acc_en", True)),
    }
    tagger_inputs = (list(input_tensor.shape), list(proj_weight.shape))
    axes.update(apply_input_taggers(INPUT_TAGGERS, tagger_inputs, axes))
    return axes


def observe(axes):
    """Record the classified cell onto the running test (no-op off-test)."""
    metrics_plugin.record_axes(axes)


def observed(*args, **kwargs):
    """Observe-only wrapper: tag the call, then dispatch the real op.

    observe() runs BEFORE dispatch so a refused cell is tagged before
    validate() raises. Tagging is best-effort: never break op dispatch.
    """
    try:
        observe(classify_call(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 — tagging must never break dispatch
        warnings.warn(f"mhc_pre classify_call failed (row will be untagged): {exc!r}")
    # op_window brackets exactly the dispatch: prep before and readback after
    # stay out of device_kernel_ns.
    with metrics_plugin.op_window():
        return _raw(*args, **kwargs)
