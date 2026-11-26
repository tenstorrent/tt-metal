# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""gen_qkv_causal_conv1d_silu — four-tap depthwise causal conv1d over tokens, SiLU, split to q/k/v.

    x[-3..-1] = history rows 0..2;  x[t] = input[0, t, offset + c]
    y[t, c]   = silu(sum_{p=0..3} tap_p[c] * x[t-3+p, c])
    q, k, v   = y[..., :Q], y[..., Q:Q+K], y[..., Q+K:]

One device program per call (ttnn.generic_op); a ROW_MAJOR input is read as row-major pages and
tilized inside that program. Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate().
"""

from __future__ import annotations

import contextlib
from math import gcd
from typing import Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .gen_qkv_causal_conv1d_silu_program_descriptor import (
    HALO_ROWS as HISTORY_ROWS,
    L1_BOUND_BYTES,
    TILE_DIM,
    Plan,
    create_program_descriptor,
    output_l1_share,
    rewrite_addresses,
    select_plan,
    select_regime,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS (verbatim from the op spec, in that order)
# ---------------------------------------------------------------------------


def tag_history_layout(inputs, axes):
    return "token_paged" if inputs[3] is None else "channel_paged"


def tag_input_window(inputs, axes):
    (_, _, width), widths, _, _, offset = inputs
    return "exact" if offset == 0 and width == sum(widths) else "column_window"


def tag_width_split(inputs, axes):
    q, k, v = inputs[1]
    return "equal" if q == k == v else "unequal"


def tag_channel_count(inputs, axes):
    return "narrow" if sum(inputs[1]) < 2048 else "wide"


def tag_seq_tiles(inputs, axes):
    seq = inputs[0][1]
    return "one" if seq == 32 else ("few" if seq < 1024 else "many")


def tag_chunk_blocks(inputs, axes):
    (q, k, v), chunk = inputs[1], inputs[2]
    if chunk == q + k + v:
        return "single"  # one channel block
    return "aligned" if gcd(gcd(q, k), v) % chunk == 0 else "straddling"
    # straddling: some block crosses the q|k or k|v boundary


INPUT_TAGGERS = {
    "history_layout": tag_history_layout,
    "input_window": tag_input_window,
    "width_split": tag_width_split,
    "channel_count": tag_channel_count,
    "seq_tiles": tag_seq_tiles,
    "chunk_blocks": tag_chunk_blocks,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED (Phase 0) — one entry per feature_spec TARGET axis
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    "in_buffer": ["dram", "l1"],
    "history_buffer": ["dram", "l1"],
    "taps_buffer": ["dram", "l1"],
    "output_memory": ["default", "dram_interleaved", "l1_interleaved"],
    "fp32_dest_acc_en": [False, True],
    "history_layout": ["token_paged"],
    "input_window": ["exact"],
    "width_split": ["unequal", "equal"],
    "channel_count": ["wide", "narrow"],
    "seq_tiles": ["few", "one", "many"],
    "chunk_blocks": ["aligned", "straddling", "single"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# Compute configuration: the single definition of "None"
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """The native default: HiFi4, precise SFPU, 16-bit DEST, no packer L1 acc, half-sync DEST."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )


# ---------------------------------------------------------------------------
# Plan reporting (BB-08) and plan pinning
# ---------------------------------------------------------------------------

_PLAN_LIMITS = {"max_block_rows": None, "max_cores_per_group": None}
_LAST_PLAN: Optional[Plan] = None


def get_last_plan() -> Optional[Plan]:
    """The Plan the most recent call ran."""
    return _LAST_PLAN


@contextlib.contextmanager
def plan_limits(max_block_rows: Optional[int] = None, max_cores_per_group: Optional[int] = None):
    """Cap block_rows / cores_per_group inside the block (enters select_plan and the cache key)."""
    saved = dict(_PLAN_LIMITS)
    _PLAN_LIMITS["max_block_rows"] = max_block_rows
    _PLAN_LIMITS["max_cores_per_group"] = max_cores_per_group
    try:
        yield
    finally:
        _PLAN_LIMITS.update(saved)


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _buffer_tag(tensor) -> str:
    return "dram" if tensor.memory_config().buffer_type == ttnn.BufferType.DRAM else "l1"


def _output_memory_tag(memory_config) -> str:
    if memory_config is None:
        return "default"
    return "dram_interleaved" if memory_config.buffer_type == ttnn.BufferType.DRAM else "l1_interleaved"


def _device_key(device):
    try:
        return device.id()
    except Exception:  # noqa: BLE001
        return id(device)


def _mesh_dims(device):
    shape = getattr(device, "shape", None)
    if shape is None:
        return []
    try:
        return [int(d) for d in shape]
    except TypeError:
        return [int(shape[i]) for i in range(shape.dims())]


def _check_device_tensor(name, tensor, device_key):
    if not isinstance(tensor, ttnn.Tensor):
        raise ValueError(f"gen_qkv_causal_conv1d_silu: {name} must be a ttnn.Tensor")
    if tensor.storage_type() != ttnn.StorageType.DEVICE or not tensor.is_allocated():
        raise ValueError(f"gen_qkv_causal_conv1d_silu: {name} must be an allocated device tensor")
    if device_key is not None and _device_key(tensor.device()) != device_key:
        raise ValueError(f"gen_qkv_causal_conv1d_silu: {name} is not on the input's device")
    if tensor.memory_config().is_sharded():
        raise ValueError(f"gen_qkv_causal_conv1d_silu: {name} must be interleaved, not sharded")
    if tensor.memory_config().buffer_type not in (ttnn.BufferType.DRAM, ttnn.BufferType.L1):
        raise ValueError(f"gen_qkv_causal_conv1d_silu: {name} must live in DRAM or L1")


def _check_malformed(
    input_tensor,
    history,
    taps,
    widths,
    program_config,
    actual_start,
    predecessor_carry,
    memory_config,
    compute_kernel_config,
    sequence_parallel_axis,
    input_column_offset,
    page_channels,
):
    """ValueError for every malformed call (## Validation of the op spec)."""
    _check_device_tensor("input", input_tensor, None)
    shape = tuple(int(d) for d in input_tensor.shape)
    if len(shape) != 3:
        raise ValueError(f"gen_qkv_causal_conv1d_silu: input must be rank 3 [1, T, W], got {list(shape)}")
    batch, seq, width = shape
    if batch != 1:
        raise ValueError(f"gen_qkv_causal_conv1d_silu: batch must be 1, got {batch}")
    if seq <= 0 or seq % TILE_DIM:
        raise ValueError(f"gen_qkv_causal_conv1d_silu: T must be a positive multiple of 32, got {seq}")
    for name, x in zip(("q_width", "k_width", "v_width"), widths):
        if not isinstance(x, int) or x <= 0 or x % TILE_DIM:
            raise ValueError(f"gen_qkv_causal_conv1d_silu: {name} must be a positive multiple of 32, got {x}")
    channels = sum(widths)
    if not isinstance(input_column_offset, int) or input_column_offset < 0 or input_column_offset % TILE_DIM:
        raise ValueError(
            f"gen_qkv_causal_conv1d_silu: input_column_offset must be a multiple of 32, got {input_column_offset}"
        )
    if width < input_column_offset + channels:
        raise ValueError(
            f"gen_qkv_causal_conv1d_silu: input width {width} < input_column_offset + C = "
            f"{input_column_offset + channels}"
        )
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"gen_qkv_causal_conv1d_silu: input must be bfloat16, got {input_tensor.dtype}")
    if input_tensor.layout not in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
        raise ValueError(f"gen_qkv_causal_conv1d_silu: input layout must be TILE or ROW_MAJOR")

    device = input_tensor.device()
    device_key = _device_key(device)
    _check_device_tensor("history", history, device_key)
    for j, tap in enumerate(taps):
        _check_device_tensor(f"tap{j}", tap, device_key)
    _check_device_tensor("actual_start", actual_start, device_key)
    _check_device_tensor("predecessor_carry", predecessor_carry, device_key)

    if history.dtype != ttnn.bfloat16 or history.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("gen_qkv_causal_conv1d_silu: history must be bfloat16 ROW_MAJOR")
    for j, tap in enumerate(taps):
        if tap.dtype != ttnn.bfloat16 or tap.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"gen_qkv_causal_conv1d_silu: tap{j} must be bfloat16 TILE")
        if int(tap.shape[-1]) != channels or int(tap.logical_volume()) != channels:
            raise ValueError(
                f"gen_qkv_causal_conv1d_silu: tap{j} must have last dimension and volume C={channels}, "
                f"got shape {list(tap.shape)}"
            )
    if actual_start.dtype != ttnn.uint32 or actual_start.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("gen_qkv_causal_conv1d_silu: actual_start must be a UINT32 ROW_MAJOR tensor")
    if int(actual_start.logical_volume()) != 1:
        raise ValueError("gen_qkv_causal_conv1d_silu: actual_start must have volume 1")

    chunk = getattr(program_config, "channel_chunk_size", None)
    if not isinstance(chunk, int) or chunk <= 0 or chunk % TILE_DIM or chunk > channels or channels % chunk:
        raise ValueError(
            f"gen_qkv_causal_conv1d_silu: channel_chunk_size must be a positive multiple of 32, at most "
            f"C={channels} and dividing it; got {chunk}"
        )

    if page_channels is None:
        expected_history = (1, HISTORY_ROWS, channels)
    else:
        if (
            not isinstance(page_channels, int)
            or page_channels <= 0
            or page_channels % TILE_DIM
            or channels % page_channels
        ):
            raise ValueError(
                f"gen_qkv_causal_conv1d_silu: page_channels must be a positive multiple of 32 dividing C, "
                f"got {page_channels}"
            )
        expected_history = (1, channels // page_channels, HISTORY_ROWS * page_channels)
    if tuple(int(d) for d in history.shape) != expected_history:
        raise ValueError(
            f"gen_qkv_causal_conv1d_silu: history shape must be {list(expected_history)}, got {list(history.shape)}"
        )
    if predecessor_carry.spec != history.spec:
        raise ValueError("gen_qkv_causal_conv1d_silu: predecessor_carry must have exactly history's spec")

    if memory_config is not None:
        if memory_config.is_sharded():
            raise ValueError("gen_qkv_causal_conv1d_silu: a sharded memory_config is refused")
        if memory_config.buffer_type not in (ttnn.BufferType.DRAM, ttnn.BufferType.L1):
            raise ValueError("gen_qkv_causal_conv1d_silu: memory_config must be DRAM or L1 interleaved")

    config = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    if getattr(config, "math_approx_mode", False):
        raise ValueError("gen_qkv_causal_conv1d_silu: math_approx_mode=True is refused (precise SiLU)")
    if getattr(config, "packer_l1_acc", False):
        raise ValueError("gen_qkv_causal_conv1d_silu: packer_l1_acc=True is refused")

    if not isinstance(sequence_parallel_axis, int) or sequence_parallel_axis < 0:
        raise ValueError(f"gen_qkv_causal_conv1d_silu: bad sequence_parallel_axis {sequence_parallel_axis}")
    dims = _mesh_dims(device)
    if dims:
        if sequence_parallel_axis >= len(dims):
            raise ValueError(
                f"gen_qkv_causal_conv1d_silu: sequence_parallel_axis {sequence_parallel_axis} outside mesh {dims}"
            )
        if dims[sequence_parallel_axis] > 1:
            raise UnsupportedAxisValue(
                "gen_qkv_causal_conv1d_silu: sequence parallelism over more than one rank is not supported"
            )
    return config


def validate(
    input_tensor,
    history,
    tap0,
    tap1,
    tap2,
    tap3,
    q_width,
    k_width,
    v_width,
    *,
    program_config,
    actual_start,
    predecessor_carry,
    memory_config=None,
    compute_kernel_config=None,
    sequence_parallel_axis=0,
    input_column_offset=0,
    page_channels=None,
):
    """ValueError checks -> axes (pinned + taggers) -> SUPPORTED -> EXCLUSIONS.

    Returns the resolved compute kernel config.
    """
    widths = (q_width, k_width, v_width)
    taps = (tap0, tap1, tap2, tap3)
    config = _check_malformed(
        input_tensor,
        history,
        taps,
        widths,
        program_config,
        actual_start,
        predecessor_carry,
        memory_config,
        compute_kernel_config,
        sequence_parallel_axis,
        input_column_offset,
        page_channels,
    )
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "in_buffer": _buffer_tag(input_tensor),
        "history_buffer": _buffer_tag(history),
        "taps_buffer": _buffer_tag(tap0),
        "output_memory": _output_memory_tag(memory_config),
        "fp32_dest_acc_en": bool(getattr(config, "fp32_dest_acc_en", False)),
    }
    inputs = (
        tuple(int(d) for d in input_tensor.shape),
        (int(q_width), int(k_width), int(v_width)),
        int(program_config.channel_chunk_size),
        page_channels,
        int(input_column_offset),
    )
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"gen_qkv_causal_conv1d_silu: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"gen_qkv_causal_conv1d_silu: unsupported combination (refinement candidate): {exc}")
    return config


# ---------------------------------------------------------------------------
# Program cache (BB-07): a hit rewrites only buffer addresses
# ---------------------------------------------------------------------------

_PROGRAM_CACHE = {}


def _tensor_key(tensor):
    config = tensor.memory_config()
    return (tuple(int(d) for d in tensor.shape), tensor.layout, tensor.dtype, config.buffer_type)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def gen_qkv_causal_conv1d_silu(
    input: ttnn.Tensor,  # noqa: A002 — the native signature
    history: ttnn.Tensor,
    tap0: ttnn.Tensor,
    tap1: ttnn.Tensor,
    tap2: ttnn.Tensor,
    tap3: ttnn.Tensor,
    q_width: int,
    k_width: int,
    v_width: int,
    *,
    program_config,
    actual_start: ttnn.Tensor,
    predecessor_carry: ttnn.Tensor,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    compute_kernel_config=None,
    sequence_parallel_axis: int = 0,
    input_column_offset: int = 0,
    page_channels: Optional[int] = None,
):
    """Returns (q, k, v): bfloat16 TILE tensors [1, T, Q], [1, T, K], [1, T, V]."""
    global _LAST_PLAN
    config = validate(
        input,
        history,
        tap0,
        tap1,
        tap2,
        tap3,
        q_width,
        k_width,
        v_width,
        program_config=program_config,
        actual_start=actual_start,
        predecessor_carry=predecessor_carry,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
        sequence_parallel_axis=sequence_parallel_axis,
        input_column_offset=input_column_offset,
        page_channels=page_channels,
    )

    device = input.device()
    widths = (int(q_width), int(k_width), int(v_width))
    taps = (tap0, tap1, tap2, tap3)
    seq = int(input.shape[1])
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    outputs = tuple(
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, seq, x]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, output_memory_config
        )
        for x in widths
    )

    math_fidelity = config.math_fidelity
    fp32_dest_acc_en = bool(getattr(config, "fp32_dest_acc_en", False))
    dst_full_sync_en = bool(getattr(config, "dst_full_sync_en", False))
    grid = device.compute_with_storage_grid_size()
    limits = (_PLAN_LIMITS["max_block_rows"], _PLAN_LIMITS["max_cores_per_group"])
    key = (
        _device_key(device),
        (grid.x, grid.y),
        _tensor_key(input),
        _tensor_key(history),
        tuple(_tensor_key(t) for t in taps),
        widths,
        output_memory_config.buffer_type,
        output_memory_config.memory_layout,
        math_fidelity,
        fp32_dest_acc_en,
        dst_full_sync_en,
        int(input_column_offset),
        # [1, 3, C] (token-paged) and [1, C/P, 3P] (channel-paged) coincide when C == 3P, so the
        # history layout is not implied by the history shape and must key the program itself.
        page_channels,
        limits,
    )
    entry = _PROGRAM_CACHE.get(key)
    if entry is None:
        tensor_seq_tiles = seq // TILE_DIM
        tensor_ch_tiles = sum(widths) // TILE_DIM
        budget = L1_BOUND_BYTES
        if output_memory_config.buffer_type == ttnn.BufferType.L1:
            num_l1_banks = ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks
            budget -= output_l1_share(tensor_seq_tiles, [x // TILE_DIM for x in widths], num_l1_banks)
        plan = select_plan(
            tensor_seq_tiles,
            tensor_ch_tiles,
            grid.x,
            grid.y,
            select_regime(input.layout),
            budget,
            max_block_rows=limits[0],
            max_cores_per_group=limits[1],
            elem_bytes=input.element_size(),
            tile_bytes=ttnn.tile_size(input.dtype),
        )
        descriptor, work = create_program_descriptor(
            input,
            history,
            taps,
            outputs,
            widths,
            plan,
            grid_x=grid.x,
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
            input_column_offset=int(input_column_offset),
        )
        entry = (plan, descriptor, work)
        _PROGRAM_CACHE[key] = entry
    else:
        plan, descriptor, work = entry
        rewrite_addresses(descriptor, work, input, history, taps, outputs)

    _LAST_PLAN = entry[0]
    ttnn.generic_op([input, history, tap0, tap1, tap2, tap3, *outputs], entry[1])
    return outputs
