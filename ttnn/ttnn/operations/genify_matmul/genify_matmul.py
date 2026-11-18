# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""genify_matmul: out = epilogue(A @ B) as ONE fused device program (one ttnn.generic_op dispatch).

Registry model (see eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS and validate() are
declared inline; the public entry point runs the contract checks (ValueError), then validate()
(NotImplementedError subclasses), then builds and dispatches the program descriptor.

Phase 0 builds regime R1 `interleaved_out_split` (op_design.md, Blocking Model): an independent
M x N output split over the device grid, each core walking its own region in blocks with a K-walk
accumulated in L1 (packer L1 accumulation), epilogue none.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .genify_matmul_program_descriptor import (
    TILE,
    BlockShardedLayoutError,
    DramShardedLayoutError,
    block_sharded_output_memory_config,
    block_sharded_topology,
    create_program_descriptor,
    dram_sharded_output_memory_config,
    height_sharded_output_memory_config,
    height_sharded_topology,
    shard_rows,
)

_ML = ttnn.TensorMemoryLayout


# ---------------------------------------------------------------------------
# Default compute config (single factory; the golden axis tagger reads it too)
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """HiFi4, fp32_dest_acc_en=True, math_approx_mode=False."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS  (inputs = (A_shape, B_shape))
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """'tile_aligned' when K, N, M are all multiples of 32; else the first non-aligned in K, N, M."""
    a_shape, b_shape = inputs[0], inputs[1]
    k, n, m = a_shape[-1], b_shape[-1], a_shape[-2]
    if k % TILE:
        return "k_non_aligned"
    if n % TILE:
        return "n_non_aligned"
    if m % TILE:
        return "m_non_aligned"
    return "tile_aligned"


def tag_weight_batch(inputs, axes):
    """'batched' when B has a leading dim > 1, else 'single'."""
    b_shape = inputs[1]
    return "batched" if any(d > 1 for d in list(b_shape)[:-2]) else "single"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "weight_batch": tag_weight_batch,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED  (every axis of feature_spec TARGET)
# ---------------------------------------------------------------------------

SUPPORTED = {
    # Refinement 1: every CB page format derives from its tensor (A, B, output) or from the DEST
    # width (partials: Float32 with fp32 DEST, Float16_b with 16-bit DEST); the matmul_block helper
    # owns the unpack/pack data-format reconfig, so mixed formats need no kernel change.
    "dtype": [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32],
    "weight_dtype": [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b],
    "layout": [ttnn.TILE_LAYOUT],
    "fp32_dest_acc_en": [True, False],
    # Refinement 4: dram_sharded = regime R4 (B streamed from its DRAM bank shards by one worker per
    # bank, A broadcast from its own width-shard cores, output in place in an op-built L1 width shard).
    # Refinement 5: block_sharded = regime R3 (A's K-slices exchanged along each grid row -- own shard
    # zero-copy, peers over a rotating-sender row multicast; B from DRAM; output in place in its shard).
    # Refinement 7: height_sharded = regime R2 (each core's resident A shard holds all of K for its rows:
    # consumed in place through cb_act_shard, B from DRAM, output in place in its L1 height shard).
    "placement": ["interleaved", "dram_sharded", "block_sharded", "height_sharded"],
    # Refinement 2: every epilogue runs inside the same program on the accumulated block
    # (matmul_block -> cb_partials -> add_bias_bcast_rows / eltwise_chain -> cb_output_tiles).
    "epilogue": ["none", "bias", "bias_relu", "swiglu", "addcmul"],
    # m_non_aligned is a knob-turn already realized by the Phase 0 geometry (op_design.md, TARGET
    # table): M is tiled with ceil per image, every image is tile-padded on its own, and padded A rows
    # only feed padded output rows. K / N non-aligned are outside TARGET.
    "alignment": ["tile_aligned", "m_non_aligned"],
    # Refinement 7: batched = regime R1d (grid_b > 1, weight_batch_stride, MatmulBlockShape.batch =
    # block_batches) on interleaved A; R2 / R3 map each A shard onto whole batches or one batch's rows.
    "weight_batch": ["single", "batched"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # Precision convention (task rule): fp32 activations need the fp32 DEST accumulator. Permanent.
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # dram_sharded with a batched weight is used by no model (task rule allows it). Permanent.
    {"placement": "dram_sharded", "weight_batch": "batched"},
]


# ---------------------------------------------------------------------------
# Axis classification (mirrors eval/golden_tests/genify_matmul/axes.py)
# ---------------------------------------------------------------------------


def _classify_placement(input_tensor, weight):
    w = weight.memory_config()
    if w.buffer_type == ttnn.BufferType.DRAM and w.memory_layout == _ML.WIDTH_SHARDED:
        return "dram_sharded"
    return {
        _ML.INTERLEAVED: "interleaved",
        _ML.HEIGHT_SHARDED: "height_sharded",
        _ML.BLOCK_SHARDED: "block_sharded",
    }.get(input_tensor.memory_config().memory_layout, "other")


def _classify_epilogue(bias, activation, residual, gate):
    if residual is not None or gate is not None:
        return "addcmul" if bias is None and activation is None else "other"
    if activation == "swiglu":
        return "swiglu" if bias is None else "other"
    if activation == "relu":
        return "bias_relu" if bias is not None else "other"
    if activation is None:
        return "bias" if bias is not None else "none"
    return "other"


def _resolve_compute_config(compute_kernel_config):
    return compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(
    input_tensor,
    weight,
    *,
    bias=None,
    activation=None,
    residual=None,
    gate=None,
    compute_kernel_config=None,
    memory_config=None,
):
    """Runtime support gate: SUPPORTED per axis, then EXCLUSIONS (both NotImplementedError).

    `input_tensor` may be a list of K segments (Refinement 14): segment 0 stands in for the per-tensor axes
    (the segments share dtype / layout / memory config) and the taggers see the logical concat shape."""
    a_shape = _logical_a_shape(input_tensor)
    if isinstance(input_tensor, (list, tuple)):
        input_tensor = input_tensor[0]
    cfg = _resolve_compute_config(compute_kernel_config)
    axes = {
        "dtype": input_tensor.dtype,
        "weight_dtype": weight.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "placement": _classify_placement(input_tensor, weight),
        "epilogue": _classify_epilogue(bias, activation, residual, gate),
    }
    inputs = (a_shape, list(weight.shape))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"genify_matmul: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"genify_matmul: unsupported combination (refinement candidate): {exc}")

    if weight.layout != ttnn.TILE_LAYOUT:
        raise UnsupportedAxisValue("genify_matmul: weight layout must be TILE_LAYOUT")
    if axes["placement"] == "dram_sharded":
        _validate_dram_sharded(input_tensor, weight, activation, memory_config)
    elif axes["placement"] == "block_sharded":
        _validate_block_sharded(input_tensor, weight, activation, memory_config)
    elif axes["placement"] == "height_sharded":
        _validate_height_sharded(input_tensor, weight, activation, memory_config)
    else:
        # R1 / R1a read B and write the output through interleaved accessors.
        if weight.memory_config().memory_layout != _ML.INTERLEAVED:
            raise UnsupportedAxisValue("genify_matmul: placement 'interleaved' needs an interleaved weight")
        if memory_config is not None and memory_config.memory_layout != _ML.INTERLEAVED:
            raise UnsupportedAxisValue(
                f"genify_matmul: placement 'interleaved' writes an interleaved output; got {memory_config.memory_layout}"
            )
    # Epilogue operands are read through interleaved accessors, tile by tile.
    for name, t in (("bias", bias), ("residual", residual), ("gate", gate)):
        if t is None:
            continue
        if t.layout != ttnn.TILE_LAYOUT:
            raise UnsupportedAxisValue(f"genify_matmul: {name} layout must be TILE_LAYOUT")
        if t.memory_config().memory_layout != _ML.INTERLEAVED:
            raise UnsupportedAxisValue(f"genify_matmul: {name} must be interleaved")
    return axes


def _validate_dram_sharded(input_tensor, weight, activation, memory_config):
    """R4 layout gate: A L1 width-sharded over K (all rows per shard, shards in core order), B DRAM
    width-sharded one column shard per bank; the output is L1 width-sharded on a spec the op builds,
    so a requested memory_config may carry no shard spec or exactly that one."""
    try:
        op_mc = dram_sharded_output_memory_config(input_tensor, weight, swiglu=activation == "swiglu")
    except DramShardedLayoutError as e:
        raise UnsupportedAxisValue(f"genify_matmul: {e}") from e
    if memory_config is None:
        return
    if memory_config.buffer_type != ttnn.BufferType.L1 or memory_config.memory_layout != _ML.WIDTH_SHARDED:
        raise UnsupportedAxisValue(
            "genify_matmul: placement 'dram_sharded' writes an L1 WIDTH_SHARDED output "
            f"(e.g. ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG); got {memory_config}"
        )
    spec = memory_config.shard_spec
    if spec is not None and (list(spec.shape) != list(op_mc.shard_spec.shape) or spec.grid != op_mc.shard_spec.grid):
        raise UnsupportedAxisValue(
            f"genify_matmul: dram_sharded builds its own output shard spec {op_mc.shard_spec}; got {spec}"
        )


def _block_sharded_output_mc(input_tensor, weight, activation, memory_config):
    if memory_config is not None:
        return memory_config
    return block_sharded_output_memory_config(input_tensor, weight, swiglu=activation == "swiglu")


def _validate_block_sharded(input_tensor, weight, activation, memory_config):
    """R3 layout gate: A L1 BLOCK_SHARDED (ROW_MAJOR, rows over grid rows, K over grid columns, the grid
    exactly covering A), B interleaved; the output is L1 BLOCK_SHARDED on A's grid with A's shard height
    (the requested spec, or the op's default when memory_config is None)."""
    if weight.memory_config().memory_layout != _ML.INTERLEAVED:
        raise UnsupportedAxisValue("genify_matmul: placement 'block_sharded' needs an interleaved weight")
    n_out = weight.shape[-1] // 2 if activation == "swiglu" else weight.shape[-1]
    try:
        out_mc = _block_sharded_output_mc(input_tensor, weight, activation, memory_config)
        topo = block_sharded_topology(input_tensor, out_mc, swiglu=activation == "swiglu", out_n=n_out)
        _check_shard_batches(input_tensor, weight, topo.shard_m_tiles)
    except BlockShardedLayoutError as e:
        raise UnsupportedAxisValue(f"genify_matmul: {e}") from e


def _height_sharded_output_mc(input_tensor, weight, activation, memory_config):
    if memory_config is not None:
        return memory_config
    return height_sharded_output_memory_config(input_tensor, weight, swiglu=activation == "swiglu")


def _validate_height_sharded(input_tensor, weight, activation, memory_config):
    """R2 layout gate: A L1 HEIGHT_SHARDED (whole-tile shards holding all of K), B interleaved; the output is
    L1 HEIGHT_SHARDED on A's cores with A's shard height and the whole output width (the requested spec, or
    the op's default when memory_config is None). A batched weight needs every shard inside one batch or
    holding whole batches."""
    if weight.memory_config().memory_layout != _ML.INTERLEAVED:
        raise UnsupportedAxisValue("genify_matmul: placement 'height_sharded' needs an interleaved weight")
    n_out = weight.shape[-1] // 2 if activation == "swiglu" else weight.shape[-1]
    try:
        out_mc = _height_sharded_output_mc(input_tensor, weight, activation, memory_config)
        topo = height_sharded_topology(input_tensor, out_mc, swiglu=activation == "swiglu", out_n=n_out)
        _check_shard_batches(input_tensor, weight, topo.shard_m_tiles)
    except BlockShardedLayoutError as e:
        raise UnsupportedAxisValue(f"genify_matmul: {e}") from e


def _check_shard_batches(input_tensor, weight, shard_m_tiles):
    """Batched weight on a row-sharded A (R2 / R3): BlockShardedLayoutError unless no block can straddle
    two batches (shard_rows is the descriptor's own mapping)."""
    lead = 1
    for d in list(input_tensor.shape)[:-2]:
        lead *= d
    if tag_weight_batch((list(input_tensor.shape), list(weight.shape)), {}) == "batched":
        shard_rows(lead, -(-list(input_tensor.shape)[-2] // TILE), shard_m_tiles)


# ---------------------------------------------------------------------------
# Contract checks (ValueError, before the support gate)
# ---------------------------------------------------------------------------


def _logical_a_shape(input_tensor):
    """A's shape: one tensor's, or concat(A_i, dim=-1)'s for a list of K segments."""
    if isinstance(input_tensor, (list, tuple)):
        a_shape = list(input_tensor[0].shape)
        return a_shape[:-1] + [sum(list(t.shape)[-1] for t in input_tensor)]
    return list(input_tensor.shape)


def _check_segments(segments, weight):
    """A as K segments (Refinement 14): every A_i is (..., M, K_i) with the same leading dims, dtype, layout and
    memory config, K_i % 32 == 0 and sum(K_i) == B[-2]; otherwise ValueError."""
    if len(segments) == 0:
        raise ValueError("genify_matmul: A as a list of K segments needs at least one segment")
    if not all(isinstance(t, ttnn.Tensor) for t in segments):
        raise ValueError("genify_matmul: every K segment of A must be a ttnn.Tensor")
    a0 = segments[0]
    s0 = list(a0.shape)
    if len(s0) < 2:
        raise ValueError(f"genify_matmul: A segments must have rank >= 2, got shape {s0}")
    for i, t in enumerate(segments):
        si = list(t.shape)
        if len(si) != len(s0) or si[:-1] != s0[:-1]:
            raise ValueError(f"genify_matmul: K segment {i} shape {si} must match segment 0's leading dims / M {s0}")
        if si[-1] % TILE:
            raise ValueError(f"genify_matmul: K segment {i} width K_i={si[-1]} must be a multiple of {TILE}")
        if t.dtype != a0.dtype or t.layout != a0.layout:
            raise ValueError(f"genify_matmul: K segment {i} must share segment 0's dtype and layout")
        mc, mc0 = t.memory_config(), a0.memory_config()
        if mc.memory_layout != mc0.memory_layout or mc.buffer_type != mc0.buffer_type:
            raise ValueError(f"genify_matmul: K segment {i} must share segment 0's placement (memory config)")
    k = sum(list(t.shape)[-1] for t in segments)
    if k != weight.shape[-2]:
        raise ValueError(f"genify_matmul: the K segments sum to K={k}, but B[-2]={weight.shape[-2]}")


def _is_row_vector(shape, n):
    return len(shape) >= 2 and shape[-1] == n and all(d == 1 for d in shape[:-1])


def _check_contract(input_tensor, weight, bias, activation, residual, gate):
    a_shape = _logical_a_shape(input_tensor)
    b_shape = list(weight.shape)
    if len(a_shape) < 2:
        raise ValueError(f"genify_matmul: A must have rank >= 2, got shape {a_shape}")
    if len(b_shape) < 2:
        raise ValueError(f"genify_matmul: B must have rank >= 2, got shape {b_shape}")
    if a_shape[-1] != b_shape[-2]:
        raise ValueError(f"genify_matmul: A[-1]={a_shape[-1]} != B[-2]={b_shape[-2]}")
    # B is one shared (K, N) matrix (all leading dims 1), or batched with leading dims equal to A's
    # (B's leading dims left-padded with 1s to A's rank, as broadcasting would).
    a_lead, b_lead = a_shape[:-2], b_shape[:-2]
    b_lead_padded = [1] * (len(a_lead) - len(b_lead)) + b_lead
    shared_b = all(d == 1 for d in b_lead)
    if len(b_shape) > 2 and not (len(b_lead) <= len(a_lead) and (shared_b or b_lead_padded == a_lead)):
        raise ValueError(
            f"genify_matmul: B must be 2-D or batched with leading dims equal to A's; A {a_shape}, B {b_shape}"
        )
    if activation not in (None, "relu", "swiglu"):
        raise ValueError(f"genify_matmul: activation must be None, 'relu' or 'swiglu', got {activation!r}")
    n = b_shape[-1]
    if activation == "swiglu" and n % 64 != 0:
        raise ValueError(f"genify_matmul: swiglu needs N % 64 == 0, got N={n}")
    if (residual is None) != (gate is None):
        raise ValueError("genify_matmul: residual and gate must be given together (addcmul)")
    if bias is not None and not _is_row_vector(list(bias.shape), n):
        raise ValueError(f"genify_matmul: bias must be (1, {n}), got {list(bias.shape)}")
    if gate is not None and not _is_row_vector(list(gate.shape), n):
        raise ValueError(f"genify_matmul: gate must be (1, {n}), got {list(gate.shape)}")
    if residual is not None and list(residual.shape) != a_shape[:-1] + [n]:
        raise ValueError(f"genify_matmul: residual must be {a_shape[:-1] + [n]}, got {list(residual.shape)}")
    if _classify_epilogue(bias, activation, residual, gate) == "other":
        raise ValueError(
            "genify_matmul: unsupported kwarg combination (relu needs bias; swiglu takes no bias; "
            "addcmul takes no bias/activation)"
        )


def _check_chunks(weight, activation, chunks):
    """Split output contract: N_out % chunks == 0 and every chunk a whole number of tiles wide."""
    if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 1:
        raise ValueError(f"genify_matmul: chunks must be an int >= 1, got {chunks!r}")
    n_out = weight.shape[-1] // 2 if activation == "swiglu" else weight.shape[-1]
    if n_out % chunks or (n_out // chunks) % TILE:
        raise ValueError(
            f"genify_matmul: chunks={chunks} must split N_out={n_out} into equal widths that are multiples of {TILE}"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def genify_matmul(
    input_tensor,
    weight: ttnn.Tensor,
    *,
    bias: ttnn.Tensor = None,
    activation: str = None,
    residual: ttnn.Tensor = None,
    gate: ttnn.Tensor = None,
    dtype: ttnn.DataType = None,
    memory_config: ttnn.MemoryConfig = None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    chunks: int = 1,
):
    """out = epilogue(A @ B) in one device program. See op_design.md.

    input_tensor: one tensor A (..., M, K), or a list [A_0, A_1, ...] of K segments read as
    concat(A_i, dim=-1) without materializing the concatenation (K = the segments' K summed).
    chunks: > 1 returns the output split along its last dim into `chunks` equal tensors, as a list
    (Refinement 13): chunk i is columns [i*w, (i+1)*w) of the output, w = N_out / chunks, each written
    directly by the same program (no full-width intermediate). Interleaved placement only.
    A list input (Refinement 14): the reader maps each K tile to its segment and local tile, so the
    concatenation is never built. Interleaved A (DRAM or L1) only.
    """
    segments = list(input_tensor) if isinstance(input_tensor, (list, tuple)) else None
    if segments is not None:
        _check_segments(segments, weight)
        if len(segments) == 1:
            input_tensor = segments[0]  # one segment is the single-tensor call
            segments = None
    _check_contract(input_tensor, weight, bias, activation, residual, gate)
    _check_chunks(weight, activation, chunks)
    axes = validate(
        input_tensor,
        weight,
        bias=bias,
        activation=activation,
        residual=residual,
        gate=gate,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )

    if chunks != 1 and axes["placement"] != "interleaved":
        raise UnsupportedAxisValue(
            f"genify_matmul: chunks > 1 (split output) is implemented for the interleaved placement only; "
            f"got {axes['placement']}"
        )
    if segments is not None and axes["placement"] != "interleaved":
        raise UnsupportedAxisValue(
            f"genify_matmul: A as K segments is implemented for the interleaved placement only; got {axes['placement']}"
        )

    a_op = segments if segments is not None else input_tensor  # what the program descriptor reads as A
    input_tensor = segments[0] if segments is not None else input_tensor  # per-tensor properties
    cfg = _resolve_compute_config(compute_kernel_config)
    device = input_tensor.device()
    out_dtype = dtype if dtype is not None else input_tensor.dtype
    placement = axes["placement"]
    if placement == "dram_sharded":
        # The op chooses the output shard spec: one width shard per bank worker.
        out_mc = dram_sharded_output_memory_config(input_tensor, weight, swiglu=activation == "swiglu")
    elif placement == "block_sharded":
        # Exactly the requested shard spec (validated), or the op's default block-shard placement.
        out_mc = _block_sharded_output_mc(input_tensor, weight, activation, memory_config)
    elif placement == "height_sharded":
        out_mc = _height_sharded_output_mc(input_tensor, weight, activation, memory_config)
    else:
        out_mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    a_shape = _logical_a_shape(a_op)
    n_out = weight.shape[-1] // 2 if activation == "swiglu" else weight.shape[-1]
    # Split output: `chunks` equal-width tensors, each with memory_config and dtype (chunks == 1: one tensor).
    out_shape = a_shape[:-1] + [n_out // chunks]

    # allocate_tensor_on_device takes positional args only.
    outputs = [
        ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), out_dtype, ttnn.TILE_LAYOUT, device, out_mc)
        for _ in range(chunks)
    ]
    output_tensor = outputs if chunks != 1 else outputs[0]

    epilogue = _classify_epilogue(bias, activation, residual, gate)
    row = bias if bias is not None else gate  # the (1, N) row the epilogue broadcasts
    program_descriptor = create_program_descriptor(
        a_op, weight, output_tensor, cfg, epilogue=epilogue, row=row, residual=residual, placement=placement
    )
    # Exactly one dispatch; the output tensor(s) last.
    io_tensors = [t for t in (*(segments or [input_tensor]), weight, row, residual) if t is not None]
    result = ttnn.generic_op(io_tensors + outputs, program_descriptor)
    return outputs if chunks != 1 else result
