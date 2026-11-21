# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for rotary_embedding_llama (regime `resident_shard_in_place`, op_design.md).

Every operand is already in the shard core's L1, so no byte moves over the NoC: each tensor's shard backs a
borrowed (zero-copy) ring, the reader only posts the credits of the resident tiles, the writer only drains the
output credits, and the compute kernel packs the result straight into the output shard.

Work distribution (op_design.md, Work Distribution):
  * cores: the input shard's core set (one user's shard per core; users_per_core = 1 in Phase 0);
  * thread plan THREAD_PLAN = (R, C, W). The head_dim tiles j of a core are dealt over its C NEOs by the
    STRIDED rings: NEO t computes tiles t, t+C, ... (neo_dim_tiles of them). C is clamped so that it divides
    tensor_dim_tiles, because a borrowed ring's capacity is pinned to the shard (capacity per NEO = Wt / C);
  * trans_mat is one tile shared by every NEO: an ALL ring.

Per NEO, per block (= one user's share):
  rotate_block          rotated = x @ T                       raw LLK matmul
  rotate_combine_block  out = rotated * sin_row + x * cos_row  eltwise_chain (row-broadcast FPU muls + SFPU add)
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

TILE = 32

# The cluster model of the target page (.claude/references/targets/quasar.md); Python has no query for them.
NEOS = 4
FREE_DM_HARTS = 6  # of 8: DM0 serves the ring interrupts, DM1 the remapper

# The thread plan (R readers, C NEOs, W writers): the op is compute-bound and its data-movement threads move no
# bytes (credits only), so one reader and one writer feed four NEOs.
THREAD_PLAN = (1, 4, 1)

# Block knobs (op_design.md, Blocking Model). Each is defined once here; every dependent count derives from it.
HEAD_ROW_TILES = 1  # a user's heads are one tile row (validate: heads <= 32)
# (contraction_tiles is not a knob: T is one 32x32 tile, so the whole contraction is one matmul_tiles call and no
# loop runs over it.)
ROTATED_DEPTH_BLOCKS = 1  # cb_rotated holds one block per NEO: the two phases are sequential in one kernel

# Ring (DFB) names, also the kernels' accessor names.
INPUT_TILES, COS_TILES, SIN_TILES, TRANS_MAT, ROTATED, OUTPUT_TILES = (
    "input_tiles", "cos_tiles", "sin_tiles", "trans_mat", "rotated", "output_tiles",
)
# Tensor parameters, in the order the op passes its tensors to generic_op.
TENSOR_PARAMETERS = ("input", "cos", "sin", "trans_mat_tensor", "output")

_ELEMENT_BYTES = {ttnn.bfloat16: 2, ttnn.float32: 4}


def validate_thread_plan(threads):
    r, c, w = threads
    if not (1 <= c <= NEOS and r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and c % r == 0 and c % w == 0):
        raise ValueError(f"thread plan (R, C, W) = {threads}: need C <= {NEOS}, R + W <= {FREE_DM_HARTS}, R | C, W | C")
    if r != 1:
        # The trans_mat ring holds one entry, and an ALL ring's capacity is num_entries / producers.
        raise ValueError(f"thread plan (R, C, W) = {threads}: the trans_mat ring has one entry, so R must be 1")


def effective_compute_threads(tensor_dim_tiles, threads):
    """The plan's C clamped to a count that divides the core's head_dim tiles (and that R and W divide): a
    borrowed STRIDED ring's capacity is the shard, num_entries / C entries per NEO (op_design.md, Mechanism caps)."""
    r, c_plan, w = threads
    return max(c for c in (4, 2, 1) if c <= c_plan and tensor_dim_tiles % c == 0 and c % r == 0 and c % w == 0)


def _compute_hw_config(config, rotated_dtype):
    """The compute kernel's Metal 2.0 config from the resolved ttnn.ComputeConfigDescriptor. The fidelity is always
    set explicitly. Every ring unpacks to the source registers: each one feeds the FPU (matmul or a row-broadcast
    multiply). A 32-bit ring under a 32-bit DEST must say so explicitly (the host refuses it otherwise); the
    bfloat16 rings take the default, which is the same mode."""
    unpack_modes = {}
    if config.fp32_dest_acc_en and rotated_dtype == ttnn.float32:
        unpack_modes[ROTATED] = E.UnpackMode.UnpackToSrc
    return E.ComputeHardwareConfig(
        fpu_math_fidelity=config.math_fidelity,
        sfpu_precision_mode=E.Precision.Approximate if config.math_approx_mode else E.Precision.Precise,
        enable_32_bit_dest=bool(config.fp32_dest_acc_en),
        unpack_modes=unpack_modes,
    )


def _dm_config(make, rings):
    # Explicit sync on every ring of the kernel: no NoC transfer fills or drains them, the kernel only posts or
    # drains credits itself.
    hw = make()
    hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=list(rings))
    return hw


def create_program_artifacts(x, cos, sin, trans_mat, out, config, *, threads=THREAD_PLAN, tag_threads=False):
    """(ProgramSpec, ProgramRunArgs, tensor indices) for one call.

    `threads` is the thread plan (R, C, W); the compute count actually used is clamped (effective_compute_threads).
    `tag_threads` is a test probe: odd NEOs subtract instead of add (out = rot*sin - x*cos), so a test sees which
    NEO computed which tile.
    """
    validate_thread_plan(threads)
    R, _, W = threads

    shard_spec = x.memory_config().shard_spec
    shard_height, shard_width = int(shard_spec.shape[0]), int(shard_spec.shape[1])
    tensor_dim_tiles = shard_width // TILE  # Wt: head_dim tiles of one user's head rows
    users_per_core = shard_height // (TILE * HEAD_ROW_TILES)  # blocks per core
    C = effective_compute_threads(tensor_dim_tiles, threads)
    neo_dim_tiles = tensor_dim_tiles // C  # exact by the clamp

    # The resident shards, borrowed: each ring *is* its tensor's shard (one entry per tile, the whole shard).
    borrowed = [
        E.dfb_spec_from_sharded_tensor(ring, tensor, borrowed_from=param)
        for ring, tensor, param in (
            (INPUT_TILES, x, TENSOR_PARAMETERS[0]),
            (COS_TILES, cos, TENSOR_PARAMETERS[1]),
            (SIN_TILES, sin, TENSOR_PARAMETERS[2]),
            (TRANS_MAT, trans_mat, TENSOR_PARAMETERS[3]),
            (OUTPUT_TILES, out, TENSOR_PARAMETERS[4]),
        )
    ]
    # The one ring the op allocates: x @ T between the two compute phases. It is kept in the input dtype, not in
    # DEST's width (a deviation from the ledger's audit 2): the same dtype as every other FPU operand keeps one
    # consistent ALU format for the whole kernel.
    rotated_dtype = x.dtype
    rotated = E.DataflowBufferSpec(
        unique_id=ROTATED,
        entry_size=TILE * TILE * _ELEMENT_BYTES[rotated_dtype],
        num_entries=neo_dim_tiles * C * ROTATED_DEPTH_BLOCKS,  # capacity per NEO x C
        data_format_metadata=rotated_dtype,
    )

    tensor_parameters = [
        E.TensorParameter(unique_id=name, spec=t.spec)
        for name, t in zip(TENSOR_PARAMETERS, (x, cos, sin, trans_mat, out))
    ]
    plan = {"reader_threads": R, "compute_threads": C, "writer_threads": W}
    blocks = {"users_per_core": users_per_core, "tensor_dim_tiles": tensor_dim_tiles}

    reader_rings = (INPUT_TILES, COS_TILES, SIN_TILES, TRANS_MAT)
    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "rotary_embedding_llama_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, reader_rings),
        num_threads=R,
        dfb_bindings=[E.producer_of(ring, ring) for ring in reader_rings],
        compile_time_args={**plan, **blocks},
    )
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "rotary_embedding_llama_compute.cpp"),
        hw_config=_compute_hw_config(config, rotated_dtype),
        num_threads=C,
        dfb_bindings=[
            E.consumer_of(INPUT_TILES, INPUT_TILES),
            E.consumer_of(COS_TILES, COS_TILES),
            E.consumer_of(SIN_TILES, SIN_TILES),
            E.all_consumer_of(TRANS_MAT, TRANS_MAT),  # one tile, seen by every NEO
            E.producer_of(ROTATED, ROTATED),  # the kernel's own ring, per NEO
            E.consumer_of(ROTATED, ROTATED),
            E.producer_of(OUTPUT_TILES, OUTPUT_TILES),
        ],
        compile_time_args={
            **plan,
            "users_per_core": users_per_core,
            "neo_dim_tiles": neo_dim_tiles,
            "tag_threads": int(tag_threads),
        },
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "rotary_embedding_llama_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config, (OUTPUT_TILES,)),
        num_threads=W,
        dfb_bindings=[E.consumer_of(OUTPUT_TILES, OUTPUT_TILES)],
        compile_time_args={**plan, **blocks},
    )
    spec = E.ProgramSpec(
        name="rotary_embedding_llama",
        kernels=[reader, compute, writer],
        dataflow_buffers=[*borrowed, rotated],
        tensor_parameters=tensor_parameters,
        work_units=[
            E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=shard_spec.grid)
        ],
    )
    run_args = E.ProgramRunArgs(kernel_run_args=[])
    return spec, run_args, {name: i for i, name in enumerate(TENSOR_PARAMETERS)}
