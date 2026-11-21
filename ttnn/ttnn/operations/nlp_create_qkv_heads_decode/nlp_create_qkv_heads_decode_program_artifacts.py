# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for nlp_create_qkv_heads_decode (regime A of op_design.md: direct gather).

One data-movement kernel, `gather`, runs on each user's core with GATHER_THREADS harts (thread plan
(R, C, W) = (GATHER_THREADS, 0, 0)): no compute kernel (the op computes nothing, R1) and no writer kernel
(every NoC read lands in its final place in the user's q / k / v shard, so nothing is written back).

The user's input row is 2 * input_width_tiles face-row runs (a tile row is the left-face run and the
right-face run). The runs are numbered flat over (section, head_dim tile, face column) and dealt
round-robin to the harts: hart t moves runs t, t+R, ... Each run is one tagged NoC read from the
interleaved input page into the shard; the harts drain their reads once per block of `heads_per_block`
sections (Phase 0: the whole user, one barrier per hart).

No dataflow buffer, scratchpad or semaphore is allocated (l1_ledger.md): the outputs are bound as tensors.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# The cluster model of the target page (.claude/references/targets/quasar.md); Python has no query for it.
FREE_DM_HARTS = 6  # of 8: DM0 serves the ring interrupts, DM1 the remapper

# Thread plan (R, C, W) = (GATHER_THREADS, 0, 0): the harts that issue this core's reads. The single source
# of the plan: the kernel's thread count and its dealing stride (CT `gather_threads`) both come from it.
GATHER_THREADS = 2

# Users per core: pinned to 1 by the output placement (rule R5: user u's shard is on core u).
USERS_PER_CORE = 1

# Heads (sections) per block, i.e. per drain: None is the whole user (all sections, one barrier per hart).
HEADS_PER_BLOCK = None

# Tile geometry (a mechanism cap: validate() refuses any other tile). The single host source of these numbers;
# the op module imports them from here.
TILE = 32  # rows and columns of a tile
FACE = 16  # rows and columns of a face
FACE_ROWS_PER_TILE = (TILE // FACE) ** 2 * FACE  # 4 faces x 16 rows = 64: a face row is page / 64 bytes


def validate_thread_plan(gather_threads):
    if not (1 <= gather_threads <= FREE_DM_HARTS):
        raise ValueError(f"gather thread plan R = {gather_threads}: need 1 <= R <= {FREE_DM_HARTS} (W = 0)")


def create_program_artifacts(
    input_tensor, q, k, v, *, cores, num_heads, num_kv_heads, head_dim,
    gather_threads=None, heads_per_block=None,
):
    """(ProgramSpec, ProgramRunArgs, tensor indices) of one dispatch that gathers every user's heads.

    `gather_threads` / `heads_per_block` default to the module constants GATHER_THREADS / HEADS_PER_BLOCK
    (read at call time, so a test can turn the knob on the module).
    """
    if gather_threads is None:
        gather_threads = GATHER_THREADS
    if heads_per_block is None:
        heads_per_block = HEADS_PER_BLOCK
    validate_thread_plan(gather_threads)

    sections = num_heads + 2 * num_kv_heads
    if heads_per_block is None:
        heads_per_block = sections  # the whole per-core assignment: one block
    if not (1 <= heads_per_block <= sections):
        raise ValueError(f"heads_per_block = {heads_per_block}: need 1 <= heads_per_block <= {sections}")

    head_dim_tiles = head_dim // TILE
    # The kernel derives the row's width in tiles as sections * head_dim_tiles (one source); check it here.
    assert -(-int(input_tensor.shape[3]) // TILE) == sections * head_dim_tiles

    in_page_bytes = input_tensor.buffer_page_size()
    face_row_bytes = in_page_bytes // FACE_ROWS_PER_TILE
    out_page_bytes = q.buffer_aligned_page_size()
    assert k.buffer_aligned_page_size() == out_page_bytes and v.buffer_aligned_page_size() == out_page_bytes
    # Every run offset (source and destination) is a multiple of face_row_bytes: aligned for an L1 source.
    if face_row_bytes % ttnn.get_l1_alignment() != 0:
        raise NotImplementedError(f"face row of {face_row_bytes} B is not L1-aligned ({ttnn.get_l1_alignment()} B)")

    # users -> cores: user u on the u-th core of the shard grid (R5), USERS_PER_CORE each.
    core_list = ttnn.corerange_to_cores(cores, None, True)
    assert USERS_PER_CORE == 1
    user_tile_row, user_row = {}, {}
    for u, core in enumerate(core_list):
        user_tile_row[core], user_row[core] = divmod(u, TILE)

    gather = E.KernelSpec(
        unique_id="gather",
        source=str(KERNEL_DIR / "nlp_create_qkv_heads_decode_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=gather_threads,
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name=name, accessor_name=name) for name in ("in", "q", "k", "v")
        ],
        compile_time_args={
            "gather_threads": gather_threads,
            "num_q_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "heads_per_block": heads_per_block,
            "head_dim_tiles": head_dim_tiles,
            "face_row_bytes": face_row_bytes,
            "out_page_bytes": out_page_bytes,
        },
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["user_tile_row", "user_row"]),
    )
    tensor_parameters = [
        E.TensorParameter(unique_id=name, spec=t.spec)
        for name, t in (("in", input_tensor), ("q", q), ("k", k), ("v", v))
    ]
    spec = E.ProgramSpec(
        name="nlp_create_qkv_heads_decode",
        kernels=[gather],
        dataflow_buffers=[],
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["gather"], target_nodes=cores)],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(
                kernel="gather",
                runtime_arg_values={"user_tile_row": user_tile_row, "user_row": user_row},
            )
        ]
    )
    return spec, run_args, {"in": 0, "q": 1, "k": 2, "v": 3}
