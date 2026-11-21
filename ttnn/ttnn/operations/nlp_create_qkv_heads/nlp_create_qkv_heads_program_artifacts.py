# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for nlp_create_qkv_heads, regime `tile_relocate` (op_design.md, Blocking Model).

With head_dim and S multiples of 32 and k not transposed the op is a tile permutation: output tile (b, h, r, c)
of section sec (q, k or v) is input tile (b, r, (first_head(sec) + h) * hd_t + c), bytes unchanged. The four
tile axes (batch, token row, head, head-dim tile) are contiguous in the input's page order and treated alike,
so they fold into one axis, the input page id, and the block is a contiguous page range:

    split_work_to_cores(grid, num_pages)  -> each cluster a contiguous range of input pages (sizes differ by <= 1)
    reader  (R harts)  reads   pages start+t, start+t+R, ...          into the ring `input_tiles` (implicit sync)
    writer  (W harts)  writes  the same pages, entry w, w+W, ...  to q / k / v at the walked destination page

No compute kernel: nothing is computed (rule R1), and a NEO copy would only add an unpack and a pack per tile.
The writer's walk state (sec, h, b, r, c) of the cluster's first page is decomposed here on the host, so the
device advances it by carries only.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# The cluster model of the target page (.claude/references/targets/quasar.md); Python has no query for them.
FREE_DM_HARTS = 6  # of 8: DM0 serves the ring interrupts, DM1 the remapper
MAX_TXN_IDS = 4  # the runtime slices an implicit-sync ring into up to 4 transaction ids

# ---- Knobs (single source of truth; every dependent quantity derives from these) ----
# Thread plan (R readers, W writers); no compute threads (the op computes nothing). Bound by DRAM: the target
# page's "Choosing the plan" gives a data-movement-bound op one thread of each kind with implicit sync.
THREAD_PLAN = (1, 1)
# Ring depth in tiles before alignment to the implicit-sync slicing: tagged reads in flight ahead of the writer.
RING_DEPTH_TILES = 8
# Blocks per cluster over its page range: 1 = the cluster's whole range is one block (the coarsest; bytes moved
# do not depend on it, and each block repays the walk-state setup and the exit flush).
BLOCKS_PER_CORE = 1


def validate_thread_plan(threads):
    r, w = threads
    if not (r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and (r % w == 0 or w % r == 0)):
        raise ValueError(f"thread plan (R, W) = {threads}: need R, W >= 1, R + W <= {FREE_DM_HARTS}, R | W or W | R")


def ring_entries(threads, depth_tiles=None):
    """Ring capacity: divisible by every transaction-id count the runtime may pick times each side's harts."""
    depth_tiles = RING_DEPTH_TILES if depth_tiles is None else depth_tiles
    quantum = MAX_TXN_IDS * max(threads)
    return -(-depth_tiles // quantum) * quantum


def split_pages(num_pages, grid_size):
    """(core set, {core: pages}, {core: first page}) over the clusters that get work, row-major, cumulative."""
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid_size, num_pages, row_wise=True
    )
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = per_core, start
            start += per_core
    return all_cores, counts, starts


def walk_state(page, *, row_tiles, width_tiles, head_dim_tiles, section_heads):
    """(sec, h, b, r, c) of input page `page`: the writer's walk position."""
    b, rest = divmod(page, row_tiles * width_tiles)
    r, col = divmod(rest, width_tiles)
    head, c = divmod(col, head_dim_tiles)
    sec = 0
    while head >= section_heads[sec]:
        head -= section_heads[sec]
        sec += 1
    return sec, head, b, r, c


def create_program_artifacts(
    input, q, k, v, *, num_heads, num_kv_heads, threads=None, blocks_per_core=None, ring_depth_tiles=None
):
    """`threads`, `blocks_per_core`, `ring_depth_tiles` override the module knobs (tests and measurements)."""
    threads = THREAD_PLAN if threads is None else tuple(threads)
    blocks_per_core = BLOCKS_PER_CORE if blocks_per_core is None else blocks_per_core
    if blocks_per_core < 1:
        raise ValueError(f"blocks_per_core must be >= 1, got {blocks_per_core}")
    validate_thread_plan(threads)
    R, W = threads
    batch, _, tokens, width = tuple(input.shape)
    sections = num_heads + 2 * num_kv_heads
    head_dim = width // sections
    tile_h, tile_w = input.tile.tile_shape
    if head_dim % tile_w != 0:
        raise NotImplementedError("tile_relocate needs head_dim a multiple of the tile width")
    head_dim_tiles = head_dim // tile_w
    row_tiles = -(-tokens // tile_h)  # per (b, 0) image: each image is tile-padded on its own
    width_tiles = sections * head_dim_tiles
    num_pages = input.buffer_num_pages()
    assert num_pages == batch * row_tiles * width_tiles, (num_pages, batch, row_tiles, width_tiles)
    page_bytes = input.buffer_page_size()
    for out in (q, k, v):
        assert out.buffer_page_size() == page_bytes
    section_heads = (num_heads, num_kv_heads, num_kv_heads)

    core_set, counts, starts = split_pages(num_pages, input.device().compute_with_storage_grid_size())

    ring = E.DataflowBufferSpec(
        unique_id="input_tiles",
        entry_size=page_bytes,
        num_entries=ring_entries(threads, ring_depth_tiles),
        data_format_metadata=input.dtype,
    )
    tensor_parameters = [
        E.TensorParameter(unique_id=name, spec=t.spec) for name, t in (("input", input), ("q", q), ("k", k), ("v", v))
    ]
    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "nlp_create_qkv_heads_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=R,
        dfb_bindings=[E.producer_of("input_tiles", "cb_input_tiles")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args={"reader_threads": R, "blocks_per_core": blocks_per_core},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["start_page", "num_pages"]),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "nlp_create_qkv_heads_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        dfb_bindings=[E.consumer_of("input_tiles", "cb_input_tiles")],
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name=name, accessor_name=name) for name in ("q", "k", "v")
        ],
        compile_time_args={
            "writer_threads": W,
            "blocks_per_core": blocks_per_core,
            "row_tiles": row_tiles,
            "head_dim_tiles": head_dim_tiles,
            "q_heads": num_heads,
            "kv_heads": num_kv_heads,
        },
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["num_pages", "start_sec", "start_h", "start_b", "start_r", "start_c"]
        ),
    )
    spec = E.ProgramSpec(
        name="nlp_create_qkv_heads",
        kernels=[reader, writer],
        dataflow_buffers=[ring],
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "writer"], target_nodes=core_set)],
    )

    walk = {
        core: walk_state(
            starts[core],
            row_tiles=row_tiles,
            width_tiles=width_tiles,
            head_dim_tiles=head_dim_tiles,
            section_heads=section_heads,
        )
        for core in counts
    }
    writer_args = {"num_pages": counts}
    for i, name in enumerate(("start_sec", "start_h", "start_b", "start_r", "start_c")):
        writer_args[name] = {core: state[i] for core, state in walk.items()}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values={"start_page": starts, "num_pages": counts}),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=writer_args),
        ]
    )
    return spec, run_args, {"input": 0, "q": 1, "k": 2, "v": 3}
