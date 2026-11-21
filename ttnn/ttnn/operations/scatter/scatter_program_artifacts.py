# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for scatter (regime `resident_line_block`, op_design.md).

One data-movement kernel, `scatter`, runs on SCATTER_THREADS user DM harts of every engaged cluster; there is
no compute kernel (the op only moves bytes, R1) and no dataflow buffer. Three scratchpads per cluster:

    scatter_out_column   the block's output line, resident: every output tile-row of the block's tile-columns
    scatter_src_rows     one chunk of `src` tile-rows for the block's tile-columns
    scatter_index_rows   one chunk of `index` row pieces (the block's columns of each source row)

Work split: units are (image, face-column) pairs, a face-column being 16 columns of the tensor along its whole
line (`dim` -2). `ttnn.split_work_to_cores` deals contiguous unit ranges to the clusters; a cluster cuts its
range into blocks of at most `block_width_faces` units (never across an image), and streams the source line
through the stages `block_src_tile_rows` tile-rows at a time. Inside a cluster the threads deal the transfers
round-robin and the source rows of each chunk in contiguous ranges.

Every blocking knob comes from `choose_blocking` (or a caller's override for tests); every scratchpad size,
loop trip count and kernel argument derives from those knobs and the tensor shapes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# --- Host constants (one source each) -------------------------------------------------------------------------
# Thread plan (R, C, W) = (6, 0, 6): the one DM kernel's harts read, scatter and write. All 6 user DM harts.
SCATTER_THREADS = 6
FREE_DM_HARTS = 6  # QUASAR_USER_DM_CORES_PER_NODE (target page cluster model); Python has no query for it
# Depth of the src/index stages. 1: a chunk's reads are not overlapped with the previous chunk's scatter. It sizes
# the stage scratchpads here and reaches the kernel as a CT arg, which refuses any other value until the ping-pong
# (two stage slots, trids 1/2) is built: a deeper stage would otherwise be allocated and never used.
SRC_STAGE_DEPTH = 1
# L1 kept back from the scratchpads' budget.
L1_RESERVE_BYTES = 64 * 1024
# DM cache line: each scratchpad region starts on a line of its own, so an invalidate of one region never
# discards another region's dirty lines (Quasar DM caches, risc_common.h).
CACHE_LINE_BYTES = 64

TILE_DIM = 32
FACE_DIM = 16
FACES_PER_TILE_ROW = TILE_DIM // FACE_DIM  # face-columns per tile-column
INDEX_ELEM_BYTES = 4  # INT32


@dataclass(frozen=True)
class Geometry:
    """Shape facts of one call (dim -2), all per image."""

    images: int  # product of the dims before -2 (1 at rank 2)
    out_rows: int  # H: real line length of input / output
    src_rows: int  # Hs: real source rows
    width: int  # W: real columns
    tile_bytes: int
    index_page_bytes: int

    @property
    def tensor_out_tile_rows(self):
        return math.ceil(self.out_rows / TILE_DIM)

    @property
    def tensor_src_tile_rows(self):
        return math.ceil(self.src_rows / TILE_DIM)

    @property
    def tensor_tile_cols(self):
        return math.ceil(self.width / TILE_DIM)

    @property
    def tensor_width_faces(self):
        return self.tensor_tile_cols * FACES_PER_TILE_ROW

    @property
    def num_units(self):
        return self.images * self.tensor_width_faces


def geometry(input, index) -> Geometry:
    shape, index_shape = tuple(input.shape), tuple(index.shape)
    return Geometry(
        images=math.prod(shape[:-2]),
        out_rows=shape[-2],
        src_rows=index_shape[-2],
        width=shape[-1],
        tile_bytes=input.buffer_page_size(),
        index_page_bytes=index.buffer_page_size(),
    )


def tile_cols_touched(width_faces: int) -> int:
    """Tile-columns a range of `width_faces` face-columns can touch (worst case: it starts on an odd face)."""
    return width_faces // FACES_PER_TILE_ROW + 1


def scratchpad_sizes(geo: Geometry, block_width_faces: int, block_src_tile_rows: int) -> dict:
    """Bytes of each scratchpad (one line of slack each for the cache-line start)."""
    tc = tile_cols_touched(block_width_faces)
    return {
        "scatter_out_column": geo.tensor_out_tile_rows * tc * geo.tile_bytes + CACHE_LINE_BYTES,
        "scatter_src_rows": block_src_tile_rows * tc * geo.tile_bytes * SRC_STAGE_DEPTH + CACHE_LINE_BYTES,
        "scatter_index_rows": block_src_tile_rows * TILE_DIM * block_width_faces * FACE_DIM * INDEX_ELEM_BYTES
        * SRC_STAGE_DEPTH + CACHE_LINE_BYTES,
    }


def footprint(geo: Geometry, block_width_faces: int, block_src_tile_rows: int) -> int:
    return sum(scratchpad_sizes(geo, block_width_faces, block_src_tile_rows).values())


def l1_budget() -> int:
    return ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES


def resident_line_fits(input, index, budget: int | None = None) -> bool:
    """Regime predicate of `resident_line_block`: the smallest block (one face-column, one source tile-row)
    with the whole output line resident fits the scratchpad budget."""
    budget = l1_budget() if budget is None else budget
    return footprint(geometry(input, index), 1, 1) <= budget


def choose_blocking(geo: Geometry, max_cluster_faces: int, budget: int) -> tuple[int, int]:
    """(block_width_faces, block_src_tile_rows): the coarsest block that fits L1 (op_design.md, choose_blocking).

    Shrinking either adds blocks or chunks but never DRAM bytes, so the source chunk shrinks first (stages only),
    the width second (it also shrinks the slab)."""
    wf0, sr0 = max_cluster_faces, max(1, geo.tensor_src_tile_rows)
    fits = lambda wf, sr: footprint(geo, wf, sr) <= budget  # noqa: E731
    if fits(wf0, sr0):
        return wf0, sr0
    if fits(wf0, 1):
        return wf0, max(sr for sr in range(1, sr0 + 1) if fits(wf0, sr))
    if fits(1, 1):
        wf = max(wf for wf in range(1, wf0 + 1) if fits(wf, 1))
        return wf, max(sr for sr in range(1, sr0 + 1) if fits(wf, sr))
    raise NotImplementedError(
        "scatter: the output line does not fit L1 (regime banded_output_line is not built)"
    )


def split_units(num_units, grid_size):
    """(core set, {core: units}, {core: first unit}) over the clusters that get work, row-major."""
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid_size, num_units, row_wise=True
    )
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = per_core, start
            start += per_core
    return all_cores, counts, starts


def create_program_artifacts(
    input,
    index,
    src,
    output,
    *,
    dim=-2,
    threads=SCATTER_THREADS,
    block_width_faces=None,
    block_src_tile_rows=None,
    budget=None,
    probe_skip_thread=None,
):
    """The ProgramSpec, its run args and the tensor indices of one scatter call.

    Knob overrides (`block_width_faces`, `block_src_tile_rows`, `budget`, `threads`) and the probe
    `probe_skip_thread` (thread t skips its scatter rows, so a test sees which thread moved which source row)
    are for tests; `scatter` passes none.
    """
    if dim != -2:
        raise NotImplementedError(f"scatter: dim={dim} has no regime built (only -2, resident_line_block)")
    if not 1 <= threads <= FREE_DM_HARTS:
        raise ValueError(f"scatter: {threads} threads, need 1..{FREE_DM_HARTS} DM harts")
    geo = geometry(input, index)
    # The kernel addresses the index as one page per source row of `width` INT32 positions (page id = row).
    if geo.index_page_bytes != geo.width * INDEX_ELEM_BYTES:
        raise ValueError(
            f"scatter: index page of {geo.index_page_bytes} B, expected one row of {geo.width} INT32 positions"
        )
    core_set, counts, starts = split_units(geo.num_units, input.device().compute_with_storage_grid_size())
    max_cluster_faces = max(counts.values())
    # A block never spans two images.
    max_cluster_faces = min(max_cluster_faces, geo.tensor_width_faces)
    budget = l1_budget() if budget is None else budget
    wf, sr = choose_blocking(geo, max_cluster_faces, budget)
    if block_width_faces is not None:
        wf = block_width_faces
    if block_src_tile_rows is not None:
        sr = block_src_tile_rows
    if not (1 <= wf and 1 <= sr):
        raise ValueError(f"scatter: block_width_faces={wf}, block_src_tile_rows={sr} must be >= 1")
    sizes = scratchpad_sizes(geo, wf, sr)
    if sum(sizes.values()) > budget:
        raise NotImplementedError(f"scatter: blocking ({wf}, {sr}) needs {sum(sizes.values())} B of L1 > {budget}")

    scratchpads = [E.ScratchpadSpec(unique_id=name, size_per_node=size) for name, size in sizes.items()]
    tensor_parameters = [
        E.TensorParameter(unique_id=name, spec=t.spec)
        for name, t in (("input", input), ("index", index), ("src", src), ("output", output))
    ]
    compile_time_args = {
        "threads": threads,
        "probe_skip_thread": threads if probe_skip_thread is None else int(probe_skip_thread),
        "elem_bytes": geo.tile_bytes // (TILE_DIM * TILE_DIM),
        "tile_bytes": geo.tile_bytes,
        "out_rows": geo.out_rows,
        "src_rows": geo.src_rows,
        "width": geo.width,
        "out_tile_rows": geo.tensor_out_tile_rows,
        "src_tile_rows": geo.tensor_src_tile_rows,
        "tile_cols": geo.tensor_tile_cols,
        "width_faces": geo.tensor_width_faces,
        "block_width_faces": wf,
        "block_src_tile_rows": sr,
        "src_stage_depth": SRC_STAGE_DEPTH,
        "cache_line_bytes": CACHE_LINE_BYTES,
    }
    kernel = E.KernelSpec(
        unique_id="scatter",
        source=str(KERNEL_DIR / "scatter.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=threads,
        scratchpad_bindings=[
            E.KernelSpec.ScratchpadBinding(scratchpad_spec_name="scatter_out_column", accessor_name="out_column"),
            E.KernelSpec.ScratchpadBinding(scratchpad_spec_name="scatter_src_rows", accessor_name="src_rows"),
            E.KernelSpec.ScratchpadBinding(scratchpad_spec_name="scatter_index_rows", accessor_name="index_rows"),
        ],
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name=name, accessor_name=name)
            for name in ("input", "index", "src", "output")
        ],
        compile_time_args=compile_time_args,
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["unit_start", "num_units"]),
    )
    spec = E.ProgramSpec(
        name="scatter",
        kernels=[kernel],
        scratchpads=scratchpads,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["scatter"], target_nodes=core_set)],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="scatter", runtime_arg_values={"unit_start": starts, "num_units": counts}),
        ]
    )
    return spec, run_args, {"input": 0, "index": 1, "src": 2, "output": 3}
