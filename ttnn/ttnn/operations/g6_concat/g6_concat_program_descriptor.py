# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_concat ProgramDescriptor: plan derivation, work split, CBs and kernel arguments.

One program per call: a reader (NCRISC / NOC0) and a writer (BRISC / NOC1) on every core of the
split, one double-buffered reader->writer CB, no compute kernel. The work split is contiguous
ranges of the row-major output page index; each core walks its range in blocks of `block_pages`.
"""

from __future__ import annotations

import bisect
import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
PAGE_READER = str(KERNEL_DIR / "g6_concat_page_reader.cpp")
GATHER_READER = str(KERNEL_DIR / "g6_concat_gather_reader.cpp")
WRITER = str(KERNEL_DIR / "g6_concat_writer.cpp")

# ---- Blocking-model knobs (single source of truth) ----------------------------------------------
# Coarsest block per core under this L1 cap (bf16: 64 pages). See op_design.md, Blocking Model.
BLOCK_BYTES_CAP = 128 * 1024
# reader->writer CB depth in blocks: the reader assembles block i+1 while the writer stores block i.
OUTPUT_PAGES_DEPTH = 2
# A page whose valid rows v satisfy v <= TRIM_ROWS_MAX moves only rows [0, v).
TRIM_ROWS_MAX = 16
# Cap on cores; None = the whole compute grid.
MAX_CORES = None

# ---- Mechanism caps ------------------------------------------------------------------------------
# Two common runtime words per input must fit the 341-word per-kernel runtime-arg budget.
K_MAX = 64
# L1 the model leaves to the op on every core (BB-05).
L1_OP_BUDGET = 1347776

TILE = 32
FACE = 16

# ---- CB slots ------------------------------------------------------------------------------------
CB_OUTPUT_PAGES = 0
CB_GATHER_STAGING = 1


def ct(x):
    return (x + TILE - 1) // TILE


def canonical_dim(dim, rank):
    return dim + rank if dim < 0 else dim


def select_regime(shapes, d, layout):
    r = len(shapes[0])
    if layout == ttnn.TILE_LAYOUT and d == r - 2:
        offset = 0
        for s in shapes[:-1]:
            offset += s[d]
            if offset % TILE != 0:
                return "row_gather"
    return "page_copy"


def trim_rows_max_for(dtype):
    # bfloat8_b keeps its exponents in a header, so rows cannot be cut out of a page.
    return 0 if dtype == ttnn.bfloat8_b else TRIM_ROWS_MAX


def _core_order(core_group_1, core_group_2):
    cores = list(ttnn.corerange_to_cores(core_group_1, None, True))
    if core_group_2.num_cores() > 0:
        cores += list(ttnn.corerange_to_cores(core_group_2, None, True))
    return cores


def build_plan(tensors, d, output_shape, output_memory_config, device):
    shapes = [list(t.shape) for t in tensors]
    r = len(shapes[0])
    K = len(shapes)
    dtype = tensors[0].dtype
    layout = tensors[0].layout
    page_bytes = ttnn.tile_size(dtype)
    rb = 0 if dtype == ttnn.bfloat8_b else (page_bytes // (TILE * TILE)) * FACE
    face_bytes = rb * FACE

    H, W = output_shape[r - 2], output_shape[r - 1]
    out_imgs = math.prod(output_shape[: r - 2])
    out_Ht, out_Wt = ct(H), ct(W)
    P_out = out_imgs * out_Ht * out_Wt
    v_last = H - TILE * (out_Ht - 1)

    regime = select_regime(shapes, d, layout)

    grid = device.compute_with_storage_grid_size()
    if MAX_CORES is not None and MAX_CORES < grid.x * grid.y:
        core_grid = ttnn.num_cores_to_corerangeset(MAX_CORES, grid, True)
        split = ttnn.split_work_to_cores(core_grid, P_out, True)
    else:
        split = ttnn.split_work_to_cores(grid, P_out, True)
    num_cores, all_cores, core_group_1, core_group_2, n1, n2 = split
    cores = _core_order(core_group_1, core_group_2)
    assert len(cores) == num_cores
    g1 = core_group_1.num_cores()
    counts = [n1 if i < g1 else n2 for i in range(num_cores)]
    starts = [0] * num_cores
    for i in range(1, num_cores):
        starts[i] = starts[i - 1] + counts[i - 1]
    assert starts[-1] + counts[-1] == P_out

    block_pages_cap = max(1, BLOCK_BYTES_CAP // page_bytes)
    block_pages = min(n1, block_pages_cap)

    # Staging only where a face-row shift can break the DRAM read alignment (bf16 on Blackhole).
    a_dram_read = ttnn.get_dram_alignment()
    use_staging = regime == "row_gather" and rb % a_dram_read != 0

    cb_output_bytes = OUTPUT_PAGES_DEPTH * block_pages * page_bytes
    cb_staging_bytes = (block_pages + 1) * page_bytes if use_staging else 0
    out_l1_per_bank = 0
    if output_memory_config.buffer_type == ttnn.BufferType.L1:
        num_l1_banks = device.num_banks(ttnn.BufferType.L1)
        out_l1_per_bank = ((P_out + num_l1_banks - 1) // num_l1_banks) * page_bytes
    l1_total = cb_output_bytes + cb_staging_bytes + out_l1_per_bank

    plan = {
        "regime": regime,
        "d": d,
        "rank": r,
        "K": K,
        "shapes": shapes,
        "output_shape": list(output_shape),
        "dtype": dtype,
        "page_bytes": page_bytes,
        "rb": rb,
        "face_bytes": face_bytes,
        "trim_rows_max": trim_rows_max_for(dtype),
        "P_out": P_out,
        "out_imgs": out_imgs,
        "out_Ht": out_Ht,
        "out_Wt": out_Wt,
        "v_last": v_last,
        "num_cores": num_cores,
        "all_cores": all_cores,
        "cores": cores,
        "counts": counts,
        "starts": starts,
        "block_pages": block_pages,
        "depth": OUTPUT_PAGES_DEPTH,
        "use_staging": use_staging,
        "cb_output_bytes": cb_output_bytes,
        "cb_staging_bytes": cb_staging_bytes,
        "l1_total": l1_total,
    }

    if regime == "page_copy":
        if d <= r - 3:
            O = math.prod(shapes[0][:d])
            inner = math.prod(shapes[0][d + 1 : r - 2]) * ct(shapes[0][r - 2]) * ct(shapes[0][r - 1])
            C_k = [s[d] * inner for s in shapes]
        elif d == r - 2:
            O = math.prod(shapes[0][: r - 2])
            C_k = [ct(s[d]) * ct(shapes[0][r - 1]) for s in shapes]
            # Every input but the last is whole tile rows, so input and output pages share v.
            assert all(s[d] % TILE == 0 for s in shapes[:-1])
        else:
            O = math.prod(shapes[0][: r - 2]) * ct(shapes[0][r - 2])
            C_k = [ct(s[d]) for s in shapes]
        C = sum(C_k)
        assert O * C == P_out
        Coff = [0] * K
        for k in range(1, K):
            Coff[k] = Coff[k - 1] + C_k[k - 1]
        cursors = []
        for p in starts:
            o, c = divmod(p, C)
            k = bisect.bisect_right(Coff, c) - 1
            cursors.append((o, k, c - Coff[k]))
        plan.update({"O": O, "C_k": C_k, "cursors": cursors})
    else:
        plan.update({"n_k": [s[d] for s in shapes], "Wt": out_Wt})

    return plan


def _out_cursor(plan, p):
    col = p % plan["out_Wt"]
    R = (p // plan["out_Wt"]) % plan["out_Ht"]
    return col, R


def _reader_common_args(plan, tensors):
    head = [plan["block_pages"], plan["K"], plan["out_Wt"], plan["out_Ht"], plan["v_last"]]
    bases = [t.buffer_address() for t in tensors]
    per_input = plan["C_k"] if plan["regime"] == "page_copy" else plan["n_k"]
    return head + bases + list(per_input)


def _writer_common_args(plan, output):
    return [plan["block_pages"], plan["out_Wt"], plan["out_Ht"], plan["v_last"], output.buffer_address()]


def create_program_descriptor(plan, tensors, output):
    all_cores = plan["all_cores"]
    page_bytes = plan["page_bytes"]
    dtype = plan["dtype"]

    cbs = [
        ttnn.CBDescriptor(
            total_size=plan["cb_output_bytes"],
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_OUTPUT_PAGES, data_format=dtype, page_size=page_bytes)
            ],
        )
    ]
    if plan["use_staging"]:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=plan["cb_staging_bytes"],
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_GATHER_STAGING, data_format=dtype, page_size=page_bytes)
                ],
            )
        )

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    for i, core in enumerate(plan["cores"]):
        p, n = plan["starts"][i], plan["counts"][i]
        col, R = _out_cursor(plan, p)
        if plan["regime"] == "page_copy":
            o, k, cp = plan["cursors"][i]
            reader_rt[core.x][core.y] = [n, o, k, cp, col, R]
        else:
            img = p // (plan["out_Ht"] * plan["out_Wt"])
            reader_rt[core.x][core.y] = [n, img, R, col]
        writer_rt[core.x][core.y] = [n, p, col, R]

    input_accessor = ttnn.TensorAccessorArgs(tensors[0]).get_compile_time_args()
    if plan["regime"] == "page_copy":
        reader_source = PAGE_READER
        reader_ct = [CB_OUTPUT_PAGES, page_bytes, plan["face_bytes"], plan["rb"], plan["trim_rows_max"]]
    else:
        reader_source = GATHER_READER
        reader_ct = [
            CB_OUTPUT_PAGES,
            CB_GATHER_STAGING,
            page_bytes,
            plan["face_bytes"],
            plan["rb"],
            int(plan["use_staging"]),
        ]
    reader_ct += list(input_accessor)

    writer_ct = [CB_OUTPUT_PAGES, page_bytes, plan["face_bytes"], plan["rb"], plan["trim_rows_max"]]
    writer_ct += list(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    reader = ttnn.KernelDescriptor(
        kernel_source=reader_source,
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=_reader_common_args(plan, tensors),
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=WRITER,
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=_writer_common_args(plan, output),
        config=ttnn.WriterConfigDescriptor(),
    )
    return reader, writer, cbs


def refresh_addresses(entry, tensors, output):
    """Cache hit: only the buffer base addresses change."""
    plan = entry["plan"]
    reader, writer = entry["kernels"]
    reader.common_runtime_args = _reader_common_args(plan, tensors)
    writer.common_runtime_args = _writer_common_args(plan, output)
    entry["descriptor"].kernels = [reader, writer]
    return entry["descriptor"]


def build_descriptor(plan, tensors, output, key_hash):
    reader, writer, cbs = create_program_descriptor(plan, tensors, output)
    descriptor = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)
    descriptor.custom_program_hash = key_hash
    return {"plan": plan, "kernels": (reader, writer), "descriptor": descriptor}
