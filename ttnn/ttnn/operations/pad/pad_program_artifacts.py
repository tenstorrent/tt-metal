# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for pad: one `ttnn.ProgramSpec`, two data-movement kernels, no compute kernel.

The op re-lays rows and computes nothing (op_design.md, Blocking Model). Output rows are of two kinds:

    data row  (D = N*C*H of them): [bw x value][input row][aw x value]; the image of exactly one input row
    pad row   (P = O - D of them): W_o x value

Two independent lists, each split over the full grid with `split_work_to_cores(row_wise=True)`:

    reader   builds one pad-row template per writer thread (iff the core has pad rows), then assembles the
             core's data rows in `dfb_out_rows`: one NoC read per input row landing at byte offset bw*e of its
             entry, the head/tail margins filled by stores on the ring's first lap. Explicit sync on both rings.
    writer   writes the core's pad rows from the template (tagged writes, one barrier per
             `pad_rows_per_barrier`), then each data row with one implicit-sync write per entry.

Block knobs (single source, all computed here): `block_rows` (data rows per reader reserve/push), `ring_blocks`
(blocks in flight, from BUFFER_DEPTH), `pad_rows_per_barrier`; the thread plan PAD_THREADS.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# Thread plan (reader threads, writer threads); no compute kernel (C = 0). The op is bound by data movement,
# so one thread of each kind is the target page's plan. Kernels deal by thread id from these counts.
PAD_THREADS = (1, 1)
FREE_DM_HARTS = 6  # of 8 per cluster: DM0 serves the ring interrupts, DM1 the remapper (target page)

BUFFER_DEPTH = 2  # blocks in flight in the rows_streamed regime
RING_ENTRY_CAP = 255  # implicit-sync txn descriptor: num_entries / num_txn_ids <= 255 (dataflow_buffer.cpp)
DFB_RING_MAX_BYTES = 1_048_560  # one ring at most 1 MiB - 16 B (target page)
L1_RESERVE_BYTES = 0
# Quasar's NoC overlay cuts every transfer into packets of at most this many bytes (NOC_OVERLAY_MAX_BYTES_IN_PACKET,
# tt_metal/hw/inc/internal/tt-2xx/quasar/noc_cmd_buf_common.h:74).
# Wider entries are drained by an explicit-sync writer instead.
NOC_PACKET_BYTES = 8 * 1024

DFB_OUT_ROWS = "out_rows"
DFB_PAD_ROW = "pad_row"
TP_INPUT = "input"
TP_OUTPUT = "output"


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def fold_to_rank4(shape, padding):
    """Shape and pairs folded to rank 4 by prepending 1s and (0, 0) pairs."""
    lead = 4 - len(shape)
    if lead < 0:
        raise NotImplementedError(f"pad: rank {len(shape)} > 4 is not supported")
    return (1,) * lead + tuple(shape), ((0, 0),) * lead + tuple(padding)


def output_shape(shape, padding):
    return [before + size + after for size, (before, after) in zip(shape, padding)]


def pad_bits(value: float, dtype) -> int:
    """`value` in the element's own encoding: bfloat16 = the top 16 bits of the RNE-rounded value; float32 = IEEE."""
    if dtype == ttnn.bfloat16:
        return int(torch.tensor([value], dtype=torch.float32).to(torch.bfloat16).view(torch.int16).item()) & 0xFFFF
    if dtype == ttnn.float32:
        return int(torch.tensor([value], dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF
    raise NotImplementedError(f"pad: no element encoding for {dtype}")


def elem_bytes_of(dtype) -> int:
    if dtype == ttnn.bfloat16:
        return 2
    if dtype == ttnn.float32:
        return 4
    raise NotImplementedError(f"pad: no element size for {dtype}")


def validate_thread_plan(threads, explicit_out_rows: bool):
    r, w = threads
    if not (r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and max(r, w) % min(r, w) == 0):
        raise ValueError(f"pad: thread plan (R, W) = {threads}: need R + W <= {FREE_DM_HARTS}, max % min == 0")
    if explicit_out_rows and r > w:
        # An explicit consumer rotates its tile counters per pop, i.e. per whole reader op: it needs W >= R.
        raise ValueError(f"pad: thread plan (R, W) = {threads}: rows wider than {NOC_PACKET_BYTES} B need W >= R")


def split_rows(num_rows: int, grid_size):
    """{core: (start, count)} over the cores that get rows, row-major; empty for no rows."""
    if num_rows == 0:
        return {}
    _, _, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid_size, num_rows, row_wise=True)
    out, start = {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            out[(core.x, core.y)] = (start, per_core)
            start += per_core
    assert start == num_rows
    return out


def pad_row_to_output_row(q: int, in_dims, pairs) -> int:
    """The output page (row-major over the output page grid) of the q-th pad page, pads enumerated in output order.

    `in_dims` = the input page grid (any rank; (N, C, Hp, Tp) here, page_geometry), `pairs` their (before, after). A recursive count, O(rank): at dim d the output range
    of one box is a before segment (b_d sub-boxes, all pad), a data segment (I_d sub-boxes, each holding
    `pads_in_sub(d)` pads), an after segment (a_d sub-boxes, all pad).
    """
    out_dims = [b + i + a for i, (b, a) in zip(in_dims, pairs)]

    def rows_in_sub(d):  # output rows in one index of dim d
        r = 1
        for k in range(d + 1, len(in_dims)):
            r *= out_dims[k]
        return r

    def data_in_sub(d):  # data rows in one data index of dim d
        r = 1
        for k in range(d + 1, len(in_dims)):
            r *= in_dims[k]
        return r

    row, d = 0, 0
    while True:
        b, a = pairs[d]
        size = in_dims[d]
        r_sub = rows_in_sub(d)
        p_sub = r_sub - data_in_sub(d)
        if q < b * r_sub:
            return row + q
        q -= b * r_sub
        if p_sub > 0 and q < size * p_sub:
            idx, q = divmod(q, p_sub)
            row += (b + idx) * r_sub
            d += 1
            continue
        q -= size * p_sub
        return row + (b + size) * r_sub + q


def page_geometry(inp, shape4, pairs4):
    """The page grid of the input and the output, and how an input page becomes an output page.

    Both layouts are a 4-dim grid of pages (N, C, Hp, Tp) -> (N_o, C_o, Hp_o, Tp_o), `before_pages` per dim:
      row-major  a page is one row: Hp = H, Tp = 1. The reader lands the input row at head_bytes of its entry
                 and the entry's head/tail margins hold `value`.
      tile       a page is one tile: Hp = ceil(H / tile_h), Tp = ceil(W / tile_w). With elements added after the
                 data in H and W, an input element keeps its tile and its place in the tile, so an input tile
                 is copied whole; in the tiles of the last tile row / column (h_last < tile_h, w_last < tile_w
                 valid rows / columns) the reader overwrites the input's tile padding with `value`, because in
                 the output those places are added elements. Output tiles past the input's are all `value`.
    """
    (N, C, H, Wd), ((bn, an), (bc, ac), (bh, ah), (bw, aw)) = shape4, pairs4
    H_o, W_o = bh + H + ah, bw + Wd + aw
    e = elem_bytes_of(inp.dtype)
    if inp.layout == ttnn.TILE_LAYOUT:
        tile = inp.get_tile()
        (th, tw), (fh, fw) = tuple(tile.tile_shape), tuple(tile.face_shape)
        if bh or bw:
            # An element-granular shift across tile boundaries: not a page copy. Outside SUPPORTED (padding_side).
            raise NotImplementedError("pad: tile layout with elements added before the data in H or W")
        Hp, Tp = -(-H // th), -(-Wd // tw)
        Hp_o, Tp_o = -(-H_o // th), -(-W_o // tw)
        tile_bytes = th * tw * e
        assert tile_bytes == inp.buffer_page_size(), (tile_bytes, inp.buffer_page_size())
        return {"in_pages": (N, C, Hp, Tp), "out_pages": (bn + N + an, bc + C + ac, Hp_o, Tp_o),
                "before_pages": (bn, bc, 0, 0), "in_page_bytes": tile_bytes, "out_page_bytes": tile_bytes,
                "head_bytes": 0, "tail_bytes": 0, "h_last": H - (Hp - 1) * th, "w_last": Wd - (Tp - 1) * tw,
                "tile": (th, tw, fh, fw)}
    return {"in_pages": (N, C, H, 1), "out_pages": (bn + N + an, bc + C + ac, H_o, 1),
            "before_pages": (bn, bc, bh, 0), "in_page_bytes": Wd * e, "out_page_bytes": W_o * e,
            "head_bytes": bw * e, "tail_bytes": aw * e, "h_last": 0, "w_last": 0, "tile": None}


def choose_blocking(max_core_data_rows: int, threads, entry_bytes: int, *, block_rows_override=None):
    """(block_rows, ring_blocks) per the design's regime selection function (rows_resident / rows_streamed)."""
    R, W = threads
    T = max(R, W)
    budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES
    template_bytes = T * entry_bytes
    ring_byte_cap = min(DFB_RING_MAX_BYTES, budget - template_bytes)
    if entry_bytes > ring_byte_cap or 2 * entry_bytes > budget:
        raise NotImplementedError(f"pad: a {entry_bytes} B row does not fit one ring entry (row_chunked regime)")
    rows_fit = min(RING_ENTRY_CAP, ring_byte_cap // entry_bytes) // T
    # Rows per op in rows_resident: one op per (reader thread, writer thread) tile counter.
    per_thread = -(-max_core_data_rows // T)
    if block_rows_override is not None:
        block_rows, ring_blocks = int(block_rows_override), BUFFER_DEPTH
        if not (1 <= block_rows and T * ring_blocks * block_rows <= min(RING_ENTRY_CAP, ring_byte_cap // entry_bytes)):
            raise ValueError(f"pad: block_rows_override={block_rows_override} does not fit the ring")
    elif per_thread <= rows_fit:
        block_rows, ring_blocks = max(1, per_thread), 1  # rows_resident
    else:
        block_rows, ring_blocks = max(1, rows_fit // BUFFER_DEPTH), BUFFER_DEPTH  # rows_streamed
    return block_rows, ring_blocks


def create_program_artifacts(inp, out, shape4, pairs4, value, *, threads=PAD_THREADS, block_rows_override=None,
                             pad_rows_per_barrier=None):
    """(ProgramSpec, ProgramRunArgs, tensor indices) for io_tensors [inp, out].

    Probes (tests): `threads` = (R, W), `block_rows_override` forces rows_streamed with that block,
    `pad_rows_per_barrier` overrides the writer's pad-row batch (default: the core's whole pad assignment).
    """
    R, W = threads
    T = max(R, W)
    (N, C, H, Wd), ((bn, an), (bc, ac), (bh, ah), (bw, aw)) = shape4, pairs4
    N_o, C_o, H_o, W_o = bn + N + an, bc + C + ac, bh + H + ah, bw + Wd + aw
    e = elem_bytes_of(inp.dtype)
    g = page_geometry(inp, shape4, pairs4)
    in_row_bytes, out_row_bytes = g["in_page_bytes"], g["out_page_bytes"]
    Hp, Tp = g["in_pages"][2:]  # page grid of the input: (N, C, Hp, Tp)
    Hp_o, Tp_o = g["out_pages"][2:]
    bhp, btp = g["before_pages"][2:]
    entry_bytes = align_up(out_row_bytes, ttnn.get_l1_alignment())
    # The implicit write sends a whole entry: it must stay inside its page's aligned slot.
    assert entry_bytes <= out.buffer_aligned_page_size(), (entry_bytes, out.buffer_aligned_page_size())
    explicit_out_rows = entry_bytes > NOC_PACKET_BYTES  # one implicit write per entry must be one packet
    validate_thread_plan(threads, explicit_out_rows)

    D = N * C * Hp * Tp
    P = N_o * C_o * Hp_o * Tp_o - D
    grid = inp.device().compute_with_storage_grid_size()
    data_split = split_rows(D, grid)
    pad_split = split_rows(P, grid)
    max_core_data_rows = max((c for _, c in data_split.values()), default=0)
    block_rows, ring_blocks = choose_blocking(max_core_data_rows, threads, entry_bytes,
                                              block_rows_override=block_rows_override)

    cores = sorted(set(data_split) | set(pad_split), key=lambda xy: (xy[1], xy[0]))
    core_objs = [ttnn.CoreCoord(x, y) for x, y in cores]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_objs])
    bits = pad_bits(value, inp.dtype)

    def per_core(fn):
        return {c: fn(xy) for c, xy in zip(core_objs, cores)}

    data_start = per_core(lambda xy: data_split.get(xy, (0, 0))[0])
    data_count = per_core(lambda xy: data_split.get(xy, (0, 0))[1])
    pad_count = per_core(lambda xy: pad_split.get(xy, (0, 0))[1])
    pad_start_out_row = per_core(
        lambda xy: pad_row_to_output_row(pad_split[xy][0], (N, C, Hp, Tp),
                                         ((bn, an), (bc, ac), (bhp, Hp_o - Hp - bhp), (btp, Tp_o - Tp - btp)))
        if xy in pad_split else 0
    )
    batch = per_core(lambda xy: max(1, pad_split.get(xy, (0, 1))[1] if pad_rows_per_barrier is None
                                    else int(pad_rows_per_barrier)))
    same = lambda v: per_core(lambda xy: v)  # noqa: E731

    dfbs = [
        E.DataflowBufferSpec(unique_id=DFB_OUT_ROWS, entry_size=entry_bytes,
                             num_entries=T * ring_blocks * block_rows, data_format_metadata=inp.dtype),
        # One template per writer thread; a STRIDED ring needs num_entries % max(R, W) == 0.
        E.DataflowBufferSpec(unique_id=DFB_PAD_ROW, entry_size=entry_bytes, num_entries=T,
                             data_format_metadata=inp.dtype),
    ]
    tensor_parameters = [E.TensorParameter(unique_id=TP_INPUT, spec=inp.spec),
                         E.TensorParameter(unique_id=TP_OUTPUT, spec=out.spec)]
    plan = {"reader_threads": R, "writer_threads": W, "elem_bytes": e}
    tile_ct = {"tile_layout": int(g["tile"] is not None),
               **{k: v for k, v in zip(("tile_h", "tile_w", "face_h", "face_w"), g["tile"] or (1, 1, 1, 1))}}

    reader_rt = ["data_start", "data_count", "pad_count", "block_rows", "ring_blocks", "in_row_bytes",
                 "out_row_bytes", "head_bytes", "tail_bytes", "pad_bits", "Hp", "Tp", "h_last", "w_last"]
    writer_rt = ["data_start", "data_count", "pad_count", "pad_start_out_row", "pad_rows_per_barrier",
                 "block_rows", "out_row_bytes", "N", "C", "H", "C_o", "H_o", "bn", "bc", "bh", "N_o",
                 "T", "T_o", "bt"]

    reader_hw = E.create_reader_datamovement_config()
    reader_hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(
        disable_dfb_implicit_sync_for=[DFB_OUT_ROWS, DFB_PAD_ROW])
    writer_hw = E.create_writer_datamovement_config()
    writer_hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(
        disable_dfb_implicit_sync_for=[DFB_PAD_ROW] + ([DFB_OUT_ROWS] if explicit_out_rows else []))

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "pad_reader.cpp"),
        hw_config=reader_hw,
        num_threads=R,
        dfb_bindings=[E.producer_of(DFB_OUT_ROWS, "out_rows"), E.producer_of(DFB_PAD_ROW, "pad_row")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=TP_INPUT, accessor_name="input")],
        compile_time_args={**plan, **tile_ct},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=reader_rt),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "pad_writer.cpp"),
        hw_config=writer_hw,
        num_threads=W,
        dfb_bindings=[E.consumer_of(DFB_OUT_ROWS, "out_rows"), E.consumer_of(DFB_PAD_ROW, "pad_row")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=TP_OUTPUT, accessor_name="output")],
        compile_time_args={**plan, "explicit_out_rows": int(explicit_out_rows)},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=writer_rt),
    )
    spec = E.ProgramSpec(
        name="pad",
        kernels=[reader, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "writer"], target_nodes=core_set)],
    )
    reader_args = {
        "data_start": data_start, "data_count": data_count, "pad_count": pad_count,
        "block_rows": same(block_rows), "ring_blocks": same(ring_blocks), "in_row_bytes": same(in_row_bytes),
        "out_row_bytes": same(out_row_bytes), "head_bytes": same(g["head_bytes"]),
        "tail_bytes": same(g["tail_bytes"]), "pad_bits": same(bits),
        "Hp": same(Hp), "Tp": same(Tp), "h_last": same(g["h_last"]), "w_last": same(g["w_last"]),
    }
    writer_args = {
        "data_start": data_start, "data_count": data_count, "pad_count": pad_count,
        "pad_start_out_row": pad_start_out_row, "pad_rows_per_barrier": batch,
        "block_rows": same(block_rows),
        "out_row_bytes": same(out_row_bytes), "N": same(N), "C": same(C), "H": same(Hp), "C_o": same(C_o),
        "H_o": same(Hp_o), "bn": same(bn), "bc": same(bc), "bh": same(bhp), "N_o": same(N_o),
        "T": same(Tp), "T_o": same(Tp_o), "bt": same(btp),
    }
    run_args = E.ProgramRunArgs(kernel_run_args=[
        E.KernelRunArgs(kernel="reader", runtime_arg_values=reader_args),
        E.KernelRunArgs(kernel="writer", runtime_arg_values=writer_args),
    ])
    return spec, run_args, {TP_INPUT: 0, TP_OUTPUT: 1}
