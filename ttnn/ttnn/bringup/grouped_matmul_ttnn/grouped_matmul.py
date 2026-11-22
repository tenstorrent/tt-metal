# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.grouped_matmul: a batched matmul with one weight per batch (group), y[g] = x[g] @ w[g], laid out one
grid row per group (deepseek_v41_flash_d_p P.81, ATTN_ROOFLINE: the attention's grouped wo_a, 8 groups x [160, 4096] @
[4096, 1024] a chip).

Why: the stock 2D multicast matmul (fuse_batch=False) splits only M over the grid rows and N over the columns and loops
the groups on every core; at M = 160 (5 row tiles) it runs on 5 x 11 = 55 cores, each core 8 x 3 x 128 tile products,
compute-bound (0.175 ms at HiFi2 vs a ~0.1 ms DRAM floor). Here every core holds ONE group:

    grid      G rows (one a group) x GX columns (GX <= the grid width, <= Nt); core (cx, g): all MT row tiles x the
              PN(cx) column tiles from n0(cx) of group g (column_split: as even as possible)
    in0       x[g] streamed in K blocks of BW tiles: the row's column-0 core reads the [MT x BW] block from DRAM and
              multicasts it along its row (READY / VALID semaphores, double-buffered CB)
    in1       each core reads its own [BW x PN] weight block (no weight tile is read twice: every (g, n) on one core)
    compute   per K block and row tile: fp32 DST accumulates the BW products, the packer adds them to Float32 partials
              in L1 (an fp32 sum over the blocks); at the end one copy + pack to the output dtype
    out       y[g] tiles written by each core (NCRISC, NoC 1)

On the 8x4 (G 8, MT 5, Kt 128, Nt 32, 11 x 10 grid): 8 x 11 = 88 cores (<= 110, the PSU matmul cap: agent rule 18),
5 x 3 x 128 tile products a core. Always dispatched as a prepared program (ttnn.bringup.prepare_program / run_prepared,
the three addresses patched; OV-23).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import ttnn

KERNELS = Path(__file__).parent / "kernels"
CB_IN0, CB_IN1, CB_PART, CB_OUT = 0, 1, 2, 16
NUM_CB_SLOTS = 64
L1_BUDGET = 1024 * 1024
_PREPARED_CAP = 64
_PREPARED = OrderedDict()  # key -> (mesh device, PreparedProgram), LRU
_TILE_BYTES = {}


def _tile_bytes(dtype) -> int:
    if dtype not in _TILE_BYTES:
        _TILE_BYTES[dtype] = ttnn.tile_size(dtype)
    return _TILE_BYTES[dtype]


def column_split(nt: int, gx: int) -> list[int]:
    """Output column tiles per grid column, as even as possible (the wider ones first)."""
    base, rem = divmod(nt, gx)
    return [base + (1 if i < rem else 0) for i in range(gx)]


def _grid(device, max_grid=None) -> tuple[int, int]:
    g = device.compute_with_storage_grid_size()
    if max_grid is None:
        return g.x, g.y
    mx, my = (max_grid.x, max_grid.y) if hasattr(max_grid, "x") else tuple(max_grid)
    return max(1, min(g.x, mx)), max(1, min(g.y, my))


def plan(x_shape, w_shape, x_dtype, w_dtype, out_dtype, device, max_grid=None, block_w=None):
    """(G, MT, KT, NT, GX, pns, BW) of a call, or None when the layout does not apply (more groups than grid rows, a
    batch shape other than [1, G], dims not tile multiples, nothing fits L1)."""
    if len(x_shape) != 4 or len(w_shape) != 4 or x_shape[0] != 1 or w_shape[0] != 1:
        return None
    G, M, K = x_shape[1], x_shape[2], x_shape[3]
    if w_shape[1] != G or w_shape[2] != K or M % 32 or K % 32 or w_shape[3] % 32:
        return None
    MT, KT, NT = M // 32, K // 32, w_shape[3] // 32
    gx, gy = _grid(device, max_grid)
    if G > gy:
        return None
    GX = min(gx, NT)
    pns = column_split(NT, GX)
    pn = max(pns)
    a, b, o = _tile_bytes(x_dtype), _tile_bytes(w_dtype), _tile_bytes(out_dtype)
    part = MT * pn * _tile_bytes(ttnn.float32)
    for bw in (block_w,) if block_w else (32, 16, 8, 4, 2, 1):
        if KT % bw:
            continue
        if 2 * MT * bw * a + 2 * bw * pn * b + part + MT * pn * o <= L1_BUDGET:
            return G, MT, KT, NT, GX, pns, bw
    return None


_SUPPORTED = {}


def supported(x_shape, w_shape, x_dtype, w_dtype, out_dtype, device, max_grid=None, block_w=None) -> bool:
    """Whether the call has a layout (``plan``); cached by its arguments (a model asks on every call)."""
    mg = None if max_grid is None else ((max_grid.x, max_grid.y) if hasattr(max_grid, "x") else tuple(max_grid))
    key = (tuple(x_shape), tuple(w_shape), str(x_dtype), str(w_dtype), str(out_dtype), device.id(), mg, block_w)
    hit = _SUPPORTED.get(key)
    if hit is None:
        hit = _SUPPORTED[key] = (
            plan(x_shape, w_shape, x_dtype, w_dtype, out_dtype, device, max_grid, block_w) is not None
        )
    return hit


def program(x, w, out, ckc, max_grid=None, block_w=None, depth=2, rt=1) -> ttnn.ProgramDescriptor:
    dev = x.device()
    p = plan(tuple(x.shape), tuple(w.shape), x.dtype, w.dtype, out.dtype, dev, max_grid, block_w)
    assert p is not None, f"grouped_matmul: no layout for {tuple(x.shape)} x {tuple(w.shape)}"
    G, MT, KT, NT, GX, pns, BW = p
    n0s = [sum(pns[:i]) for i in range(GX)]
    a, b, o = _tile_bytes(x.dtype), _tile_bytes(w.dtype), _tile_bytes(out.dtype)
    pn_max = max(pns)
    allc = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, G - 1))])

    def cb(idx, dtype, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=allc,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)],
        )

    # (the multicast target CB_IN0 has the same size and position on every core of the grid)
    cbs = [
        cb(CB_IN0, x.dtype, a, depth * MT * BW),
        cb(CB_IN1, w.dtype, b, depth * BW * pn_max),
        cb(CB_PART, ttnn.float32, _tile_bytes(ttnn.float32), MT * pn_max),
        cb(CB_OUT, out.dtype, o, MT * pn_max),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=allc, initial_value=0) for i in range(2)]
    phys = {(cx, g): dev.worker_core_from_logical_core(ttnn.CoreCoord(cx, g)) for cx in range(GX) for g in range(G)}
    x_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    w_ct = list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
    y_ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    assert depth >= 2, f"grouped_matmul: depth {depth} (>= 2)"
    fp32 = bool(getattr(ckc, "fp32_dest_acc_en", True))
    rt = max(1, min(rt, MT))
    dst = (4 if fp32 else 8) * (2 if rt * pn_max > (4 if fp32 else 8) else 1)
    assert rt * pn_max <= dst, f"grouped_matmul: rt {rt} x {pn_max} column tiles exceed DEST ({dst} tiles)"
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ckc, "math_fidelity", ttnn.MathFidelity.HiFi2),
        fp32_dest_acc_en=fp32,
        math_approx_mode=bool(getattr(ckc, "math_approx_mode", False)),
        dst_full_sync_en=rt * pn_max > (4 if fp32 else 8),
    )
    if fp32:
        modes = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
        modes[CB_PART] = ttnn.UnpackToDestMode.UnpackToDestFp32  # exact fp32 copies of the partials into DEST
        cfg.unpack_to_dest_mode = modes
    groups = {}
    for cx, pn in enumerate(pns):
        groups.setdefault(pn, []).append(cx)
    kernels = []
    for pn, cols in groups.items():
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, 0), ttnn.CoreCoord(cx, G - 1)) for cx in cols])
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for cx in cols:
            for g in range(G):
                s = phys[(0, g)]
                if GX > 1:
                    lo, hi = phys[(1, g)], phys[(GX - 1, g)]
                    mc = [lo.x, hi.x, lo.y]
                else:
                    mc = [0, 0, 0]
                rd[cx][g] = [g, n0s[cx], 1 if cx == 0 else 0] + mc + [s.x, s.y]
                wr[cx][g] = [g, n0s[cx]]
                cp[cx][g] = []
        kernels += [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "gmm_reader.cpp"),
                core_ranges=crs,
                compile_time_args=[MT, KT, NT, BW, pn, GX, CB_IN0, CB_IN1, a, b, pn_max] + x_ct + w_ct,
                runtime_args=rd,
                common_runtime_args=[x.buffer_address(), w.buffer_address()],
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "gmm_writer.cpp"),
                core_ranges=crs,
                compile_time_args=[MT, NT, pn, CB_OUT, o] + y_ct,
                runtime_args=wr,
                common_runtime_args=[out.buffer_address()],
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "gmm_compute.cpp"),
                core_ranges=crs,
                compile_time_args=[MT, BW, pn, KT // BW, CB_IN0, CB_IN1, CB_PART, CB_OUT, rt, pn_max],
                runtime_args=cp,
                config=cfg,
            ),
        ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)


def _ckc_key(ckc):
    return tuple(str(getattr(ckc, k, None)) for k in ("math_fidelity", "fp32_dest_acc_en", "math_approx_mode"))


def _prepared(x, w, out, ckc, max_grid=None, block_w=None, depth=2, rt=1):
    mesh = x.device()
    mg = None if max_grid is None else ((max_grid.x, max_grid.y) if hasattr(max_grid, "x") else tuple(max_grid))
    key = (
        mesh.id(),
        tuple(x.shape),
        tuple(w.shape),
        str(x.dtype),
        str(w.dtype),
        str(out.dtype),
        str(x.memory_config()),
        str(w.memory_config()),
        str(out.memory_config()),
        mg,
        block_w,
        depth,
        rt,
        _ckc_key(ckc),
    )
    hit = _PREPARED.get(key)
    if hit is not None:
        _PREPARED.move_to_end(key)
        return hit[1]
    desc = program(x, w, out, ckc, max_grid, block_w, depth, rt)
    shape = mesh.shape
    md = ttnn.MeshProgramDescriptor()
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(shape[0] - 1, shape[1] - 1))] = desc
    pp = ttnn.bringup.prepare_program(md, [x, w, out])
    # the three common address slots of each kernel group (x, w: reader; y: writer); anything else matched a frozen value
    want = 3 * (len(desc.kernels) // 3)
    assert pp.patch_count == want, f"grouped_matmul: {pp.patch_count} patched slots, {want} expected"
    if len(_PREPARED) >= _PREPARED_CAP:
        _PREPARED.popitem(last=False)
    _PREPARED[key] = (mesh, pp)
    return pp


def grouped_matmul(
    x: ttnn.Tensor,
    w: ttnn.Tensor,
    *,
    dtype=None,
    compute_kernel_config=None,
    output_tensor: ttnn.Tensor | None = None,
    memory_config=None,
    max_grid=None,
    block_w: int | None = None,
    depth: int = 2,
    rt: int = 1,
) -> ttnn.Tensor:
    """x [1, G, M, K] @ w [1, G, K, N] -> [1, G, M, N] (TILE, DRAM interleaved; any tile dtypes, e.g. bfloat8_b
    operands and output). ``dtype`` (default x's): the output dtype. ``compute_kernel_config``: its math fidelity,
    fp32 DEST and approx mode (default HiFi2, fp32 DEST). ``output_tensor``: the caller's buffer of exactly that spec,
    returned as is. ``max_grid`` (default the whole compute grid): the cores come from (0, 0) .. (max_grid - 1) (a
    Galaxy's 110-core matmul cap). ``block_w`` (default the widest of 32 / 16 / 8 / .. tiles that fits L1): K tiles a
    block. ``depth`` (default 2): K blocks of x and of w buffered a core. ``rt`` (default 1): row tiles a DST block
    (rt x the core's column tiles; full-sync DEST when that exceeds half of it). Needs G <= the grid's rows
    (``supported``)."""
    for t in (x, w):
        assert t.layout == ttnn.TILE_LAYOUT, t.layout
        mc = t.memory_config()
        assert mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, mc
    odt = dtype or x.dtype
    shape = [1, x.shape[1], x.shape[2], w.shape[3]]
    if output_tensor is None:
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), odt, ttnn.TILE_LAYOUT, x.device(), memory_config or ttnn.DRAM_MEMORY_CONFIG
        )
    else:
        o = output_tensor
        assert o.dtype == odt and o.layout == ttnn.TILE_LAYOUT and list(o.shape) == shape, (o.dtype, o.shape)
        assert o.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, o.memory_config()
    pp = _prepared(x, w, output_tensor, compute_kernel_config, max_grid, block_w, depth, rt)
    ttnn.bringup.run_prepared(pp, [x, w, output_tensor])
    return output_tensor
