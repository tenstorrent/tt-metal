# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""dispatch_prep — what the sparse 2D dispatch (sparse_all_gather_2d) needs on each chip, from the device's routing,
in one generic_op per chip:

  route core   the route word of each own token t: the destination chips (every chip holding one of its K experts,
               the own chip excluded), the order (X or Y first: the tree with fewer links, ties by a hash of t) and
               the link (hash bit 29 of t), from the replicated joint routing rows (ids, then weights: [N S, 2 K]
               uint16, combine_plan's joint=True input) -> route [1, 1, 1, S] uint32
  copy cores   the own rows (the latent projection's [S, W] row-major output) into their slots idx * S + t of the
               dispatch buffer [1, 1, N S, W]
"""

import os

import ttnn

from .sparse_all_gather_2d import _COMMON, mesh_plan, route_words, span_subs

_ROUTE = (
    _COMMON
    + r"""
constexpr uint32_t K = __K__;
constexpr uint32_t ME = __ME__;
constexpr uint32_t ROWMATES = __ROWMATES__;
constexpr uint8_t CHIP_OF[__E__] = {__CHIPS__};
constexpr auto ids_args = TensorAccessorArgs<0>();
constexpr auto route_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
constexpr uint32_t ROW = 4 * K;  // a joint routing row: K ids, K weights (uint16)
constexpr uint32_t PT = __PT__;  // routing rows per page (router_topk: 32)
static_assert(K % 2 == 0);

FORCE_INLINE uint32_t links(uint32_t s) { return (s & 15) + (s >> 4); }

void kernel_main() {
    const uint32_t ids_addr = get_arg_val<uint32_t>(0), route_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2), t1 = get_arg_val<uint32_t>(3);  // (this core's tokens)
    const uint32_t scratch = get_write_ptr(0);
    const auto ids = TensorAccessor(ids_args, ids_addr, ROW * PT);
    const auto route = TensorAccessor(route_args, route_addr, S * 4 * RW);
    constexpr uint32_t RS = (ROW + 63) / 64 * 64;  // (64-byte landing: a DRAM read keeps its source alignment)
    const uint32_t rows_l1 = (scratch + 63) & ~63u, words_l1 = rows_l1 + (t1 - t0) * RS;
    auto put = [&](uint32_t t, uint32_t m, uint32_t flags) {  // (flags: order | link << 1)
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words_l1);
        if constexpr (RW == 2) {
            w[2 * (t - t0)] = m;
            w[2 * (t - t0) + 1] = flags;
        } else {
            w[t - t0] = m | (flags << 16);
        }
    };
    if constexpr (__NOROUTE__) {  // (probe: no route words)
        return;
    }
    // (a DRAM read keeps its source's 64 B alignment in L1: a row at an odd 32 B of its page lands 32 B into its slot)
    auto so = [&](uint32_t t) { return ((ME * S + t) % PT * ROW) & 63u; };
    for (uint32_t t = t0; t < t1; ++t) {
        const uint32_t g = ME * S + t;
        noc_async_read(ids.get_noc_addr(g / PT) + (g % PT) * ROW, rows_l1 + (t - t0) * RS + so(t), ROW);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t r0 = ME / C, c0 = ME % C;
    for (uint32_t t = t0; t < t1; ++t) {
        const uint32_t* id2 = reinterpret_cast<const uint32_t*>(rows_l1 + (t - t0) * RS + so(t));  // two ids a word
        uint32_t m = 0;
        for (uint32_t k = 0; k < K / 2; ++k) {
            const uint32_t v = id2[k];
            m |= (1u << CHIP_OF[v & 0xFFFF]) | (1u << CHIP_OF[v >> 16]);
        }
        m |= ROWMATES;  // (shared_rows: every token also to the other chips of my mesh row - the TP shared expert's x)
        m &= ~(1u << ME);
        if (m == 0) {
            put(t, 0, 0);
            continue;
        }
        // links of the two dimension-order trees (X first: along my row to every column with a destination, then
        // down each column; Y first: along my column to every row with one, then along each row)
        const uint32_t cm = cols_of(m), rm = rows_of(m);
        uint32_t hx = links(xspan(c0, cm & ~(1u << c0), t)), hy = links(yspan(r0, rm & ~(1u << r0), t));
        for (uint32_t c = 0; c < C; ++c) {
            if (((cm | (1u << c0)) >> c) & 1) hx += links(yspan(r0, rows_in_col(m, c) & ~(1u << r0), t));
        }
        for (uint32_t r = 0; r < R; ++r) {
            if (((rm | (1u << r0)) >> r) & 1) hy += links(xspan(c0, cols_in_row(m, r) & ~(1u << c0), t));
        }
        const uint32_t hsh = t * 2654435761u;
        const uint32_t order = hx < hy ? 0 : (hy < hx ? 1 : (hsh >> 31) & 1);
        put(t, m, order | (((hsh >> 29) & 1) << 1));
    }
    if (t1 > t0) noc_async_write(words_l1, route.get_noc_addr(0) + t0 * 4 * RW, (t1 - t0) * 4 * RW);
    noc_async_write_barrier();
}
"""
)

_COPY = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t ROW = get_compile_time_arg_val(0);
constexpr uint32_t BATCH = get_compile_time_arg_val(1);
constexpr auto src_args = TensorAccessorArgs<2>();
constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
// own rows t0 .. t1 of the latent projection into the dispatch buffer's slots base + t
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0), dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2), t1 = get_arg_val<uint32_t>(3), base = get_arg_val<uint32_t>(4);
    const uint32_t buf = get_write_ptr(1);
    const auto src = TensorAccessor(src_args, src_addr, ROW);
    const auto dst = TensorAccessor(dst_args, dst_addr, ROW);
    if constexpr (__NOCOPY__) {  // (probe: no own-row copy)
        return;
    }
    // two halves: batch b's writes overlap batch b + 1's reads
    uint32_t h = 0;
    for (uint32_t t = t0; t < t1; t += BATCH, h ^= 1) {
        const uint32_t n = t1 - t < BATCH ? t1 - t : BATCH, b = buf + h * BATCH * ROW;
        for (uint32_t i = 0; i < n; ++i) noc_async_read(src.get_noc_addr(t + i), b + i * ROW, ROW);
        noc_async_read_barrier();
        noc_async_writes_flushed();  // (the other half's writes have left L1: it may be refilled next)
        for (uint32_t i = 0; i < n; ++i) noc_async_write(b + i * ROW, dst.get_noc_addr(base + t + i), ROW);
    }
    noc_async_write_barrier();
}
"""


def _mates_mask(R, C, r, c, axis):
    """The chips of (r, c)'s group as a bit mask over r C + c: its mesh row (axis 1) or column (axis 0)."""
    return ((1 << C) - 1) << (r * C) if axis == 1 else sum(1 << (rr * C + c) for rr in range(R))


def dispatch_prep_prepare(
    mesh_device,
    *,
    S,
    K,
    EPC,
    n_route=8,
    n_copy=16,
    batch=4,
    torus=(False, False),
    rows=(8, 9),
    tokens_per_page=1,
    copy=True,
    row_mates=False,
    mates_axis=1,
):
    """Returns run(joint_routing, own_rows, xbuf, route) (one generic_op: route words + own-row copy).
    joint_routing [N S, 2 K] uint16 row-major (replicated), own_rows [S, W] bf16 row-major, xbuf [1, 1, N S, W]
    row-major, route [1, 1, 1, RW S] uint32 (RW = route_words(N); outputs written in place). Cores (logical): n_route route cores on row
    rows[0] (each a token range: 8-word multiples, 32-byte aligned route writes), n_copy copy cores on row rows[1].
    copy=False: route words only (the own rows already in xbuf, e.g. layout_ops' untilize with a row offset; own None).
    row_mates: every own token also goes to the other chips of this chip's mesh row (mates_axis 1; 0: its mesh column) -
    the split shared expert's input: after the dispatch each chip holds its group's tokens at slots (r C + c) S + t.
    """
    R, C = tuple(mesh_device.shape)
    N = R * C
    E = N * EPC
    _, _, (wrap_x, wrap_y) = mesh_plan(mesh_device, 2, torus)
    RW = route_words(N)
    assert N <= 32
    chips_of = ", ".join(str(e // EPC) for e in range(E))
    grid = mesh_device.compute_with_storage_grid_size()
    assert n_route <= grid.x and n_copy <= 2 * grid.x and max(rows) < grid.y and n_route + n_copy <= 2 * grid.x
    rcs = [ttnn.CoreCoord(i, rows[0]) for i in range(n_route)]
    cps = [
        ttnn.CoreCoord(i % grid.x, rows[1]) if i < grid.x else ttnn.CoreCoord(n_route + i - grid.x, rows[0])
        for i in range(n_copy)
    ]
    rper = (-(-S // n_route) + 7) // 8 * 8  # (tokens per route core: 8-word multiples, 32-byte aligned writes)
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    dm = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default)
    state = {}

    def build(jr, own, xbuf, route, per_chip=False):
        W = int(xbuf.shape[-1])
        progs = {}
        row = W * 2
        route_l1 = rper * (((4 * K) + 63) // 64 * 64) + rper * 4 * RW + 128
        mesh_desc = ttnn.MeshProgramDescriptor()
        jr_ct = list(ttnn.TensorAccessorArgs(jr).get_compile_time_args())
        rt_ct = list(ttnn.TensorAccessorArgs(route).get_compile_time_args())
        own_ct = list(ttnn.TensorAccessorArgs(own).get_compile_time_args()) if copy else []
        xb_ct = list(ttnn.TensorAccessorArgs(xbuf).get_compile_time_args())
        per = -(-S // n_copy)
        for me in range(N):
            r, c = me // C, me % C
            src = _ROUTE
            for k_, v_ in span_subs(R, C, wrap_x, wrap_y).items():
                src = src.replace(k_, v_)
            src = (
                src.replace("__R__", str(R))
                .replace("__C__", str(C))
                .replace("__S__", str(S))
                .replace("__GMAX__", str(max(1, R + C - 3)))
                .replace("__RW__", str(RW))
                .replace("__K__", str(K))
                .replace("__ME__", str(me))
                .replace("__ROWMATES__", str(_mates_mask(R, C, r, c, mates_axis) if row_mates else 0) + "u")
                .replace("__E__", str(E))
                .replace("__CHIPS__", chips_of)
                .replace("__PT__", str(tokens_per_page))
                .replace("__NOROUTE__", "1" if os.environ.get("DPREP_PROBE") == "noroute" else "0")
            )
            copy_src = _COPY.replace("__NOCOPY__", "1" if os.environ.get("DPREP_PROBE") == "nocopy" else "0")
            program = ttnn.ProgramDescriptor()
            rset = ttnn.CoreRangeSet([ttnn.CoreRange(rcs[0], rcs[-1])])
            cset = ttnn.CoreRangeSet([ttnn.CoreRange(c_, c_) for c_ in cps])
            program.cbs = [
                ttnn.CBDescriptor(
                    total_size=route_l1,
                    core_ranges=rset,
                    format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, route_l1)],
                ),
            ] + (
                [
                    ttnn.CBDescriptor(
                        total_size=2 * batch * row,
                        core_ranges=cset,
                        format_descriptors=[ttnn.CBFormatDescriptor(1, ttnn.uint32, row)],
                    )
                ]
                if copy
                else []
            )
            rrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for i, rc in enumerate(rcs):
                rrt[rc.x][rc.y] = [
                    int(jr.buffer_address()),
                    int(route.buffer_address()),
                    min(S, i * rper),
                    min(S, (i + 1) * rper),
                ]
            for i, cc in enumerate(cps if copy else []):
                t0, t1 = min(S, i * per), min(S, (i + 1) * per)
                crt[cc.x][cc.y] = [int(own.buffer_address()), int(xbuf.buffer_address()), t0, t1, me * S, 0]
            program.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=src,
                    source_type=st,
                    core_ranges=rset,
                    compile_time_args=jr_ct + rt_ct,
                    runtime_args=rrt,
                    config=dm,
                ),
            ] + (
                [
                    ttnn.KernelDescriptor(
                        kernel_source=copy_src,
                        source_type=st,
                        core_ranges=cset,
                        compile_time_args=[row, batch] + own_ct + xb_ct,
                        runtime_args=crt,
                        config=dm,
                    )
                ]
                if copy
                else []
            )
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
            progs[(r, c)] = program
        return progs if per_chip else mesh_desc

    def run(jr, own, xbuf, route):
        ts = [jr] + ([own] if copy else []) + [xbuf, route]
        k = tuple(int(t.buffer_address()) for t in ts)
        if state.get("key") != k:
            state["key"], state["desc"] = k, build(jr, own, xbuf, route)
            ttnn.freeze_program_hashes(state["desc"])  # (hash stored once: not re-hashed per call)
        return ttnn.generic_op(ts, state["desc"])

    run.programs = lambda jr, own, xbuf, route: build(jr, own, xbuf, route, per_chip=True)
    run.cores = rcs + (cps if copy else [])  # (the cores its kernels take)
    return run


def own_rows_copy_prepare(mesh_device, *, S, n_copy=4, batch=8, row0=8):
    """run(own [S, W] row-major, buf [1, 1, N S, W] row-major): this chip's rows into slots idx * S + t (one generic_op;
    e.g. the routing rows before an all-chips sparse_all_gather_2d)."""
    R, C = tuple(mesh_device.shape)
    N = R * C
    cps = [ttnn.CoreCoord(i, row0) for i in range(n_copy)]
    cset = ttnn.CoreRangeSet([ttnn.CoreRange(c_, c_) for c_ in cps])
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    dm = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default)
    src = _COPY.replace("__NOCOPY__", "0")
    state = {}

    def build(own, buf):
        row = int(own.buffer_aligned_page_size())
        own_ct = list(ttnn.TensorAccessorArgs(own).get_compile_time_args()) if copy else []
        b_ct = list(ttnn.TensorAccessorArgs(buf).get_compile_time_args())
        per = -(-S // n_copy)
        mesh_desc = ttnn.MeshProgramDescriptor()
        for me in range(N):
            program = ttnn.ProgramDescriptor()
            program.cbs = [
                ttnn.CBDescriptor(
                    total_size=2 * batch * row,
                    core_ranges=cset,
                    format_descriptors=[ttnn.CBFormatDescriptor(1, ttnn.uint32, row)],
                )
            ]
            rt = ttnn.RuntimeArgs()
            for i, cc in enumerate(cps):
                rt[cc.x][cc.y] = [
                    int(own.buffer_address()),
                    int(buf.buffer_address()),
                    min(S, i * per),
                    min(S, (i + 1) * per),
                    me * S,
                    0,
                ]
            program.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=src,
                    source_type=st,
                    core_ranges=cset,
                    compile_time_args=[row, batch] + own_ct + b_ct,
                    runtime_args=rt,
                    config=dm,
                )
            ]
            mesh_desc[
                ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(me // C, me % C), ttnn.MeshCoordinate(me // C, me % C))
            ] = program
        return mesh_desc

    def run(own, buf):
        k = (int(own.buffer_address()), int(buf.buffer_address()))
        if state.get("key") != k:
            state["key"], state["desc"] = k, build(own, buf)
            ttnn.freeze_program_hashes(state["desc"])  # (hash stored once: not re-hashed per call)
        return ttnn.generic_op([own, buf], state["desc"])

    return run


def all_chips_route_words(mesh_device, S):
    """[1, 1, 1, S] uint32 per chip (a mesh tensor): every own row to every other chip (order / link by a hash of t)."""
    import torch

    R, C = tuple(mesh_device.shape)
    N = R * C
    RW = route_words(N)
    w = torch.zeros(N, 1, 1, RW * S, dtype=torch.int64)
    for me in range(N):
        for t in range(S):
            h = (t * 2654435761) & 0xFFFFFFFF
            m, f = ((1 << N) - 1) & ~(1 << me), ((h >> 31) & 1) | (((h >> 29) & 1) << 1)
            if RW == 2:
                w[me, 0, 0, 2 * t], w[me, 0, 0, 2 * t + 1] = m - (1 << 32 if m >= 1 << 31 else 0), f
            else:
                w[me, 0, 0, t] = m | (f << 16)
    return ttnn.from_torch(
        w.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
