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

from .sparse_all_gather_2d import _COMMON, mesh_plan, span_table

_ROUTE = (
    _COMMON
    + r"""
constexpr uint32_t K = __K__;
constexpr uint32_t ME = __ME__;
constexpr uint8_t CHIP_OF[__E__] = {__CHIPS__};
constexpr auto ids_args = TensorAccessorArgs<0>();
constexpr auto route_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
constexpr uint32_t ROW = 4 * K;  // a joint routing row: K ids, K weights (uint16)
static_assert(K % 2 == 0);

FORCE_INLINE uint32_t links(uint32_t s) { return (s & 15) + (s >> 4); }

void kernel_main() {
    const uint32_t ids_addr = get_arg_val<uint32_t>(0), route_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2), t1 = get_arg_val<uint32_t>(3);  // (this core's tokens)
    const uint32_t scratch = get_write_ptr(0);
    const auto ids = TensorAccessor(ids_args, ids_addr, ROW);
    const auto route = TensorAccessor(route_args, route_addr, S * 4);
    constexpr uint32_t RS = (ROW + 63) / 64 * 64;  // (64-byte landing: a DRAM read keeps its source alignment)
    const uint32_t rows_l1 = (scratch + 63) & ~63u, words_l1 = rows_l1 + (t1 - t0) * RS;
    if constexpr (__NOROUTE__) {  // (probe: no route words)
        return;
    }
    for (uint32_t t = t0; t < t1; ++t) noc_async_read(ids.get_noc_addr(ME * S + t), rows_l1 + (t - t0) * RS, ROW);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t r0 = ME / C, c0 = ME % C;
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words_l1);
    for (uint32_t t = t0; t < t1; ++t) {
        const uint32_t* id2 = reinterpret_cast<const uint32_t*>(rows_l1 + (t - t0) * RS);  // two ids a word
        uint32_t m = 0;
        for (uint32_t k = 0; k < K / 2; ++k) {
            const uint32_t v = id2[k];
            m |= (1u << CHIP_OF[v & 0xFFFF]) | (1u << CHIP_OF[v >> 16]);
        }
        m &= ~(1u << ME);
        if (m == 0) {
            words[t - t0] = 0;
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
        words[t - t0] = m | (order << 16) | (((hsh >> 29) & 1) << 17);
    }
    if (t1 > t0) noc_async_write(words_l1, route.get_noc_addr(0) + t0 * 4, (t1 - t0) * 4);
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


def dispatch_prep_prepare(mesh_device, *, S, K, EPC, n_route=8, n_copy=16, batch=4, torus=(False, False), rows=(8, 9)):
    """Returns run(joint_routing, own_rows, xbuf, route) (one generic_op: route words + own-row copy).
    joint_routing [N S, 2 K] uint16 row-major (replicated), own_rows [S, W] bf16 row-major, xbuf [1, 1, N S, W]
    row-major, route [1, 1, 1, S] uint32 (outputs written in place). Cores (logical): n_route route cores on row
    rows[0] (each a token range: 8-word multiples, 32-byte aligned route writes), n_copy copy cores on row rows[1]."""
    R, C = tuple(mesh_device.shape)
    N = R * C
    E = N * EPC
    _, _, (wrap_x, wrap_y) = mesh_plan(mesh_device, 2, torus)
    chips_of = ", ".join(str(e // EPC) for e in range(E))
    grid = mesh_device.compute_with_storage_grid_size()
    assert n_route <= grid.x and n_copy <= 2 * grid.x and max(rows) < grid.y and n_route + n_copy <= 2 * grid.x
    rcs = [ttnn.CoreCoord(i, rows[0]) for i in range(n_route)]
    cps = [ttnn.CoreCoord(i % grid.x, rows[1]) if i < grid.x else ttnn.CoreCoord(n_route + i - grid.x, rows[0]) for i in range(n_copy)]
    rper = (-(-S // n_route) + 7) // 8 * 8  # (tokens per route core: 8-word multiples, 32-byte aligned writes)
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    dm = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default)
    state = {}

    def build(jr, own, xbuf, route):
        W = int(own.shape[-1])
        row = W * 2
        route_l1 = rper * (((4 * K) + 63) // 64 * 64) + rper * 4 + 128
        mesh_desc = ttnn.MeshProgramDescriptor()
        jr_ct = list(ttnn.TensorAccessorArgs(jr).get_compile_time_args())
        rt_ct = list(ttnn.TensorAccessorArgs(route).get_compile_time_args())
        own_ct = list(ttnn.TensorAccessorArgs(own).get_compile_time_args())
        xb_ct = list(ttnn.TensorAccessorArgs(xbuf).get_compile_time_args())
        per = -(-S // n_copy)
        for me in range(N):
            r, c = me // C, me % C
            src = (
                _ROUTE.replace("__R__", str(R)).replace("__C__", str(C)).replace("__S__", str(S))
                .replace("__GMAX__", str(max(1, R + C - 3)))
                .replace("__XSPAN__", ", ".join(map(str, span_table(C, wrap_x))))
                .replace("__YSPAN__", ", ".join(map(str, span_table(R, wrap_y))))
                .replace("__K__", str(K)).replace("__ME__", str(me)).replace("__E__", str(E))
                .replace("__CHIPS__", chips_of)
                .replace("__NOROUTE__", "1" if os.environ.get("DPREP_PROBE") == "noroute" else "0")
            )
            copy_src = _COPY.replace("__NOCOPY__", "1" if os.environ.get("DPREP_PROBE") == "nocopy" else "0")
            program = ttnn.ProgramDescriptor()
            rset = ttnn.CoreRangeSet([ttnn.CoreRange(rcs[0], rcs[-1])])
            cset = ttnn.CoreRangeSet([ttnn.CoreRange(c_, c_) for c_ in cps])
            program.cbs = [
                ttnn.CBDescriptor(total_size=route_l1, core_ranges=rset,
                                  format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, route_l1)]),
                ttnn.CBDescriptor(total_size=2 * batch * row, core_ranges=cset,
                                  format_descriptors=[ttnn.CBFormatDescriptor(1, ttnn.uint32, row)]),
            ]
            rrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for i, rc in enumerate(rcs):
                rrt[rc.x][rc.y] = [int(jr.buffer_address()), int(route.buffer_address()), min(S, i * rper), min(S, (i + 1) * rper)]
            for i, cc in enumerate(cps):
                t0, t1 = min(S, i * per), min(S, (i + 1) * per)
                crt[cc.x][cc.y] = [int(own.buffer_address()), int(xbuf.buffer_address()), t0, t1, me * S, 0]
            program.kernels = [
                ttnn.KernelDescriptor(kernel_source=src, source_type=st, core_ranges=rset,
                                      compile_time_args=jr_ct + rt_ct, runtime_args=rrt, config=dm),
                ttnn.KernelDescriptor(kernel_source=copy_src, source_type=st, core_ranges=cset,
                                      compile_time_args=[row, batch] + own_ct + xb_ct, runtime_args=crt, config=dm),
            ]
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
        return mesh_desc

    def run(jr, own, xbuf, route):
        k = tuple(int(t.buffer_address()) for t in (jr, own, xbuf, route))
        if state.get("key") != k:
            state["key"], state["desc"] = k, build(jr, own, xbuf, route)
        return ttnn.generic_op([jr, own, xbuf, route], state["desc"])

    return run
