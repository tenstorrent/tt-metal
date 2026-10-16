# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""combine_plan — the flat expert's and the router-forwarded combine's per-call tables, built on the device.

From the routing (every token's top-k global expert ids and gate weights, replicated on every chip: [1, N * S * K]
uint16 ids and bf16 weights, token-major), each chip's planner (one core) builds, for its local experts and its home
tokens:
  counts / regions [1, NG] uint32       the flat expert's per-global-expert token counts and first rows (padded to 32)
  token_index [1, rows_cap] uint32      per flat row, the gathered-token row it reads (home * S + t)
  hand-off table [rows_cap / 32, 64]    per flat row (destination | gate weight bf16 << 16, header word): the y writers'
                                        L1 hand-off (se_yrm.hpp SE_Y_L1); header = home | t << 8 | local expert << 24
  plan [1, 1024] uint32                 the combine's per-call counts: n_inj[8], n_ret[8] per port id, n_local[16] per
                                        accumulator, want[16][32] per accumulator and (source chip, link)
Rows of a local expert are in (home, token, k) scan order. A row goes to home over link (t + k) % L, through this
chip's first-hop port toward home (Y first, then X; port_lut[direction * 2 + link] = port id); a local row (home is
this chip) goes to accumulator t % A (destination id P + t % A). The same rules as sparse_combine_rf_prepare's host
tables (plan_reference below is the host model)."""

import torch

import ttnn

PLAN_WORDS = 1024
P_NINJ, P_NRET, P_NLOC, P_WANT = 0, 8, 16, 32  # plan word offsets
P_ROWS, P_OVER = 600, 601

_PLANNER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t ME = get_compile_time_arg_val(0), R = get_compile_time_arg_val(1), C = get_compile_time_arg_val(2);
constexpr uint32_t S = get_compile_time_arg_val(3), K = get_compile_time_arg_val(4), EPC = get_compile_time_arg_val(5);
constexpr uint32_t NG = get_compile_time_arg_val(6), A = get_compile_time_arg_val(7), L = get_compile_time_arg_val(8);
constexpr uint32_t P = get_compile_time_arg_val(9), ROWS_CAP = get_compile_time_arg_val(10);
constexpr uint32_t N = R * C, T = N * S * K;
constexpr uint32_t cb_in = 0, cb_out = 1;
// direction index (sparse_all_gather_2d DIRS): 0 X+, 1 X-, 2 Y+, 3 Y-; Y first, then X (lines)
FORCE_INLINE uint32_t first_hop(uint32_t src, uint32_t dst) {
    const uint32_t sr = src / C, sc = src % C, dr = dst / C, dc = dst % C;
    if (sr != dr) return dr > sr ? 2 : 3;
    return dc > sc ? 0 : 1;
}
void kernel_main() {
    size_t a = 0;
    const uint32_t ids_addr = get_arg_val<uint32_t>(a++), w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(a++), reg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tidx_addr = get_arg_val<uint32_t>(a++), tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t plan_addr = get_arg_val<uint32_t>(a++);
    uint32_t lut[8];
    for (uint32_t i = 0; i < 8; ++i) lut[i] = get_arg_val<uint32_t>(a++);
    // L1: ids, weights | counts, regions, token index, table, plan
    cb_reserve_back(cb_in, 1);  // (64-byte aligned starts: DRAM transfers)
    const uint32_t ids_l1 = (get_write_ptr(cb_in) + 63) & ~63u, w_l1 = ids_l1 + ((T * 2 + 63) / 64) * 64;
    cb_reserve_back(cb_out, 1);
    const uint32_t cnt_l1 = (get_write_ptr(cb_out) + 63) & ~63u;
    const uint32_t reg_l1 = cnt_l1 + ((NG * 4 + 63) / 64) * 64;
    const uint32_t tidx_l1 = reg_l1 + ((NG * 4 + 63) / 64) * 64;
    const uint32_t tab_l1 = tidx_l1 + ROWS_CAP * 4;
    const uint32_t plan_l1 = tab_l1 + ROWS_CAP * 8;
    const InterleavedAddrGen<true> ids_g = {.bank_base_address = ids_addr, .page_size = T * 2};
    const InterleavedAddrGen<true> w_g = {.bank_base_address = w_addr, .page_size = T * 2};
    noc_async_read(get_noc_addr(0, ids_g), ids_l1, T * 2);
    noc_async_read(get_noc_addr(0, w_g), w_l1, T * 2);
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_l1);
    volatile tt_l1_ptr uint32_t* reg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reg_l1);
    volatile tt_l1_ptr uint32_t* tidx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tidx_l1);
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab_l1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_l1);
    for (uint32_t i = 0; i < NG; ++i) cnt[i] = reg[i] = 0;
    for (uint32_t i = 0; i < ROWS_CAP; ++i) tidx[i] = 0;
    for (uint32_t i = 0; i < 2 * ROWS_CAP; ++i) tab[i] = 0;
    for (uint32_t i = 0; i < 1024; ++i) plan[i] = 0;
    noc_async_read_barrier();
    const volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(ids_l1);
    const volatile tt_l1_ptr uint16_t* wts = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(w_l1);
    constexpr uint32_t E0 = ME * EPC;
    // pass 1: tokens per local expert
    uint32_t lc[EPC];
    for (uint32_t e = 0; e < EPC; ++e) lc[e] = 0;
    for (uint32_t i = 0; i < T; ++i) {
        const uint32_t e = ids[i] - E0;
        if (e < EPC) ++lc[e];
    }
    uint32_t fill[EPC], off = 0;
    for (uint32_t e = 0; e < EPC; ++e) {
        cnt[E0 + e] = lc[e];
        reg[E0 + e] = off;
        fill[e] = off;
        off += (lc[e] + 31) / 32 * 32;
    }
    plan[600] = off;
    plan[601] = off > ROWS_CAP ? 1 : 0;
    // pass 2: place the local experts' rows; count the combine's rows (this chip as source and as home)
    for (uint32_t o = 0; o < N; ++o) {
        const uint32_t dir_home = o == ME ? 0 : first_hop(ME, o);
        for (uint32_t t = 0; t < S; ++t) {
            const uint32_t base = (o * S + t) * K;
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t g = ids[base + k], src = g / EPC, l = (t + k) % L;
                if (src == ME) {
                    const uint32_t e = g - E0, r = fill[e]++;
                    if (r < ROWS_CAP) {
                        tidx[r] = o * S + t;
                        uint32_t dest;
                        if (o == ME) {
                            dest = P + t % A;
                        } else {
                            dest = lut[dir_home * 2 + l];
                            ++plan[0 + dest];  // n_inj
                        }
                        tab[2 * r] = dest | (static_cast<uint32_t>(wts[base + k]) << 16);
                        tab[2 * r + 1] = o | (t << 8) | (e << 24);
                    }
                }
                if (o == ME) {  // a row for my token
                    if (src == ME) {
                        ++plan[16 + t % A];  // n_local
                    } else {
                        ++plan[32 + (t % A) * 32 + src * L + l];  // want
                        ++plan[8 + lut[first_hop(ME, src) * 2 + l]];  // n_ret: the port facing the source
                    }
                }
            }
        }
    }
    // out
    const InterleavedAddrGen<true> cg = {.bank_base_address = cnt_addr, .page_size = NG * 4};
    const InterleavedAddrGen<true> rg = {.bank_base_address = reg_addr, .page_size = NG * 4};
    const InterleavedAddrGen<true> ig = {.bank_base_address = tidx_addr, .page_size = ROWS_CAP * 4};
    const InterleavedAddrGen<true> tg = {.bank_base_address = tab_addr, .page_size = 256};
    const InterleavedAddrGen<true> pg = {.bank_base_address = plan_addr, .page_size = 4096};
    noc_async_write(cnt_l1, get_noc_addr(0, cg), NG * 4);
    noc_async_write(reg_l1, get_noc_addr(0, rg), NG * 4);
    noc_async_write(tidx_l1, get_noc_addr(0, ig), ROWS_CAP * 4);
    for (uint32_t p = 0; p < ROWS_CAP / 32; ++p) noc_async_write(tab_l1 + p * 256, get_noc_addr(p, tg), 256);
    noc_async_write(plan_l1, get_noc_addr(0, pg), 4096);
    noc_async_write_barrier();
}
"""


def first_hop_idx(src, dst, C):
    sr, sc, dr, dc = src // C, src % C, dst // C, dst % C
    if sr != dr:
        return 2 if dr > sr else 3
    return 0 if dc > sc else 1


def plan_reference(ids, wbits, me, R, C, S, K, EPC, NG, A, L, P, lut, rows_cap):
    """Host model of the planner for chip `me`: ids / wbits [N, S, K] (global expert ids, bf16 bits)."""
    N = R * C
    E0 = me * EPC
    lc = [0] * EPC
    flat = ids.reshape(-1).tolist()
    for g in flat:
        if E0 <= g < E0 + EPC:
            lc[g - E0] += 1
    cnt, reg = [0] * NG, [0] * NG
    fill, off = [0] * EPC, 0
    for e in range(EPC):
        cnt[E0 + e], reg[E0 + e], fill[e] = lc[e], off, off
        off += (lc[e] + 31) // 32 * 32
    tidx = [0] * rows_cap
    tab = [0] * (2 * rows_cap)
    plan = [0] * 1024
    plan[P_ROWS], plan[P_OVER] = off, int(off > rows_cap)
    idsl, wl = ids.tolist(), wbits.tolist()
    for o in range(N):
        dh = 0 if o == me else first_hop_idx(me, o, C)
        for t in range(S):
            for k in range(K):
                g, wb = idsl[o][t][k], wl[o][t][k]
                src, l = g // EPC, (t + k) % L
                if src == me:
                    e = g - E0
                    r = fill[e]
                    fill[e] += 1
                    if r < rows_cap:
                        tidx[r] = o * S + t
                        if o == me:
                            dest = P + t % A
                        else:
                            dest = lut[dh * 2 + l]
                            plan[P_NINJ + dest] += 1
                        tab[2 * r] = dest | (wb << 16)
                        tab[2 * r + 1] = o | (t << 8) | (e << 24)
                if o == me:
                    if src == me:
                        plan[P_NLOC + t % A] += 1
                    else:
                        plan[P_WANT + (t % A) * 32 + src * L + l] += 1
                        plan[P_NRET + lut[first_hop_idx(me, src, C) * 2 + l]] += 1
    return dict(counts=cnt, regions=reg, token_index=tidx, table=tab, plan=plan)


def port_luts(mesh_device, num_links):
    """Per chip coord: lut[direction * 2 + link] = port id (the chip's ports in DIRS / link order, as
    sparse_combine_rf_prepare numbers them), and the port count."""
    from .sparse_all_gather_2d import DIRS, mesh_plan

    chips, _, _ = mesh_plan(mesh_device, num_links)
    luts, n = {}, {}
    for ch, ports in chips.items():
        lut, j = [0xFF] * 8, 0
        for di, d in enumerate(DIRS):
            if d in ports:
                for li in range(num_links):
                    lut[di * 2 + li] = j
                    j += 1
        luts[ch], n[ch] = lut, j
    return luts, n


def combine_planner_prepare(mesh_device, *, S, K, EPC, A, L, port_luts, n_ports, rows_cap, core=(0, 0)):
    """port_luts[coord] = 8 port ids (direction * 2 + link; 0xFF none), n_ports[coord] = P. Returns run(ids, wts) ->
    dict of the per-chip output tensors (counts, regions, token_index, table, plan), allocated once."""
    R, C = tuple(mesh_device.shape)
    N = R * C
    NG = N * EPC
    assert rows_cap % 32 == 0 and N * S * K < (1 << 16) * 64
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1))
    zeros = lambda w: ttnn.from_torch(  # noqa: E731
        torch.zeros(R, C, 1, w, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    outs = dict(
        counts=zeros(NG),
        regions=zeros(NG),
        token_index=zeros(rows_cap),
        table=ttnn.from_torch(
            torch.zeros(R, C, rows_cap // 32, 64, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        ),
        plan=zeros(1024),
    )
    T = N * S * K
    in_bytes = 2 * ((T * 2 + 63) // 64) * 64 + 64
    out_bytes = 2 * ((NG * 4 + 63) // 64) * 64 + rows_cap * 12 + 4096 + 64
    c = ttnn.CoreCoord(*core)
    cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c)])

    def build(ids, wts):
        mesh_desc = ttnn.MeshProgramDescriptor()
        for r in range(R):
            for cc in range(C):
                me = r * C + cc
                pd = ttnn.ProgramDescriptor()
                pd.cbs = [
                    ttnn.CBDescriptor(
                        total_size=in_bytes,
                        core_ranges=cs,
                        format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, in_bytes)],
                    ),
                    ttnn.CBDescriptor(
                        total_size=out_bytes,
                        core_ranges=cs,
                        format_descriptors=[ttnn.CBFormatDescriptor(1, ttnn.uint32, out_bytes)],
                    ),
                ]
                rt = ttnn.RuntimeArgs()
                rt[c.x][c.y] = [
                    int(ids.buffer_address()),
                    int(wts.buffer_address()),
                    int(outs["counts"].buffer_address()),
                    int(outs["regions"].buffer_address()),
                    int(outs["token_index"].buffer_address()),
                    int(outs["table"].buffer_address()),
                    int(outs["plan"].buffer_address()),
                ] + list(port_luts[(r, cc)])
                pd.kernels = [
                    ttnn.KernelDescriptor(
                        kernel_source=_PLANNER,
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=cs,
                        compile_time_args=[me, R, C, S, K, EPC, NG, A, L, n_ports[(r, cc)], rows_cap],
                        runtime_args=rt,
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                        ),
                    )
                ]
                mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, cc), ttnn.MeshCoordinate(r, cc))] = pd
        return mesh_desc

    state = {}

    def run(ids, wts):
        k = (int(ids.buffer_address()), int(wts.buffer_address()))
        if state.get("key") != k:
            state["key"], state["desc"] = k, build(ids, wts)
        ttnn.generic_op(
            [ids, wts, outs["counts"], outs["regions"], outs["token_index"], outs["table"], outs["plan"]], state["desc"]
        )
        return outs

    run.outs = outs
    return run
