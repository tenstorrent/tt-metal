// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end flat expert: x relay reader (BRISC). Extracts every expert's tokens from the row-major bf16 dispatch
// buffer (one row per page, interleaved) and stages them for tilizing: for each virtual expert (a sub-block of MT row
// tiles of expert e), each super-block of 32 K tiles (1024 columns) and each row tile m, one chunk of 32 rows x 2 KB
// goes into the row-major CB (32 pages of 2 KB = 32 rows of 1024 bf16). Rows at or past the expert's token count are
// not read (their tiles are padding: their outputs are never written back).
// CT: 0 RM_CB, 1 X_PAGE_BYTES (the token row's bytes / PPR), 2 NUM_EXPERTS, 3 MT, 4 NSB (super-blocks per row),
//     5 MAX_SUB (sub-blocks per expert at most), 6 BATCH (chunks per read barrier), 7 see XRD_INDEXED, 8 PPR (pages
//     per token row)
// XRD_INDEXED (CT 7: the runtime-arg index of the token index address): flat row r (region space) reads x row
// token_index[r] (x = the all-gathered tokens rather than a dispatch buffer). Each sub-block's MT x 32 indices are
// read once (one small read + barrier) into this RISC's dynamic-schedule scratch (free after se_dyn_load).
// RT: 0 dispatch buffer address, 1 STRIDE, 2 OFF (this relay takes super-blocks OFF, OFF + STRIDE, ... in stream
// order),
//     then per expert: region row offset, token count
#include <stdint.h>
#ifndef SE_SBT
#define SE_SBT 32  // K tiles per super-block (the row-major chunk width / 32 columns)
#endif
#include "api/dataflow/dataflow_api.h"
#ifdef SE_DYN
#include "se_dyn.hpp"
#endif
#ifdef SE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define ZW(name) DeviceZoneScopedN(name)
#else
#define ZW(name)
#endif

void kernel_main() {
    constexpr uint32_t rm_cb = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);  // x page bytes
    constexpr uint32_t num_e = get_compile_time_arg_val(2);
    constexpr uint32_t mt = get_compile_time_arg_val(3);
    constexpr uint32_t nsb = get_compile_time_arg_val(4);
    constexpr uint32_t batch = get_compile_time_arg_val(6);
    constexpr uint32_t seg = SE_SBT * 64;  // one row of a super-block: SE_SBT tiles x 32 bf16
    // x page = row_bytes (CT 1) / pages per row (CT 8): a token row is PPR consecutive pages (PPR 8 at H 4096 spreads
    // every row over the 8 DRAM banks); a segment is one read (page >= seg) or seg / page reads
    constexpr uint32_t ppr = get_compile_time_arg_val(8);
    constexpr uint32_t pread = row_bytes < seg ? row_bytes : seg;
    const InterleavedAddrGen<true> xg = {.bank_base_address = get_arg_val<uint32_t>(0), .page_size = row_bytes};
    auto read_seg = [&](uint32_t row, uint32_t j, uint32_t dst) {
        const uint32_t b = j * seg;
        for (uint32_t q = 0; q < seg; q += pread) {
            noc_async_read(get_noc_addr(row * ppr + (b + q) / row_bytes, xg, (b + q) % row_bytes), dst + q, pread);
        }
    };
#ifdef XRD_INDEXED
    // one page holds the whole [1, rows] index row: address it as bank 0's page 0 plus a byte offset
    const uint64_t idx_base = get_noc_addr(
        0,
        InterleavedAddrGen<true>{
            .bank_base_address = get_arg_val<uint32_t>(get_compile_time_arg_val(7)), .page_size = 4});
    const uint32_t idx_l1 = get_write_ptr(tt::CBIndex::c_7);  // MT x 32 words (<= 512 B <= 2 SE_DYN_HALF)
#endif
    uint32_t in_batch = 0, l1 = 0;
    auto flush = [&]() {
        {
            ZW("XRD_DRAM");
            noc_async_read_barrier();
        }
        cb_push_back(rm_cb, 32 * in_batch);
        in_batch = 0;
    };
    uint32_t stride = get_arg_val<uint32_t>(1), soff = get_arg_val<uint32_t>(2);
    uint32_t g = 0;  // super-block index in stream order
#ifdef SE_DYN
    // Dynamic counts: the active experts' regions (RT 3.. are the se_dyn.hpp args, CB 7 this RISC's scratch); the
    // tilizer gets this relay's super-block count in CB 6.
    SeDyn dyn;
    se_dyn_load<num_e>(dyn, 3, get_write_ptr(tt::CBIndex::c_7), mt * 32);
#if defined(XHELP_SMALL) && defined(SE_SMALL_T)
    // small-M role split with helper relays: the primary (offset 0) reads all of x itself, the helper nothing
    if (dyn.small) {
        if (soff) {
            dyn.n_act = 0;
            dyn.num_v = 0;
        }
        stride = 1;
        soff = 0;
    }
#endif
    {
        const uint32_t tot = dyn.num_v * nsb;
        cb_reserve_back(tt::CBIndex::c_6, 1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(tt::CBIndex::c_6)) =
            tot > soff ? (tot - soff + stride - 1) / stride : 0;
        cb_push_back(tt::CBIndex::c_6, 1);
    }
    for (uint32_t a = 0; a < dyn.n_act; ++a) {
        const uint32_t off = dyn.off[a], count = dyn.cnt[a];
#else
    for (uint32_t e = 0; e < num_e; ++e) {
        const uint32_t off = get_arg_val<uint32_t>(3 + 2 * e), count = get_arg_val<uint32_t>(4 + 2 * e);
#endif
        const uint32_t subs = (count + mt * 32 - 1) / (mt * 32);
        for (uint32_t s = 0; s < subs; ++s) {
#ifdef XRD_INDEXED
            volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
            {
                bool mine = false;  // does this relay take any of the sub-block's super-blocks
                for (uint32_t j = 0; j < nsb && !mine; ++j) {
                    mine = (g + j) % stride == soff;
                }
                if (mine) {
                    const uint32_t rows = count - s * mt * 32 < mt * 32 ? count - s * mt * 32 : mt * 32;
                    // whole 32-row groups: the region is padded to 32 rows, so the read stays inside it
                    noc_async_read(
                        idx_base + (off + s * mt * 32) * 4, reinterpret_cast<uint32_t>(idx), (rows + 31) / 32 * 128);
                    noc_async_read_barrier();
                }
            }
#endif
            for (uint32_t j = 0; j < nsb; ++j, ++g) {
                if (g % stride != soff) {
                    continue;
                }
                for (uint32_t m = 0; m < mt; ++m) {
                    if (in_batch == 0) {
                        ZW("XRD_FULL");
                        cb_reserve_back(rm_cb, 32 * batch);
                        l1 = get_write_ptr(rm_cb);
                    }
                    const uint32_t dst = l1 + in_batch * 32 * seg;
                    const uint32_t r0 = s * mt * 32 + m * 32;
#ifdef XRD_SKIP_READS
                    // perf experiment only: no DRAM traffic (the tiles are garbage)
                    for (uint32_t r = 0; r < XRD_SKIP_READS && r0 + r < count; ++r) {
#else
                    for (uint32_t r = 0; r < 32 && r0 + r < count; ++r) {
#endif
#ifdef XRD_INDEXED
                        read_seg(idx[m * 32 + r], j, dst + r * seg);
#else
                        read_seg(off + r0 + r, j, dst + r * seg);
#endif
                    }
                    if (++in_batch == batch) {
                        flush();
                    }
                }
            }
        }
    }
    if (in_batch) {
        flush();
    }
    // leave no NoC transaction in flight (reads, writes, atomics, posted writes): the next program starts clean
    noc_async_full_barrier();
}
