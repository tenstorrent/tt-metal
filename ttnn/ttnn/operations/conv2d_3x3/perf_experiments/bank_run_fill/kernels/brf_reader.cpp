// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// bank_run_fill bench, reader (NCRISC, NoC0): the conv2d_3x3 ROW_MAJOR interleaved slab fill of one core, M-block by
// M-block, three kh bands per block (band kh = input pixels [o_first + (kh - pad_top) W - pad_left, o_end + ... + 2),
// clipped to the image), and nothing else.
//
// Variants (compile-time arg 0):
//   0 base       one NoC read per pixel (one DRAM page) straight into the slab: the op's read_pages / slab_bands.
//   1 bank       per chunk of CH pixels, one read per DRAM bank (pages a + s, a + s + 8, ... are contiguous in that
//                bank: InterleavedAddrGen offset = (page / 8) * aligned_page_size), landing bank-major in a staging
//                slot. No de-interleave: the pure DRAM fill.
//   2 bank_noc   bank + de-interleave by one self-aimed NoC read per pixel (stateful: two address writes per read).
//   3 bank_risc  bank + de-interleave by RISC word copies.
//   4 bank_trisc bank, the staging chunk handed to compute (cb_stage), which de-interleaves into cb_slab.
//   5 noc_only / 6 risc_only / 7 trisc_only: the de-interleave of 2 / 3 / 4 with no DRAM read (staging as is).
// Staged slot s of a chunk [a, e) holds pixels a + s + 8 k, k < ceil((e - a - s) / 8), at slot base + k * pb.
// Raw NoC (no kernel_lib helper): the bank-run read is a plain multi-packet noc_async_read; the stateful per-pixel
// copy writes both addresses per read (dataflow_kernel_lib::read_with_state fixes the source).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "brf_geom.hpp"

constexpr uint32_t CB_SLAB = 0;
constexpr uint32_t CB_STAGE = 1;
constexpr uint32_t TRID_CHECK_EVERY = 64;
constexpr uint32_t TRID_MIN_FREE = (NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2;

void kernel_main() {
    constexpr uint32_t variant = get_compile_time_arg_val(0);
    using G = brf::Geom<
        get_compile_time_arg_val(1),
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7)>;
    constexpr uint32_t pb = get_compile_time_arg_val(8);
    constexpr uint32_t band_cap = get_compile_time_arg_val(9);
    constexpr uint32_t CH = get_compile_time_arg_val(10);
    constexpr uint32_t NSTG = get_compile_time_arg_val(11);
    constexpr auto in_args = TensorAccessorArgs<12>();
    constexpr bool dram = variant >= 1 && variant <= 4;
    constexpr bool noc_deint = variant == 2 || variant == 5;
    constexpr bool risc_deint = variant == 3 || variant == 6;
    constexpr bool trisc = variant == 4 || variant == 7;
    constexpr uint32_t slot_rows = CH / 8;
    constexpr uint32_t slot_bytes = slot_rows * pb;
    constexpr uint32_t chunk_bytes = CH * pb;
    constexpr uint32_t chunk_pages = chunk_bytes / 2048;  // cb_stage pages are 2 KB (one bf16 tile)
    constexpr uint32_t slab_pages = 3 * band_cap * pb / 2048;
    static_assert(CH % 256 == 0 && chunk_bytes % 2048 == 0, "chunks are whole 256-pixel groups");

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_tile_start = get_arg_val<uint32_t>(1);
    const uint32_t actual_m_tiles = get_arg_val<uint32_t>(2);
    const auto input = TensorAccessor(in_args, in_addr, pb);

    const uint32_t stage_base = get_write_ptr(CB_STAGE);
    const uint32_t slab_base = get_write_ptr(CB_SLAB);
    const uint32_t noc = noc_index;

    const uint32_t num_blocks = (actual_m_tiles + G::bm - 1) / G::bm;
    uint32_t packets = 0;
    auto throttle = [&](uint32_t trid) {
        if (++packets >= TRID_CHECK_EVERY) {
            packets = 0;
            while (noc_available_transactions(noc, trid) < TRID_MIN_FREE) {
            }
        }
    };

    if constexpr (variant == 0) {
        for (uint32_t m = 0; m < num_blocks; ++m) {
            brf::Bands b = G::bands(m_tile_start, actual_m_tiles, m);
            cb_reserve_back(CB_SLAB, slab_pages);
            for (uint32_t kh = 0; kh < 3; ++kh) {
                noc_async_read_set_trid(kh + 1, noc);
                packets = 0;
                uint32_t l1 = slab_base + kh * band_cap * pb;
                for (int32_t v = b.lo[kh]; v < b.hi[kh]; ++v) {
                    noc_async_read(input.get_noc_addr(v), l1, pb);
                    l1 += pb;
                    throttle(kh + 1);
                }
            }
            for (uint32_t kh = 0; kh < 3; ++kh) {
                noc_async_read_barrier_with_trid(kh + 1, noc);
            }
            cb_push_back(CB_SLAB, slab_pages);
        }
        noc_async_read_set_trid(0, noc);
        return;
    }

    // Chunk walk: blocks x bands x chunks of CH pixels, in order; two cursors (issue runs ahead of process).
    brf::ChunkCursor<G, CH> issue_c, proc_c;
    issue_c.start(m_tile_start, actual_m_tiles);
    proc_c.start(m_tile_start, actual_m_tiles);

    // DRAM trid of staging slot q: 1 + q; local-copy trid: 1 + NSTG + q.
    auto issue_chunk = [&](uint32_t slot) {
        const uint32_t base = stage_base + slot * chunk_bytes;
        noc_async_read_set_trid(1 + slot, noc);
        const int32_t a = issue_c.a, e = issue_c.e;
        for (int32_t s = 0; s < 8 && a + s < e; ++s) {
            const uint32_t run = static_cast<uint32_t>(e - a - s + 7) / 8;
            noc_async_read(input.get_noc_addr(a + s), base + s * slot_bytes, run * pb);
        }
    };

    if constexpr (trisc) {
        // The staging CB is NSTG chunks; the reader keeps up to NSTG - 1 chunks reserved and in flight. Compute pops.
        uint32_t issued = 0, pushed = 0;
        uint32_t wr_slot = 0;  // slot of the oldest reserved, unpushed chunk
        while (!proc_c.done) {
            while (!issue_c.done && issued - pushed < NSTG - 1) {
                cb_reserve_back(CB_STAGE, (issued - pushed + 1) * chunk_pages);
                if constexpr (dram) {
                    issue_chunk((wr_slot + issued - pushed) % NSTG);
                }
                ++issued;
                issue_c.next();
            }
            if constexpr (dram) {
                noc_async_read_barrier_with_trid(1 + wr_slot, noc);
            }
            cb_push_back(CB_STAGE, chunk_pages);
            ++pushed;
            wr_slot = (wr_slot + 1) % NSTG;
            proc_c.next();
        }
        noc_async_read_set_trid(0, noc);
        return;
    }

    // Variants 1-3, 5, 6: the reader de-interleaves into the slab (cb_slab: one slab per block, the writer pops).
    uint32_t issued = 0, processed = 0;
    int32_t cur_block = -1;
    auto deint_noc = [&](uint32_t slot, uint32_t dst_px) {
        const uint32_t base = stage_base + slot * chunk_bytes;
        noc_async_read_set_trid(1 + NSTG + slot, noc);
        packets = 0;
        const int32_t n = proc_c.e - proc_c.a;
        ncrisc_noc_read_set_state<DM_DEDICATED_NOC, true>(noc, NCRISC_RD_CMD_BUF, get_noc_addr(base), pb);
        for (int32_t s = 0; s < 8 && s < n; ++s) {
            uint32_t src = base + s * slot_bytes;
            uint32_t dst = slab_base + (dst_px + s) * pb;
            for (int32_t q = s; q < n; q += 8) {
                ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, true>(noc, NCRISC_RD_CMD_BUF, src, dst);
                src += pb;
                dst += 8 * pb;
                throttle(1 + NSTG + slot);
            }
        }
    };
    auto deint_risc = [&](uint32_t slot, uint32_t dst_px) {
        const uint32_t base = stage_base + slot * chunk_bytes;
        invalidate_l1_cache();
        const int32_t n = proc_c.e - proc_c.a;
        for (int32_t s = 0; s < 8 && s < n; ++s) {
            const volatile uint32_t* src = reinterpret_cast<const volatile uint32_t*>(base + s * slot_bytes);
            volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(slab_base + (dst_px + s) * pb);
            for (int32_t q = s; q < n; q += 8) {
#pragma GCC unroll 64
                for (uint32_t w = 0; w < pb / 4; w += 4) {
                    const uint32_t x0 = src[w], x1 = src[w + 1], x2 = src[w + 2], x3 = src[w + 3];
                    dst[w] = x0;
                    dst[w + 1] = x1;
                    dst[w + 2] = x2;
                    dst[w + 3] = x3;
                }
                src += pb / 4;
                dst += 8 * pb / 4;
            }
        }
    };

    for (uint32_t i = 0; i < NSTG - 1 && !issue_c.done; ++i) {
        if constexpr (dram) {
            issue_chunk(issued % NSTG);
        }
        ++issued;
        issue_c.next();
    }
    while (!proc_c.done) {
        if (static_cast<int32_t>(proc_c.m) != cur_block) {
            if (cur_block >= 0) {
                if constexpr (noc_deint) {
                    for (uint32_t q = 0; q < NSTG; ++q) {
                        noc_async_read_barrier_with_trid(1 + NSTG + q, noc);
                    }
                }
                cb_push_back(CB_SLAB, slab_pages);
            }
            cb_reserve_back(CB_SLAB, slab_pages);
            cur_block = proc_c.m;
        }
        const uint32_t slot = processed % NSTG;
        if constexpr (dram) {
            noc_async_read_barrier_with_trid(1 + slot, noc);
        }
        const uint32_t dst_px = proc_c.kh * band_cap + static_cast<uint32_t>(proc_c.a - proc_c.band_lo);
        if constexpr (noc_deint) {
            deint_noc(slot, dst_px);
        } else if constexpr (risc_deint) {
            deint_risc(slot, dst_px);
        }
        ++processed;
        proc_c.next();
        // Refill the slot processed one step earlier (its copies had a whole chunk's time to land).
        if (!issue_c.done) {
            const uint32_t free_slot = issued % NSTG;
            if constexpr (noc_deint) {
                noc_async_read_barrier_with_trid(1 + NSTG + free_slot, noc);
            }
            if constexpr (dram) {
                issue_chunk(free_slot);
            }
            ++issued;
            issue_c.next();
        }
    }
    if constexpr (noc_deint) {
        for (uint32_t q = 0; q < NSTG; ++q) {
            noc_async_read_barrier_with_trid(1 + NSTG + q, noc);
        }
    }
    if (cur_block >= 0) {
        cb_push_back(CB_SLAB, slab_pages);
    }
    noc_async_read_set_trid(0, noc);
}
