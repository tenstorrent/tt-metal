// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_to_all_matmul compute-core reader (RISCV_1 / NoC1: keeps NoC0 free for the port readers).
//
// Per head slot of this core, in slot order (local-source heads first, then remote heads by source-local index j, then
// helper slots of the tail split):
//   load_weight_block(slot, nb)  Kt x block_n weight tiles -> cb_weight for the first w_prefetch_blocks n-blocks
//                                (issued before the slot lands: the dominant DRAM read fills the fabric wait),
//                                weight_chunk_tiles reads in flight at a time
//   land_block(slot)             cb_a_landed is backed on the persistent landing scratch (zero-copy): slot s sits at
//                                base + s * slot_bytes, where the remote port writes it.
//                                local head:  read the own quarter of the head's rows (all batches, all K) from DRAM
//                                remote head: wait for the slot's arrival counter (fused inc on the entry's last
//                                             packet), re-arm it (-1)
//                                then push the slot (compute retains it across the head's n-blocks)
//   forward (tail split)         a late-landing head's rows are copied to a helper core's slot (+ arrival inc), which
//                                multiplies the head's other n-blocks: this core keeps [nb0, nb1) of the head
//   load_weight_block(slot, nb)  the head's remaining n-blocks of [nb0, nb1) (beyond the prefetch)
//
// Regime R1' (Mq % 32 != 0): the local quarter is re-tiled per image through cb_retile_stage
// (all_to_all_matmul_retile.hpp), exactly as the port reader does for the remote quarters.
//
// Slot tile order (b, mt, k) = matmul_block's in0 order (rows of Kt tiles).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_to_all_matmul_retile.hpp"

#if defined(ABLATE_ZONES)
#include "tools/profiler/kernel_profiler.hpp"
#define A2AMM_ZONE(n) DeviceZoneScopedN(n)
#else
#define A2AMM_ZONE(n)
#endif

void kernel_main() {
    constexpr uint32_t cb_a_landed = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Nt = get_compile_time_arg_val(3);
    constexpr uint32_t block_n = get_compile_time_arg_val(4);
    [[maybe_unused]] constexpr uint32_t num_n_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t B = get_compile_time_arg_val(6);
    constexpr uint32_t Mq_t = get_compile_time_arg_val(7);
    constexpr uint32_t St = get_compile_time_arg_val(8);
    constexpr uint32_t Hl = get_compile_time_arg_val(9);
    constexpr uint32_t in_tile_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t Mq = get_compile_time_arg_val(12);  // rows per device
    constexpr uint32_t cb_retile_stage = get_compile_time_arg_val(13);
    // Weight reads in flight per core (OVERLAP, Refinement 3): one barrier per weight_chunk_tiles reads. 96 cores
    // each flooding the NoC with a whole 128 KiB block stretched the fabric exchange by ~20 us (Kimi); a few tiles in
    // flight per core keep DRAM busy without starving the landing / port traffic.
    constexpr uint32_t weight_chunk_tiles = get_compile_time_arg_val(14);
    static_assert(weight_chunk_tiles >= 1, "weight_chunk_tiles >= 1");
    // Weight blocks of a slot read before its landing wait (<= cb_weight depth): the rest follow the landing.
    constexpr uint32_t w_prefetch_blocks = get_compile_time_arg_val(15);
    static_assert(w_prefetch_blocks >= 1, "w_prefetch_blocks >= 1");
    // Weight-read NoC split (OVERLAP, landing_contention): the fabric router writes the landing rows into compute
    // L1 on NoC1 (edm_to_local_chip_noc) and the port senders reach the router on NoC1, so weight responses on NoC1
    // share the row links with the landing path. Weight tile i of a block goes on the OTHER NoC (the writer's) when
    // (i % wnoc_den) < wnoc_num; wnoc_den = 0 keeps every weight read on this kernel's NoC. A slot whose pace word has
    // bit 31 set reads all of its weights on the other NoC. Only READS go on the other NoC: the writer issues only
    // writes there, so the two RISCs never share a transaction counter (read-response counter vs write counters).
    constexpr uint32_t wnoc_den = get_compile_time_arg_val(16);
    constexpr uint32_t wnoc_num = get_compile_time_arg_val(17);
    // Weight-read rate cap (OVERLAP, landing_contention): the weight chunks a core reads while the exchange is still
    // landing (every block of a local slot, the pre-landing blocks of a remote slot) start at least wgap_cycles apart
    // (a schedule from the kernel start: a late chunk does not push the later ones back). The ~25 us weight burst at
    // DRAM peak loads the NoC1 rows that the router's landing writes also cross; spread over the exchange, the same
    // bytes load those rows less. Post-landing blocks of a remote slot (on the tail) are never capped. 0 = off.
    // The cap holds only for chunks scheduled within the first wcap_window cycles of the kernel (cores with many
    // pre-landing chunks, e.g. a local head + tail helper slots, would otherwise be stretched past the exchange).
    constexpr uint32_t wgap_cycles = get_compile_time_arg_val(18);
    constexpr uint32_t wcap_window = get_compile_time_arg_val(19);
    // DRAM endpoint of the weight reads (OVERLAP, landing_contention). Every Blackhole DRAM channel has 3 NoC
    // endpoints in one DRAM column; the NoC1 endpoint the bank table picks for the first channel of each column sits on
    // physical row 1, the Ethernet row. NoC1 routes north then west, so read requests to those two channels run west
    // along row 1, which the port senders' writes to their router and the routers' 2-hop forwarding also cross; with
    // DRAM saturated by the weight reads, the requests back up there. wdram_ytrig / wdram_yalt (NoC1 coordinates, from
    // the host; 0xFF = off): a weight read whose endpoint y is wdram_ytrig goes to the same channel's endpoint at
    // wdram_yalt instead (same memory, another endpoint of the same DRAM tile).
    constexpr uint32_t wdram_port = get_compile_time_arg_val(20);  // (ytrig | yalt << 8), 0xFFFF = off
    constexpr uint32_t wdram_ytrig = wdram_port & 0xFF, wdram_yalt = (wdram_port >> 8) & 0xFF;
    // Static VC of this RISC's read REQUESTS on its NoC (the port senders' writes use NOC_UNICAST_WRITE_VC = 1). In
    // dedicated-NoC mode the read command buffer's VC is preset by firmware (noc_init), so it is set here and restored
    // at the end.
    constexpr uint32_t wreq_vc = get_compile_time_arg_val(21);
    constexpr bool retile = (Mq % 32) != 0;
    constexpr auto in_args = TensorAccessorArgs<22>();
    constexpr auto w_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr uint32_t quarter_tiles = Mq_t * Kt;       // one batch's rows of one head quarter
    constexpr uint32_t slot_tiles = B * quarter_tiles;  // = block_m * Kt
    constexpr uint32_t weight_block_tiles = Kt * block_n;

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t my_q = get_common_arg_val<uint32_t>(2);  // this chip's position in its group
    constexpr uint32_t arrival_base = 3;                    // arrival counter address of slot s
    // Per slot: (h | wait << 16), (nb0 | nb1 << 16), forward, helper x, helper y, helper slot addr, helper arrival
    // addr, weight pace (cycles to wait before the slot's first weight read; 0 = none), weight blocks read before the
    // landing wait (tail M-split piece slots read all of theirs; else w_prefetch_blocks)
    constexpr uint32_t SLOT_WORDS = 8;

    const uint32_t n_slots = get_arg_val<uint32_t>(0);

    const auto in = TensorAccessor(in_args, in_addr, in_tile_bytes);
    const auto w = TensorAccessor(w_args, w_addr, w_tile_bytes);

    auto set_rd_vc = [](uint32_t vc) {
        NOC_CMD_BUF_WRITE_REG(
            noc_index,
            read_cmd_buf,
            NOC_CTRL,
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc));
    };
    if constexpr (wreq_vc != NOC_UNICAST_WRITE_VC) {
        set_rd_vc(wreq_vc);
    }
    constexpr uint8_t other_noc = 1 - noc_index;
    // This RISC's software read counter of the other NoC starts from the NIU's count (firmware inits only noc_index).
    noc_reads_num_issued[other_noc] = NOC_STATUS_READ_REG(other_noc, NIU_MST_RD_RESP_RECEIVED);
    bool slot_other = false;  // this slot's weights all on the other NoC (pace word bit 31)
    bool slot_skip = false;   // measurement only: this slot's weight reads are stubbed (pace word bit 30)
    bool core_east = false;   // this core sits east of the middle DRAM column (pace word bit 29)
    auto wnoc_of = [&](uint32_t i) -> uint8_t {
        if (slot_other) {
            return other_noc;
        }
        if constexpr (wnoc_den > 0 && wnoc_den != 255) {
            return (i % wnoc_den) < wnoc_num ? other_noc : noc_index;
        }
        return noc_index;
    };
    auto w_barrier = [&]() {
        noc_async_read_barrier(noc_index);
        if (slot_other || wnoc_den > 0) {
            noc_async_read_barrier(other_noc);
        }
    };

    volatile tt_reg_ptr uint32_t* wall_clock =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t cap_start = wall_clock[0];
    uint32_t next_chunk_at = cap_start;
    auto cap_rate = [&](bool capped) {
        if constexpr (wgap_cycles > 0) {
            if (capped && next_chunk_at - cap_start < wcap_window) {
                while (static_cast<int32_t>(wall_clock[0] - next_chunk_at) < 0) {
                }
                next_chunk_at += wgap_cycles;
            }
        }
    };

    auto load_weight_block = [&](uint32_t h, uint32_t nb, bool capped) {
        cb_reserve_back(cb_weight, weight_block_tiles);
        const uint32_t wbase = get_write_ptr(cb_weight);
        const uint32_t page0 = h * Kt * Nt + nb * block_n;
        // Issue order rotated per head: every head's block starts on the same DRAM bank otherwise (Kt * Nt is a
        // multiple of the bank count), and all cores would queue on one bank at a time.
        uint32_t t = h % weight_block_tiles;
        uint32_t k = t / block_n, n = t - k * block_n;
        for (uint32_t i = 0; i < weight_block_tiles; ++i) {
            if (i % weight_chunk_tiles == 0) {
                cap_rate(capped);
            }
#ifndef ABLATE_NOWEIGHT
            if (!slot_skip) {
                const uint32_t page = page0 + k * Nt + n;
                uint8_t wn = wnoc_of(i);
                if constexpr (wnoc_den == 255) {
                    // Nearest-direction NoC per (core, DRAM column): NoC0 runs east from the DRAM column, NoC1 runs
                    // west from it, so the response never crosses the other DRAM column / the far half of the row.
                    const uint32_t dram_x = static_cast<uint32_t>(w.get_noc_addr(page, 0, 0) >> NOC_ADDR_LOCAL_BITS) &
                                            ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
                    wn = ((dram_x != 0) == core_east) ? 0 : 1;
                }
                uint64_t src = w.get_noc_addr(page, 0, wn);
                if constexpr (wdram_ytrig != 0xFF) {
                    constexpr uint32_t y_shift = NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS;
                    constexpr uint64_t y_mask = static_cast<uint64_t>((1u << NOC_ADDR_NODE_ID_BITS) - 1) << y_shift;
                    if (wn == 1 && ((src & y_mask) >> y_shift) == wdram_ytrig) {
                        src = (src & ~y_mask) | (static_cast<uint64_t>(wdram_yalt) << y_shift);
                    }
                }
                noc_async_read(src, wbase + (k * block_n + n) * w_tile_bytes, w_tile_bytes, wn, wreq_vc);
            }
#endif
            if constexpr (weight_chunk_tiles < weight_block_tiles) {
                if ((i + 1) % weight_chunk_tiles == 0) {
                    w_barrier();
                }
            }
            if (++n == block_n) {
                n = 0;
                if (++k == Kt) {
                    k = 0;
                }
            }
        }
        w_barrier();
        cb_push_back(cb_weight, weight_block_tiles);
    };

    for (uint32_t s = 0; s < n_slots; ++s) {
        const uint32_t rec = 1 + s * SLOT_WORDS;
        const uint32_t word = get_arg_val<uint32_t>(rec);
        const uint32_t h = word & 0xFFFF;
        const bool remote =
            ((word >> 16) & 0xFF) != 0;  // remote head, or a helper slot (rows forwarded by the primary)
        const uint32_t nbw = get_arg_val<uint32_t>(rec + 1);
        const uint32_t nb0 = nbw & 0xFF, nb1 = (nbw >> 8) & 0xFF;  // this core's n-blocks of the head
        const bool forward = get_arg_val<uint32_t>(rec + 2) != 0;

        const uint32_t pace_word = get_arg_val<uint32_t>(rec + 7);
        slot_other = (pace_word >> 31) != 0;
        slot_skip = ((pace_word >> 30) & 1) != 0;
        core_east = ((pace_word >> 29) & 1) != 0;
        const uint32_t pace = pace_word & 0x1FFFFFFF;
        if (pace != 0) {
            riscv_wait(pace);  // paced weight start (spreads the cores' weight bursts over the exchange)
        }
        const uint32_t pre = word >> 24;
        const uint32_t nb_pre = (nb1 - nb0 < pre) ? nb1 : nb0 + pre;
        {
            A2AMM_ZONE("rd_w0");
            for (uint32_t nb = nb0; nb < nb_pre; ++nb) {
                load_weight_block(h, nb, true);
            }
        }

        cb_reserve_back(cb_a_landed, slot_tiles);
        const uint32_t slot_l1 = get_write_ptr(cb_a_landed);
        if (remote) {
            A2AMM_ZONE("rd_wait_remote");
            volatile tt_l1_ptr uint32_t* arrived =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_val<uint32_t>(arrival_base + s));
            noc_semaphore_wait_min(arrived, 1);
            noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(arrived)), 0u - 1u);
            noc_async_atomic_barrier();
        } else {
            A2AMM_ZONE("rd_local_a");
            const uint32_t j = h - my_q * Hl;
            uint32_t ap = get_write_ptr(cb_a_landed);
            const auto qr = a2amm_retile::quarter_of<Mq>(my_q);  // aligned: t0 = my_q * Mq_t, shift 0
            for (uint32_t b = 0; b < B; ++b) {
                const uint32_t base = ((b * Hl + j) * St + qr.t0) * Kt;
                if constexpr (retile) {
                    const uint32_t stage = get_write_ptr(cb_retile_stage);  // scratch (never pushed)
                    a2amm_retile::stage_rows<in_tile_bytes, Kt>(in, base, qr.nsrc, stage);
                    for (uint32_t r = 0; r < quarter_tiles; ++r) {
                        a2amm_retile::retile_tile<in_tile_bytes, Kt, Mq>(stage, ap, qr.shift, r / Kt, r % Kt);
                        ap += in_tile_bytes;
                    }
                } else {
                    for (uint32_t r = 0; r < quarter_tiles; ++r) {
                        noc_async_read(in.get_noc_addr(base + r), ap, in_tile_bytes);
                        ap += in_tile_bytes;
                    }
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_a_landed, slot_tiles);

        if (forward) {
            // Tail split (OVERLAP): the helper core multiplies the head's other n-blocks. Copy the landed rows into
            // its helper slot (core-local NoC), then increment that slot's arrival counter. Read-only on this slot
            // while compute unpacks it; nothing rewrites the slot before the program ends.
            A2AMM_ZONE("rd_forward");
            const uint32_t fx = get_arg_val<uint32_t>(rec + 3), fy = get_arg_val<uint32_t>(rec + 4);
            noc_async_write(slot_l1, get_noc_addr(fx, fy, get_arg_val<uint32_t>(rec + 5)), slot_tiles * in_tile_bytes);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(fx, fy, get_arg_val<uint32_t>(rec + 6)), 1);
        }

        for (uint32_t nb = nb_pre; nb < nb1; ++nb) {
            load_weight_block(h, nb, !remote);
        }
    }
    noc_async_atomic_barrier();
    if constexpr (wreq_vc != NOC_UNICAST_WRITE_VC) {
        set_rd_vc(NOC_UNICAST_WRITE_VC);
    }
}
