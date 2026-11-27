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
// q-stem mode (RT > 0): input tiles are addressed with the strides of all_to_all_matmul_args.hpp (the q_b_proj
// layout); a local head's slot also copies the head's own-quarter rope tiles (RT per tile row) from the input into
// rope_seq through cb_rope_stage (the remote quarters' rope tiles come from the port cores).
//
// Slot tile order (b, mt, k) = matmul_block's in0 order (rows of Kt tiles).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_to_all_matmul_retile.hpp"
#include "ttnn/cpp/ttnn/operations/all_to_all_matmul/device/kernels/all_to_all_matmul_args.hpp"

using namespace all_to_all_matmul;

#if defined(ABLATE_ZONES)
#include "api/debug/kernel_profiler.hpp"
#define A2AMM_ZONE(n) DeviceZoneScopedN(n)
#else
#define A2AMM_ZONE(n)
#endif

void kernel_main() {
    constexpr uint32_t cb_a_landed = get_compile_time_arg_val(reader_ct::CB_A_LANDED);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(reader_ct::CB_WEIGHT);
    constexpr uint32_t Kt = get_compile_time_arg_val(reader_ct::KT);
    constexpr uint32_t Nt = get_compile_time_arg_val(reader_ct::NT);
    constexpr uint32_t block_n = get_compile_time_arg_val(reader_ct::BLOCK_N);
    [[maybe_unused]] constexpr uint32_t num_n_blocks = get_compile_time_arg_val(reader_ct::NUM_N_BLOCKS);
    constexpr uint32_t B = get_compile_time_arg_val(reader_ct::B);
    constexpr uint32_t Mq_t = get_compile_time_arg_val(reader_ct::MQ_T);
    constexpr uint32_t St = get_compile_time_arg_val(reader_ct::ST);
    constexpr uint32_t Hl = get_compile_time_arg_val(reader_ct::HL);
    constexpr uint32_t in_tile_bytes = get_compile_time_arg_val(reader_ct::IN_TILE_BYTES);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(reader_ct::W_TILE_BYTES);
    constexpr uint32_t Mq = get_compile_time_arg_val(reader_ct::MQ);  // rows per device
    constexpr uint32_t cb_retile_stage = get_compile_time_arg_val(reader_ct::CB_RETILE_STAGE);
    constexpr uint32_t in_batch_stride = get_compile_time_arg_val(reader_ct::IN_BATCH_STRIDE);
    constexpr uint32_t in_head_stride = get_compile_time_arg_val(reader_ct::IN_HEAD_STRIDE);
    constexpr uint32_t in_row_stride = get_compile_time_arg_val(reader_ct::IN_ROW_STRIDE);
    constexpr uint32_t Rt = get_compile_time_arg_val(reader_ct::RT);  // q-stem: rope tiles per row; 0 = off
    constexpr uint32_t RtD = Rt > 0 ? Rt : 1;                         // divisor (the rope code is dead when Rt = 0)
    constexpr uint32_t H = get_compile_time_arg_val(reader_ct::H);
    constexpr uint32_t cb_rope_stage = get_compile_time_arg_val(reader_ct::CB_ROPE_STAGE);
    constexpr uint32_t rope_stage_tiles = get_compile_time_arg_val(reader_ct::ROPE_STAGE_TILES);
    static_assert(Rt == 0 || (rope_stage_tiles >= 1 && (Mq % 32) == 0), "q-stem: stage tiles, tile-aligned rows");
    static_assert(Rt > 0 || (in_row_stride == Kt && in_head_stride == St * Kt), "plain input is (B, Hl, S, K)");
    // Weight reads in flight per core (OVERLAP, Refinement 3): one barrier per weight_chunk_tiles reads. 96 cores
    // each flooding the NoC with a whole 128 KiB block stretched the fabric exchange by ~20 us (Kimi); a few tiles in
    // flight per core keep DRAM busy without starving the landing / port traffic.
    constexpr uint32_t weight_chunk_tiles = get_compile_time_arg_val(reader_ct::WEIGHT_CHUNK_TILES);
    static_assert(weight_chunk_tiles >= 1, "weight_chunk_tiles >= 1");
    // Weight blocks of a slot read before its landing wait (<= cb_weight depth): the rest follow the landing.
    constexpr uint32_t w_prefetch_blocks = get_compile_time_arg_val(reader_ct::W_PREFETCH_BLOCKS);
    static_assert(w_prefetch_blocks >= 1, "w_prefetch_blocks >= 1");
    constexpr bool retile = (Mq % 32) != 0;
    constexpr auto in_args = TensorAccessorArgs<reader_ct::COUNT>();
    constexpr auto w_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto rope_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();  // RT = 0: unused
    constexpr uint32_t quarter_tiles = Mq_t * Kt;       // one batch's rows of one head quarter
    constexpr uint32_t slot_tiles = B * quarter_tiles;  // = block_m * Kt
    constexpr uint32_t weight_block_tiles = Kt * block_n;

    const uint32_t in_addr = get_common_arg_val<uint32_t>(reader_common::IN_ADDR);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(reader_common::W_ADDR);
    const uint32_t my_q = get_common_arg_val<uint32_t>(reader_common::MY_Q);  // this chip's position in its group
    const uint32_t landing_addr = get_common_arg_val<uint32_t>(reader_common::LANDING_ADDR);
    constexpr uint32_t arrival_base = reader_common::ARRIVAL_BASE;  // arrival counter address of slot s
    // Per slot: (h | wait << 16), (nb0 | nb1 << 16), forward, helper x, helper y, helper slot addr, helper arrival
    // addr, weight pace (cycles to wait before the slot's first weight read; 0 = none), weight blocks read before the
    // landing wait (tail M-split piece slots read all of theirs; else w_prefetch_blocks)
    constexpr uint32_t SLOT_WORDS = reader_slot::COUNT;

    const uint32_t n_slots = get_arg_val<uint32_t>(reader_rt::N_SLOTS);

    const auto in = TensorAccessor(in_args, in_addr, in_tile_bytes);
    const auto w = TensorAccessor(w_args, w_addr, w_tile_bytes);
    const auto rope = TensorAccessor(rope_args, get_common_arg_val<uint32_t>(reader_common::ROPE_ADDR), in_tile_bytes);

    // q-stem: head h (source-local j, this chip's quarter) rope tiles (b, mt, rt) -> rope_seq page
    // ((b * H + h) * Mq_t + mt) * Rt + rt, staged rope_stage_tiles at a time.
    auto copy_rope = [&](uint32_t h, uint32_t j) {
        constexpr uint32_t total = B * Mq_t * Rt;
        const uint32_t stage = get_write_ptr(cb_rope_stage);  // scratch (never pushed)
        const uint32_t row0 = my_q * Mq_t;
        for (uint32_t i0 = 0; i0 < total; i0 += rope_stage_tiles) {
            const uint32_t n = (total - i0) < rope_stage_tiles ? (total - i0) : rope_stage_tiles;
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t e = i0 + i;
                const uint32_t rt = e % RtD;
                const uint32_t mt = (e / RtD) % Mq_t;
                const uint32_t b = e / (RtD * Mq_t);
                const uint32_t page = b * in_batch_stride + j * in_head_stride + (row0 + mt) * in_row_stride + Kt + rt;
                noc_async_read(in.get_noc_addr(page), stage + i * in_tile_bytes, in_tile_bytes);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t e = i0 + i;
                const uint32_t b = e / (RtD * Mq_t);
                const uint32_t page = (b * H + h) * Mq_t * Rt + (e - b * RtD * Mq_t);
                noc_async_write(stage + i * in_tile_bytes, rope.get_noc_addr(page), in_tile_bytes);
            }
            noc_async_writes_flushed();  // the stage is reused by the next chunk
        }
    };

    auto load_weight_block = [&](uint32_t h, uint32_t nb) {
        cb_reserve_back(cb_weight, weight_block_tiles);
        const uint32_t wbase = get_write_ptr(cb_weight);
        const uint32_t page0 = h * Kt * Nt + nb * block_n;
        // Issue order rotated per head: every head's block starts on the same DRAM bank otherwise (Kt * Nt is a
        // multiple of the bank count), and all cores would queue on one bank at a time.
        uint32_t t = h % weight_block_tiles;
        uint32_t k = t / block_n, n = t - k * block_n;
        for (uint32_t i = 0; i < weight_block_tiles; ++i) {
#ifndef ABLATE_NOWEIGHT
            noc_async_read(w.get_noc_addr(page0 + k * Nt + n), wbase + (k * block_n + n) * w_tile_bytes, w_tile_bytes);
#endif
            if constexpr (weight_chunk_tiles < weight_block_tiles) {
                if ((i + 1) % weight_chunk_tiles == 0) {
                    noc_async_read_barrier();
                }
            }
            if (++n == block_n) {
                n = 0;
                if (++k == Kt) {
                    k = 0;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_weight, weight_block_tiles);
    };

    for (uint32_t s = 0; s < n_slots; ++s) {
        const uint32_t rec = reader_rt::SLOTS_BASE + s * SLOT_WORDS;
        const uint32_t word = get_arg_val<uint32_t>(rec + reader_slot::WORD);
        const uint32_t h = word & reader_slot::HEAD_MASK;
        // remote head, or a helper slot (rows forwarded by the primary)
        const bool remote = ((word >> reader_slot::WAIT_SHIFT) & reader_slot::WAIT_MASK) != 0;
        const uint32_t nbw = get_arg_val<uint32_t>(rec + reader_slot::NBW);
        // this core's n-blocks of the head
        const uint32_t nb0 = (nbw >> nbw_bits::NB0_SHIFT) & nbw_bits::FIELD_MASK;
        const uint32_t nb1 = (nbw >> nbw_bits::NB1_SHIFT) & nbw_bits::FIELD_MASK;
        const bool forward = get_arg_val<uint32_t>(rec + reader_slot::FWD) != 0;

        const uint32_t pace = get_arg_val<uint32_t>(rec + reader_slot::PACE);
        if (pace != 0) {
            riscv_wait(pace);  // paced weight start (spreads the cores' weight bursts over the exchange)
        }
        const uint32_t pre = word >> reader_slot::PRE_SHIFT;
        const uint32_t nb_pre = (nb1 - nb0 < pre) ? nb1 : nb0 + pre;
        {
            A2AMM_ZONE("rd_w0");
            for (uint32_t nb = nb0; nb < nb_pre; ++nb) {
                load_weight_block(h, nb);
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
                const uint32_t base = b * in_batch_stride + j * in_head_stride + qr.t0 * in_row_stride;
                if constexpr (retile) {
                    const uint32_t stage = get_write_ptr(cb_retile_stage);  // scratch (never pushed)
                    a2amm_retile::stage_rows<in_tile_bytes, Kt>(in, base, qr.nsrc, stage);
                    for (uint32_t r = 0; r < quarter_tiles; ++r) {
                        a2amm_retile::retile_tile<in_tile_bytes, Kt, Mq>(stage, ap, qr.shift, r / Kt, r % Kt);
                        ap += in_tile_bytes;
                    }
                } else {
                    for (uint32_t mt = 0; mt < Mq_t; ++mt) {
                        const uint32_t row = base + mt * in_row_stride;
                        for (uint32_t k = 0; k < Kt; ++k) {
                            noc_async_read(in.get_noc_addr(row + k), ap, in_tile_bytes);
                            ap += in_tile_bytes;
                        }
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
            const uint32_t fx = get_arg_val<uint32_t>(rec + reader_slot::FWD_X);
            const uint32_t fy = get_arg_val<uint32_t>(rec + reader_slot::FWD_Y);
            const uint32_t fwd_slot = landing_addr + get_arg_val<uint32_t>(rec + reader_slot::FWD_SLOT_OFFSET);
            noc_async_write(slot_l1, get_noc_addr(fx, fy, fwd_slot), slot_tiles * in_tile_bytes);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(fx, fy, get_arg_val<uint32_t>(rec + reader_slot::FWD_ARRIVAL_ADDR)), 1);
        }

        for (uint32_t nb = nb_pre; nb < nb1; ++nb) {
            load_weight_block(h, nb);
        }
        if constexpr (Rt > 0) {
            if (!remote) {  // while compute multiplies the slot
                A2AMM_ZONE("rd_local_rope");
                copy_rope(h, h - my_q * Hl);
            }
        }
    }
    noc_async_atomic_barrier();
    if constexpr (Rt > 0) {
        noc_async_write_barrier();  // the local rope copy has landed before the program ends
    }
}
