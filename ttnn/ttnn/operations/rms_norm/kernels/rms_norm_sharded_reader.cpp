// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for rms_norm — WIDTH_SHARDED / BLOCK_SHARDED cross-core reduction
// (R5 TILE; R5a adds the ROW_MAJOR leg).
//
// This is the dataflow face of the design's dependent-axis SCHEME-CHANGE: the
// hidden W is split across a reduction GROUP of cores, so the RMS denominator
// spans core boundaries. Each core owns a contiguous W-slice of its tile-rows
// (a resident L1 shard). Per tile-row the flow is:
//
//   PASS 1  : stream the LOCAL W-slice -> cb_input_tiles (TILE) or cb_input_rm
//             (ROW_MAJOR; compute tilizes). Compute squares + reduces it into
//             cb_partial = local Σx² / W_global (this core's partial
//             contribution to the GLOBAL mean(x²)).
//   COMBINE : reduce-root gather + broadcast-back (this file's cross-core core):
//               * non-root: unicast cb_partial -> root's cb_gather[my_index],
//                 bump the root's `progress` semaphore, then act as a
//                 ReceiverPipe for the finalized 1/rms broadcast -> cb_sumsq.
//               * root: copy its own partial into cb_gather[my_index], wait for
//                 all (GROUP_SIZE-1) peers, push cb_gather (compute folds the
//                 partials -> global mean(x²) -> transform_in_place rsqrt ->
//                 cb_rms_src), then SenderPipe-broadcast cb_rms_src -> cb_sumsq
//                 on the whole group rectangle (loopback fills its own).
//   PASS 2  : re-stream the LOCAL W-slice + local gamma slice; compute
//             normalizes x·(1/rms)·gamma -> cb_output_tiles / cb_output_rm;
//             writer drains its own output shard.
//
// R5a ROW_MAJOR leg (this refinement): RM width-sharding splits each logical
// row's W across cores at sub-tile (stick) granularity, so a row is not
// contiguous in any one core's L1 and the TILE global-page addressing can't be
// used. Instead each core reads its OWN resident shard directly from local L1
// (buffer_address is the same L1 offset on every shard-grid core; the shard is
// `shard_h` sticks of `shard_w` elements, stride shard_w*elt) into cb_input_rm,
// zero-padding the sub-tile W-tail (and any H-tail) so compute always tilizes a
// clean full tile. The Σx² partial over the zero-padded tile equals the partial
// over the core's real columns (zeros add nothing) and ×1/W_global makes the
// group-sum the true global mean(x²) — no partial scaler needed. Gamma on this leg
// is read as a column-slice stick at the core's global W-offset (w_col_start),
// matching the input's local column layout: RM gamma directly (a linear stick read),
// TILE gamma (R5c, fp32/bf16) by extracting the containing gamma tile(s)' row-0
// sub-columns into the same cb_gamma_rm (see the GAMMA_TILE_EXTRACT pass-2 leg).
//
// The gather is many->one (raw unicast + a `progress` counter). The broadcast is
// one->many, chosen by group geometry:
//   * RECTANGULAR group (ncores == nx*ny): mcast_pipe SenderPipe/ReceiverPipe + a
//     host Mcast2D rectangle (the R5/R5a fast path).
//   * RAGGED WIDTH group (R5b — auto_shard_config pads a non-tile-aligned W into
//     ceil(W/w_gran) cores, overflowing a full row into a partial one): the bbox
//     holds phantom cores no mcast should touch, so the root UNICASTS 1/rms to each
//     member individually + a per-member ready flag (USE_UNICAST_BCAST). The non-root
//     ReceiverPipe is topology-agnostic and unchanged — it acks the root and waits its
//     own flag whether the root mcasts or unicasts.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_gamma_rm = 3;
constexpr uint32_t cb_gamma_tiles = 4;
constexpr uint32_t cb_gamma_src = 5;  // R5c: L1 scratch for the containing global gamma tile(s)
constexpr uint32_t cb_sumsq = 25;
constexpr uint32_t cb_partial = 27;
constexpr uint32_t cb_gather = 28;
constexpr uint32_t cb_rms_src = 29;
constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
// Arch DRAM read source alignment (Wormhole/Blackhole = 32B). The RM-gamma
// column-slice read at a sub-tile w_col_start can be sub-aligned, so it is read
// from the aligned-down base and shifted (see the RM gamma leg below).
constexpr uint32_t GAMMA_DRAM_ALIGN = 32;

// Zero an L1 page word-by-word (RM stick padding tail).
FORCE_INLINE void zero_l1_page(uint32_t l1, uint32_t nbytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < nbytes / 4; ++i) {
        p[i] = 0;
    }
}

// R5d: decode ONE bfloat8_b datum (block-float) to fp32 bits. Mirrors the host
// convert_bfp_to_u32 Bfp8_b branch (blockfloat_common.cpp): the shared exponent
// byte is already the fp32-biased exponent (bias 127 — NO rebias for the _b
// variant); the mantissa byte is [sign:1][magnitude:7] with an implied leading 1
// at bit 6. Normalize the magnitude (shift left until bit 6 is set, decrementing
// the exponent), drop the hidden bit, and assemble the fp32 word. bf8b -> fp32 is
// a lossless widening, so this reconstructs exactly what the hardware unpacker
// would produce (== the R2 INTERLEAVED bf8b-gamma FPU-unpack path).
FORCE_INLINE uint32_t bfp8b_datum_to_f32_bits(uint8_t shared_exp, uint8_t mantissa_byte) {
    uint32_t sign = static_cast<uint32_t>(mantissa_byte) >> 7;
    uint32_t man = static_cast<uint32_t>(mantissa_byte) & 0x7f;
    if (man == 0) {
        return sign << 31;  // signed zero
    }
    uint32_t exp = shared_exp;
    while ((man & 0x40) == 0) {
        man <<= 1;
        --exp;  // exp = shared_exp - shift_cnt
    }
    man = (man << 1) & 0x7f;  // drop the hidden leading 1
    return (sign << 31) | (exp << 23) | (man << 16);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(0);
    constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(1);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(2);  // 1/W_global bits
    constexpr uint32_t origin_W = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);  // GLOBAL tiled W stride (TILE leg)
    constexpr uint32_t W_BLOCK_TILES = get_compile_time_arg_val(5);
    constexpr uint32_t num_w_blocks = get_compile_time_arg_val(6);  // LOCAL W-block count
    constexpr uint32_t gamma_elt = get_compile_time_arg_val(7);
    constexpr uint32_t GAMMA_IS_ROW_MAJOR = get_compile_time_arg_val(8);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(9);
    constexpr uint32_t progress_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t shard_h = get_compile_time_arg_val(11);    // Hs (RM leg): shard rows
    constexpr uint32_t shard_w = get_compile_time_arg_val(12);    // Ws (RM leg): shard cols
    constexpr uint32_t input_elt = get_compile_time_arg_val(13);  // input element bytes (RM leg)
    // R5b: ragged WIDTH group -> broadcast 1/rms by UNICAST (root -> each member) instead
    // of one mcast to the bbox rectangle (which would hit phantom cores). The gather leg
    // and the non-root ReceiverPipe are topology-agnostic and unchanged.
    constexpr uint32_t USE_UNICAST_BCAST = get_compile_time_arg_val(14);
    // R5c/R5d: RM cross-core + TILE gamma. This core owns a SUB-TILE global W-column
    // offset (w_col_start), so a TILE-stored gamma can't be read as whole tiles aligned
    // to local col 0. When set, the reader extracts the gamma's ROW 0 sub-columns from the
    // containing global gamma tile(s) into cb_gamma_rm (see pass-2 gamma leg), and the SAME
    // RM-gamma compute tilize leg (GAMMA_IS_ROW_MAJOR=1 on the compute side) produces
    // cb_gamma_tiles unchanged. Three-valued: 0 = off; 1 = fp32/bf16 face-aware byte copy
    // (R5c); 2 = bf8b block-float dequant of the row-0 sub-columns into the float
    // cb_gamma_rm (R5d — a bf8b tile shares one exponent per 16 elements, so a byte copy is
    // meaningless; the reader decodes each datum via bfp8b_datum_to_f32_bits).
    constexpr uint32_t GAMMA_TILE_EXTRACT = get_compile_time_arg_val(15);
    // mcast wire (host Mcast2D) at CT 16; TensorAccessors chained after it.
    constexpr auto mc = McastArgs</*CT=*/16, /*RT=*/10>();
    constexpr uint32_t ta_ct_base = mc.next_compile_time_args_offset();
    // Ragged unicast broadcast: the root reads its group's member virtual coords from the
    // RT tail (2 words each, slot order), starting right after the 4-word mcast RT block.
    constexpr uint32_t member_rt_base = mc.next_runtime_args_offset();
    constexpr auto input_args = TensorAccessorArgs<ta_ct_base>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    uint32_t input_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    uint32_t start_tile_row = get_arg_val<uint32_t>(2);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(3);
    uint32_t w_tile_start = get_arg_val<uint32_t>(4);  // this core's W-tile offset (TILE leg)
    uint32_t my_index = get_arg_val<uint32_t>(5);      // this core's slot in the reduction group
    uint32_t is_root = get_arg_val<uint32_t>(6);
    uint32_t root_x = get_arg_val<uint32_t>(7);  // root virtual coords (gather target)
    uint32_t root_y = get_arg_val<uint32_t>(8);
    [[maybe_unused]] uint32_t w_col_start = get_arg_val<uint32_t>(9);  // this core's global W-col start (RM gamma)

    // Reduce scaler (SUM AccumulateViaAdd does not consume it, but the compute
    // reduce<> template references cb_scaler; prepare it once so the CB is valid).
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scaler_f, TILE_W);

    constexpr uint32_t wblock_cols = W_BLOCK_TILES * TILE_W;
    constexpr uint32_t in_padded_bytes = wblock_cols * input_elt;     // RM stick page (input)
    constexpr uint32_t shard_row_bytes = shard_w * input_elt;         // RM local shard stride
    constexpr uint32_t gamma_padded_bytes = wblock_cols * gamma_elt;  // RM stick page (gamma)
    const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
    const uint32_t sumsq_bytes = get_tile_size(cb_sumsq);
    const auto in_acc = TensorAccessor(input_args, input_addr, tile_bytes);

    Noc noc;
    Semaphore<> progress(progress_sem_id);

    // Read one W-block of the LOCAL RM shard into cb_input_rm (32 stick-pages).
    // Reads the core's OWN resident shard from local L1 (self NoC read): stick
    // for local row `lr` is at input_addr + lr*shard_row_bytes + block col offset.
    // Rows >= num_real_rows (H-tail) and the sub-tile W-tail are zero-padded so
    // compute tilizes a clean full tile.
    auto read_local_input_wblock = [&](uint32_t local_row_base, uint32_t num_real_rows, uint32_t b) {
        uint32_t cols = shard_w - b * wblock_cols;  // real cols of this block within the shard
        if (cols > wblock_cols) {
            cols = wblock_cols;
        }
        const uint32_t real_bytes = cols * input_elt;
        const uint32_t col_off = b * wblock_cols * input_elt;
        const bool w_tail = real_bytes < in_padded_bytes;
        for (uint32_t r = 0; r < TILE_H; ++r) {
            cb_reserve_back(cb_input_rm, 1);
            uint32_t l1 = get_write_ptr(cb_input_rm);
            const bool is_real = r < num_real_rows;
            if (!is_real || w_tail) {
                zero_l1_page(l1, in_padded_bytes);
            }
            if (is_real) {
                uint32_t src = input_addr + (local_row_base + r) * shard_row_bytes + col_off;
                noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], src), l1, real_bytes);
                noc_async_read_barrier();
            }
            cb_push_back(cb_input_rm, 1);
        }
    };

    // Read one W-block of the LOCAL W-slice as TILES into cb_input_tiles (TILE leg).
    auto read_tile_input_wblock = [&](uint32_t tr, uint32_t b) {
        cb_reserve_back(cb_input_tiles, W_BLOCK_TILES);
        uint32_t wp = get_write_ptr(cb_input_tiles);
        uint32_t base_tile = tr * Wt + w_tile_start + b * W_BLOCK_TILES;
        for (uint32_t wt = 0; wt < W_BLOCK_TILES; ++wt) {
            noc_async_read_page(base_tile + wt, in_acc, wp + wt * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, W_BLOCK_TILES);
    };

    for (uint32_t t = 0; t < num_tile_rows; ++t) {
        uint32_t tr = start_tile_row + t;
        // RM leg: local shard row base for this tile-row + real (non-padding) rows.
        uint32_t local_row_base = t * TILE_H;
        uint32_t num_real_rows = shard_h > local_row_base ? (shard_h - local_row_base) : 0;
        if (num_real_rows > TILE_H) {
            num_real_rows = TILE_H;
        }

        // ---------- PASS 1: stream local W-slice ----------
        for (uint32_t b = 0; b < num_w_blocks; ++b) {
            if constexpr (IS_ROW_MAJOR) {
                read_local_input_wblock(local_row_base, num_real_rows, b);
            } else {
                read_tile_input_wblock(tr, b);
            }
        }

        // ---------- COMBINE: reduce-root gather + broadcast-back ----------
        cb_wait_front(cb_partial, 1);
        uint32_t partial_rp = get_read_ptr(cb_partial);
        if (is_root) {
            cb_reserve_back(cb_gather, GROUP_SIZE);
            uint32_t gather_wp = get_write_ptr(cb_gather);
            // copy own partial into its own gather slot (local L1 read from self)
            noc_async_read(
                get_noc_addr(my_x[noc_index], my_y[noc_index], partial_rp),
                gather_wp + my_index * sumsq_bytes,
                sumsq_bytes);
            noc_async_read_barrier();
            cb_pop_front(cb_partial, 1);
            // wait for all peers' partials to land, then reset for the next round
            if constexpr (GROUP_SIZE > 1) {
                progress.wait(GROUP_SIZE - 1);
                progress.set(0);
            }
            cb_push_back(cb_gather, GROUP_SIZE);  // compute folds -> cb_rms_src

            // broadcast the finalized 1/rms to the whole group
            cb_wait_front(cb_rms_src, 1);
            uint32_t rms_rp = get_read_ptr(cb_rms_src);
            cb_reserve_back(cb_sumsq, 1);
            uint32_t sumsq_wp = get_write_ptr(cb_sumsq);
            if constexpr (USE_UNICAST_BCAST) {
                // R5b ragged group: the mcast rectangle would hit phantom bbox cores, so
                // UNICAST 1/rms to each member individually + a per-member ready flag —
                // the SAME handshake the non-root ReceiverPipe expects (cross_core_
                // reduction_design.md §8 option 3). Mirrors the unicast gather leg.
                Semaphore<> data_ready(mc.data_ready);          // members' S->R data-ready flag
                Semaphore<> consumer_ready(mc.consumer_ready);  // members' R->S readiness ack
                if constexpr (mc.pre_handshake) {
                    if (GROUP_SIZE > 1) {
                        // gate the send on every non-root member having drained its cb_sumsq
                        consumer_ready.wait(GROUP_SIZE - 1);
                        consumer_ready.set(0);
                    }
                }
                // fill own cb_sumsq (the root is a group member) via a local self-read
                noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], rms_rp), sumsq_wp, sumsq_bytes);
                noc_async_read_barrier();
                // unicast 1/rms to each other member's cb_sumsq (identical L1 offset everywhere)
                for (uint32_t m = 0; m < GROUP_SIZE; ++m) {
                    if (m == my_index) {
                        continue;
                    }
                    uint32_t mx = get_arg_val<uint32_t>(member_rt_base + 2 * m + 0);
                    uint32_t my = get_arg_val<uint32_t>(member_rt_base + 2 * m + 1);
                    noc_async_write(rms_rp, get_noc_addr(mx, my, sumsq_wp), sumsq_bytes);
                }
                noc_async_write_barrier();  // data lands before any flag (§9 barrier-before-signal)
                // raise each other member's data-ready flag (Flag: 0 -> VALID; receiver clears it)
                for (uint32_t m = 0; m < GROUP_SIZE; ++m) {
                    if (m == my_index) {
                        continue;
                    }
                    uint32_t mx = get_arg_val<uint32_t>(member_rt_base + 2 * m + 0);
                    uint32_t my = get_arg_val<uint32_t>(member_rt_base + 2 * m + 1);
                    data_ready.up(noc, mx, my, VALID);
                }
            } else {
                // rectangular group: one mcast to the group rect (loopback fills own cb_sumsq)
                auto sender = mc.sender(noc);
                sender.send(rms_rp, sumsq_wp, sumsq_bytes);
            }
            cb_pop_front(cb_rms_src, 1);
            cb_push_back(cb_sumsq, 1);
        } else {
            // unicast own partial to the root's gather slot; signal after it lands
            uint32_t gather_base = get_write_ptr(cb_gather);  // same L1 offset as the root's cb_gather
            noc_async_write(
                partial_rp, get_noc_addr(root_x, root_y, gather_base + my_index * sumsq_bytes), sumsq_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_partial, 1);
            progress.up(noc, root_x, root_y, 1);
            // receive the broadcast 1/rms into cb_sumsq
            cb_reserve_back(cb_sumsq, 1);
            auto receiver = mc.receiver(noc);
            receiver.receive();
            cb_push_back(cb_sumsq, 1);
        }

        // ---------- PASS 2: re-stream local W-slice + local gamma slice ----------
        for (uint32_t b = 0; b < num_w_blocks; ++b) {
            if constexpr (IS_ROW_MAJOR) {
                read_local_input_wblock(local_row_base, num_real_rows, b);
            } else {
                read_tile_input_wblock(tr, b);
            }

            if constexpr (HAS_GAMMA) {
                if constexpr (GAMMA_IS_ROW_MAJOR) {
                    // RM gamma: one W-block-wide stick slice at the core's global W-col
                    // offset (w_col_start) so it matches the input's local column layout.
                    const auto g_acc = TensorAccessor(gamma_args, gamma_addr);
                    cb_reserve_back(cb_gamma_rm, 1);
                    uint32_t gl1 = get_write_ptr(cb_gamma_rm);
                    // Global column start of this block, and real cols available.
                    uint32_t g_col0 = (IS_ROW_MAJOR ? w_col_start : w_tile_start * TILE_W) + b * wblock_cols;
                    uint32_t cols = origin_W > g_col0 ? (origin_W - g_col0) : 0;
                    if (cols > wblock_cols) {
                        cols = wblock_cols;
                    }
                    uint32_t real_bytes = cols * gamma_elt;
                    zero_l1_page(gl1, gamma_padded_bytes);
                    if (real_bytes > 0) {
                        if constexpr (IS_ROW_MAJOR) {
                            // R5a: a sub-tile w_col_start makes the gamma DRAM source
                            // byte offset sub-32B-aligned, but DRAM reads require a
                            // 32B-aligned source (odd sub-tile cores read garbage
                            // otherwise). Read from the 32B-aligned base into the page
                            // (sized with GAMMA_DRAM_ALIGN slack) and shift the window
                            // left to local col 0 with an L1 byte copy (any alignment).
                            uint32_t byte_off = g_col0 * gamma_elt;
                            uint32_t aligned_off = byte_off & ~(GAMMA_DRAM_ALIGN - 1);
                            uint32_t shift = byte_off - aligned_off;
                            uint32_t page_bytes = origin_W * gamma_elt;
                            uint32_t span = shift + real_bytes;
                            uint32_t aligned_span = (span + GAMMA_DRAM_ALIGN - 1) & ~(GAMMA_DRAM_ALIGN - 1);
                            if (aligned_off + aligned_span > page_bytes) {
                                aligned_span = page_bytes - aligned_off;
                            }
                            noc_async_read(g_acc.get_noc_addr(0, aligned_off), gl1, aligned_span);
                            noc_async_read_barrier();
                            if (shift > 0) {
                                volatile tt_l1_ptr uint8_t* p = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(gl1);
                                for (uint32_t i = 0; i < real_bytes; ++i) {
                                    p[i] = p[i + shift];  // forward copy (dst < src): safe
                                }
                                for (uint32_t i = real_bytes; i < gamma_padded_bytes; ++i) {
                                    p[i] = 0;  // re-zero the W-tail after the shift
                                }
                            }
                        } else {
                            noc_async_read(g_acc.get_noc_addr(0, g_col0 * gamma_elt), gl1, real_bytes);
                            noc_async_read_barrier();
                        }
                    }
                    cb_push_back(cb_gamma_rm, 1);
                } else if constexpr (GAMMA_TILE_EXTRACT) {
                    // R5c/R5d: RM cross-core + TILE gamma. This core owns global cols
                    // [g_col0, g_col0+cols) at a SUB-TILE column offset, so a TILE-stored
                    // gamma can't be read as whole tiles aligned to local col 0. Read the
                    // containing global gamma tile(s) into an L1 scratch (whole tile pages ->
                    // tile-aligned DRAM reads, no sub-32B source issue), then extract their
                    // ROW 0 sub-columns into cb_gamma_rm at local col 0, zero-padding the
                    // W-tail. The compute then tilizes cb_gamma_rm exactly like the RM-gamma
                    // leg (GAMMA_IS_ROW_MAJOR=1) -> cb_gamma_tiles unchanged.
                    //   * GAMMA_TILE_EXTRACT==1 (R5c, fp32/bf16): face-aware alignment-free
                    //     L1->L1 byte copy (cols 0..15 in face 0 at byte c*elt; cols 16..31
                    //     in face 1 at byte (256+(c-16))*elt).
                    //   * GAMMA_TILE_EXTRACT==2 (R5d, bf8b): a bf8b tile is block-float, so a
                    //     byte copy is meaningless — DEQUANT each row-0 datum to the dest
                    //     float (gamma_elt = 2 -> bf16, 4 -> fp32). Row-0 layout in a 1088B
                    //     bf8b tile: exponent section bytes 0..63 (one byte per face-row: face
                    //     f row r -> byte f*16+r), mantissa section bytes 64..1087 (face f base
                    //     64+f*256, row-major 16x16). Row 0 -> cols 0..15 in face 0 (exp byte 0,
                    //     man bytes 64..79), cols 16..31 in face 1 (exp byte 16, man 320..335).
                    const uint32_t g_tile_bytes = get_tile_size(cb_gamma_src);
                    const auto g_acc = TensorAccessor(gamma_args, gamma_addr, g_tile_bytes);
                    uint32_t g_col0 = w_col_start + b * wblock_cols;
                    uint32_t cols = origin_W > g_col0 ? (origin_W - g_col0) : 0;
                    if (cols > wblock_cols) {
                        cols = wblock_cols;
                    }
                    cb_reserve_back(cb_gamma_rm, 1);
                    uint32_t gl1 = get_write_ptr(cb_gamma_rm);
                    zero_l1_page(gl1, gamma_padded_bytes);
                    if (cols > 0) {
                        uint32_t gt_first = g_col0 / TILE_W;
                        uint32_t gt_last = (g_col0 + cols - 1) / TILE_W;
                        uint32_t span = gt_last - gt_first + 1;
                        // cb_gamma_src is a reader-LOCAL L1 scratch (single-thread: the reader
                        // both fills and reads it). It has no consumer kernel, so it uses NO CB
                        // flow control — get_write_ptr returns its fixed L1 base (nothing is ever
                        // pushed). Read the span containing tiles, then extract row 0 in place.
                        uint32_t src_base = get_write_ptr(cb_gamma_src);
                        for (uint32_t s = 0; s < span; ++s) {
                            noc_async_read_page(gt_first + s, g_acc, src_base + s * g_tile_bytes);
                        }
                        noc_async_read_barrier();
                        for (uint32_t j = 0; j < cols; ++j) {
                            uint32_t g = g_col0 + j;
                            uint32_t local_t = (g / TILE_W) - gt_first;
                            uint32_t sc = g % TILE_W;
                            volatile tt_l1_ptr uint8_t* dptr =
                                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(gl1 + j * gamma_elt);
                            if constexpr (GAMMA_TILE_EXTRACT == 2) {
                                // bf8b: dequant the row-0 datum, then write dest float bytes.
                                volatile tt_l1_ptr uint8_t* tile =
                                    reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src_base + local_t * g_tile_bytes);
                                uint32_t exp_idx = (sc < 16) ? 0u : 16u;
                                uint32_t man_idx = (sc < 16) ? (64u + sc) : (320u + (sc - 16u));
                                uint32_t f32 = bfp8b_datum_to_f32_bits(tile[exp_idx], tile[man_idx]);
                                if (gamma_elt == 2) {
                                    uint16_t bf16 = static_cast<uint16_t>(f32 >> 16);  // bf8b->bf16 lossless
                                    dptr[0] = static_cast<uint8_t>(bf16 & 0xff);
                                    dptr[1] = static_cast<uint8_t>(bf16 >> 8);
                                } else {
                                    dptr[0] = static_cast<uint8_t>(f32 & 0xff);
                                    dptr[1] = static_cast<uint8_t>((f32 >> 8) & 0xff);
                                    dptr[2] = static_cast<uint8_t>((f32 >> 16) & 0xff);
                                    dptr[3] = static_cast<uint8_t>((f32 >> 24) & 0xff);
                                }
                            } else {
                                // fp32/bf16 (R5c): face-aware byte copy from the containing tile.
                                uint32_t face_off = (sc < 16 ? sc : (256 + (sc - 16))) * gamma_elt;
                                volatile tt_l1_ptr uint8_t* sptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(
                                    src_base + local_t * g_tile_bytes + face_off);
                                for (uint32_t k = 0; k < gamma_elt; ++k) {
                                    dptr[k] = sptr[k];
                                }
                            }
                        }
                    }
                    cb_push_back(cb_gamma_rm, 1);
                } else {
                    // TILE gamma direct: raw tile copy at the global gamma tile offset (TILE
                    // INPUT cross-core; the whole-tile W offset w_tile_start is tile-aligned).
                    const auto g_acc = TensorAccessor(gamma_args, gamma_addr, get_tile_size(cb_gamma_tiles));
                    uint32_t g_base_tile = w_tile_start + b * W_BLOCK_TILES;
                    for (uint32_t wt = 0; wt < W_BLOCK_TILES; ++wt) {
                        cb_reserve_back(cb_gamma_tiles, 1);
                        noc_async_read_page(g_base_tile + wt, g_acc, get_write_ptr(cb_gamma_tiles));
                        noc_async_read_barrier();
                        cb_push_back(cb_gamma_tiles, 1);
                    }
                }
            }
        }
    }
    // cb_scaler is prepared once above (producer=reader) and consumed once by the
    // compute kernel at row-loop end (consumer=compute) — the reader never pops it.
}
