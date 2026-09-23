// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — writer (NoC1): membership build, stats all-gather, output store.
//
//  build_membership_block  once: K*Ng fp32 0/1 tiles, M_T[c][g'] = 1 iff channel
//                          c0+32T+c (c < c_valid, the core's valid lanes) belongs to group
//                          32*k+g' (slot k), and the transposed copy MT_T[g'][c] (the in1 of
//                          the expansion matmuls, so every matmul in the op is the same
//                          non-transposed body). Lanes beyond c_valid (a 40-channel shard's
//                          second tile, the clipped last column shard, channels >= C) get
//                          all-zero rows / columns — the trailing-C mask for free. No helper
//                          builds a 0/1 matrix from (C, G, T): prepare_reduce_mask /
//                          generate_mask_w are prefix-only masks, so this is a raw fill (NoC
//                          zero-fill + 32*K scalar stores, never a CPU tile fill).
//  zero_unused_gather_rows once: rows >= num_active of the last gather row-tile
//                          (both round halves) — disjoint from any remote write.
//  fill_inv_rows           once (root): GT fp32 tiles with row 0 = 1/(HW*Cg) — the in0
//                          of the combine matmul, so the gathered sums come out as
//                          mean / variance directly. Re-pushed every round (the matmul
//                          pops its in0; the bytes persist in the ring).
//  fill_inv32_row          once (two_pass programs): one fp32 tile with row 0 = 1/32 — the in0 of
//                          compute's per-channel shift matmul over the first chunk's tile-row 0.
//  push_masked_mean        per image (hw_mask programs): masked copies of the landed group-mean
//                          tiles for a first / last tile-row whose sticks are only partly inside
//                          this image (RM shard heights that are stick counts, HW % 32 != 0 in any
//                          placement) — the mean in the valid rows, 0 elsewhere.
//  combine_round_block     twice per image (mean, then centered variance): every
//                          active core unicasts its partial row 0 (2 x 64 B per slot
//                          tile) into row `core_linear_idx` of the ROOT's cb_gather
//                          (row-tile-major: tile rt*Ng + k, the matmul's K x N layout)
//                          and bumps the root's monotone round counter; the root waits
//                          for num_active rows, its compute sums them with the group
//                          aggregation matmul ((1 x GT) @ (GT x Ng) -> cb_stats_bcast)
//                          and the root's writer multicasts the Ng stat tiles into
//                          every core's cb_group_mean / cb_group_var with mcast_pipe's
//                          SenderPipe (one sender per round; receivers ack readiness
//                          after reserving the landing CB). Idle cores inside the
//                          rectangle run the receive loop so the dense ack count
//                          holds. A sharded core whose shard has no sticks in image n
//                          sends a ZERO row instead (its compute has nothing to add), so the
//                          root's row count is the same every image. One landing region
//                          suffices: a core sends its round-r rows only after receiving the
//                          previous round's broadcast, which the root sends only after its
//                          compute has consumed the previous gather (op_design.md "Gather + multicast":
//                          one sender per round instead of a flat all-to-all).
//  store_chunk             per pass-3 chunk: interleaved TILE -> raw TensorAccessor page
//                          writes (nominal Q*K pages, pad popped unread); interleaved RM ->
//                          write_sticks_after_untilize per tile-row (the sb valid sticks of the
//                          row, c_valid*elem bytes each); TILE shard -> nothing
//                          (compute packs straight into the output shard); RM shard -> the
//                          valid sticks of each untilized tile-row are copied L1 -> L1 into
//                          the output shard (c_valid*elem bytes each); RM shard, direct view ->
//                          nothing (the untilize packs straight into the output shard, which IS
//                          cb_output_sticks).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"  // McastArgs / SenderPipe / ReceiverPipe
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
#include "ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C/kernels/groupnorm_sc_N_1_HW_C_geometry.hpp"

namespace {

constexpr uint32_t FACE_BYTES_FP32 = 16 * 16 * 4;
constexpr uint32_t FACE_ROW_BYTES_FP32 = 16 * 4;
constexpr uint32_t ONE_F32_BITS = 0x3F800000u;
constexpr uint32_t INV32_F32_BITS = 0x3D000000u;  // 1/32 = 2^-5

// Byte offset of element (row, col) inside a face-major 32x32 fp32 tile.
FORCE_INLINE uint32_t fp32_tile_elem_offset(uint32_t row, uint32_t col) {
    const uint32_t face = (row >> 4) * 2 + (col >> 4);
    return face * FACE_BYTES_FP32 + (row & 15) * FACE_ROW_BYTES_FP32 + (col & 15) * 4;
}

// push_masked_mean (hw_mask programs): Ng copies of the landed group-mean tiles with every row
// outside [r_lo, r_hi) zeroed. The group-mean tiles are FULL tiles here (cb_inv_rows has 1/n in all
// 32 rows), so the expansion matmul (32 x Ng) @ (Ng x 1) turns this into mean_full[r][c] =
// mask[r] * mean_c — the pass-2 chain subtracts it as a plain tile and the zero-staged pad sticks
// give (0 - 0)^2 = 0. Local NoC copy of Ng tiles + <= 8 NoC zero writes per tile (face-row ranges).
__attribute__((noinline, noclone)) void push_masked_mean(
    uint32_t cb_masked, uint32_t src_base, uint32_t Ng, uint32_t r_lo, uint32_t r_hi) {
    CircularBuffer masked(cb_masked);
    Noc noc;
    const uint32_t T4 = get_tile_size(cb_masked);
    masked.reserve_back(Ng);
    const uint32_t dst = masked.get_write_ptr();
    noc_async_write(src_base, get_noc_addr(my_x[noc_index], my_y[noc_index], dst), Ng * T4);
    noc_async_write_barrier();
    for (uint32_t k = 0; k < Ng; ++k) {
        for (uint32_t face = 0; face < 4; ++face) {
            // rows of this face: [f0, f0 + 16); zero the part below r_lo and the part at/above r_hi
            const uint32_t f0 = (face >> 1) * 16;
            const uint32_t face_off = k * T4 + face * FACE_BYTES_FP32;
            const uint32_t lo_end = (r_lo < f0 + 16 ? r_lo : f0 + 16);
            if (lo_end > f0) {
                noc.async_write_zeros(masked, (lo_end - f0) * FACE_ROW_BYTES_FP32, {.offset_bytes = face_off});
            }
            const uint32_t hi_start = (r_hi > f0 ? r_hi : f0);
            if (hi_start < f0 + 16) {
                noc.async_write_zeros(
                    masked,
                    (f0 + 16 - hi_start) * FACE_ROW_BYTES_FP32,
                    {.offset_bytes = face_off + (hi_start - f0) * FACE_ROW_BYTES_FP32});
            }
        }
    }
    noc.write_zeros_l1_barrier();
    masked.push_back(Ng);
}

}  // namespace

void kernel_main() {
    // ---- compile-time knobs / CB ids ------------------------------------
    constexpr bool is_rm = get_compile_time_arg_val(0) == 1;
    [[maybe_unused]] constexpr bool input_resident = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t K = get_compile_time_arg_val(2);  // block_c_tiles
    constexpr uint32_t Q = get_compile_time_arg_val(3);  // chunk_hw_tiles
    constexpr bool sharded = get_compile_time_arg_val(6) == 1;
    // hw_mask: some tile-row of some core is only partly inside its image (RM shards) -> the pass-2
    // chain carries a row-mask multiply for EVERY chunk and this writer feeds it [head][ones][tail].
    constexpr bool hw_mask = get_compile_time_arg_val(7) == 1;
    // rm_direct (host-derived): a ROW_MAJOR block shard consumed IN PLACE as the row-major
    // block of width lcm(shard_w, 32) = K*32 elements (m sticks per block row): the shard itself is the
    // tilize's input CB and the output shard the untilize's output CB (tile-sized pages), and block
    // lane j is channel c0 + j % c_period. No stick staging, no stick write-back.
    constexpr bool rm_direct = get_compile_time_arg_val(8) == 1;
    // two_pass (host-derived): streaming programs whose compute takes both statistics from one
    // read of each chunk; this writer only adds the constant 1/32 row tile their shift matmul reads.
    constexpr bool two_pass = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t cb_membership = get_compile_time_arg_val(10);
    constexpr uint32_t cb_partial_rows = get_compile_time_arg_val(11);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(12);
    constexpr uint32_t cb_stats_bcast = get_compile_time_arg_val(13);
    constexpr uint32_t cb_group_mean = get_compile_time_arg_val(14);
    constexpr uint32_t cb_group_var = get_compile_time_arg_val(15);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(16);
    constexpr uint32_t cb_output_sticks = get_compile_time_arg_val(17);
    constexpr uint32_t sem_round0_id = get_compile_time_arg_val(18);
    constexpr uint32_t sem_round1_id = get_compile_time_arg_val(19);
    constexpr uint32_t cb_inv_rows = get_compile_time_arg_val(20);
    constexpr uint32_t cb_membership_t = get_compile_time_arg_val(21);
    [[maybe_unused]] constexpr uint32_t cb_masked_mean = get_compile_time_arg_val(22);
    [[maybe_unused]] constexpr uint32_t cb_zero_row = get_compile_time_arg_val(23);
    [[maybe_unused]] constexpr uint32_t cb_output_shard = get_compile_time_arg_val(24);
    [[maybe_unused]] constexpr uint32_t cb_inv32_row = get_compile_time_arg_val(25);
    constexpr uint32_t mcast_ct_base = 26;
    constexpr uint32_t mcast_rt_base = 11;  // after the 11 per-core op args
    constexpr auto mc = dataflow_kernel_lib::McastArgs<mcast_ct_base, mcast_rt_base>();
    constexpr auto output_args = TensorAccessorArgs<mc.next_compile_time_args_offset()>();

    // ---- runtime args: grid-wide constants are COMMON, per-core geometry per core ----
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t N = get_common_arg_val<uint32_t>(1);
    const uint32_t HWt = get_common_arg_val<uint32_t>(2);
    [[maybe_unused]] const uint32_t Ct = get_common_arg_val<uint32_t>(3);
    const uint32_t HW = get_common_arg_val<uint32_t>(4);
    [[maybe_unused]] const uint32_t C = get_common_arg_val<uint32_t>(5);
    const uint32_t Cg = get_common_arg_val<uint32_t>(6);
    const uint32_t Ng = get_common_arg_val<uint32_t>(7);
    const uint32_t GT = get_common_arg_val<uint32_t>(8);
    const uint32_t num_active = get_common_arg_val<uint32_t>(9);
    const uint32_t root_vx = get_common_arg_val<uint32_t>(10);
    const uint32_t root_vy = get_common_arg_val<uint32_t>(11);
    const uint32_t inv_bits = get_common_arg_val<uint32_t>(12);  // fp32 bits of 1/(HW*Cg)
    // The output's aligned page (buffer_aligned_page_size): RM shard stick / RM full-row stick / tile.
    const uint32_t page_bytes = get_common_arg_val<uint32_t>(13);
    [[maybe_unused]] const uint32_t hw_stride = get_common_arg_val<uint32_t>(14);  // stick pitch between images
    const uint32_t c_period = get_common_arg_val<uint32_t>(15);  // channel period of a block lane (>= K*32: identity)
    const uint32_t is_active = get_arg_val<uint32_t>(0);
    const uint32_t r0 = get_arg_val<uint32_t>(1);
    const uint32_t H_core = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3);
    const uint32_t core_idx = get_arg_val<uint32_t>(4);
    const bool is_root = get_arg_val<uint32_t>(5) == 1;
    const uint32_t c0 = get_arg_val<uint32_t>(6);
    const uint32_t c_valid = get_arg_val<uint32_t>(7);
    [[maybe_unused]] const uint32_t s0 = get_arg_val<uint32_t>(8);
    [[maybe_unused]] const uint32_t sticks_valid = get_arg_val<uint32_t>(9);
    const uint32_t row_hi = get_arg_val<uint32_t>(10);  // valid sticks of this block's last tile-row (interleaved)

    constexpr uint32_t T4 = get_tile_size(cb_membership);  // fp32 tile bytes
    constexpr uint32_t output_tile_bytes = get_tile_size(cb_output_tiles);
    constexpr uint32_t output_elem = output_tile_bytes / 1024;

    Noc noc;

    // Idle cores inside the rectangle (is_active == 0) run the same round loop: they
    // skip the partial-row send and the landing-CB reserve/push, but take part in every
    // broadcast handshake so the root's dense ack count holds (one code path — the
    // program must fit the kernel-config ring in the --dev build).
    CircularBuffer membership(cb_membership);
    CircularBuffer membership_t(cb_membership_t);
    CircularBuffer partial(cb_partial_rows);
    CircularBuffer gather(cb_gather);

    // ---- build_membership_block (once, active cores): M and its transpose ------
    if (is_active) {
        membership.reserve_back(K * Ng);
        membership_t.reserve_back(K * Ng);
        noc.async_write_zeros(membership, K * Ng * T4);
        noc.async_write_zeros(membership_t, K * Ng * T4);
        noc.write_zeros_l1_barrier();
        const uint32_t base = membership.get_write_ptr();
        const uint32_t base_t = membership_t.get_write_ptr();
        for (uint32_t T = 0; T < K; ++T) {
            for (uint32_t c = 0; c < 32; ++c) {
                // Block lane 32T + c is channel c0 + (lane % c_period) — periodic on the RM direct view
                // (c_period = shard_w), the identity elsewhere (c_period >= K*32).
                const uint32_t p = (32 * T + c) % c_period;
                if (p >= c_valid) {
                    continue;  // lanes beyond this core's valid channels: all-zero rows / columns
                }
                const uint32_t g = (c0 + p) / Cg;
                const uint32_t slot = g >> 5;
                const uint32_t lane = g & 31;
                const uint32_t tile_off = (T * Ng + slot) * T4;
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + tile_off + fp32_tile_elem_offset(c, lane)) =
                    ONE_F32_BITS;
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_t + tile_off + fp32_tile_elem_offset(lane, c)) =
                    ONE_F32_BITS;
            }
        }
        membership.push_back(K * Ng);
        membership_t.push_back(K * Ng);
    }

    // ---- zero row (sharded): the partial a core sends for an image its shard misses ------
    [[maybe_unused]] uint32_t zero_row_addr = 0;
    if constexpr (sharded) {
        CircularBuffer zero_row(cb_zero_row);
        zero_row_addr = zero_row.get_write_ptr();
        noc.async_write_zeros(zero_row, 2 * FACE_ROW_BYTES_FP32);
        noc.write_zeros_l1_barrier();
    }

    // ---- zero_unused_gather_rows + fill_inv_rows (root, once) ---------------
    // Gather rows >= num_active of the last row-tile of every slot (the write
    // pointer sits at base here — nothing pushed yet — and returns there every
    // round: Ng*GT pushed and popped per round). Then the combine matmul's in0:
    // GT tiles whose row 0 is 1/(HW*Cg) in every lane, rows 1..31 zero, so the
    // (1 x GT) @ (GT x Ng) product is the per-group mean / variance directly.
    const uint32_t gather_base = gather.get_write_ptr();
    CircularBuffer inv_rows(cb_inv_rows);
    if (is_root) {
        const uint32_t first_pad_row = num_active & 31;
        if (first_pad_row != 0) {
            for (uint32_t k = 0; k < Ng; ++k) {
                const uint32_t tile_off = ((GT - 1) * Ng + k) * T4;
                for (uint32_t face = 0; face < 4; ++face) {
                    const uint32_t face_row0 = (face >> 1) * 16;
                    const uint32_t a = first_pad_row > face_row0 ? first_pad_row : face_row0;
                    const uint32_t end = face_row0 + 16;
                    if (a < end) {
                        noc.async_write_zeros(
                            gather,
                            (end - a) * FACE_ROW_BYTES_FP32,
                            {.offset_bytes =
                                 tile_off + face * FACE_BYTES_FP32 + (a - face_row0) * FACE_ROW_BYTES_FP32});
                    }
                }
            }
            noc.write_zeros_l1_barrier();
        }
        inv_rows.reserve_back(GT);
        noc.async_write_zeros(inv_rows, GT * T4);
        noc.write_zeros_l1_barrier();
        // Row 0 = 1/n makes the combine product a row-0 stat tile (what the bcast-Row chains read).
        // hw_mask programs fill ALL 32 rows so the stat tiles are full tiles: the masked-mean
        // expansion (push_masked_mean) needs the mean in every row it keeps.
        constexpr uint32_t inv_fill_rows = hw_mask ? 32 : 1;
        const uint32_t inv_base = inv_rows.get_write_ptr();
        for (uint32_t t = 0; t < GT; ++t) {
            for (uint32_t r = 0; r < inv_fill_rows; ++r) {
                for (uint32_t lane = 0; lane < 32; ++lane) {
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        inv_base + t * T4 + fp32_tile_elem_offset(r, lane)) = inv_bits;
                }
            }
        }
        inv_rows.push_back(GT);
    }

    // ---- fill_inv32_row (two_pass programs, once, every active core) -------
    // ONE fp32 tile with row 0 = 1/32 in every lane, rows 1..31 zero: the in0 of compute's shift matmul
    // (1 x 1) @ (1 x K) against tile-row 0 of pass A's first chunk, whose product is the per-channel column
    // mean of those 32 sticks — the shift every centered square of the image is taken against. Never popped.
    if constexpr (two_pass) {
        if (is_active) {
            CircularBuffer inv32(cb_inv32_row);
            inv32.reserve_back(1);
            noc.async_write_zeros(inv32, T4);
            noc.write_zeros_l1_barrier();
            const uint32_t inv32_base = inv32.get_write_ptr();
            for (uint32_t lane = 0; lane < 32; ++lane) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inv32_base + fp32_tile_elem_offset(0, lane)) =
                    INV32_F32_BITS;
            }
            inv32.push_back(1);
        }
    }

    // ---- combine geometry ---------------------------------------------------
    Semaphore<> sem_round0(sem_round0_id);
    Semaphore<> sem_round1(sem_round1_id);
    auto sender = mc.sender(noc);      // used by the root only (degenerate 1x1 -> local copy)
    auto receiver = mc.receiver(noc);  // used by every non-root core
    const uint32_t my_row_tile = core_idx >> 5;
    const uint32_t my_row = core_idx & 31;

    [[maybe_unused]] const auto output = TensorAccessor(output_args, output_addr, page_bytes);
    [[maybe_unused]] const uint32_t shard_out_base =
        sharded && is_rm && !rm_direct ? get_write_ptr(cb_output_shard) : 0;

    for (uint32_t image = 0; image < N; ++image) {
        // Which tile-rows of my block belong to this image (all of them when interleaved).
        const auto w = sharded ? groupnorm_geometry::image_work_sharded(image, s0, sticks_valid, HW, hw_stride)
                               : groupnorm_geometry::image_work_interleaved(H_core, row_hi);
        const bool active_img = is_active && w.active;

        // ---- combine_round_block, rounds 0 (mean) and 1 (variance) --------
        for (uint32_t round = 0; round < 2; ++round) {
            Semaphore<>& sem = (round == 0) ? sem_round0 : sem_round1;
            const uint32_t cb_target = (round == 0) ? cb_group_mean : cb_group_var;

            // 1. my partial rows -> root's gather (unicast; CB addresses are
            //    identical on every core, so the root's landing address is our
            //    own gather base + the row-tile-major tile offset rt*Ng + k).
            //    A core without work in this image contributes zeros.
            if (is_active) {
                uint32_t src_base = zero_row_addr;
                uint32_t src_stride_k = 0, src_stride_face = 0;
                if (active_img) {
                    partial.wait_front(Ng);
                    src_base = partial.get_read_ptr();
                    src_stride_k = T4;
                    src_stride_face = FACE_BYTES_FP32;  // row 0 of face 0/1
                }
                for (uint32_t k = 0; k < Ng; ++k) {
                    for (uint32_t face = 0; face < 2; ++face) {
                        const uint32_t src = src_base + k * src_stride_k + face * src_stride_face;
                        const uint32_t dst =
                            gather_base + (my_row_tile * Ng + k) * T4 + fp32_tile_elem_offset(my_row, face * 16);
                        noc_async_write(src, get_noc_addr(root_vx, root_vy, dst), FACE_ROW_BYTES_FP32);
                    }
                }
                noc_async_write_barrier();  // rows landed before the root is told
                sem.up(noc, root_vx, root_vy, 1);
                if (active_img) {
                    partial.pop_front(Ng);
                }
            }

            // 2. receive the reduced stats (root: combine, then broadcast). Idle cores never
            //    reserve/push the landing CB (nobody pops it there); its write pointer stays at
            //    base, which is where every active core's is too (capacity Ng, pushed and popped
            //    Ng per image). EVERY active core reserves — also for an image its shard has no
            //    work in: the reserve is the flow control that keeps the broadcast from landing
            //    while this core's compute is still reading the previous image's stats in pass 3.
            if (is_active) {
                cb_reserve_back(cb_target, Ng);  // blocks until compute popped the previous image's stats
            }
            const uint32_t landing = get_write_ptr(cb_target);  // identical on every core
            if (is_root) {
                gather.reserve_back(Ng * GT);
                sem.wait_min((image + 1) * num_active);  // monotone counter, never reset
                gather.push_back(Ng * GT);               // root compute: combine matmul -> cb_stats_bcast
                cb_wait_front(cb_stats_bcast, Ng);
                sender.send(get_read_ptr(cb_stats_bcast), landing, Ng * T4);
                cb_pop_front(cb_stats_bcast, Ng);
                // The matmul popped its GT in0 tiles; re-arm them for the next round
                // (capacity == GT, so the pointer is back at base and the bytes persist).
                inv_rows.reserve_back(GT);
                inv_rows.push_back(GT);
            } else {
                receiver.receive();
            }
            if (is_active) {
                cb_push_back(cb_target, Ng);  // compute pops it: consumed in pass 3, or drained for a no-work image
            }
            // ---- push_masked_mean (hw_mask programs, after the mean landed): the head / tail
            //      tile-rows that are only partly inside this image get group-mean tiles whose
            //      invalid rows are zero; compute expands them into the pass-2 mean operand of
            //      that segment (the body segment expands the unmasked landing tiles itself).
            if constexpr (hw_mask) {
                if (round == 0 && active_img) {
                    if (groupnorm_geometry::head_masked(w)) {
                        push_masked_mean(cb_masked_mean, landing, Ng, w.lo, w.rows == 1 ? w.hi : 32);
                    }
                    if (groupnorm_geometry::tail_masked(w)) {
                        push_masked_mean(cb_masked_mean, landing, Ng, 0, w.hi);
                    }
                }
            }
        }

        if (!active_img) {
            continue;
        }

        // ---- store_chunk (pass 3 output) ------------------------------------
        if constexpr (sharded && (!is_rm || rm_direct)) {
            // TILE shard: compute packs straight into the output shard (cb_output_tiles is placed
            // on it); RM direct view: the untilize packs straight into it (cb_output_sticks is the
            // output shard). Nothing to move.
        } else if constexpr (sharded) {
            // RM shard: each untilized tile-row (K tile pages, sticks at a K*64 B stride) -> the
            // valid sticks of shard tile-row row_off+i, c_valid*elem bytes each (L1 -> L1).
            constexpr uint32_t stride = K * 32 * output_elem;
            const uint32_t copy_bytes = c_valid * output_elem;
            for (uint32_t i = 0; i < w.rows; ++i) {
                uint32_t sa, sb;
                groupnorm_geometry::row_sticks(w, i, sa, sb);
                cb_wait_front(cb_output_sticks, K);
                const uint32_t l1 = get_read_ptr(cb_output_sticks);
                const uint32_t dst0 = shard_out_base + ((w.row_off + i) * 32) * page_bytes;
                for (uint32_t s = sa; s < sb; ++s) {
                    noc_async_write(
                        l1 + s * stride,
                        get_noc_addr(my_x[noc_index], my_y[noc_index], dst0 + s * page_bytes),
                        copy_bytes);
                }
                noc_async_write_barrier();
                cb_pop_front(cb_output_sticks, K);
            }
        } else {
            const uint32_t num_chunks = (H_core + Q - 1) / Q;
            for (uint32_t c = 0; c < num_chunks; ++c) {
                const uint32_t row_base = r0 + c * Q;
                const uint32_t q = (H_core - c * Q) < Q ? (H_core - c * Q) : Q;
                if constexpr (is_rm) {
                    // The sb valid sticks of each tile-row (32 except the image's ragged last row),
                    // c_valid*elem bytes each; the helper pops the K untilized pages either way.
                    for (uint32_t r = 0; r < q; ++r) {
                        uint32_t sa, sb;
                        groupnorm_geometry::row_sticks(w, c * Q + r, sa, sb);
                        dataflow_kernel_lib::write_sticks_after_untilize<cb_output_sticks>(
                            output, sb, c_valid * output_elem, image * HW + 32 * (row_base + r), c0 * output_elem);
                    }
                } else {
                    cb_wait_front(cb_output_tiles, q * K);
                    uint32_t l1 = get_read_ptr(cb_output_tiles);
                    for (uint32_t r = 0; r < q; ++r) {
                        const uint32_t page_row = (image * HWt + row_base + r) * Ct + t0;
                        for (uint32_t col = 0; col < K; ++col) {
                            noc_async_write(l1, output.get_noc_addr(page_row + col), output_tile_bytes);
                            l1 += output_tile_bytes;
                        }
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_output_tiles, q * K);
                    const uint32_t pad = (Q - q) * K;
                    if (pad > 0) {
                        cb_wait_front(cb_output_tiles, pad);
                        cb_pop_front(cb_output_tiles, pad);
                    }
                }
            }
        }
    }
    // The round-counter atomics are fire-and-forget on the critical path; drain
    // their acks once before exit (the firmware asserts an idle NoC at kernel end).
    noc_async_atomic_barrier();
}
