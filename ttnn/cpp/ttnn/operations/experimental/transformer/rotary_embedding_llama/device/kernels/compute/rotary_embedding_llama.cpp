// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

// Rotary kernel structure (Llama-style):
//   out = (x @ trans_mat) * sin + x * cos
//
// The original implementation ran this as 4 sequential ACQ/REL phases with
// 3 intermediate L1 materializations (rotated_in_interm_cb -> sin_interim_cb
// + cos_interim_cb -> out_cb). At MATH=4% on the full grid (Qwen3 prefill,
// bs=32 ISL=512), the kernel is bandwidth-bound on the intermediate
// CB round-trips, not on FPU work.
//
// This rewrite collapses the 4 phases into 2 by reusing DST registers across
// FPU ops via EltwiseBinaryReuseDestType::DEST_TO_SRCA:
//
//   Phase 1 (single ACQ/REL):
//     dst[j] = matmul(x[j], trans_mat)           // FPU matmul, dst <- rotated
//     dst[j] *= sin[j]                            // FPU mul DEST_TO_SRCA
//     pack(dst[j] -> sin_interm_cb)               // -> rotated*sin in CB
//
//   Phase 2 (single ACQ/REL):
//     dst[j] = x[j] * cos[j]                      // FPU mul, dst <- x*cos
//     dst[j] += sin_interm_cb[j]                  // FPU add DEST_TO_SRCA
//     pack(dst[j] -> out_cb)                      // -> final output
//
// rotated_in_interm_cb and cos_interm_cb are no longer used; they remain
// allocated by the program factory but are inert. The arithmetic is
// bit-identical to the original (all ops are FPU add/mul/matmul on the same
// bf16 tiles). Each phase still uses Wt DST tiles, matching the original's
// budget so fp32_dest_acc_en behavior is unchanged.

void kernel_main() {
    uint32_t argrt = 0;
    uint32_t batch_start = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_end = get_arg_val<uint32_t>(argrt++);
    uint32_t seq_t_start = get_arg_val<uint32_t>(argrt++);
    uint32_t seq_t_end = get_arg_val<uint32_t>(argrt++);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);

    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t n_heads = get_compile_time_arg_val(9);
    constexpr uint32_t rotary_Ht = get_compile_time_arg_val(10);

    const uint32_t rotary_seq_t_end = seq_t_end < rotary_Ht ? seq_t_end : rotary_Ht;
    const uint32_t my_rotary_seq_tiles = seq_t_start < rotary_seq_t_end ? rotary_seq_t_end - seq_t_start : 0;
    const uint32_t my_cos_sin_tiles = my_rotary_seq_tiles * Wt;

    mm_init(in_cb, trans_mat_cb, out_cb);
    binary_op_init_common(rotated_in_interm_cb, cos_cb, out_cb);  // General Init for all binary ops

    // Get the trans_mat
    cb_wait_front(trans_mat_cb, onetile);

    uint32_t in0_index = 0;
    uint32_t in1_index = 0;
    uint32_t interm_index = 0;

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            cb_wait_front(sin_cb, my_cos_sin_tiles);
            cb_wait_front(cos_cb, my_cos_sin_tiles);
        }
#endif
        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            uint32_t sin_cos_row_cnt = 0;
            for (uint32_t seq_tile = seq_t_start; seq_tile < rotary_seq_t_end; ++seq_tile) {
                // input cb wait and reserve
                cb_wait_front(in_cb, Wt);
#if RELOAD_IMPL == 1
                cb_wait_front(sin_cb, Wt);
                cb_wait_front(cos_cb, Wt);
#endif

                cb_reserve_back(sin_interm_cb, Wt);
                cb_reserve_back(out_cb, Wt);

                // ---------- Phase 1: matmul + sin-mul fused via DEST_TO_SRCA ----------
                // dst[j] = (x[j] @ trans_mat) * sin[j];  pack -> sin_interm_cb
                mm_init_short(in_cb, trans_mat_cb);
                ACQ();
                for (uint32_t j = 0; j < Wt; ++j) {
                    matmul_tiles(in_cb, trans_mat_cb, j, in1_index, j);
                }
                // Re-init to read first operand from DST (the matmul result) instead of a CB.
                binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    sin_cb);
                for (uint32_t j = 0; j < Wt; ++j) {
                    binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        sin_cb, j + (sin_cos_row_cnt * Wt), j);
                    pack_tile(j, sin_interm_cb, j);
                }
                REL();
                cb_push_back(sin_interm_cb, Wt);
                cb_wait_front(sin_interm_cb, Wt);

                // ---------- Phase 2: cos-mul + add fused via DEST_TO_SRCA ----------
                // dst[j] = x[j] * cos[j] + sin_interm_cb[j];  pack -> out_cb
                mul_tiles_init(in_cb, cos_cb);
                ACQ();
                for (uint32_t j = 0; j < Wt; ++j) {
                    mul_tiles(in_cb, cos_cb, j, j + (sin_cos_row_cnt * Wt), j);
                }
                binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    sin_interm_cb);
                for (uint32_t j = 0; j < Wt; ++j) {
                    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        sin_interm_cb, j, j);
                    pack_tile(j, out_cb, j);
                }
                REL();
                cb_push_back(out_cb, Wt);
                cb_pop_front(sin_interm_cb, Wt);
                cb_pop_front(in_cb, Wt);  // Done with input
#if RELOAD_IMPL == 1
                cb_pop_front(sin_cb, Wt);
                cb_pop_front(cos_cb, Wt);
#endif

#if RELOAD_IMPL == 0
                // no-reload needs to increment this counter
                // Used a sin/cos row
                sin_cos_row_cnt++;
#endif
            }
        }

#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            cb_pop_front(sin_cb, my_cos_sin_tiles);
            cb_pop_front(cos_cb, my_cos_sin_tiles);
        }
#endif
    }

    // Done with the transformation matrix, so remove from CB
    cb_pop_front(trans_mat_cb, onetile);
}
