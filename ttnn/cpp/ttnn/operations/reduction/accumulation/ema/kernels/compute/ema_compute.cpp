// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
inline void ema_sfpi_face(
    uint32_t inp_dst_index,
    uint32_t prv_dst_index,
    uint32_t out_dst_index,
    float alpha,
    float beta,
    bool first_sample) {
    constexpr uint32_t n_vector_in_tile = 32;

    const uint32_t inp_base_idx = inp_dst_index * n_vector_in_tile;
    const uint32_t prv_base_idx = prv_dst_index * n_vector_in_tile;
    const uint32_t out_base_idx = out_dst_index * n_vector_in_tile;

    constexpr size_t vectors_per_face = 8;
    for (size_t i = 0; i < vectors_per_face; i++) {
        vFloat inp = dst_reg[inp_base_idx + i];
        vFloat prv = dst_reg[prv_base_idx + i];
        vFloat result = first_sample ? inp * beta : inp * beta + prv * alpha;
        dst_reg[out_base_idx + i] = result;
        dst_reg[prv_base_idx + i] = result;
    }
}
#endif

inline void ema_sfpi_tile(
    uint32_t inp_dst_index,
    uint32_t prv_dst_index,
    uint32_t out_dst_index,
    float alpha,
    float beta,
    bool first_sample) {
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(
        ema_sfpi_face, inp_dst_index, prv_dst_index, out_dst_index, VectorMode::RC, alpha, beta, first_sample));
}

namespace NAMESPACE {
void MAIN {
    // Compile time args
    // -----------------
    constexpr auto total_batches_per_core = get_compile_time_arg_val(0);
    constexpr auto tiles_per_channel = get_compile_time_arg_val(1);
    constexpr auto alpha_bits = get_compile_time_arg_val(2);
    constexpr auto beta_bits = get_compile_time_arg_val(3);

    // We have the bit representation of the alpha and beta values, get the float values
    union {
        uint32_t bits;
        float value;
    } alpha_union{alpha_bits}, beta_union{beta_bits};

    const auto alpha = alpha_union.value;
    const auto beta = beta_union.value;

    // CB indices
    // ----------
    constexpr auto src_cb = tt::CBIndex::c_0;
    constexpr auto dst_cb = tt::CBIndex::c_1;
    constexpr auto prv_cb = tt::CBIndex::c_2;

    // DST indices
    // -----------
    constexpr auto inp_dst_index = 0;
    constexpr auto prv_dst_index = 1;
    constexpr auto out_dst_index = 2;

    // Tell the SFPU that we will be using circular buffers src_cb and dst_cb
    // to perform the computation.
    init_sfpu(src_cb, dst_cb);

    //-------------------------------------------------------------------------
    // Main loop - compute ema for each batch
    for (uint32_t batch_id = 0; batch_id < total_batches_per_core; ++batch_id) {
        // For the first tile, just multiply the input by beta
        cb_wait_front(src_cb, 1);
        tile_regs_acquire();
        copy_tile(src_cb, 0, inp_dst_index);
        ema_sfpi_tile(inp_dst_index, prv_dst_index, out_dst_index, alpha, beta, /*first_sample=*/true);
        tile_regs_commit();
        cb_pop_front(src_cb, 1);

        cb_reserve_back(dst_cb, 1);
        cb_reserve_back(prv_cb, 1);
        tile_regs_wait();
        pack_tile(out_dst_index, dst_cb);
        pack_tile(prv_dst_index, prv_cb);
        tile_regs_release();
        cb_push_back(dst_cb, 1);
        cb_push_back(prv_cb, 1);

        // For each successive tile, multiply the input by beta, prv by alpha and add them together
        for (uint32_t tile_id = 1; tile_id < tiles_per_channel; ++tile_id) {
            cb_wait_front(src_cb, 1);
            cb_wait_front(prv_cb, 1);
            tile_regs_acquire();
            copy_tile(src_cb, 0, inp_dst_index);
            copy_tile(prv_cb, 0, prv_dst_index);
            ema_sfpi_tile(inp_dst_index, prv_dst_index, out_dst_index, alpha, beta, /*first_sample=*/false);
            tile_regs_commit();
            cb_pop_front(src_cb, 1);
            cb_pop_front(prv_cb, 1);

            cb_reserve_back(dst_cb, 1);
            cb_reserve_back(prv_cb, 1);
            tile_regs_wait();
            pack_tile(out_dst_index, dst_cb);
            pack_tile(prv_dst_index, prv_cb);
            tile_regs_release();
            cb_push_back(dst_cb, 1);
            cb_push_back(prv_cb, 1);
        }

        // Clear the previous output, we don't need it anymore
        cb_wait_front(prv_cb, 1);
        cb_pop_front(prv_cb, 1);
    }
}
}  // namespace NAMESPACE
