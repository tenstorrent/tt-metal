// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem compute (every compute-rect core). Block ops:
//   s1_matmul_block     (stage-1 cores) q block = x round(s) (m x Q_t, in s1_kb K-blocks) @ resident q_b slice
//                       (Q_t x cols) -> cb_q_slice (m x cols, row-major), never popped here (the ports and the
//                       local-delivery writer read it in place)
//   prefix_matmul_block per landed row: nope (1 x Dn_t) @ wkv_b1[h] (Dn_t x s3n) -> cb_prefix_out, per N-block
//   rope_block (NoPE)   copy of the row's Dr_t rope tiles -> cb_rope_out
// All through kernel_lib helpers (matmul_block, eltwise copy).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

using namespace compute_kernel_lib;

namespace {
// in1 (resident q_b slice, fronted once) offset of K-block kb: kb * s1_kb rows of `cols` tiles.
// Fixed in1 offset (a resident head's weights inside cb_wkv).
struct TileOffset {
    uint32_t tiles;
    ALWI uint32_t operator()(uint32_t) const { return tiles; }
};
struct QbKBlockOffset {
    uint32_t tiles_per_kblock;
    ALWI uint32_t operator()(uint32_t block) const { return block * tiles_per_kblock; }
};
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x_round = get_compile_time_arg_val(0);
    constexpr uint32_t cb_qb_slice = get_compile_time_arg_val(1);
    constexpr uint32_t cb_q_slice = get_compile_time_arg_val(2);
    constexpr uint32_t cb_q_partial = get_compile_time_arg_val(3);
    constexpr uint32_t cb_landing = get_compile_time_arg_val(4);
    constexpr uint32_t cb_wkv = get_compile_time_arg_val(5);
    constexpr uint32_t cb_prefix_out = get_compile_time_arg_val(6);
    constexpr uint32_t cb_rope_out = get_compile_time_arg_val(7);
    constexpr uint32_t Q_t = get_compile_time_arg_val(8);
    constexpr uint32_t s1_kb = get_compile_time_arg_val(9);
    constexpr uint32_t nkb = get_compile_time_arg_val(10);
    constexpr uint32_t m = get_compile_time_arg_val(11);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(12);
    constexpr uint32_t rows_per_head = get_compile_time_arg_val(13);
    constexpr uint32_t Dn_t = get_compile_time_arg_val(14);
    constexpr uint32_t Dr_t = get_compile_time_arg_val(15);
    constexpr uint32_t N_t = get_compile_time_arg_val(16);
    constexpr uint32_t s3n = get_compile_time_arg_val(17);
    constexpr uint32_t n_nb = get_compile_time_arg_val(18);
    constexpr bool wkv_resident = get_compile_time_arg_val(19) != 0;
    constexpr uint32_t sbw3 = get_compile_time_arg_val(20);
    constexpr bool k_blocked = nkb > 1;

    const uint32_t cols = get_arg_val<uint32_t>(0);
    const uint32_t sbw1 = get_arg_val<uint32_t>(1);
    const uint32_t hpc = get_arg_val<uint32_t>(2);
    const uint32_t n_r = get_arg_val<uint32_t>(3);

    CircularBuffer x_buf(cb_x_round), qb_buf(cb_qb_slice), q_buf(cb_q_slice), q_partial_buf(cb_q_partial);
    CircularBuffer landing_buf(cb_landing), wkv_buf(cb_wkv), prefix_buf(cb_prefix_out);

    if (cols > 0) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_round, cb_qb_slice, cb_q_slice);
    } else {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_landing, cb_wkv, cb_prefix_out);
    }

    // ---- stage 1 ----------------------------------------------------------------------------------------------------
    if (cols > 0) {
        DeviceZoneScopedN("S1");
        cb_wait_front(cb_qb_slice, Q_t * cols);  // resident in1 (NoWaitNoPop below: the lifecycle is mine)
        const auto shape = MatmulBlockShape::of(m, cols / sbw1, 1, sbw1, s1_kb, nkb);
        for (uint32_t j = 0; j < num_blocks; ++j) {
#ifdef ABLATE_S1_COMPUTE
            for (uint32_t kb = 0; kb < nkb; ++kb) {
                cb_wait_front(cb_x_round, m * s1_kb);
                cb_pop_front(cb_x_round, m * s1_kb);
            }
            cb_reserve_back(cb_q_slice, m * cols);
            cb_push_back(cb_q_slice, m * cols);
            continue;
#endif
            matmul_block<
                false,
                false,
                LastBlockTarget::Out,
                OutputCBLayout::SubblockMajor,
                matmul_config::InitMode::Short,
                InputPolicy::WaitAndPopPerKBlock,
                InputPolicy::NoWaitNoPop,
                NoPostCompute,
                NoPreKBlock,
                NoPostKBlock,
                0,
                NoKBlockInnerDimFn,
                NoIn0Source,
                QbKBlockOffset>(
                x_buf,
                qb_buf,
                q_buf,
                k_blocked ? q_partial_buf : q_buf,
                shape,
                {},
                {},
                cols,
                0,
                {},
                {},
                {},
                QbKBlockOffset{s1_kb * cols});
        }
        cb_pop_front(cb_qb_slice, Q_t * cols);
    }

    // ---- stage 3 ----------------------------------------------------------------------------------------------------
    // Consumption order = landing order: the n_r remote heads row-major across heads (R1: all their weights resident,
    // head u at in1 offset u * Dn_t * N_t), then the local heads head-major.
    const auto shape3 = MatmulBlockShape::of(1, s3n / sbw3, 1, sbw3, Dn_t, 1);
    constexpr uint32_t head_tiles = Dn_t * N_t;
    DeviceZoneScopedN("S3");
    auto row_block = [&](uint32_t wkv_offset) {
        for (uint32_t nb = 0; nb < n_nb; ++nb) {
#ifdef ABLATE_S3
            if constexpr (!wkv_resident) {
                cb_wait_front(cb_wkv, Dn_t * s3n);
                cb_pop_front(cb_wkv, Dn_t * s3n);
            }
            cb_wait_front(cb_landing, Dn_t);
            cb_reserve_back(cb_prefix_out, s3n);
            cb_push_back(cb_prefix_out, s3n);
            continue;
#endif
            if constexpr (wkv_resident) {
                matmul_block<
                    false,
                    false,
                    LastBlockTarget::Out,
                    OutputCBLayout::SubblockMajor,
                    matmul_config::InitMode::Short,
                    InputPolicy::WaitAndRetainOnLastBlock,
                    InputPolicy::NoWaitNoPop,
                    NoPostCompute,
                    NoPreKBlock,
                    NoPostKBlock,
                    0,
                    NoKBlockInnerDimFn,
                    NoIn0Source,
                    TileOffset>(
                    landing_buf,
                    wkv_buf,
                    prefix_buf,
                    prefix_buf,
                    shape3,
                    {},
                    {},
                    0,
                    0,
                    {},
                    {},
                    {},
                    TileOffset{wkv_offset});
            } else {
                matmul_block<
                    false,
                    false,
                    LastBlockTarget::Out,
                    OutputCBLayout::SubblockMajor,
                    matmul_config::InitMode::Short,
                    InputPolicy::WaitAndRetainOnLastBlock,
                    InputPolicy::WaitAndPopPerKBlock>(landing_buf, wkv_buf, prefix_buf, prefix_buf, shape3);
            }
        }
        cb_pop_front(cb_landing, Dn_t);  // the row's nope, retained across the N-blocks
#ifdef ABLATE_S3
        cb_wait_front(cb_landing, Dr_t);
        cb_pop_front(cb_landing, Dr_t);
        cb_reserve_back(cb_rope_out, Dr_t);
        cb_push_back(cb_rope_out, Dr_t);
        return;
#endif
        copy<input(cb_landing), output(cb_rope_out)>(IterationShape::tiles(Dr_t));
    };
    if (n_r > 0) {
        if constexpr (wkv_resident) {
            cb_wait_front(cb_wkv, n_r * head_tiles);
        }
        for (uint32_t t = 0; t < rows_per_head; ++t) {
            for (uint32_t u = 0; u < n_r; ++u) {
                row_block(u * head_tiles);
            }
        }
        if constexpr (wkv_resident) {
            cb_pop_front(cb_wkv, n_r * head_tiles);
        }
    }
    for (uint32_t u = n_r; u < hpc; ++u) {
        if constexpr (wkv_resident) {
            cb_wait_front(cb_wkv, head_tiles);
        }
        for (uint32_t t = 0; t < rows_per_head; ++t) {
            row_block(0);
        }
        if constexpr (wkv_resident) {
            cb_pop_front(cb_wkv, head_tiles);
        }
    }
}
