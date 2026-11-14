// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Thin wrapper around compute_kernel_lib::reduce<>. The input data format is deduced from the input
// buffer inside the helper, so Int32 MAX, MIN and SUM, and bf16 MIN, are routed to the SFPU path
// automatically; otherwise FPU/GMPOOL. Accurate fp32 also uses the SFPU. Every other MIN (bfloat8_b,
// fast-mode fp32) never reaches here: the host rewrites it as -MAX(-x) and dispatches
// reduce_{h,w}_neg instead.

#include <cstdint>
#include "api/compute/cb_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_compute_common.hpp"
#ifdef REDUCE_W_COMBINE
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#endif

void kernel_main() {
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t NC = get_arg(args::NC);
    // Accurate fp32: the host sets enable_fp32_sfpu to route Float32 through the SFPU (full fp32)
    // vs the FPU (tf32).
    constexpr auto fp32_mode = get_arg(args::enable_fp32_sfpu) != 0 ? ReduceFp32Mode::Accurate : ReduceFp32Mode::Fast;

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_scaler(dfb::scaler);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in0, dfb::scaler, dfb::out);

    compute_kernel_lib::reduce<
        REDUCE_OP,
        REDUCE_DIM,
        dfb::in0,
        dfb::scaler,
        dfb::out,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT,
        fp32_mode>(
        dfb_in0,
        dfb_scaler,
        dfb_out,
        compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
#ifdef REDUCE_POST_MUL
        // GMPOOL only respects the scaler's exponent for MAX/MIN and SFPU reduce ignores the
        // scaler buffer entirely, so both paths apply the user scalar here per output tile.
        // reduce_post_mul_tile handles Int32 (typecast-bracketed) and float formats uniformly.
        [](uint32_t dst_idx) {
            const auto post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
            if (post_mul_scaler_bits == k_identity_scaler_bits) {
                return;
            }
            // The data format has to be a constant expression here (it is a template argument), so it
            // is read from the JIT descriptor array indexed by the DFB handle rather than off a
            // DataflowBuffer object: DataflowBuffer's constructor is not constexpr, so no such object
            // is usable in a constant expression.
            constexpr auto reduce_format = static_cast<DataFormat>(unpack_src_format[dfb::in0]);
            compute_kernel_lib::detail::reduce_post_mul_tile<reduce_format>(dst_idx, post_mul_scaler_bits);
        }
#else
        compute_kernel_lib::NoOp{}
#endif
    );

    // The reduce helper waits on the scaler buffer but never pops it (the single scaler tile is
    // reused for the whole reduction). Pop it here so the buffer is left balanced.
    dfb_scaler.pop_front(1);

#ifdef REDUCE_W_COMBINE
    // Column split: this core folds the partials of its tile row, its own included, into one tile.
    constexpr uint32_t num_parts = get_arg(args::num_parts);
    DataflowBuffer dfb_gather(dfb::gather);
    DataflowBuffer dfb_result(dfb::result);
    dfb_gather.wait_front(num_parts);
    dfb_result.reserve_back(1);
    reconfig_data_format_srca(dfb::gather);
    pack_reconfig_data_format(dfb::result);
    if constexpr (REDUCE_OP == PoolType::MAX) {
        binary_max_tile_init();
    } else if constexpr (REDUCE_OP == PoolType::MIN) {
        binary_min_tile_init();
    } else {
        add_binary_tile_init();
    }
    copy_init(dfb::gather);
    tile_regs_acquire();
    copy_tile(dfb::gather, 0, 0);
    for (uint32_t i = 1; i < num_parts; ++i) {
        copy_tile(dfb::gather, i, 1);
        if constexpr (REDUCE_OP == PoolType::MAX) {
            binary_max_tile(0, 1, 0);
        } else if constexpr (REDUCE_OP == PoolType::MIN) {
            binary_min_tile(0, 1, 0);
        } else {
            add_binary_tile(0, 1, 0);
        }
    }
#ifdef REDUCE_W_COMBINE_POST_MUL
    const auto post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
    if (post_mul_scaler_bits != k_identity_scaler_bits) {
        compute_kernel_lib::detail::reduce_post_mul_tile<DataFormat::Float32>(0, post_mul_scaler_bits);
    }
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::result);
    tile_regs_release();
    dfb_result.push_back(1);
    dfb_gather.pop_front(num_parts);
#endif
}
