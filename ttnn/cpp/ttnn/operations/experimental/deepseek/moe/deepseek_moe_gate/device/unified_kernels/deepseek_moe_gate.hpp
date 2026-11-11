// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#elif defined(COMPILE_FOR_TRISC)
#include <cstdint>
#ifndef REDUCE_OP
#define REDUCE_OP PoolType::SUM
#endif
#ifndef REDUCE_DIM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#endif
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/deepseek_moe_gate.h"
#include "api/dataflow/dataflow_buffer.h"
#endif

namespace deepseek_v3_ops {

// ============================================================================
// DeepseekMoeGate micro-op
//
// Computes top-8 expert selection with normalized scores.
// Input: router logits [16, 16] + bias [16, 16] + indices [16, 16]
// Output: top8 scores [1, 16] + top8 indices [1, 16] (only first 8 valid)
//
// DFB States:
//   NCRISC: No-op (sharded buffer setup done in kernel file via setup_sharded_buffer)
//   BRISC: Waits for output DFBs (scores, indices)
//   TRISC: Computes sigmoid (optional), bias add, sorting, normalization
// ============================================================================
struct DeepseekMoeGate {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC) - empty, sharded buffer setup done in kernel file
    struct ReaderCTArgs {};

    // Writer CTArgs (BRISC)
    template <const auto& output_dfb_, const auto& output_indices_dfb_>
    struct WriterCTArgs {
        static constexpr auto output_dfb = output_dfb_;
        static constexpr auto output_indices_dfb = output_indices_dfb_;
    };

    // Compute CTArgs (TRISC)
    // enable_sigmoid must be compile-time (template parameter for deepseek_moe_gate<>)
    template <
        const auto& input_dfb_,
        const auto& bias_dfb_,
        const auto& input_indices_dfb_,
        const auto& output_dfb_,
        const auto& output_indices_dfb_,
        uint32_t eps_,
        uint32_t scaling_factor_,
        uint32_t enable_sigmoid_>
    struct ComputeCTArgs {
        static constexpr auto input_dfb = input_dfb_;
        static constexpr auto bias_dfb = bias_dfb_;
        static constexpr auto input_indices_dfb = input_indices_dfb_;
        static constexpr auto output_dfb = output_dfb_;
        static constexpr auto output_indices_dfb = output_indices_dfb_;
        static constexpr uint32_t eps = eps_;
        static constexpr uint32_t scaling_factor = scaling_factor_;
        static constexpr bool enable_sigmoid = enable_sigmoid_ == 1;
    };

    // ========================================================================
    // Op - the actual operation, templated on CTArgs and IsActiveCore
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore>
    class Op {
    public:
        void operator()() {
            if constexpr (IsActiveCore) {
                impl();
            }
        }

    private:
        void impl() {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC: No-op - sharded buffer setup done in kernel file
            // ================================================================

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC: Wait for compute to finish
            // ================================================================
            DataflowBuffer output_indices_dfb(CTArgs::output_indices_dfb);
            DataflowBuffer output_dfb(CTArgs::output_dfb);
            output_indices_dfb.wait_front(1);
            output_dfb.wait_front(1);

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC: Compute gate logic
            // ================================================================

            // Input indices DFB should have the same tile shape as the input DFB
            reconfig_full_operand(CTArgs::input_indices_dfb, CTArgs::bias_dfb);
            // Output indices DFB should have the same tile shape as the output DFB
            pack_reconfig_data_format<true>(CTArgs::output_dfb);

            DataflowBuffer input_indices_dfb(CTArgs::input_indices_dfb);
            DataflowBuffer bias_dfb(CTArgs::bias_dfb);
            DataflowBuffer input_dfb(CTArgs::input_dfb);
            DataflowBuffer output_dfb(CTArgs::output_dfb);
            DataflowBuffer output_indices_dfb(CTArgs::output_indices_dfb);

            // Init portion
            input_indices_dfb.wait_front(1);
            bias_dfb.wait_front(1);

            // Compute portion
            copy_init(CTArgs::input_indices_dfb);

            tile_regs_acquire();

            // Copy indices (already transposed to cols)
            copy_tile(CTArgs::input_indices_dfb, 0, 1);

            reconfig_data_format_srca(CTArgs::input_dfb);  // Assumes same tile shape as input indices DFB
            deepseek_moe_gate_init<CTArgs::enable_sigmoid>(CTArgs::input_dfb, CTArgs::bias_dfb);
            input_dfb.wait_front(1);
            deepseek_moe_gate<CTArgs::enable_sigmoid>(
                CTArgs::input_dfb, CTArgs::bias_dfb, CTArgs::eps, CTArgs::scaling_factor);
            // Pop input tile
            input_dfb.pop_front(1);

            tile_regs_commit();

            output_dfb.reserve_back(1);
            output_indices_dfb.reserve_back(1);

            tile_regs_wait();

            pack_tile(0, CTArgs::output_dfb);
            output_dfb.push_back(1);

            pack_reconfig_data_format(CTArgs::output_indices_dfb);  // Assumes same tile shape as output DFB
            pack_tile(1, CTArgs::output_indices_dfb);
            output_indices_dfb.push_back(1);

            tile_regs_release();
#endif
        }
    };  // class Op

};  // struct DeepseekMoeGate

}  // namespace deepseek_v3_ops
