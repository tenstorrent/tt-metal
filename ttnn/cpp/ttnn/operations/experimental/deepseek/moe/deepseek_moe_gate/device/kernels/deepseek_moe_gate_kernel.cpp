// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Deepseek MoE Gate unified kernel
// Single kernel file, compiles for all RISC cores
//
// NCRISC: Sets up sharded DFBs (input, bias, indices)
// BRISC: Waits for output DFBs
// TRISC: Computes gate logic (sigmoid, bias add, sorting, normalization)

#include "../unified_kernels/kernel_op_api.hpp"
#include "../unified_kernels/kernel_utils.hpp"
#include "../unified_kernels/deepseek_moe_gate.hpp"
#include "experimental/kernel_args.h"

// Compile-time role flag for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_active_core = get_arg(args::moe_gate_is_active_core) == 1;
};

void kernel_main() {
// ============================================================================
// Define CTArgs per RISC
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using MoeGateCTArgs = deepseek_v3_ops::DeepseekMoeGate::ReaderCTArgs;

    // Named DFB bindings for sharded buffer setup
    constexpr auto input_dfb = dfb::moe_gate_input;
    constexpr auto bias_dfb = dfb::moe_gate_bias;
    constexpr auto input_indices_dfb = dfb::moe_gate_input_indices;

    // Setup sharded persistent buffers (all tensor-backed)
    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(input_dfb, 1);
        unified_kernels::setup_sharded_buffer(bias_dfb, 1);
        unified_kernels::setup_sharded_buffer(input_indices_dfb, 1);
    }

#elif defined(COMPILE_FOR_BRISC)
    using MoeGateCTArgs =
        deepseek_v3_ops::DeepseekMoeGate::WriterCTArgs<dfb::moe_gate_output, dfb::moe_gate_output_indices>;

#elif defined(COMPILE_FOR_TRISC)
    using MoeGateCTArgs = deepseek_v3_ops::DeepseekMoeGate::ComputeCTArgs<
        dfb::moe_gate_input,
        dfb::moe_gate_bias,
        dfb::moe_gate_input_indices,
        dfb::moe_gate_output,
        dfb::moe_gate_output_indices,
        get_arg(args::moe_gate_eps),
        get_arg(args::moe_gate_scaling_factor),
        get_arg(args::moe_gate_enable_sigmoid)>;
    deepseek_compute_kernel_init();
#endif

    // ========================================================================
    // Deepseek MoE Gate operation
    // ========================================================================
    deepseek_v3_ops::DeepseekMoeGate::Op<MoeGateCTArgs, Core::is_active_core> moe_gate;
    moe_gate();
}
