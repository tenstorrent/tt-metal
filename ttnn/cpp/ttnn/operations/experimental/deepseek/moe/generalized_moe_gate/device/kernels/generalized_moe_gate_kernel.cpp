// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Deepseek MoE Gate unified kernel
// Single kernel file, compiles for all RISC cores
//
// NCRISC: Sets up sharded CBs (input, bias, indices)
// BRISC: Waits for output CBs
// TRISC: Computes gate logic (sigmoid, bias add, sorting, normalization)

// MODE SELECT — moe_gate_ungrouped_top8 is a named compile-time arg set by the device op (NOT hardcoded here):
//   = 1: true global top-8 over all 256 experts (ungrouped). The proven 4-group merge runs twice
//        (topA=top8(groups 0-3) at cols {0,2}, topB=top8(groups 4-7) at {4,6}, with FPU copy4rows stashing
//        the idle half in rows 8-15), then finalize fully bitonic-sorts the 16 candidates -> global top-8.
//   = 0: the DeepSeek grouped gate (8 groups × 32 -> top-2-sum -> top-4 groups -> top-8).
// The descriptor builder sets it from operation_attrs.grouped (false -> 1, true -> 0); it flows to the
// compute API as the defaultless `ungrouped_top8` template parameter of generalized_moe_gate<>.

#include <cstdint>
#include "../unified_kernels/kernel_op_api.hpp"
#include "../unified_kernels/kernel_utils.hpp"
#include "../unified_kernels/generalized_moe_gate.hpp"

// Compile-time role flag for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("moe_gate_is_active_core") == 1;
};

void kernel_main() {
// ============================================================================
// Define CTArgs per RISC
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using MoeGateCTArgs = deepseek_v3_ops::GeneralizedMoeGate::ReaderCTArgs;

    // Named compile-time args for sharded buffer setup
    constexpr std::uint32_t input_cb = get_named_compile_time_arg_val("moe_gate_input_cb");
    constexpr std::uint32_t bias_cb = get_named_compile_time_arg_val("moe_gate_bias_cb");
    constexpr std::uint32_t input_indices_cb = get_named_compile_time_arg_val("moe_gate_input_indices_cb");
    constexpr std::uint32_t num_blocks = get_named_compile_time_arg_val("moe_gate_num_blocks");

    // Setup sharded persistent buffers (all tensor-backed). input + bias each have num_blocks tiles/core
    // (one 256-expert block per tile); input_indices likewise has num_blocks tiles/core — block b's tile
    // holds that block's GLOBAL expert ids (arange + b*256), uploaded by the host.
    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(input_cb, num_blocks);
        unified_kernels::setup_sharded_buffer(bias_cb, num_blocks);
        unified_kernels::setup_sharded_buffer(input_indices_cb, num_blocks);
    }

#elif defined(COMPILE_FOR_BRISC)
    using MoeGateCTArgs = deepseek_v3_ops::GeneralizedMoeGate::WriterCTArgs<
        get_named_compile_time_arg_val("moe_gate_output_cb"),
        get_named_compile_time_arg_val("moe_gate_output_indices_cb")>;

#elif defined(COMPILE_FOR_TRISC)
    using MoeGateCTArgs = deepseek_v3_ops::GeneralizedMoeGate::ComputeCTArgs<
        get_named_compile_time_arg_val("moe_gate_input_cb"),
        get_named_compile_time_arg_val("moe_gate_bias_cb"),
        get_named_compile_time_arg_val("moe_gate_input_indices_cb"),
        get_named_compile_time_arg_val("moe_gate_output_cb"),
        get_named_compile_time_arg_val("moe_gate_output_indices_cb"),
        get_named_compile_time_arg_val("moe_gate_eps"),
        get_named_compile_time_arg_val("moe_gate_scaling_factor"),
        get_named_compile_time_arg_val("moe_gate_enable_sigmoid"),
        get_named_compile_time_arg_val("moe_gate_ungrouped_top8"),
        get_named_compile_time_arg_val("moe_gate_num_blocks"),
        get_named_compile_time_arg_val("moe_gate_run_scores_cb"),
        get_named_compile_time_arg_val("moe_gate_run_idx_cb"),
        get_named_compile_time_arg_val("moe_gate_run_bias_cb"),
        get_named_compile_time_arg_val("moe_gate_cb_tilize"),
        get_named_compile_time_arg_val("moe_gate_cb_tilize_idx"),
        get_named_compile_time_arg_val("moe_gate_topk"),
        get_named_compile_time_arg_val("moe_gate_softmax")>;
    deepseek_compute_kernel_init<false /* enable_math_reconfig_remap */>();
#endif

    // ========================================================================
    // Deepseek MoE Gate operation
    // ========================================================================
    deepseek_v3_ops::GeneralizedMoeGate::Op<MoeGateCTArgs, Core::is_active_core> moe_gate;
    moe_gate();
}
