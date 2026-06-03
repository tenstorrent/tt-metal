// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Deepseek MoE Gate unified kernel
// Single kernel file, compiles for all RISC cores
//
// NCRISC: Sets up sharded CBs (input, bias, indices)
// BRISC: Waits for output CBs
// TRISC: Computes gate logic (sigmoid, bias add, sorting, normalization)

// DEBUG PROBE: uncomment to dump the per-group top-8 layout in DEST right after sum_top2.
// When enabled, the pipeline stops after sum_top2 and packs:
//   bias region (dst tile 2, ranking keys) -> output_tensor
//   indices region (dst tile 1, expert ids) -> output_indices_tensor
// Group selection (sort_top4) and top8 are skipped, so this is a layout probe, not a result.
//
// MODE SELECT — enable AT MOST ONE of the following:
//   GMG_UNGROUPED_TOP8      : real change — true global top-8 over all 8 groups (no group select).
//   GMG_DUMP_AFTER_SUM_TOP2 : debug probe — stop after sum_top2 (8 groups' top-8 at cols 0,2,...,14).
//   GMG_DUMP_AFTER_STEP1    : debug probe — stop after step1.
// (none enabled -> original grouped DeepSeek gate.)
//   GMG_PROBE_LANEMAP : P1 geometry calibration — pure load/store probe after sum_top2.
// NOTE: GMG_UNGROUPED_TOP8 v0 is INCORRECT (skips the cross-column rotation); off until fixed.
// #define GMG_PROBE_LANEMAP 1
// #define GMG_PROBE_OFFSETS 1
// #define GMG_DUMP_AFTER_STEP0 1
// #define GMG_DUMP_AFTER_SUM_TOP2 1
// #define GMG_SKIP_SORT_TOP4 1
// (diagnostic done: FPU MOVB2D offset-8 relocates; SFPU SFPLOAD/SFPSTORE offset>=8 wraps to rows 0-7.)
// #define GMG_TEST_HI_GROUPS 1
// #define GMG_HI_D2B_DST 4
// #define GMG_HI_B2D_BASE 8
//
// ===== ACTIVE: true global top-8 (ungrouped), assembled in rows 0-7 with FPU copy4rows stashing
// the 4-row source/topA into rows 8-15 between the two half-transposes. See generalized_moe_gate.h.
#define GMG_UNGROUPED_TOP8 1
// (diag: DIAG_TOPA showed topA clean via SFPU-tile parking.)
// #define GMG_DIAG_TOPA 1
// #define GMG_DIAG_TOPB 1
// #define GMG_DUMP_SRC 1

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
    constexpr uint32_t input_cb = get_named_compile_time_arg_val("moe_gate_input_cb");
    constexpr uint32_t bias_cb = get_named_compile_time_arg_val("moe_gate_bias_cb");
    constexpr uint32_t input_indices_cb = get_named_compile_time_arg_val("moe_gate_input_indices_cb");

    // Setup sharded persistent buffers (all tensor-backed)
    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(input_cb, 1);
        unified_kernels::setup_sharded_buffer(bias_cb, 1);
        unified_kernels::setup_sharded_buffer(input_indices_cb, 1);
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
        get_named_compile_time_arg_val("moe_gate_enable_sigmoid")>;
    deepseek_compute_kernel_init();
#endif

    // ========================================================================
    // Deepseek MoE Gate operation
    // ========================================================================
    deepseek_v3_ops::GeneralizedMoeGate::Op<MoeGateCTArgs, Core::is_active_core> moe_gate;
    moe_gate();
}
