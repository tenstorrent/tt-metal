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
#if defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Gathers token row `token` of a [.., 256] tile tensor into face 0 of the input tile. One read per tile brings
// both faces' 32 B rows ([row & ~63, + 576) of the tile) into the CB's scratch pages, while faces 1 to 3 are zeroed.
template <typename Accessor>
void gather_token_row(const Accessor& input, uint32_t input_cb, uint32_t token, uint32_t width_tiles) {
    constexpr uint32_t span = 512 + 64;
    CircularBuffer cb(input_cb);
    cb.reserve_back(1);
    const uint32_t page = cb.get_write_ptr();
    const uint32_t slots_l1 = page + get_tile_size(input_cb);
    const uint32_t r = token % 32;
    const uint32_t face_row = (r / 16) * 2 * 512 + (r % 16) * 32;
    Noc noc;
    for (uint32_t t = 0; t < width_tiles; ++t) {
        noc.async_read(
            input,
            CoreLocalMem<uint32_t>(slots_l1 + t * span),
            span,
            {.page_id = (token / 32) * width_tiles + t, .offset_bytes = face_row & ~63u},
            {});
    }
    noc.async_write_zeros(cb, 3 * 512, {.offset_bytes = 512});
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
    // Eight loads ahead of eight stores, so the loads overlap.
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots_l1 + (face_row & 63u));
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page);
    for (uint32_t s = 0; s < width_tiles * 2; ++s, dst += 8) {
        volatile tt_l1_ptr uint32_t* row = src + (s / 2) * (span / 4) + (s % 2) * (512 / 4);
        const uint32_t w0 = row[0], w1 = row[1], w2 = row[2], w3 = row[3];
        const uint32_t w4 = row[4], w5 = row[5], w6 = row[6], w7 = row[7];
        dst[0] = w0;
        dst[1] = w1;
        dst[2] = w2;
        dst[3] = w3;
        dst[4] = w4;
        dst[5] = w5;
        dst[6] = w6;
        dst[7] = w7;
    }
    cb.push_back(1);
}
#endif

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
    constexpr bool input_interleaved = get_named_compile_time_arg_val("moe_gate_input_interleaved") == 1;
    constexpr std::uint32_t input_width_tiles = get_named_compile_time_arg_val("moe_gate_input_width_tiles");

    // Setup sharded persistent buffers (all tensor-backed). bias has num_blocks tiles/core (one 256-expert
    // block per tile); input_indices likewise — block b's tile holds that block's GLOBAL expert ids
    // (arange + b*256), uploaded by the host. The input is either sharded the same way or gathered here
    // from the logits tile rows.
    if constexpr (Core::is_active_core) {
        if constexpr (input_interleaved) {
            // Bias and indices go first, so compute sets up while the token row is in flight.
            unified_kernels::setup_sharded_buffer(bias_cb, num_blocks);
            unified_kernels::setup_sharded_buffer(input_indices_cb, num_blocks);
            constexpr auto input_args = TensorAccessorArgs<0>();
            const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(0));
            gather_token_row(input, input_cb, get_arg_val<uint32_t>(1), input_width_tiles);
        } else {
            unified_kernels::setup_sharded_buffer(input_cb, num_blocks);
            unified_kernels::setup_sharded_buffer(bias_cb, num_blocks);
            unified_kernels::setup_sharded_buffer(input_indices_cb, num_blocks);
        }
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
