// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // Rsqrt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Typecast
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // AddBinary, SubBinary, MulBinary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"     // OptionalChainElement
#include "api/dataflow/circular_buffer.h"

// batchnorm_bcast_tiles: for each output tile in [tile_start, freq), computes
//
//   cb_den         = rsqrt(cb_batch_var + cb_eps)                  // Stage 1, one tile
//   running        = (cb_other - cb_bcast) * cb_den                // Stage 2
//                    [* cb_weight if WeightHas]                    // Stage 3
//                    [+ cb_bias  if BiasHas]                       // Stage 4
//   cb_final_out   = [typecast(running) if NeedsTypecast]          // Stage 5
//                    else running
//
// SFPU variant: every multiplicand other than the running operand is loaded into
// DEST::D1 via CopyTile (which the chain emits before the binary). The binary
// SFPU helpers (AddBinary/SubBinary/MulBinary) consume D0+D1 and write D0, so the
// running result stays in D0 across the whole chain — no intermediate CB writes.
// Original code staged through cb_tmp_1 between separate tile_regs windows; with
// the fused chain that bridge is unnecessary and cb_tmp_1 is dropped.

template <
    bool WeightHas,
    bool BiasHas,
    bool NeedsTypecast,
    uint32_t TcInFmt,
    uint32_t TcOutFmt,
    uint32_t cb_bcast,
    uint32_t cb_other,
    uint32_t cb_batch_var,
    uint32_t cb_eps,
    uint32_t cb_den,
    uint32_t cb_weight,
    uint32_t cb_bias,
    uint32_t cb_output_0,
    uint32_t cb_output_final>
ALWI void batchnorm_bcast_tiles(uint32_t freq, uint32_t tile_start) {
    using namespace compute_kernel_lib;

    // Stage 1: cb_den = rsqrt(cb_batch_var + cb_eps).
    // cb_batch_var: Bulk + Scalar — chain emits 1-tile wait+pop per call (window_1d<Scalar>).
    // cb_eps:        CallerManaged + Scalar — held by kernel_main for the whole kernel.
    eltwise_chain(
        1,
        CopyTile<cb_batch_var, Dst::D0, Bulk, OperandKind::Scalar, CopyTileReconfig::Input>{},
        CopyTile<cb_eps, Dst::D1, CallerManaged, OperandKind::Scalar, CopyTileReconfig::Input>{},
        AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
        Rsqrt<>{},
        PackTile<cb_den, Dst::D0, OutStreaming, PackTileReconfig::Output>{});

    const uint32_t inner_count = freq - tile_start;

    // Output CB depends on whether the typecast tail runs: when NeedsTypecast is true the
    // typecast tile writes the cast result to cb_output_final directly, so the fused chain's
    // pack target is cb_output_final; otherwise it's cb_output_0.
    constexpr uint32_t cb_final_out = NeedsTypecast ? cb_output_final : cb_output_0;

    // Stage 2..5 fused — DEST[0] threaded through Sub → Mul(den) → [Mul(weight)] → [Add(bias)]
    // → [Typecast] → Pack. CopyTile<…, D1> loads each new operand into D1; the SFPU binaries
    // then consume (D0, D1) and write back to D0.
    //
    // Lifecycles: every held single-tile operand uses Bulk + Scalar — chain's
    // window_1d<Scalar> collapses to 1, so each side emits a single
    // cb_wait_front(cb, 1) at the chain head and cb_pop_front(cb, 1) at the tail.
    // For cb_weight / cb_bias, wrapping the CopyTile in OptionalChainElement also
    // makes the wait/pop conditional on the compile-time flag — when the option
    // is off the wrapped element collapses to a tag with a_policy() == CallerManaged,
    // so the chain emits NOTHING for the inactive branch (CB ids, wait, pop all
    // suppressed).
    eltwise_chain(
        inner_count,
        CopyTile<cb_other, Dst::D0, Streaming, OperandKind::Scalar, CopyTileReconfig::Input>{},
        CopyTile<cb_bcast, Dst::D1, Bulk, OperandKind::Scalar, CopyTileReconfig::Input>{},
        SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
        CopyTile<cb_den, Dst::D1, Bulk, OperandKind::Scalar, CopyTileReconfig::Input>{},
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
        OptionalChainElement<
            WeightHas,
            CopyTile<cb_weight, Dst::D1, Bulk, OperandKind::Scalar, CopyTileReconfig::Input>>{},
        OptionalChainElement<WeightHas, MulBinary<Dst::D0, Dst::D1, Dst::D0>>{},
        OptionalChainElement<BiasHas, CopyTile<cb_bias, Dst::D1, Bulk, OperandKind::Scalar, CopyTileReconfig::Input>>{},
        OptionalChainElement<BiasHas, AddBinary<Dst::D0, Dst::D1, Dst::D0>>{},
        OptionalChainElement<NeedsTypecast, Typecast<TcInFmt, TcOutFmt, Dst::D0>>{},
        PackTile<cb_final_out, Dst::D0, OutStreaming, PackTileReconfig::Output>{});
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    constexpr bool weight_has_value = get_compile_time_arg_val(0) == 1;
    constexpr bool bias_has_value = get_compile_time_arg_val(1) == 1;

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_input = get_compile_time_arg_val(2);       // input
    constexpr auto cb_batch_mean = get_compile_time_arg_val(3);  // batch_mean
    constexpr auto cb_output_0 = get_compile_time_arg_val(4);    // pre-typecast staging (or final output)
    constexpr auto cb_batch_var = get_compile_time_arg_val(5);
    constexpr auto cb_eps = get_compile_time_arg_val(6);
    constexpr auto cb_den = get_compile_time_arg_val(7);
    constexpr auto cb_weight = get_compile_time_arg_val(8);
    // get_compile_time_arg_val(9) used to be cb_tmp_1 — no longer referenced (the fused chain
    // keeps the running result in DEST instead of staging through cb_tmp_1). CT-arg slot kept
    // for ABI compatibility with the program factory.
    constexpr auto cb_bias = get_compile_time_arg_val(10);
    constexpr auto cb_output_final = get_compile_time_arg_val(11);
    constexpr bool needs_output_typecast = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t tc_in_fmt = get_compile_time_arg_val(13);
    constexpr uint32_t tc_out_fmt = get_compile_time_arg_val(14);

    compute_kernel_hw_startup(cb_input, cb_batch_mean, cb_output_0);

    cb_wait_front(cb_eps, 1);

    const uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    const uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles<
            weight_has_value,
            bias_has_value,
            needs_output_typecast,
            tc_in_fmt,
            tc_out_fmt,
            cb_batch_mean,
            cb_input,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_weight,
            cb_bias,
            cb_output_0,
            cb_output_final>(tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles<
            weight_has_value,
            bias_has_value,
            needs_output_typecast,
            tc_in_fmt,
            tc_out_fmt,
            cb_batch_mean,
            cb_input,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_weight,
            cb_bias,
            cb_output_0,
            cb_output_final>(remaining_iterations, tile_start);
    }

    cb_pop_front(cb_eps, 1);
}
