// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"      // Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"      // Negative

// Per gathered tile: out = -input [* weight] [* (1/divisor)]
//   has_weight  → multiply the negated input (held in DEST) by the per-target weight tile
//   has_divisor → multiply by the scalar 1/divisor (mean reduction), broadcast from a single tile
// The 1/divisor reciprocal is computed once up front and held across the loop. The weight
// multiply is an inert OptionalChainElement when absent; the divisor stage is gated by
// `if constexpr` (it needs its own DEST-sync window — the bcast reads the prior stage from a CB).
void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    constexpr bool has_weight = get_compile_time_arg_val(1) == 1;
    constexpr bool has_divisor = get_compile_time_arg_val(2) == 1;

    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;
    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp_input = tt::CBIndex::c_25;
    constexpr uint32_t cb_tmp = tt::CBIndex::c_26;
    constexpr uint32_t cb_divisor_recip = tt::CBIndex::c_27;  // 1/divisor
    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    constexpr uint32_t onetile = 1;

    // Stage-1 packs straight to the output unless a divisor multiply follows, in which case it
    // stages to a temp tile that the scalar-broadcast multiply reads back from a CB.
    constexpr uint32_t cb_stage1_out = has_divisor ? cb_tmp : cb_output;

    // Engine boot (moreh inner-loop pattern — covers every chain call; the chain owns per-element init).
    binary_op_init_common(cb_tmp_input, cb_tmp_weight, cb_output);

    if constexpr (has_divisor) {
        // 1/divisor — computed once, held in cb_divisor_recip for the whole loop.
        eltwise_chain(
            onetile,
            CopyTile<cb_divisor, Dst::D0, InputLifecycle::Streaming, OperandKind::Scalar, CopyTileReconfig::Input>{},
            Recip<>{},
            PackTile<cb_divisor_recip, OutputLifecycle::Streaming>{});
        // Hold the reciprocal across the loop; per-iter chains read it as a CallerManaged operand.
        cb_wait_front(cb_divisor_recip, onetile);
    }

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        // -input [* weight]  →  cb_stage1_out
        eltwise_chain(
            onetile,
            CopyTile<cb_tmp_input, Dst::D0, InputLifecycle::Streaming, OperandKind::Scalar, CopyTileReconfig::Input>{},
            Negative<>{},
            OptionalChainElement<
                has_weight,
                DestReuseBinary<cb_tmp_weight, BinaryFpuOp::Mul, DestReuseType::DEST_TO_SRCA>>{},
            PackTile<cb_stage1_out, OutputLifecycle::Streaming>{});

        if constexpr (has_divisor) {
            // * (1/divisor), broadcast as a scalar  →  cb_output
            eltwise_chain(
                onetile,
                BinaryFpu<
                    cb_stage1_out,
                    cb_divisor_recip,
                    BinaryFpuOp::Mul,
                    BroadcastDim::Scalar,
                    BinaryDataFormatReconfig::Input,
                    InputLifecycle::Streaming,      // cb_stage1_out: wait + pop per iter
                    InputLifecycle::CallerManaged,  // cb_divisor_recip: held, kernel-managed wait/pop
                    OperandKind::Scalar,
                    Dst::D0,
                    OperandKind::Scalar>{},
                PackTile<cb_output, OutputLifecycle::Streaming>{});
        }
    }

    if constexpr (has_divisor) {
        cb_pop_front(cb_divisor_recip, onetile);
    }
}
