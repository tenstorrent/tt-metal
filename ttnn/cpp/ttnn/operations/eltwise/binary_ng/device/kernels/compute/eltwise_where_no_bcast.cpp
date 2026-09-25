// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "eltwise_utils_common.hpp"
#if HAS_ACTIVATIONS(LHS)
#define TT_POLY_WHERE_CONDITION_FACTOR_CONTEXT 1
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#undef TT_POLY_WHERE_CONDITION_FACTOR_CONTEXT
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/fill.h"
#include "eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#endif

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/special.hpp"    // Where
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"  // FillBitcast / FillInt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"    // Optional

namespace ckl = compute_kernel_lib;

constexpr bool kIsInt = get_compile_time_arg_val(1) == 1;
constexpr bool kIsFloat = !kIsInt;

constexpr DataFormat kWhereDF = DataFormat::WHERE_DATA_FORMAT;

#if HAS_ACTIVATIONS(LHS)
#if HAS_ACTIVATIONS(LHS)
#if DST_ACCUM_MODE || !WHERE_TTS || HAS_ACTIVATIONS(RHS) || HAS_ACTIVATIONS(POST) || defined(TT_POLY_LLK_DISABLE)
#error "Condition preprocessing requires enabled BF16 TTS with untouched gradient and no post chain"
#endif
#endif

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);
    const auto scalar_val = reinterpret_cast<const float*>(&scalar_value);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
#if HAS_ACTIVATIONS(LHS)
    static_assert(num_tiles_per_cycle == 1, "Selected condition factor owns destination tile zero");
#endif

    constexpr auto condition_cb = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : tt::CBIndex::c_0;
    DataflowBuffer dfb_in0(condition_cb);
    DataflowBuffer dfb_in1(tt::CBIndex::c_1);
    DataflowBuffer dfb_out(tt::CBIndex::c_2);

    compute_kernel_hw_startup(dfb_in0.get_id(), dfb_out.get_id());
    copy_init(dfb_in0.get_id());
    BINARY_SFPU_INIT

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        PREPROCESS(
            LHS,
            CircularBuffer(tt::CBIndex::c_0),
            CircularBuffer(condition_cb),
            CircularBuffer(dfb_out.get_id()),
            num_tiles_per_cycle);
#if HAS_ACTIVATIONS(LHS)
        BINARY_SFPU_INIT;
#endif
        dfb_in0.wait_front(num_tiles_per_cycle);
        dfb_in1.wait_front(num_tiles_per_cycle);
        dfb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();
        copy_init(dfb_in0.get_id());
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(dfb_in0.get_id(), i, i * 3);
        }
        copy_init(dfb_in1.get_id());
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            // TTS: tensor is true value, goes to dst_reg 1
#if WHERE_TTS
            copy_tile(dfb_in1.get_id(), i, i * 3 + 1);  // Copy true tensor to dst_reg 1
            fill_tile_init();
            // TTS: scalar is false value, goes to dst_reg 2
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(i * 3 + 2, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(i * 3 + 2, scalar_value);
#endif
#endif

// TST: tensor is false value, goes to dst_reg 2
#if WHERE_TST
            copy_tile(dfb_in1.get_id(), i, i * 3 + 2);  // Copy false tensor to dst_reg 2
            fill_tile_init();
            // TST: scalar is true value, goes to dst_reg 1
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(i * 3 + 1, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(i * 3 + 1, scalar_value);
#endif
#endif

            BINARY_SFPU_OP(i * 3, i * 3 + 1, i * 3 + 2, i * 3);
        }

        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 3, dfb_out.get_id());
        }
        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);
        dfb_in0.pop_front(num_tiles_per_cycle);
        dfb_in1.pop_front(num_tiles_per_cycle);
    }
}

#else
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto dfb_cond_id = tt::CBIndex::c_0;
    constexpr auto dfb_tensor_id = tt::CBIndex::c_1;
    constexpr auto dfb_out_id = tt::CBIndex::c_2;

#if WHERE_TTS
    // TTS: tensor is true value, goes to dst_reg 1
    // TTS: scalar is false value, goes to dst_reg 2
    constexpr auto kTensorSlot = ckl::Dst::D1;
    constexpr auto kFillSlot = ckl::Dst::D2;
#else
    // TST: tensor is false value, goes to dst_reg 2
    // TST: scalar is true value, goes to dst_reg 1
    constexpr auto kTensorSlot = ckl::Dst::D2;
    constexpr auto kFillSlot = ckl::Dst::D1;
#endif

    compute_kernel_hw_startup(dfb_cond_id, dfb_tensor_id, dfb_out_id);

    ckl::eltwise_chain(
        ckl::IterationShape::tiles(num_tiles).block_size(num_tiles_per_cycle),
        // cond -> D0 (block read, init_short for dfb_cond_id).
        ckl::CopyTile<
            ckl::input(
                dfb_cond_id, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
            ckl::Dst::D0>{},
        // tensor -> D1 (TTS) / D2 (TST) (block read, init_short for dfb_tensor_id).
        ckl::CopyTile<
            ckl::input(
                dfb_tensor_id,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block),
            kTensorSlot>{},
        // scalar fill -> the other slot. Inactive flavor folds to a no-op.
        ckl::Optional<kIsInt, ckl::FillInt<kWhereDF, kFillSlot>>{scalar_value},
        ckl::Optional<kIsFloat, ckl::FillBitcast<kFillSlot>>{scalar_value},
        // where(D0, D1, D2) -> D0.
        ckl::Where<kWhereDF, ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            dfb_out_id,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{});
}

#endif
