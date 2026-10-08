// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/special.hpp"    // Where
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"  // FillBitcast / FillInt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"    // Optional
#if defined(ARCH_BLACKHOLE)
#include "api/dataflow/dataflow_buffer.h"
#include "eltwise_utils_common.hpp"
#endif

namespace ckl = compute_kernel_lib;

constexpr bool kIsInt = get_compile_time_arg_val(1) == 1;
constexpr bool kIsFloat = !kIsInt;

constexpr DataFormat kWhereDF = DataFormat::WHERE_DATA_FORMAT;

#if defined(ARCH_BLACKHOLE)
// Condition and tensor of a tile under one unpack-to-dest handshake: the tensor to reg 1 and the scalar to reg 2 in
// both variants
template <uint32_t cb_cond, uint32_t cb_tensor, uint32_t cb_out, uint32_t block>
ALWI void where_operand_pairs(uint32_t num_tiles, uint32_t scalar_value) {
    constexpr uint32_t true_slot = WHERE_TTS ? 1 : 2;
    DataflowBuffer dfb_cond(cb_cond);
    DataflowBuffer dfb_tensor(cb_tensor);
    DataflowBuffer dfb_out(cb_out);
    copy_init(cb_cond);
    where_tile_init();
    for (uint32_t t = 0; t < num_tiles; t += block) {
        const uint32_t n = (num_tiles - t < block) ? num_tiles - t : block;
        dfb_cond.wait_front(n);
        dfb_tensor.wait_front(n);
        dfb_out.reserve_back(n);
        tile_regs_acquire();
        copy_init(cb_cond);
        for (uint32_t i = 0; i < n; ++i) {
            copy_operands_to_dest<2>({cb_cond, cb_tensor}, {i, i}, i * 3, 1);
        }
        fill_tile_init();
        for (uint32_t i = 0; i < n; ++i) {
            if constexpr (kIsInt) {
                fill_tile_int<kWhereDF>(i * 3 + 2, scalar_value);
            } else {
                fill_tile_bitcast(i * 3 + 2, scalar_value);
            }
            where_tile<kWhereDF>(i * 3, i * 3 + true_slot, i * 3 + 3 - true_slot, i * 3);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i * 3, cb_out);
        }
        tile_regs_release();
        dfb_out.push_back(n);
        dfb_cond.pop_front(n);
        dfb_tensor.pop_front(n);
    }
}
#endif

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
#if defined(ARCH_BLACKHOLE)
    if constexpr (operands_to_dest<dfb_cond_id, dfb_tensor_id>()) {
        where_operand_pairs<dfb_cond_id, dfb_tensor_id, dfb_out_id, num_tiles_per_cycle>(num_tiles, scalar_value);
        return;
    }
#endif

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
