// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/mask.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_mask.hpp
 * @brief `Mask<DF, DataSlot>` and `MaskPosInf<DataSlot>` chain elements.
 *
 * Hardcoded LLK contract: `mask_tile` reads the mask from `DataSlot + 1` —
 * baked into the BinaryOp instantiation (lessons §1.4) so callers can't
 * point it at the wrong slot.
 */

namespace compute_kernel_lib::eltwise {

using namespace ckernel;

template <DataFormat DF = DataFormat::Float16_b, Dst DataSlot = Dst::D0>
struct Mask : BinaryOp<Mask<DF, DataSlot>, DataSlot, static_cast<Dst>(static_cast<uint32_t>(DataSlot) + 1), DataSlot> {
    static_assert(
        static_cast<uint32_t>(DataSlot) + 1 < DST_HW_CEILING, "Mask requires DataSlot + 1 < 16 (mask lives at slot+1)");
    ALWI void init() const { mask_tile_init(); }
    ALWI void call(uint32_t data, uint32_t /*mask*/, uint32_t /*out*/) const {
        // LLK reads mask from data+1 regardless of `mask` arg.
        mask_tile(data, /*idst2_mask=*/0);
    }
};

template <Dst DataSlot = Dst::D0>
struct MaskPosInf
    : BinaryOp<MaskPosInf<DataSlot>, DataSlot, static_cast<Dst>(static_cast<uint32_t>(DataSlot) + 1), DataSlot> {
    static_assert(
        static_cast<uint32_t>(DataSlot) + 1 < DST_HW_CEILING,
        "MaskPosInf requires DataSlot + 1 < 16 (mask lives at slot+1)");
    ALWI void init() const { mask_tile_init(); }
    ALWI void call(uint32_t data, uint32_t /*mask*/, uint32_t /*out*/) const {
        mask_posinf_tile(data, /*idst2_mask=*/0);
    }
};

}  // namespace compute_kernel_lib::eltwise
