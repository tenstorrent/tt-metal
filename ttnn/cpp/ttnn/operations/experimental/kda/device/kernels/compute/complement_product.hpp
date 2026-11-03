// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"

namespace kda {

constexpr uint32_t dst_tile_capacity =
    ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();

// Tiles of a row-major tile matrix whose rows hold Stride tiles, from tile column Column on: a whole buffer, or the E
// or B half of packed [E | B] rows.
template <typename DFB, uint32_t Stride, uint32_t Column = 0>
struct TileWindow {
    DFB& buffer;
    FORCE_INLINE uint32_t id() const { return buffer.get_id(); }
    FORCE_INLINE uint32_t tile(uint32_t row, uint32_t column) const { return row * Stride + Column + column; }
};

template <uint32_t Stride, uint32_t Column = 0, typename DFB>
FORCE_INLINE TileWindow<DFB, Stride, Column> window(DFB& buffer) {
    return {buffer};
}

// Output tiles per complement-product block: half of DST holds the result, the other half the second addend.
template <uint32_t Columns>
constexpr uint32_t complement_block_columns = largest_divisor_at_most(Columns, dst_tile_capacity / 2);

// Complement-form update of output tiles (row, column .. column + Block) into DST slots 0 .. Block:
//
//     first + second + later @ earlier
//
// e.g. E' = E2 + E1 + E2 @ E1, B' = B2 + B1 + E2 @ B1, or S + B + E @ S. The two addends are summed on the SFPU in FP32
// DST, and the FPU accumulates the product onto that sum, so only the product's operands pass through source
// registers: an addend read from an UnpackToDest buffer stays exact, and the rounding of the small product scales with
// E, never with A = I + E. Call it between tile_regs_acquire and tile_regs_commit, with SrcA configured for `earlier`
// and SrcB for `later` (reconfig_data_format(earlier, later)); that configuration holds again on return.
template <uint32_t Kt, uint32_t Block, typename Later, typename Earlier, typename First, typename Second>
FORCE_INLINE void complement_product_block(
    const Later& later,
    const Earlier& earlier,
    const First& first,
    const Second& second,
    uint32_t row,
    uint32_t column) {
    static_assert(2 * Block <= dst_tile_capacity);
    reconfig_data_format_srca(earlier.id(), first.id());
    copy_init(first.id());
    for (uint32_t slot = 0; slot < Block; ++slot) {
        copy_tile(first.id(), first.tile(row, column + slot), slot);
    }
    reconfig_data_format_srca(first.id(), second.id());
    copy_init(second.id());
    for (uint32_t slot = 0; slot < Block; ++slot) {
        copy_tile(second.id(), second.tile(row, column + slot), Block + slot);
    }
    add_binary_tile_init();
    for (uint32_t slot = 0; slot < Block; ++slot) {
        add_binary_tile(slot, Block + slot, slot);
    }
    // The FPU accumulates onto the SFPU's sums: LLK orders SFPU after FPU work, not FPU after SFPU work.
    MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));
    reconfig_data_format_srca(second.id(), earlier.id());
    // matmul's in1 (earlier) is SrcA and in0 (later) SrcB; one output row of Block tiles accumulates onto the sum.
    matmul_block_init(later.id(), earlier.id(), false, Block, 1, Kt);
    for (uint32_t k = 0; k < Kt; ++k) {
        matmul_block(later.id(), earlier.id(), later.tile(row, k), earlier.tile(k, column), 0, false, Block, 1, Kt);
    }
}

}  // namespace kda
