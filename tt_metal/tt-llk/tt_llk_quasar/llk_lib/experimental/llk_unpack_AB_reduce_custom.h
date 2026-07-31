// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "experimental/llk_unpack_AB_reduce_custom_runtime.h"
#include "tensor_shape.h"

// Compile-time-block_ct_dim block reduce_max_row (unpack). Mirrors the Blackhole file layout: a
// compile-time lib alongside the runtime one, reusing the runtime helpers (cfg / mop_config / execute).
//
// Quasar bakes the SrcA buffer descriptor into the unpack MOP, and the descriptor is only known when
// the operand is bound (at execute). So -- unlike the runtime path, whose init programs the MOP -- the
// compile-time init here only enables transpose, and the execute programs the MOP (block_ct_dim is a
// compile-time constant here) and then runs it.

using namespace ckernel;

/**
 * @brief Compile-time-block_ct_dim init for the block reduce_max_row unpacker.
 *
 * Enables the UNPACKER0 transpose only; the MOP is programmed at execute (it needs the operand's
 * buffer descriptor). tensor_shape is accepted to mirror the Blackhole signature.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false, bool respect_trigger = false>
inline void _llk_unpack_AB_reduce_block_max_row_init_(const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(!respect_trigger, "respect_trigger is not supported on Quasar");
    (void)tensor_shape; // MOP (which uses the shape) is programmed at execute, once the operand is known
    _llk_unpack_AB_reduce_block_max_row_cfg_(true);
}

/**
 * @brief Compile-time-block_ct_dim execute for the block reduce_max_row unpacker.
 *
 * Programs the SrcA MOP for this block (now that the buffer descriptor is known), then unpacks the
 * scaler once, sets the block-start tile and fires the MOP.
 *
 * @param start_l1_tile_idx_0  L1 tile index of the block's first SrcA tile.
 * @param start_l1_tile_idx_1  L1 tile index of the SrcB scaler tile.
 * @param buf_desc_id_0        SrcA operand buffer-descriptor identifier.
 * @param buf_desc_id_1        SrcB scaler operand buffer-descriptor identifier.
 * @param tensor_shape         Operand tile shape driving the MOP.
 */
template <std::uint32_t block_ct_dim, bool respect_trigger = false>
inline void _llk_unpack_AB_reduce_block_max_row_(
    const std::uint32_t start_l1_tile_idx_0,
    const std::uint32_t start_l1_tile_idx_1,
    const std::uint32_t buf_desc_id_0,
    const std::uint32_t buf_desc_id_1,
    const ckernel::TensorShape& tensor_shape)
{
    _llk_unpack_AB_reduce_block_max_row_mop_config_runtime_(block_ct_dim, buf_desc_id_0, buf_desc_id_1, tensor_shape, respect_trigger);
    _llk_unpack_AB_reduce_block_max_row_runtime_(
        block_ct_dim, start_l1_tile_idx_0, start_l1_tile_idx_1, buf_desc_id_1, tensor_shape, respect_trigger, false /*overlap_first_half*/);
}

template <bool respect_trigger = false>
inline void _llk_unpack_AB_reduce_block_max_row_uninit_()
{
    _llk_unpack_AB_reduce_block_max_row_uninit_runtime_(respect_trigger, false /*overlap_first_half*/);
}
