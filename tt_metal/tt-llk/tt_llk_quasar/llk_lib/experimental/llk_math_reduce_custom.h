// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "experimental/llk_math_reduce_runtime_custom.h"
#include "tensor_shape.h"

// Compile-time-block_ct_dim block reduce_max_row (math). Mirrors the Blackhole file layout: a
// compile-time lib alongside the runtime one. Quasar's pool/transpose implementation is identical for
// both, so these template wrappers forward to the runtime lib (baking block_ct_dim as a compile-time
// constant into the runtime MOP), reusing the shared helpers rather than duplicating them.

using namespace ckernel;

/**
 * @brief Compile-time-block_ct_dim MOP config for block reduce_max_row (math).
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_mop_config_(const ckernel::TensorShape& tensor_shape)
{
    _llk_math_reduce_block_max_row_mop_config_runtime_<is_fp32_dest_acc_en>(block_ct_dim, tensor_shape);
}

/**
 * @brief Compile-time-block_ct_dim init for block reduce_max_row (math).
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_init_(const ckernel::TensorShape& tensor_shape)
{
    _llk_math_reduce_block_max_row_init_runtime_<is_fp32_dest_acc_en>(block_ct_dim, tensor_shape);
}

/**
 * @brief Compile-time-block_ct_dim execute for block reduce_max_row (math). block_ct_dim is baked into
 *        the MOP by init, so it is unused here but kept as a template param to mirror Blackhole.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape)
{
    _llk_math_reduce_block_max_row_runtime_<is_fp32_dest_acc_en>(dst_index, tensor_shape);
}

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_uninit_()
{
    _llk_math_reduce_block_max_row_uninit_runtime_<is_fp32_dest_acc_en>();
}
