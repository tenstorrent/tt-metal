// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose.hpp"

#include "clone/clone.hpp"
#include "device/transpose_device_operation.hpp"
#include "device/transpose_utils.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"

#include <tt-metalium/hal.hpp>

namespace ttnn::operations::data_movement::transpose {

namespace detail {

using namespace tt::tt_metal::experimental;
using namespace tt;
using tt::tt_metal::BufferType;
using ttnn::operations::data_movement::transpose::adjust_shard_spec_to_shape;
using ttnn::operations::data_movement::transpose::is_native_transpose_sharding;

inline Tensor transpose_(
    const Tensor& a,
    ttnn::prim::TransposeOpDim transpose_dim,
    const std::optional<MemoryConfig>& output_mem_config,
    float pad_value = 0.0f) {
    MemoryConfig output_mem_constructed;
    if (!output_mem_config.has_value() ||
        (output_mem_config.value().is_sharded() && !output_mem_config.value().shard_spec().has_value())) {
        // Native sharded subset: derive output shard_spec from input's. Otherwise fall back to L1
        // interleaved (the interleaved factories handle non-native via TensorAccessor).
        const bool native = is_native_transpose_sharding(a.tensor_spec());
        if (a.is_sharded() && native) {
            // Seed: honor user's requested sharded layout (spec gets synthesized below); else
            // inherit input config so downstream can promote (e.g. N=C=1 → WIDTH_SHARDED).
            const bool user_requested_layout = output_mem_config.has_value() && output_mem_config.value().is_sharded();
            output_mem_constructed = user_requested_layout ? output_mem_config.value() : a.memory_config();
            // If shard geometry can't scale to a valid output shard, hand back a shard-spec-less
            // sharded MemoryConfig (device op synthesizes via generate_transpose_shard_spec) when
            // the user requested sharded; otherwise fall back to L1 interleaved.
            const auto shard_derivation_fallback = [&]() {
                if (user_requested_layout) {
                    output_mem_constructed = MemoryConfig(
                        output_mem_config.value().memory_layout(), output_mem_config.value().buffer_type());
                } else {
                    output_mem_constructed = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1);
                }
            };
            const auto& input_padded_shape = a.padded_shape();
            if (transpose_dim == ttnn::prim::TransposeOpDim::WH) {
                const uint32_t W = input_padded_shape[3], C = input_padded_shape[1], N = input_padded_shape[0];
                auto shard_spec = a.shard_spec().value();
                // N=C=1 height-sharded-with-full-width → WIDTH_SHARDED (skip if user requested layout).
                const bool can_promote_to_width_sharded =
                    !user_requested_layout && N == 1 && C == 1 && shard_spec.shape[1] == W;
                if (can_promote_to_width_sharded) {
                    std::swap(shard_spec.shape[0], shard_spec.shape[1]);
                    if (a.layout() == Layout::TILE && (shard_spec.shape[0] % tt::constants::TILE_HEIGHT != 0 ||
                                                       shard_spec.shape[1] % tt::constants::TILE_WIDTH != 0)) {
                        shard_derivation_fallback();
                    } else {
                        output_mem_constructed = MemoryConfig(
                            TensorMemoryLayout::WIDTH_SHARDED, output_mem_constructed.buffer_type(), shard_spec);
                    }
                } else {
                    auto output_padded_shape = input_padded_shape;
                    std::swap(output_padded_shape[2], output_padded_shape[3]);
                    auto adjusted = adjust_shard_spec_to_shape(shard_spec, input_padded_shape, output_padded_shape);
                    if (!adjusted.has_value() ||
                        (a.layout() == Layout::TILE && (adjusted->shape[0] % tt::constants::TILE_HEIGHT != 0 ||
                                                        adjusted->shape[1] % tt::constants::TILE_WIDTH != 0))) {
                        shard_derivation_fallback();
                    } else {
                        output_mem_constructed = MemoryConfig(
                            output_mem_constructed.memory_layout(),
                            output_mem_constructed.buffer_type(),
                            std::move(adjusted));
                    }
                }
            } else if (transpose_dim == ttnn::prim::TransposeOpDim::HC && a.layout() == Layout::TILE) {
                auto shard_spec = a.shard_spec().value();
                // HC TILE padded-shape contract: dim[1] = logical H, dim[2] = round_up(logical C, TILE_HEIGHT).
                auto output_padded_shape = input_padded_shape;
                output_padded_shape[1] = a.logical_shape()[2];
                output_padded_shape[2] = tt::round_up(a.logical_shape()[1], tt::constants::TILE_HEIGHT);
                auto adjusted = adjust_shard_spec_to_shape(shard_spec, input_padded_shape, output_padded_shape);
                if (!adjusted.has_value() || adjusted->shape[0] % tt::constants::TILE_HEIGHT != 0 ||
                    adjusted->shape[1] % tt::constants::TILE_WIDTH != 0) {
                    shard_derivation_fallback();
                } else {
                    output_mem_constructed = MemoryConfig(
                        output_mem_constructed.memory_layout(),
                        output_mem_constructed.buffer_type(),
                        std::move(adjusted));
                }
            }
        } else if (output_mem_config.has_value()) {
            // User-requested sharded output (no spec): honor the layout; device op will synthesize
            // the spec. Must precede the non-native fallback below so it isn't overridden.
            output_mem_constructed = output_mem_config.value();
        } else if (a.is_sharded()) {
            // Non-native sharded input → mirror input's memory_layout (no shard_spec, device op
            // synthesizes via adjust_shard_spec_to_shape / generate_transpose_shard_spec)
            output_mem_constructed = MemoryConfig(a.memory_config().memory_layout(), a.memory_config().buffer_type());
        } else {
            output_mem_constructed = a.memory_config();
        }
    } else {
        output_mem_constructed = output_mem_config.value();
    }

    auto prim_permute = [&](const ttnn::Tensor& input, const ttnn::SmallVector<uint32_t>& dims) -> ttnn::Tensor {
        return ttnn::prim::permute(input, dims, output_mem_constructed, std::nullopt, pad_value);
    };

    bool interleaved_rm = !a.is_sharded() && a.layout() == Layout::ROW_MAJOR;
    switch (transpose_dim) {
        case ttnn::prim::TransposeOpDim::HC:
            if (interleaved_rm) {
                return prim_permute(a, ttnn::SmallVector<uint32_t>{0, 2, 1, 3});
            }
            break;
        case ttnn::prim::TransposeOpDim::NH:
            return ttnn::permute(
                (const ttnn::Tensor)a, ttnn::SmallVector<int64_t>({2, 1, 0, 3}), output_mem_config, pad_value);
        case ttnn::prim::TransposeOpDim::NW:
            return ttnn::permute(
                (const ttnn::Tensor)a, ttnn::SmallVector<int64_t>({3, 1, 2, 0}), output_mem_config, pad_value);
        case ttnn::prim::TransposeOpDim::CW:
            return ttnn::permute(
                (const ttnn::Tensor)a, ttnn::SmallVector<int64_t>({0, 3, 2, 1}), output_mem_config, pad_value);
        case ttnn::prim::TransposeOpDim::CN:
            if (interleaved_rm) {
                return prim_permute(a, ttnn::SmallVector<uint32_t>({1, 0, 2, 3}));
            }
            break;
        case ttnn::prim::TransposeOpDim::WH:
            if (interleaved_rm) {
                return prim_permute(a, ttnn::SmallVector<uint32_t>({0, 1, 3, 2}));
            }
            if (a.layout() == Layout::ROW_MAJOR) {
                // Only compute the RM WH CB-vs-L1 budget when actually on the RM WH path:
                // the allocator query and padded-shape arithmetic are wasted work otherwise.
                const uint32_t W_padded = round_up(a.logical_shape()[3], tt::constants::TILE_WIDTH);
                const uint32_t H_padded = round_up(a.logical_shape()[2], tt::constants::TILE_HEIGHT);
                const uint32_t cb_size_for_rm = (2 * W_padded + 2 * H_padded + H_padded * W_padded) * a.element_size();
                auto* device = a.device();
                auto lowest_address = device->lowest_occupied_compute_l1_address();
                uint32_t max_l1_space =
                    lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
                max_l1_space =
                    max_l1_space - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
                if (cb_size_for_rm > max_l1_space) {
                    return prim_permute(a, ttnn::SmallVector<uint32_t>({0, 1, 3, 2}));
                }
            }
            break;
        default: break;
    }
    return ttnn::prim::transpose(a, transpose_dim, output_mem_constructed, pad_value);
}

ttnn::Tensor transpose_nd(
    const ttnn::Tensor& input_tensor,
    uint32_t dim1,
    uint32_t dim2,
    const std::optional<MemoryConfig>& memory_config_arg,
    float pad_value = 0.0f) {
    const auto rank = input_tensor.logical_shape().rank();
    ttnn::SmallVector<int64_t> permutation;
    permutation.reserve(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        permutation.push_back(i);
    }
    std::swap(permutation[dim1], permutation[dim2]);
    return ttnn::permute(input_tensor, permutation, memory_config_arg, pad_value);
}

inline bool is_block_or_width_sharded_mc(const tt::tt_metal::MemoryConfig& mc) {
    return mc.is_sharded() && (mc.memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED ||
                               mc.memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED);
}

}  // namespace detail

ttnn::Tensor transpose_impl(
    const ttnn::Tensor& input_tensor,
    int64_t dim1,
    int64_t dim2,
    const std::optional<MemoryConfig>& memory_config_arg,
    float pad_value = 0.0f) {
    {
        // Irregular RM block/width sharded hits a pages_per_shard misread in noc_async_*_sharded.
        // Unshard until the kernel-side helpers handle irregular RM geometries.
        const bool rm = input_tensor.layout() == Layout::ROW_MAJOR;
        const bool in_sharded = detail::is_block_or_width_sharded_mc(input_tensor.memory_config());
        const auto& input_logical = input_tensor.logical_shape();
        const bool irregular_hw = input_logical.rank() >= 2 && (input_logical[-1] % tt::constants::TILE_WIDTH != 0 ||
                                                                input_logical[-2] % tt::constants::TILE_HEIGHT != 0);
        if (rm && in_sharded && irregular_hw) {
            const auto interleaved_l1 =
                tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1);
            Tensor x = ttnn::to_memory_config(input_tensor, interleaved_l1, std::nullopt);
            return transpose_impl(x, dim1, dim2, memory_config_arg, pad_value);
        }
    }
    const auto& input_shape = input_tensor.logical_shape();
    uint32_t normalized_dim1 = input_shape.get_normalized_index(dim1);
    uint32_t normalized_dim2 = input_shape.get_normalized_index(dim2);

    Tensor input_unsqueezed = input_tensor;
    uint32_t initial_rank = input_shape.rank();
    if (initial_rank < 4) {
        input_unsqueezed = ttnn::unsqueeze_to_4D(input_tensor);
        uint32_t rank_diff = 4 - initial_rank;
        normalized_dim1 += rank_diff;
        normalized_dim2 += rank_diff;
    } else if (initial_rank > 4) {
        return detail::transpose_nd(input_tensor, normalized_dim1, normalized_dim2, memory_config_arg, pad_value);
    }

    bool wh = (normalized_dim1 == 2 && normalized_dim2 == 3) || (normalized_dim2 == 2 && normalized_dim1 == 3);
    bool cn = (normalized_dim1 == 0 && normalized_dim2 == 1) || (normalized_dim2 == 0 && normalized_dim1 == 1);
    bool bfloat8_supported = cn || wh;
    bool typecast = input_unsqueezed.dtype() == DataType::BFLOAT8_B and !bfloat8_supported;
    Tensor input_typecasted = typecast ? ttnn::typecast(input_unsqueezed, DataType::BFLOAT16) : input_unsqueezed;

    TT_FATAL(normalized_dim1 <= 3, "dimension has to be 0-3 only corresponding to N,C,H,W");
    TT_FATAL(normalized_dim2 <= 3, "dimension has to be 0-3 only corresponding to N,C,H,W");

    Tensor output;
    if ((normalized_dim1 == normalized_dim2) || (input_typecasted.padded_shape()[normalized_dim1] == 1 &&
                                                 input_typecasted.padded_shape()[normalized_dim2] == 1)) {
        if (memory_config_arg.has_value() && input_typecasted.memory_config() != memory_config_arg.value()) {
            output = ttnn::clone(
                input_typecasted,
                std::nullopt,
                memory_config_arg.value_or(input_typecasted.memory_config()),
                std::nullopt);
        } else {
            output = input_typecasted;
        }
    } else {
        if (normalized_dim1 > normalized_dim2) {
            std::swap(normalized_dim1, normalized_dim2);
        }

        ttnn::prim::TransposeOpDim transpose_dim = ttnn::prim::TransposeOpDim::NW;

        if (normalized_dim2 == 3 && normalized_dim1 == 0) {
            transpose_dim = ttnn::prim::TransposeOpDim::NW;
        } else if (normalized_dim2 == 3 && normalized_dim1 == 1) {
            transpose_dim = ttnn::prim::TransposeOpDim::CW;
        } else if (normalized_dim2 == 3 && normalized_dim1 == 2) {
            transpose_dim = ttnn::prim::TransposeOpDim::WH;
        } else if (normalized_dim2 == 2 && normalized_dim1 == 0) {
            transpose_dim = ttnn::prim::TransposeOpDim::NH;
        } else if (normalized_dim2 == 2 && normalized_dim1 == 1) {
            transpose_dim = ttnn::prim::TransposeOpDim::HC;
        } else if (normalized_dim2 == 1 && normalized_dim1 == 0) {
            transpose_dim = ttnn::prim::TransposeOpDim::CN;
        } else {
            TT_ASSERT(false, "Unsupported transpose dims");
        }
        output = detail::transpose_(input_typecasted, transpose_dim, memory_config_arg, pad_value);
    }
    output = initial_rank < 4u ? ttnn::squeeze_from_4D(output, initial_rank) : output;
    return typecast ? ttnn::typecast(output, DataType::BFLOAT8_B) : output;
}

}  // namespace ttnn::operations::data_movement::transpose

namespace ttnn {

ttnn::Tensor transpose(
    const ttnn::Tensor& input_tensor,
    int64_t dim1,
    int64_t dim2,
    const std::optional<MemoryConfig>& memory_config,
    float pad_value) {
    return operations::data_movement::transpose::transpose_impl(input_tensor, dim1, dim2, memory_config, pad_value);
}

ttnn::Tensor transpose(const ttnn::Tensor& input_tensor, int64_t dim1, int64_t dim2, float pad_value) {
    return transpose(input_tensor, dim1, dim2, std::nullopt, pad_value);
}

}  // namespace ttnn
