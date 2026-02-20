// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "transpose.hpp"

#include "clone/clone.hpp"
#include "device/transpose_device_operation.hpp"
#include "ttnn/decorators.hpp"
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

inline Tensor transpose_(
    const Tensor& a,
    ttnn::prim::TransposeOpDim transpose_dim,
    const MemoryConfig& output_mem_config,
    float pad_value = 0.0f) {
    auto prim_permute = [&](const ttnn::Tensor& input, const ttnn::SmallVector<uint32_t>& dims) -> ttnn::Tensor {
        return ttnn::prim::permute(input, dims, output_mem_config, std::nullopt, pad_value);
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
            break;
        default: break;
    }
    return ttnn::prim::transpose(a, transpose_dim, output_mem_config, pad_value);
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

}  // namespace detail

using OwnedTransposeArgs = std::tuple<ttnn::Tensor, int64_t, int64_t, std::optional<MemoryConfig>, float>;
using BaseTransposeType =
    std::function<ttnn::Tensor(const ttnn::Tensor&, int64_t, int64_t, const std::optional<MemoryConfig>&, float)>;

using MassagedTranspose = MassagedOperation<
    ttnn::Tensor,
    const ttnn::Tensor&,
    int64_t,
    int64_t,
    const std::optional<MemoryConfig>&,
    const float>;
using MassagedTransposeParams = MassagedOperationParams<
    ttnn::Tensor,
    const ttnn::Tensor&,
    int64_t,
    int64_t,
    const std::optional<MemoryConfig>&,
    const float>;

bool shard_not_supported(const ttnn::Tensor& input_tensor, int64_t dim1, int64_t dim2) {
    const auto& input_shape = input_tensor.logical_shape();
    uint32_t normalized_dim1 = input_shape.get_normalized_index(dim1);
    uint32_t normalized_dim2 = input_shape.get_normalized_index(dim2);

    uint32_t initial_rank = input_shape.rank();
    if (initial_rank < 4) {
        uint32_t rank_diff = 4 - initial_rank;
        normalized_dim1 += rank_diff;
        normalized_dim2 += rank_diff;
    }

    if (normalized_dim1 > normalized_dim2) {
        std::swap(normalized_dim1, normalized_dim2);
    }

    // diff output shard spec is not supported for HC transpose
    bool not_supported = normalized_dim2 == 2 && normalized_dim1 == 1;
    return not_supported;
}

// transpose::hc does not use the shard spec of the output memory config
// massage the op to use the output memory config properly
MassagedTranspose build_memory_config_transpose(BaseTransposeType base_transpose) {
    auto target_memory_config = std::make_shared<std::optional<MemoryConfig>>();
    return MassagedTranspose(MassagedTransposeParams{
        .predicate = [target_memory_config](
                         const ttnn::Tensor& input_tensor,
                         int64_t dim1,
                         int64_t dim2,
                         const std::optional<MemoryConfig>& memory_config,
                         const float /*pad_value*/) -> bool {
            *target_memory_config = memory_config;
            if (!memory_config.has_value()) {
                return false;
            }
            auto input_mem_config_sharded = input_tensor.memory_config().is_sharded();
            auto output_mem_config_sharded = memory_config.value().is_sharded();
            bool massage_op = input_tensor.memory_config() != memory_config.value() && output_mem_config_sharded &&
                              input_mem_config_sharded;
            massage_op = massage_op && shard_not_supported(input_tensor, dim1, dim2);
            return massage_op;
        },
        .pre_transform = [](const ttnn::Tensor& input_tensor,
                            int64_t dim1,
                            int64_t dim2,
                            const std::optional<MemoryConfig>& /*memory_config*/,
                            const float pad_value) -> OwnedTransposeArgs {
            return std::make_tuple(input_tensor, dim1, dim2, input_tensor.memory_config(), pad_value);
        },
        .post_transform = [target_memory_config](const ttnn::Tensor& output) -> ttnn::Tensor {
            if (target_memory_config->has_value() && output.memory_config() != target_memory_config->value()) {
                return ttnn::to_memory_config(output, target_memory_config->value());
            }
            return output;
        },
        .operation = std::move(base_transpose)});
}

ttnn::Tensor transpose_impl(
    const ttnn::Tensor& input_tensor,
    int64_t dim1,
    int64_t dim2,
    const std::optional<MemoryConfig>& memory_config_arg,
    float pad_value = 0.0f) {
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
    bool typecast =
        input_unsqueezed.dtype() == DataType::BFLOAT8_B and !bfloat8_supported and !input_unsqueezed.is_sharded();
    Tensor input_typecasted = typecast ? ttnn::typecast(input_unsqueezed, DataType::BFLOAT16) : input_unsqueezed;

    auto memory_config = memory_config_arg.value_or(input_typecasted.memory_config());

    TT_FATAL(normalized_dim1 <= 3, "dimension has to be 0-3 only corresponding to N,C,H,W");
    TT_FATAL(normalized_dim2 <= 3, "dimension has to be 0-3 only corresponding to N,C,H,W");

    Tensor output;
    if ((normalized_dim1 == normalized_dim2) || (input_typecasted.padded_shape()[normalized_dim1] == 1 &&
                                                 input_typecasted.padded_shape()[normalized_dim2] == 1)) {
        if (input_typecasted.memory_config() != memory_config) {
            output = ttnn::clone(input_typecasted, std::nullopt, memory_config, std::nullopt);
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
        output = detail::transpose_(input_typecasted, transpose_dim, memory_config, pad_value);
    }
    output = initial_rank < 4u ? ttnn::squeeze_from_4D(output, initial_rank) : output;
    return typecast ? ttnn::typecast(output, DataType::BFLOAT8_B) : output;
}

ttnn::Tensor ExecuteTranspose::invoke(
    const ttnn::Tensor& input_tensor,
    int64_t dim1,
    int64_t dim2,
    const std::optional<MemoryConfig>& memory_config,
    float pad_value) {
    auto base_transpose = [](const ttnn::Tensor& input_tensor,
                             int64_t dim1,
                             int64_t dim2,
                             const std::optional<MemoryConfig>& memory_config,
                             float pad_value) {
        return transpose_impl(input_tensor, dim1, dim2, memory_config, pad_value);
    };

    return build_memory_config_transpose(base_transpose)(input_tensor, dim1, dim2, memory_config, pad_value);
}

ttnn::Tensor ExecuteTranspose::invoke(const ttnn::Tensor& input_tensor, int64_t dim1, int64_t dim2, float pad_value) {
    return invoke(input_tensor, dim1, dim2, std::nullopt, pad_value);
}

}  // namespace ttnn::operations::data_movement::transpose
