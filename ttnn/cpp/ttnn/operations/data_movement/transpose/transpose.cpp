// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "device/transpose_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

#include <tt-metalium/hal.hpp>

namespace ttnn::operations::data_movement {

namespace detail {

using namespace tt::tt_metal::experimental;
using namespace tt;
using namespace tt::tt_metal::operation;

inline Tensor transpose_(
    const Tensor& a,
    TransposeOpDim transpose_dim,
    const MemoryConfig& output_mem_config,
    const std::optional<float>& pad_value) {
    auto prim_permute = [&](const ttnn::Tensor& input, ttnn::SmallVector<uint32_t> dims) -> ttnn::Tensor {
        return ttnn::prim::permute(input, dims, output_mem_config, std::nullopt, pad_value);
    };

    bool interleaved_rm = !a.is_sharded() && a.layout() == Layout::ROW_MAJOR;
    switch (transpose_dim) {
        case TransposeOpDim::HC:
            if (interleaved_rm) {
                return prim_permute(a, ttnn::SmallVector<uint32_t>{0, 2, 1, 3});
            }
            break;
        case TransposeOpDim::NH:
            return ttnn::permute(
                (const ttnn::Tensor)a, ttnn::SmallVector<int64_t>({2, 1, 0, 3}), output_mem_config, pad_value);
        case TransposeOpDim::NW:
            return ttnn::permute(
                (const ttnn::Tensor)a, ttnn::SmallVector<int64_t>({3, 1, 2, 0}), output_mem_config, pad_value);
        case TransposeOpDim::CW:
            return ttnn::permute(
                (const ttnn::Tensor)a, ttnn::SmallVector<int64_t>({0, 3, 2, 1}), output_mem_config, pad_value);
        case TransposeOpDim::CN:
            if (interleaved_rm) {
                return prim_permute(a, ttnn::SmallVector<uint32_t>({1, 0, 2, 3}));
            }
            break;
        case TransposeOpDim::WH:
            if (interleaved_rm) {
                return prim_permute(a, ttnn::SmallVector<uint32_t>({0, 1, 3, 2}));
            }
            break;
        default: break;
    }
    return tt::tt_metal::operation::run(Transpose{transpose_dim, output_mem_config, pad_value}, {a}).at(0);
}

ttnn::Tensor transpose_nd(
    const ttnn::Tensor& input_tensor,
    const uint32_t dim1,
    const uint32_t dim2,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<float>& pad_value) {
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

ttnn::Tensor ExecuteTranspose::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const int64_t& dim1,
    const int64_t& dim2,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<float>& pad_value) {
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
        output = ttnn::operations::experimental::auto_format::AutoFormat::move_tensor_to_mem_config(
            input_typecasted, memory_config);
    } else {
        if (normalized_dim1 > normalized_dim2) {
            std::swap(normalized_dim1, normalized_dim2);
        }

        TransposeOpDim transpose_dim = TransposeOpDim::NW;

        if (normalized_dim2 == 3 && normalized_dim1 == 0) {
            transpose_dim = TransposeOpDim::NW;
        } else if (normalized_dim2 == 3 && normalized_dim1 == 1) {
            transpose_dim = TransposeOpDim::CW;
        } else if (normalized_dim2 == 3 && normalized_dim1 == 2) {
            transpose_dim = TransposeOpDim::WH;
        } else if (normalized_dim2 == 2 && normalized_dim1 == 0) {
            transpose_dim = TransposeOpDim::NH;
        } else if (normalized_dim2 == 2 && normalized_dim1 == 1) {
            transpose_dim = TransposeOpDim::HC;
        } else if (normalized_dim2 == 1 && normalized_dim1 == 0) {
            transpose_dim = TransposeOpDim::CN;
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
    const int64_t& dim1,
    const int64_t& dim2,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<float>& pad_value) {
    return invoke(DefaultQueueId, input_tensor, dim1, dim2, memory_config, pad_value);
}

ttnn::Tensor ExecuteTranspose::invoke(
    const ttnn::Tensor& input_tensor, const int64_t& dim1, const int64_t& dim2, const std::optional<float>& pad_value) {
    return invoke(DefaultQueueId, input_tensor, dim1, dim2, std::nullopt, pad_value);
}

}  // namespace ttnn::operations::data_movement
