// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn::operations::data_movement {
namespace detail {

ttnn::Tensor permute_impl(
    const ttnn::Tensor& a,
    const ttnn::SmallVector<uint32_t>& dims,
    const MemoryConfig& output_mem_config,
    const std::optional<float>& pad_value) {
    // Get the device
    IDevice* device = a.device();
    uint32_t rank = a.logical_shape().rank();

    auto prim_permute = [&](const ttnn::Tensor& input) -> ttnn::Tensor {
        return ttnn::prim::permute(input, dims, output_mem_config, std::nullopt, pad_value);
    };

    if (rank > 4) {
        return prim_permute(a);
    }

    TT_FATAL(dims.size() == 4, "Only 4D tensor are supported for permute.");
    uint32_t N = dims[0], C = dims[1], H = dims[2], W = dims[3];

    auto formatted_input_tensor = a;
    // WH and CN should be supported without typecast
    bool wh = N == 0 && C == 1 && H == 3 && W == 2;
    bool cn = N == 1 && C == 0 && H == 2 && W == 3;
    bool cnwh = N == 1 && C == 0 && H == 3 && W == 2;
    bool bfloat8_supported = wh || cn || cnwh;
    bool typecast = formatted_input_tensor.dtype() == DataType::BFLOAT8_B and !bfloat8_supported && !a.is_sharded();
    formatted_input_tensor =
        typecast ? ttnn::typecast(formatted_input_tensor, DataType::BFLOAT16) : formatted_input_tensor;

    auto output = formatted_input_tensor;
    auto transpose_wh = [&](const ttnn::Tensor& input) -> ttnn::Tensor {
        return ttnn::transpose(input, -2, -1, output_mem_config, std::nullopt);
    };

    auto transpose_hc = [&](const ttnn::Tensor& input) -> ttnn::Tensor {
        return ttnn::transpose(input, 1, -2, output_mem_config, pad_value);
    };

    auto transpose_cn = [&](const ttnn::Tensor& input) -> ttnn::Tensor {
        return ttnn::transpose(input, 0, 1, output_mem_config, std::nullopt);
    };

    // Keep limited sharding support with recursive calls
    if (a.is_sharded()) {
        if (N == 0 && C == 1 && H == 2 && W == 3) {
            output = formatted_input_tensor;
        } else if (N == 0 && C == 1 && H == 3 && W == 2) {
            output = transpose_wh(formatted_input_tensor);
        } else if (N == 0 && C == 2 && H == 1 && W == 3) {
            output = transpose_hc(formatted_input_tensor);
        } else if (N == 0 && C == 2 && H == 3 && W == 1) {
            output = transpose_wh(transpose_hc(formatted_input_tensor));
        } else if (N == 0 && C == 3 && H == 1 && W == 2) {
            output = transpose_hc(transpose_wh(formatted_input_tensor));
        } else if (N == 0 && C == 3 && H == 2 && W == 1) {
            output = transpose_wh(transpose_hc(transpose_wh(formatted_input_tensor)));
        } else {
            TT_FATAL(false, "Sharded permute not supported for this permutation");
        }
    } else {
        if (N == 0 && C == 1 && H == 2 && W == 3) {
            output = formatted_input_tensor;
        } else if (N == 0 && C == 1 && H == 3 && W == 2) {
            output = transpose_wh(formatted_input_tensor);
        } else if (N == 0 && C == 2 && H == 1 && W == 3) {
            output = transpose_hc(formatted_input_tensor);
        } else if (N == 1 && C == 0 && H == 2 && W == 3) {
            output = transpose_cn(formatted_input_tensor);
        } else {
            output = prim_permute(formatted_input_tensor);
        }
    }
    // Convert tensor back to original dtype if typecast was performed
    output = typecast ? ttnn::typecast(output, DataType::BFLOAT8_B) : output;
    return output;
}

ttnn::Tensor permute_launch(
    const ttnn::Tensor& a,
    const ttnn::SmallVector<uint32_t>& dims,
    const MemoryConfig& output_mem_config,
    const std::optional<float>& pad_value) {
    return permute_impl(a, dims, output_mem_config, pad_value);
}

bool is_permute_nop(const ttnn::Tensor& a, const ttnn::SmallVector<uint32_t>& dims) {
    // 1) Trivial early-out for rank <= 1
    const auto rank = a.logical_shape().rank();
    if (rank <= 1) {
        return true;
    }

    // 2) Check for identity permutation
    ttnn::SmallVector<uint32_t> seq_dims(rank);
    std::iota(seq_dims.begin(), seq_dims.end(), 0);
    if (dims == seq_dims) {
        return true;
    }

    // 3) Otherwise, when the input is tiled, it is never a NOP if the last two dimensions are permuted. When it is row
    // major, it is never a NOP if the last dimension is permuted.
    if (a.layout() == Layout::TILE && (dims[rank - 1] != rank - 1 || dims[rank - 2] != rank - 2)) {
        return false;
    } else if (a.layout() == Layout::ROW_MAJOR && dims[rank - 1] != rank - 1) {
        return false;
    }

    // Build permuted shape
    const auto& shape = a.logical_shape();
    ttnn::SmallVector<uint32_t> perm_shape(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        perm_shape[i] = shape[dims[i]];
    }

    // 4) If the shape changed, definitely not a no-op
    if (perm_shape != shape) {
        return false;
    }

    // 5) If the shape stayed the same, ensure we didn't
    //    relocate a dimension with size > 1
    for (uint32_t i = 0; i < rank; ++i) {
        const uint32_t j = dims[i];
        if (i != j && shape[i] > 1) {
            // Moved a dimension that has > 1 elements
            // => layout changed => not a no-op.
            return false;
        }
    }

    // If we made it here, we either
    //    - only moved dimensions of size 1, or
    //    - didn't move anything at all
    return true;
}

}  // namespace detail

ttnn::Tensor ExecutePermute::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<float>& pad_value) {
    const auto input_rank = input_tensor.logical_shape().rank();
    TT_FATAL(
        input_rank == dims.size(),
        "The number of dimensions in the tensor input does not match the length of the desired ordering");
    TT_FATAL(is_device_tensor(input_tensor), "Tensor must already be on device");

    SmallVector<uint32_t> normalized_dims(dims.size());
    std::transform(dims.begin(), dims.end(), normalized_dims.begin(), [input_tensor](std::int64_t idx) {
        return input_tensor.logical_shape().get_normalized_index(idx);
    });
    if (detail::is_permute_nop(input_tensor, normalized_dims)) {
        return ttnn::to_memory_config(input_tensor, memory_config.value_or(input_tensor.memory_config()));
    }

    auto adjust_order = [](const ttnn::SmallVector<uint32_t>& dims) {
        ttnn::SmallVector<uint32_t> new_order;
        TT_FATAL(dims.size() <= 4, "Minimum rank of tensor required is 4");
        int additional_ranks = 4 - dims.size();
        for (int i = 0; i < additional_ranks; i++) {
            new_order.push_back(i);
        }
        for (int i = 0; i < dims.size(); i++) {
            new_order.push_back(dims[i] + additional_ranks);
        }
        return new_order;
    };
    auto itensor = (input_tensor.logical_shape().rank() < 4) ? ttnn::unsqueeze_to_4D(input_tensor) : input_tensor;
    auto iorder = normalized_dims.size() < 4 ? adjust_order(normalized_dims) : normalized_dims;

    const auto input_layout = input_tensor.layout();
    auto output_tensor =
        detail::permute_launch(itensor, iorder, memory_config.value_or(input_tensor.memory_config()), pad_value);
    output_tensor = ttnn::to_layout(output_tensor, input_layout);

    if (input_rank < 4) {
        output_tensor = ttnn::squeeze_from_4D(output_tensor, input_rank);
    }

    return output_tensor;
}

ttnn::Tensor ExecutePermute::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<float>& pad_value) {
    return invoke(DefaultQueueId, input_tensor, dims, memory_config, pad_value);
}

ttnn::Tensor ExecutePermute::invoke(
    const ttnn::Tensor& input_tensor, const ttnn::SmallVector<int64_t>& dims, const std::optional<float>& pad_value) {
    return invoke(input_tensor, dims, std::nullopt, pad_value);
}

}  // namespace ttnn::operations::data_movement
