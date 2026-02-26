// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "permute.hpp"

#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"

#include <tt-metalium/hal.hpp>
#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn::operations::data_movement {
namespace detail {

ttnn::Tensor permute_impl(
    const ttnn::Tensor& a,
    const ttnn::SmallVector<uint32_t>& dims,
    const MemoryConfig& output_mem_config,
    float pad_value = 0.0f) {
    // Get the device
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
        return ttnn::transpose(input, -2, -1, output_mem_config, 0.0f);
    };

    auto transpose_hc = [&](const ttnn::Tensor& input) -> ttnn::Tensor {
        // some permute tests assume transpose hc uses the input shard spec
        // avoid the intermediate memory configuration mismatch
        auto mem_config = output_mem_config;
        if (input.memory_config().is_sharded() && output_mem_config.is_sharded()) {
            mem_config = input.memory_config();
        }
        return ttnn::transpose(input, 1, -2, mem_config, pad_value);
    };

    auto transpose_cn = [&](const ttnn::Tensor& input) -> ttnn::Tensor {
        return ttnn::transpose(input, 0, 1, output_mem_config, 0.0f);
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
            output = prim_permute(formatted_input_tensor);
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
    float pad_value = 0.0f) {
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
    if ((a.layout() == Layout::TILE && (dims[rank - 1] != rank - 1 || dims[rank - 2] != rank - 2)) ||
        (a.layout() == Layout::ROW_MAJOR && dims[rank - 1] != rank - 1)) {
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
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    float pad_value) {
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
        for (unsigned int dim : dims) {
            new_order.push_back(dim + additional_ranks);
        }
        return new_order;
    };
    auto itensor = (input_tensor.logical_shape().rank() < 4) ? ttnn::unsqueeze_to_4D(input_tensor) : input_tensor;
    auto iorder = normalized_dims.size() < 4 ? adjust_order(normalized_dims) : normalized_dims;

    const auto input_layout = input_tensor.layout();
    const auto output_memory_config = memory_config.value_or(input_tensor.memory_config());

    if (input_layout == Layout::ROW_MAJOR) {
        uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
        TT_FATAL(
            !output_memory_config.is_sharded() ||
                (*output_memory_config.shard_spec()).shape[1] * input_tensor.element_size() % (l1_alignment) == 0,
            "Shard page size must be aligned to {}B for L1 Tensor",
            l1_alignment);
    }
    auto output_tensor = detail::permute_launch(itensor, iorder, output_memory_config, pad_value);
    output_tensor = ttnn::to_layout(output_tensor, input_layout);

    if (input_rank < 4) {
        output_tensor = ttnn::squeeze_from_4D(output_tensor, input_rank);
    }

    return output_tensor;
}

ttnn::Tensor ExecutePermute::invoke(
    const ttnn::Tensor& input_tensor, const ttnn::SmallVector<int64_t>& dims, float pad_value) {
    return invoke(input_tensor, dims, std::nullopt, pad_value);
}

}  // namespace ttnn::operations::data_movement
