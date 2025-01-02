// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/cpp/ttnn/operations/copy.hpp"

namespace ttnn::operations::data_movement {
namespace detail {

inline bool is_on_device(const Tensor& t) {
    return ttnn::has_storage_type_of(t, ttnn::StorageType::DEVICE) or
           ttnn::has_storage_type_of(t, ttnn::StorageType::MULTI_DEVICE);
}

ttnn::Tensor permute_impl(
    const ttnn::Tensor& a,
    const ttnn::SmallVector<uint32_t>& dims,
    const MemoryConfig& output_mem_config,
    const std::optional<float>& pad_value) {
    using ttnn::operations::experimental::auto_format::AutoFormat;

    // Get the device
    Device* device = a.device();
    uint32_t rank = a.get_shape().rank();
    if (rank > 4) {
        if (a.get_layout() == Layout::TILE && ((dims[rank - 1] == rank - 1 && dims[rank - 2] == rank - 2)) ||
            (dims[rank - 1] == rank - 2 && dims[rank - 2] == rank - 1)) {
            return ttnn::prim::permute(a, dims, output_mem_config, std::nullopt);
        }

        auto input = a.get_layout() == Layout::TILE
                         ? ttnn::to_layout(a, Layout::ROW_MAJOR, std::nullopt, std::nullopt, (Device*)nullptr)
                         : a;
        TT_FATAL(
            !(pad_value.has_value() && pad_value.value() != 0.0f),
            "Non-zero padding is not supported for permute on tensors with rank > 4.");
        input = ttnn::prim::permute(input, dims, output_mem_config, std::nullopt);
        return ttnn::to_layout(input, a.get_layout(), std::nullopt, std::nullopt, (Device*)nullptr);
    }

    TT_FATAL(dims.size() == 4, "Only 4D tensor are supported for permute.");
    uint32_t N = dims[0], C = dims[1], H = dims[2], W = dims[3];

    auto formatted_input_tensor = a;
    // WH and CN should be supported without typecast
    bool wh = N == 0 && C == 1 && H == 3 && W == 2;
    bool cn = N == 1 && C == 0 && H == 2 && W == 3;
    bool cnwh = N == 1 && C == 0 && H == 3 && W == 2;
    bool bfloat8_supported = wh || cn || cnwh;
    bool typecast = formatted_input_tensor.get_dtype() == DataType::BFLOAT8_B and !bfloat8_supported && !a.is_sharded();
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
    } else if (N == 1 && C == 0 && H == 2 && W == 3) {
        output = transpose_cn(formatted_input_tensor);
    } else if (N == 1 && C == 0 && H == 3 && W == 2) {
        output = transpose_wh(transpose_cn(formatted_input_tensor));
    } else if (N == 1 && C == 2 && H == 0 && W == 3) {
        output = transpose_hc(transpose_cn(formatted_input_tensor));
    } else if (N == 1 && C == 2 && H == 3 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_cn(formatted_input_tensor)));
    } else if (N == 1 && C == 3 && H == 0 && W == 2) {
        output = transpose_hc(transpose_wh(transpose_cn(formatted_input_tensor)));
    } else if (N == 1 && C == 3 && H == 2 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_wh(transpose_cn(formatted_input_tensor))));
    } else if (N == 2 && C == 0 && H == 1 && W == 3) {
        output = transpose_cn(transpose_hc(formatted_input_tensor));
    } else if (N == 2 && C == 0 && H == 3 && W == 1) {
        output = transpose_wh(transpose_cn(transpose_hc(formatted_input_tensor)));
    } else if (N == 2 && C == 1 && H == 0 && W == 3) {
        output = transpose_cn(transpose_hc(transpose_cn(formatted_input_tensor)));
    } else if (N == 2 && C == 1 && H == 3 && W == 0) {
        output = transpose_wh(transpose_cn(transpose_hc(transpose_cn(formatted_input_tensor))));
    } else if (N == 2 && C == 3 && H == 0 && W == 1) {
        output = transpose_hc(transpose_wh(transpose_cn(transpose_hc(formatted_input_tensor))));
    } else if (N == 2 && C == 3 && H == 1 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_wh(transpose_cn(transpose_hc(formatted_input_tensor)))));
    } else if (N == 3 && C == 0 && H == 1 && W == 2) {
        output = transpose_cn(transpose_hc(transpose_wh(formatted_input_tensor)));
    } else if (N == 3 && C == 0 && H == 2 && W == 1) {
        output = transpose_wh(transpose_cn(transpose_hc(transpose_wh(formatted_input_tensor))));
    } else if (N == 3 && C == 1 && H == 0 && W == 2) {
        output = transpose_cn(transpose_hc(transpose_cn(transpose_wh(formatted_input_tensor))));
    } else if (N == 3 && C == 1 && H == 2 && W == 0) {
        output = transpose_wh(transpose_cn(transpose_hc(transpose_cn(transpose_wh(formatted_input_tensor)))));
    } else if (N == 3 && C == 2 && H == 0 && W == 1) {
        output = transpose_hc(transpose_wh(transpose_cn(transpose_hc(transpose_wh(formatted_input_tensor)))));
    } else if (N == 3 && C == 2 && H == 1 && W == 0) {
        output =
            transpose_wh(transpose_hc(transpose_wh(transpose_cn(transpose_hc(transpose_wh(formatted_input_tensor))))));
    } else {
        TT_ASSERT(false, "Illegal permute args");
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
    std::vector<ttnn::Tensor> output_tensors = {ttnn::Tensor(operation::get_workers_for_op_output({a}))};
    operation::launch_with_autoformat(
        [dims, output_mem_config, pad_value](
            const std::vector<ttnn::Tensor>& input_tensors,
            const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
            const std::vector<std::optional<ttnn::Tensor>>& optional_output_tensors) mutable
        -> std::vector<ttnn::Tensor> {
            auto& a = input_tensors.at(0);
            return {permute_impl(a, dims, output_mem_config, pad_value)};
        },
        {a},
        output_tensors);
    return output_tensors.at(0);
}

bool is_permute_nop(const ttnn::Tensor& a, const ttnn::SmallVector<uint32_t>& dims) {
    if (a.get_shape().rank() <= 1) {
        return true;
    }
    auto normalized_dims = ttnn::SmallVector<uint32_t>(dims.begin(), dims.end());
    ttnn::SmallVector<uint32_t> seq_dims(dims.size());
    std::iota(seq_dims.begin(), seq_dims.end(), 0);
    return normalized_dims == seq_dims;
}

}  // namespace detail

ttnn::Tensor ExecutePermute::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<float>& pad_value) {
    const auto input_rank = input_tensor.get_logical_shape().rank();
    TT_FATAL(
        input_rank == dims.size(),
        "The number of dimensions in the tensor input does not match the length of the desired ordering");
    TT_FATAL(detail::is_on_device(input_tensor), "Tensor must already be on device");

    SmallVector<uint32_t> normalized_dims(dims.size());
    std::transform(dims.begin(), dims.end(), normalized_dims.begin(), [input_tensor](std::int64_t idx) {
        return input_tensor.get_logical_shape().get_normalized_index(idx);
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
    auto itensor = (input_tensor.get_logical_shape().rank() < 4) ? ttnn::unsqueeze_to_4D(input_tensor) : input_tensor;
    auto iorder = normalized_dims.size() < 4 ? adjust_order(normalized_dims) : normalized_dims;

    const auto input_layout = input_tensor.get_layout();
    auto output_tensor =
        detail::permute_launch(itensor, iorder, memory_config.value_or(input_tensor.memory_config()), pad_value);
    output_tensor = ttnn::to_layout(output_tensor, input_layout, std::nullopt, std::nullopt, (Device*)nullptr);

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
