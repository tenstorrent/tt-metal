// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"


namespace ttnn::operations::data_movement {
namespace detail {

inline bool is_on_device(const Tensor& t) {
    return ttnn::has_storage_type_of(t, ttnn::StorageType::DEVICE) or
    ttnn::has_storage_type_of(t, ttnn::StorageType::MULTI_DEVICE);
}

inline bool has_tile_padding(const Tensor& t) {
    if (t.get_shape().rank() > 1) {
        auto the_shape = t.get_shape();
        auto the_shape_with_padding = t.get_shape().with_tile_padding();
        return the_shape[-1] != the_shape_with_padding[-1] or the_shape[-2] != the_shape_with_padding[-2];
    }
    return false;
}

ttnn::Tensor permute_impl(const ttnn::Tensor &a, const std::vector<uint32_t>& dims, const MemoryConfig& output_mem_config) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    Device * device;

    // Get the device
    if (a.storage_type() != StorageType::DEVICE) {
        device = AutoFormat::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = a.device();
    }

    TT_FATAL(dims.size() == 4, "Only 4D tensor are supported for permute.");
    uint32_t N = dims[0], C = dims[1], H = dims[2], W = dims[3];

    bool pad_n = H == 0 || W == 0;
    bool pad_c = H == 1 || W == 1;
    // Convert tensor back to original
    auto a_pad_shape = AutoFormat::pad_to_tile_shape(a.get_legacy_shape(), pad_c, pad_n);
    auto out_shape = a.get_legacy_shape();
    out_shape = {out_shape[N], out_shape[C], out_shape[H], out_shape[W]};

    auto formatted_input_tensor = a;
    if (!AutoFormat::check_input_tensor_format(a, a_pad_shape)) {
        formatted_input_tensor = AutoFormat::format_input_tensor(a, device, a_pad_shape, 0.0, Layout::TILE);
    }
    auto output = formatted_input_tensor;
    static auto transpose_wh = std::bind(ttnn::transpose, std::placeholders::_1, -2, -1, output_mem_config);
    static auto transpose_hc = std::bind(ttnn::transpose, std::placeholders::_1, 1, -2, output_mem_config);
    static auto transpose_cn = std::bind(ttnn::transpose, std::placeholders::_1, 0, 1, output_mem_config);
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
        output = transpose_wh(transpose_hc(transpose_wh(transpose_cn(transpose_hc(transpose_wh(formatted_input_tensor))))));
    } else {
        TT_ASSERT(false, "Illegal permute args");
    }
    return AutoFormat::format_output_tensor(output, out_shape, device, Layout::TILE);
}

ttnn::Tensor permute_launch(const ttnn::Tensor &a, const std::vector<std::int64_t>& dims, const MemoryConfig& output_mem_config) {
    std::vector<ttnn::Tensor> output_tensors = {ttnn::Tensor(operation::get_workers_for_op_output({a}))};
    operation::launch_with_autoformat(
        [dims, output_mem_config]  (const std::vector<ttnn::Tensor>& input_tensors, const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors, const std::vector<std::optional<ttnn::Tensor>>& optional_output_tensors) mutable -> std::vector<ttnn::Tensor> {
            auto& a = input_tensors.at(0);
            std::vector<uint32_t> normalized_dims(dims.size());
            std::transform(dims.begin(), dims.end(), normalized_dims.begin(), [a](std::int64_t idx) {return a.get_legacy_shape().get_normalized_index(idx);});
            std::vector<uint32_t> seq_dims(dims.size());
            std::iota(seq_dims.begin(), seq_dims.end(), 0);
            if (normalized_dims == seq_dims) {
                return {ttnn::operations::experimental::auto_format::AutoFormat::move_tensor_to_mem_config(a, output_mem_config)};
            }
            return {permute_impl(a, normalized_dims, output_mem_config)};
        }, {a}, output_tensors);
    return output_tensors.at(0);
}

Tensor composite_invoke(
    const ttnn::Tensor& input_tensor,
    const std::vector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config) {

    auto output_tensor = permute_launch(input_tensor, dims, memory_config.value_or(input_tensor.memory_config()));
    return output_tensor;
}

} // detail namespace

ttnn::Tensor ExecutePermute::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::vector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    bool composite) {

    if (composite)
        return detail::composite_invoke(input_tensor, dims, memory_config);

    const bool initial_input_tensor_on_device = detail::is_on_device(input_tensor);
    const auto input_layout = input_tensor.get_layout();
    const auto input_rank = input_tensor.get_shape().rank();

    TT_FATAL(input_rank <= 4);
    TT_FATAL(
        input_rank == dims.size(),
        "The number of dimensions in the tensor input does not match the length of the desired ordering");

    auto adjust_order = [](const std::vector<int64_t>& dims) {
        std::vector<std::int64_t> new_order;
        TT_FATAL(dims.size() <= 4);
        int additional_ranks = 4 - dims.size();
        for (int i = 0; i < additional_ranks; i++) {
            new_order.push_back(i);
        }
        for (int i = 0; i < dims.size(); i++) {
            new_order.push_back(dims.at(i) + additional_ranks);
        }
        return new_order;
    };
    auto itensor = (input_tensor.get_shape().rank() < 4) ? ttnn::unsqueeze_to_4D(input_tensor) : input_tensor;
    auto iorder = adjust_order(dims);  // internals of permute_impl already adjust negative indices

    if (detail::has_tile_padding(itensor)) {
        itensor = ttnn::to_layout(itensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
    }

    TT_FATAL(detail::is_on_device(itensor) and itensor.get_shape().rank() == 4);
    auto output_tensor = detail::permute_launch(itensor, iorder, memory_config.value_or(input_tensor.memory_config()));
    output_tensor = ttnn::to_layout(output_tensor, input_layout, std::nullopt, std::nullopt, (Device*)nullptr);

    if (input_rank < 4) {
        const auto shape = output_tensor.get_shape();
        const auto full_shape = output_tensor.get_shape().with_tile_padding();
        std::vector<uint32_t> shape_vec{};
        std::vector<uint32_t> full_shape_vec{};
        int i = 0;
        while (i < 3 and shape[i] == 1) i++;
        for (; i < shape.rank(); i++) {
            shape_vec.push_back(shape[i]);
            full_shape_vec.push_back(full_shape[i]);
        }
        output_tensor = ttnn::reshape(output_tensor, ttnn::Shape(shape_vec, full_shape_vec));
    }

    if (initial_input_tensor_on_device and not detail::is_on_device(output_tensor)) {
        output_tensor = ttnn::to_device(
            output_tensor, input_tensor.device(), memory_config.value_or(input_tensor.memory_config()));
    }

    return output_tensor;
}

ttnn::Tensor ExecutePermute::invoke(
    const ttnn::Tensor& input_tensor,
    const std::vector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config) {
    return invoke(DefaultQueueId, input_tensor, dims, memory_config);
}

ttnn::Tensor ExecutePermute::invoke(const ttnn::Tensor& input_tensor, const std::vector<int64_t>& dims) {
    return invoke(input_tensor, dims, std::nullopt);
}

} // ttnn::operations::data_movement namespace
