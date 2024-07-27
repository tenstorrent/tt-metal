// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute.h"

#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/copy/copy_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"


namespace ttnn {
namespace operations::data_movement {

namespace permute {

namespace {

Tensor permute_launch(const Tensor &a, std::vector<std::int64_t> dims, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({a}))};
    operation::launch_with_autoformat(
        [dims, output_mem_config]  (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto& a = input_tensors.at(0);
            std::vector<uint32_t> normalized_dims(dims.size());
            std::transform(dims.begin(), dims.end(), normalized_dims.begin(), [a](std::int64_t idx) {return a.get_legacy_shape().get_normalized_index(idx);});
            std::vector<uint32_t> seq_dims(dims.size());
            std::iota(seq_dims.begin(), seq_dims.end(), 0);
            if (normalized_dims == seq_dims) {
                return {AutoFormat::move_tensor_to_mem_config(a, output_mem_config)};
            }
            return {operation::decorate_as_composite(__func__, permute_)(a, normalized_dims, output_mem_config)};
        }, {a}, output_tensors);
    return output_tensors.at(0);
}
}

ttnn::Tensor ExecutePermute::operator()(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::vector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config) {

    const bool initial_input_tensor_on_device = permute::is_on_device(input_tensor);
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

    if (permute::has_tile_padding(itensor)) {
        itensor = ttnn::to_layout(itensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
    }

    TT_FATAL(permute::is_on_device(itensor) and itensor.get_shape().rank() == 4);
    auto output_tensor =
        permute::permute_launch(itensor, iorder, memory_config.value_or(input_tensor.memory_config()));
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

    if (initial_input_tensor_on_device and not permute::is_on_device(output_tensor)) {
        output_tensor = ttnn::to_device(
            output_tensor, input_tensor.device(), memory_config.value_or(input_tensor.memory_config()));
    }

    return output_tensor;
}

auto ExecutePermute::operator()(
    const ttnn::Tensor& input_tensor,
    const std::vector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config) {
    return operator()(0, input_tensor, dims, memory_config);
}

auto ExecutePermute::operator()(const ttnn::Tensor& input_tensor, const std::vector<int64_t>& dims) {
    return operator()(input_tensor, dims, std::nullopt);
}


}
}
}  // namespace ttnn
