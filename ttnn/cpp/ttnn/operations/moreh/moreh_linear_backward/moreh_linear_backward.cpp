// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_linear_backward.hpp"

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/cpp/ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp"
#include "ttnn/cpp/ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"

namespace ttnn::operations::moreh::moreh_linear_backward {

std::tuple<bool, bool, bool> MorehLinearBackward::get_required_outputs(const std::vector<bool>& are_required_outputs) {
    if (are_required_outputs.size() != 3) {
        TT_FATAL(are_required_outputs.size() == 3, "are_required_outputs size must be 3");
    }

    return {are_required_outputs[0], are_required_outputs[1], are_required_outputs[2]};
}

void get_tensor_dim(std::vector<uint32_t>& dim, const tt::tt_metal::LegacyShape& shape) {
    const auto rank = shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;

        // last 2-dim
        if (idx == rank - 1 || idx == rank - 2) {
            dim[i] = shape[idx] / tt::constants::TILE_HEIGHT;
        } else {
            dim[i] = shape[idx];
        }
    }

    log_debug(tt::LogOp, "rank {}", rank);
    for (auto i = 0; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        log_debug(tt::LogOp, "dim[{}] = {}", i, dim[i]);
    }
}

inline void moreh_linear_backward_validate(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<const Tensor>& input_grad,
    const std::optional<const Tensor>& weight_grad,
    const std::optional<const Tensor>& bias_grad) {
    if (input_grad.has_value()) {
        const auto& input_grad_tensor = input_grad.value();
        TT_FATAL(
            tt::operations::primary::is_same_shape(input, input_grad_tensor), "both tensors should be the same shape");
    }

    if (weight_grad.has_value()) {
        const auto& weight_grad_tensor = weight_grad.value();
        TT_FATAL(
            tt::operations::primary::is_same_shape(weight, weight_grad_tensor),
            "both tensors should be the same shape");
    }

    if (bias_grad.has_value()) {
        const auto& bias_grad_tensor = bias_grad.value();
        TT_FATAL(
            tt::operations::primary::is_scalar(bias_grad_tensor) ||
                tt::operations::primary::is_1d_tensor(bias_grad_tensor),
            "bias_grad tensor should be 1d or scalar");
    }
}

std::vector<int64_t> find_reduce_dim(
    const tt::tt_metal::LegacyShape& a_shape, const tt::tt_metal::LegacyShape& b_shape) {
    std::vector<uint32_t> a_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    std::vector<uint32_t> b_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(a_dim, a_shape);
    get_tensor_dim(b_dim, b_shape);
    int32_t rank = std::max(a_shape.rank(), b_shape.rank());
    log_debug(tt::LogOp, "find_reduce_dim :{} rank {} a {} b {}", __LINE__, rank, a_shape.rank(), b_shape.rank());
    std::vector<int64_t> dims;
    // batch dims
    for (int i = 0; i < rank - 2; ++i) {
        int idx = rank - 1 - i;
        TT_FATAL(idx >= 0, "find_reduce_dim idx must be >= 0");
        if (a_dim[idx] != b_dim[idx]) {
            dims.push_back(i);
            log_debug(tt::LogOp, "find_reduce_dim :{} push {} dim", __LINE__, i);
        }
    }
    return dims;
}

bool is_same_batch_dim(const Tensor& tensor_a, const Tensor& tensor_b) {
    // check batch dims
    const auto& a_shape = tensor_a.get_shape().value;
    const auto& b_shape = tensor_b.get_shape().value;
    std::vector<uint32_t> a_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    std::vector<uint32_t> b_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(a_dim, a_shape);
    get_tensor_dim(b_dim, b_shape);
    for (auto i = 2; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        if (a_dim[i] != b_dim[i]) {
            log_debug(tt::LogOp, "{}:{} {} a_dim {} - b_dim {}", __func__, __LINE__, i, a_dim[i], b_dim[i]);
            return false;
        }
    }
    log_debug(tt::LogOp, "{}:{} batch dims are the same.", __func__, __LINE__);
    return true;
}

std::vector<Tensor> MorehLinearBackward::create_async_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
    const auto& output_grad = input_tensors.at(0);
    const auto& input = input_tensors.at(1);
    const auto& weight = input_tensors.at(2);
    return {
        Tensor(operation::get_workers_for_op_output({output_grad, input, weight})),
        Tensor(operation::get_workers_for_op_output({output_grad, input, weight})),
        Tensor(operation::get_workers_for_op_output({output_grad, input, weight}))};
}

std::vector<std::optional<Tensor>> MorehLinearBackward::invoke(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& weight,
    const std::vector<bool>& are_required_outputs,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& input_grad,
    const std::optional<Tensor>& weight_grad,
    const std::optional<Tensor>& bias_grad,
    const std::optional<ttnn::MemoryConfig>& input_grad_memory_config,
    const std::optional<ttnn::MemoryConfig>& weight_grad_memory_config,
    const std::optional<ttnn::MemoryConfig>& bias_grad_memory_config,
    const DeviceComputeKernelConfig compute_kernel_config) {
    std::vector<std::optional<Tensor>> result(3);
    const auto [input_required_grad, weight_required_grad, bias_required_grad] =
        get_required_outputs(are_required_outputs);

    TT_FATAL(
        output_grad.storage_type() == StorageType::DEVICE && input.storage_type() == StorageType::DEVICE &&
            weight.storage_type() == StorageType::DEVICE,
        "input and weight tensors need to be on device");

    TT_FATAL(output_grad.storage_type() == StorageType::DEVICE, "Error");
    auto kernel_config_val =
        init_device_compute_kernel_config(output_grad.device()->arch(), compute_kernel_config, MathFidelity::HiFi4);
    moreh_linear_backward_validate(output_grad, input, weight, input_grad, weight_grad, bias_grad);

    if (input_required_grad) {
        TT_FATAL(input_grad.has_value(), "input_grad tensor should not be std::nullopt");
        result[0] = ttnn::moreh_matmul(
            output_grad, weight, false, false, input_grad, std::nullopt, input_grad_memory_config, kernel_config_val);
    }

    if (weight_required_grad) {
        TT_FATAL(weight_grad.has_value(), "weight_grad tensor should not be std::nullopt");
        const auto& weight_grad_tensor = weight_grad.value();

        if (is_same_batch_dim(output_grad, weight_grad_tensor)) {
            ttnn::moreh_matmul(
                output_grad,
                input,
                true,
                false,
                weight_grad_tensor,
                std::nullopt,
                weight_grad_memory_config,
                kernel_config_val);
        } else {
            const auto& temp_weight_grad = ttnn::moreh_matmul(
                output_grad,
                input,
                true,
                false,
                std::nullopt,
                std::nullopt,
                weight_grad_memory_config,
                kernel_config_val);
            TT_FATAL(weight_grad.has_value(), "weight_grad tensor should not be std::nullopt");
            std::vector<int64_t> dims =
                find_reduce_dim(temp_weight_grad.get_legacy_shape(), weight_grad.value().get_legacy_shape());
            ttnn::moreh_sum(
                temp_weight_grad, dims, true, weight_grad.value(), weight_grad_memory_config, kernel_config_val);
        }
        result[1] = weight_grad_tensor;
    }

    if (bias_required_grad) {
        Tensor output_tensor = ttnn::prim::moreh_bias_add_backward(
            output_grad, bias, bias_grad, bias_grad_memory_config, kernel_config_val);
        result[2] = std::make_optional(output_tensor);
    }

    return result;
}

std::vector<bool> MorehLinearBackward::create_async_return_flag(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& weight,
    const std::vector<bool>& are_required_outputs,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& input_grad,
    const std::optional<Tensor>& weight_grad,
    const std::optional<Tensor>& bias_grad,
    const std::optional<ttnn::MemoryConfig>& input_grad_memory_config,
    const std::optional<ttnn::MemoryConfig>& weight_grad_memory_config,
    const std::optional<ttnn::MemoryConfig>& bias_grad_memory_config,
    const DeviceComputeKernelConfig compute_kernel_config) {
    return are_required_outputs;
}

}  // namespace ttnn::operations::moreh::moreh_linear_backward
