// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/moreh_softmax/moreh_softmax_op.hpp"
#include "device/softmax_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_device_operation.hpp"

namespace ttnn::operations::normalization {

ttnn::Tensor SoftmaxOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const int dim_arg,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {

    auto input_shape = input_tensor.get_shape();
    auto rank = input_shape.size();
    auto dim = dim_arg;
    if (dim < 0) {
        dim = rank + dim;
    }

    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    if (dim == rank - 1) {
        auto output_tensor = ttnn::prim::softmax(
            SoftmaxDeviceOperation::operation_attributes_t{
                .memory_config = memory_config.value_or(input_tensor.memory_config()),
                .compute_kernel_config = compute_kernel_config.value_or(DeviceComputeKernelConfig{}),
            },
            SoftmaxDeviceOperation::tensor_args_t{
                .input_tensor = input_tensor_4D,
            }
        );
        return ttnn::reshape(output_tensor, input_shape);
    } else {
        auto dim_4D = dim + 4 - rank;
        auto output_tensor = tt::operations::primary::moreh_softmax(input_tensor_4D, dim_4D);
        return ttnn::reshape(output_tensor, input_shape);
    }
}

ttnn::Tensor ScaleMaskSoftmaxOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<float> scale,
    const std::optional<const Tensor> mask,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const bool is_causal_mask,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {

    auto input_shape = input_tensor.get_shape();

    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    auto output_tensor = ttnn::prim::softmax(
        SoftmaxDeviceOperation::operation_attributes_t{
            .scale = scale,
            .memory_config = memory_config.value_or(input_tensor.memory_config()),
            .is_causal_mask = is_causal_mask,
            .compute_kernel_config = compute_kernel_config
        },
        SoftmaxDeviceOperation::tensor_args_t{
            .input_tensor = input_tensor_4D,
            .mask = mask
        }
    );
    return ttnn::reshape(output_tensor, input_shape);
}

ttnn::Tensor SoftmaxInPlaceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const SoftmaxProgramConfig& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {

    auto input_shape = input_tensor.get_shape();

    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    auto output_tensor = ttnn::prim::softmax(
        SoftmaxDeviceOperation::operation_attributes_t{
            .inplace = true,
            .program_config = program_config,
            .compute_kernel_config = compute_kernel_config
        },
        SoftmaxDeviceOperation::tensor_args_t{
            .input_tensor = input_tensor_4D
        }
    );
    return ttnn::reshape(output_tensor, input_shape);
}

ttnn::Tensor ScaleMaskSoftmaxInPlaceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<float> scale,
    const std::optional<const Tensor> mask,
    const SoftmaxProgramConfig& program_config,
    const bool is_causal_mask,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {

    auto input_shape = input_tensor.get_shape();

    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    auto output_tensor = ttnn::prim::softmax(
        SoftmaxDeviceOperation::operation_attributes_t{
            .scale = scale,
            .inplace = true,
            .program_config = program_config,
            .is_causal_mask = is_causal_mask,
            .compute_kernel_config = compute_kernel_config
        },
        SoftmaxDeviceOperation::tensor_args_t{
            .input_tensor = input_tensor_4D,
            .mask = mask
        }
    );
    return ttnn::reshape(output_tensor, input_shape);
}

ttnn::Tensor ScaleCausalMaskHWSoftmaxInPlaceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<float> scale,
    const std::optional<const Tensor> mask,
    const SoftmaxProgramConfig& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {

    auto input_shape = input_tensor.get_shape();

    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    auto output_tensor = ttnn::prim::softmax(
        SoftmaxDeviceOperation::operation_attributes_t{
            .scale = scale,
            .inplace = true,
            .program_config = program_config,
            .is_causal_mask = true,
            .compute_kernel_config = compute_kernel_config,
            .is_scale_causal_mask_hw_dims_softmax = true
        },
        SoftmaxDeviceOperation::tensor_args_t{
            .input_tensor = input_tensor_4D,
            .mask = mask
        }
    );
    return ttnn::reshape(output_tensor, input_shape);
}

}  // namespace ttnn::operations::normalization
