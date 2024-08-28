// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_bmm_backward/moreh_bmm_backward_op.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace operations {
namespace primary {

namespace {
inline void moreh_bmm_backward_validate(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mat2,
    std::optional<const Tensor> &input_grad,
    std::optional<const Tensor> &mat2_grad) {
    TT_ASSERT(
        output_grad.storage_type() == StorageType::DEVICE && input.storage_type() == StorageType::DEVICE &&
            mat2.storage_type() == StorageType::DEVICE,
        "input tensors need to be on device");

    const auto &output_grad_shape = output_grad.get_legacy_shape();
    const auto &input_shape = input.get_legacy_shape();
    const auto &mat2_shape = mat2.get_legacy_shape();
    TT_ASSERT(output_grad_shape.rank() == 3, "output_grad must be a 3D tensor");
    TT_ASSERT(input_shape.rank() == 3, "input must be a 3D tensor");
    TT_ASSERT(mat2_shape.rank() == 3, "mat2 must be a 3D tensor");

    if (input_grad.has_value()) {
        const auto &input_grad_shape = input_grad.value().get_legacy_shape();
        TT_ASSERT(input_grad_shape.rank() == 3, "input_grad must be a 3D tensor");
    }

    if (mat2_grad.has_value()) {
        const auto &mat2_grad_shape = mat2_grad.value().get_legacy_shape();
        TT_ASSERT(mat2_grad_shape.rank() == 3, "mat2_grad must be a 3D tensor");
    }
}
}

std::vector<std::optional<Tensor>> moreh_bmm_backward(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mat2,
    const std::vector<bool> &are_required_outputs,
    std::optional<const Tensor> input_grad,
    std::optional<const Tensor> mat2_grad,
    const MemoryConfig &input_grad_mem_config,
    const MemoryConfig &mat2_grad_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    std::vector<std::optional<Tensor>> outputs(2);
    outputs.reserve(2);

    const bool input_requires_grad = are_required_outputs.at(0);
    const bool mat2_requires_grad = are_required_outputs.at(1);

    if (input_requires_grad) {
        TT_ASSERT(input_grad.has_value());
        const auto &input_grad_tensor = input_grad.value();
        outputs[0] = moreh_matmul(
            output_grad,
            mat2,
            false,
            true,
            input_grad_tensor,
            std::nullopt,
            input_grad_mem_config,
            compute_kernel_config);
    }

    if (mat2_requires_grad) {
        TT_ASSERT(mat2_grad.has_value());
        const auto &mat2_grad_tensor = mat2_grad.value();
        outputs[1] = moreh_matmul(
            input,
            output_grad,
            true,
            false,
            mat2_grad_tensor,
            std::nullopt,
            mat2_grad_mem_config,
            compute_kernel_config);
    }

    return outputs;
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
