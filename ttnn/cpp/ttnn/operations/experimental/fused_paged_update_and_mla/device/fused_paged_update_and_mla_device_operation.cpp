// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/fused_paged_update_and_mla/device/fused_paged_update_and_mla_device_operation.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::constants;

namespace ttnn::operations::experimental::fused_paged_update_and_mla {

void FusedPagedUpdateAndMlaDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() >= 2, "Error");
}

std::vector<ttnn::Shape> FusedPagedUpdateAndMlaDeviceOperation::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.logical_shape()};
}

std::vector<Tensor> FusedPagedUpdateAndMlaDeviceOperation::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return {input_tensors.at(0)};
}

tt::tt_metal::operation::ProgramWithCallbacks FusedPagedUpdateAndMlaDeviceOperation::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    // Factory method overrides this
    return {tt::tt_metal::CreateProgram(), {}};
}

} // namespace ttnn::operations::experimental::fused_paged_update_and_mla