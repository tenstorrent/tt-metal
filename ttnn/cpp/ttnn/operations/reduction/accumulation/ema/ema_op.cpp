// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ema_op.hpp"

#include <optional>

#include "ttnn/operations/math.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::reduction::accumulation {

void Ema::validate(const std::vector<Tensor>& input_tensors) const {}

std::vector<TensorSpec> Ema::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.tensor_spec()};
}

std::vector<Tensor> Ema::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    return operation::default_create_output_tensors(*this, input_tensors, optional_output_tensors);
}

operation::ProgramWithCallbacks Ema::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    return ema_multi_core(a, output_tensor, this->alpha, this->grid_size, this->compute_kernel_config);
}

}  // namespace ttnn::operations::reduction::accumulation
