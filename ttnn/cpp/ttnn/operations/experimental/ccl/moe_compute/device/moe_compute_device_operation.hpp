// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "moe_compute_device_operation_types.hpp"
#include "moe_compute_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct MoEComputeDeviceOperation {
    using operation_attributes_t = MoEComputeParams;
    using tensor_args_t = MoEComputeInputs;
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<ttnn::Tensor>;
    using program_factory_t = std::variant<MoEComputeMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> moe_compute(
    const ttnn::Tensor& tilize_input_tensor,
    const ttnn::Tensor& tilize_expert_indices_tensor,
    const ttnn::Tensor& tilize_expert_scores_tensor,
    const ttnn::Tensor& tilize_expert_mapping_tensor,
    const ttnn::Tensor& matmul_w0_w1_tensor,
    const ttnn::Tensor& matmul_w2_tensor,
    const uint32_t layer_id,
    const std::optional<uint32_t> cluster_axis);

}  // namespace ttnn::prim
