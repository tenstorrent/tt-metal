// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "rmsnorm_fw_program_factory.hpp"
#include "rmsnorm_fw_device_operation_types.hpp"

namespace ttnn::operations::experimental::rmsnorm_fw {

struct RMSNormForwardDeviceOperation {
    using operation_attributes_t = rmsnorm_fw::operation_attributes_t;
    using tensor_args_t = rmsnorm_fw::tensor_args_t;
    using spec_return_value_t = rmsnorm_fw::spec_return_value_t;
    using tensor_return_value_t = rmsnorm_fw::tensor_return_value_t;
    using program_factory_t = std::variant<program::RMSNormForwardProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& gamma_tensor,
        bool return_intermediates,
        float epsilon = 1e-6F,
        const std::optional<Tensor>& preallocated_rms = std::nullopt,
        const std::optional<Tensor>& preallocated_output = std::nullopt);
};

}  // namespace ttnn::operations::experimental::rmsnorm_fw

namespace ttnn::prim {
constexpr auto rmsnorm_fw = ttnn::register_operation<
    "ttnn::prim::rmsnorm_fw",
    ttnn::operations::experimental::rmsnorm_fw::RMSNormForwardDeviceOperation>();
}  // namespace ttnn::prim
