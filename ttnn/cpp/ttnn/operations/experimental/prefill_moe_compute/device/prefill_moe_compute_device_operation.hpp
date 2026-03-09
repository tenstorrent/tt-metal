// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "prefill_moe_compute_device_operation_types.hpp"
#include "prefill_moe_compute_program_factory.hpp"

namespace ttnn::operations::experimental::prefill_moe_compute {

struct PrefillMoeComputeDeviceOperation {
    using operation_attributes_t = prefill_moe_compute::operation_attributes_t;
    using tensor_args_t = prefill_moe_compute::tensor_args_t;
    using tensor_return_value_t = prefill_moe_compute::tensor_return_value_t;
    using spec_return_value_t = prefill_moe_compute::spec_return_value_t;

    using program_factory_t = std::variant<PrefillMoeComputeMeshFactory>;

    static void validate_on_program_cache_miss(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& hidden_states,
        const std::vector<Tensor>& gate_up_weights,
        const std::vector<Tensor>& down_weights,
        const Tensor& pkt_buf,
        const Tensor& inter_buf,
        const std::vector<Tensor>& out_bufs,
        const Tensor& output,
        const std::vector<std::vector<uint32_t>>& per_device_combine_metadata,
        uint32_t num_experts,
        uint32_t num_cores,
        uint32_t grid_x,
        uint32_t grid_y,
        const std::optional<Tensor>& reduce_recv_buf,
        bool enable_fabric_reduce,
        const std::optional<Tensor>& hidden_states_rm = std::nullopt,
        const std::optional<Tensor>& staging_buf = std::nullopt,
        bool enable_fabric_dispatch = false,
        const std::vector<std::vector<uint32_t>>& dispatch_metadata = {},
        const std::vector<uint32_t>& dispatch_target_cols = {});
};

}  // namespace ttnn::operations::experimental::prefill_moe_compute

namespace ttnn::experimental {

constexpr auto prefill_moe_compute = ttnn::register_operation<
    "ttnn::experimental::prefill_moe_compute",
    ttnn::operations::experimental::prefill_moe_compute::PrefillMoeComputeDeviceOperation>();

}  // namespace ttnn::experimental
