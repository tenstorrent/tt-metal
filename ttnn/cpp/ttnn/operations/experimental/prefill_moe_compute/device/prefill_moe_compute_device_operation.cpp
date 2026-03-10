// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_moe_compute_device_operation.hpp"

namespace ttnn::operations::experimental::prefill_moe_compute {

void PrefillMoeComputeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

void PrefillMoeComputeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    TT_FATAL(attrs.num_experts > 0 && attrs.num_experts <= 4, "num_experts must be 1-4");
    TT_FATAL(attrs.grid_x * attrs.grid_y == attrs.num_cores, "grid_x * grid_y must equal num_cores");
    TT_FATAL(tensor_args.gate_up_weights.size() == attrs.num_experts, "gate_up_weights size must match num_experts");
    TT_FATAL(tensor_args.down_weights.size() == attrs.num_experts, "down_weights size must match num_experts");
    TT_FATAL(tensor_args.out_bufs.size() == attrs.num_experts, "out_bufs size must match num_experts");
    TT_FATAL(!attrs.per_device_combine_metadata.empty(), "per_device_combine_metadata must have at least one entry");
    if (attrs.enable_fabric_reduce) {
        TT_FATAL(
            tensor_args.reduce_recv_buf.has_value(), "reduce_recv_buf is required when enable_fabric_reduce is true");
    }
    if (attrs.enable_fabric_dispatch) {
        TT_FATAL(
            tensor_args.hidden_states_rm.has_value(),
            "hidden_states_rm is required when enable_fabric_dispatch is true");
        TT_FATAL(tensor_args.staging_buf.has_value(), "staging_buf is required when enable_fabric_dispatch is true");
        TT_FATAL(!attrs.dispatch_metadata.empty(), "dispatch_metadata is required when enable_fabric_dispatch is true");
    }
}

spec_return_value_t PrefillMoeComputeDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.output.tensor_spec();
}

tensor_return_value_t PrefillMoeComputeDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.output;
}

std::tuple<PrefillMoeComputeDeviceOperation::operation_attributes_t, PrefillMoeComputeDeviceOperation::tensor_args_t>
PrefillMoeComputeDeviceOperation::invoke(
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
    const std::optional<Tensor>& hidden_states_rm,
    const std::optional<Tensor>& staging_buf,
    bool enable_fabric_dispatch,
    const std::vector<std::vector<uint32_t>>& dispatch_metadata,
    const std::vector<uint32_t>& dispatch_target_cols,
    const std::optional<std::vector<std::vector<uint32_t>>>& per_expert_dispatch_sources,
    const std::optional<std::vector<std::vector<uint32_t>>>& multi_dest_dispatch_metadata,
    bool enable_fpu_combine) {
    return {
        operation_attributes_t{
            num_experts,
            num_cores,
            grid_x,
            grid_y,
            per_device_combine_metadata,
            enable_fabric_reduce,
            enable_fabric_dispatch,
            dispatch_metadata,
            dispatch_target_cols,
            per_expert_dispatch_sources,
            multi_dest_dispatch_metadata,
            enable_fpu_combine},
        tensor_args_t{
            hidden_states,
            pkt_buf,
            inter_buf,
            output,
            gate_up_weights,
            down_weights,
            out_bufs,
            reduce_recv_buf,
            hidden_states_rm,
            staging_buf}};
}

}  // namespace ttnn::operations::experimental::prefill_moe_compute
