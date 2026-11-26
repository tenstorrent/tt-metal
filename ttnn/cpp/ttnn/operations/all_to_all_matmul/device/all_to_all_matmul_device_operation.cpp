// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_matmul_device_operation.hpp"

#include <string_view>

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::all_to_all_matmul {

using namespace tt::tt_metal;

namespace {

void refuse_unless(bool supported, std::string_view axis, const auto& value, std::string_view allowed) {
    if (!supported) {
        throw UnsupportedAxisValue(fmt::format("all_to_all_matmul: {}={} not in SUPPORTED [{}]", axis, value, allowed));
    }
}

std::string_view topology_name(tt::tt_fabric::Topology topology) {
    switch (topology) {
        case tt::tt_fabric::Topology::NeighborExchange: return "NeighborExchange";
        case tt::tt_fabric::Topology::Linear: return "Linear";
        case tt::tt_fabric::Topology::Ring: return "Ring";
        case tt::tt_fabric::Topology::Mesh: return "Mesh";
        case tt::tt_fabric::Topology::Torus: return "Torus";
    }
    return "Unknown";
}

// A scratch tensor sharing memory with another tensor of the call: same buffer type, overlapping per-bank ranges.
bool shares_memory(const Tensor& scratch, const Tensor& other) {
    const Buffer* a = scratch.buffer();
    const Buffer* b = other.buffer();
    if (a->buffer_type() != b->buffer_type()) {
        return false;
    }
    const uint64_t a0 = a->address();  // smuggled-rta-ok: aliasing check, not a runtime arg
    const uint64_t b0 = b->address();  // smuggled-rta-ok: aliasing check, not a runtime arg
    return a0 < b0 + b->aligned_size_per_bank() && b0 < a0 + a->aligned_size_per_bank();
}

void validate_scratch_placement(const AllToAllMatmulInputs& t) {
    TT_FATAL(t.scratch.has_value(), "all_to_all_matmul: scratch_tensor is required");
    const Tensor& scratch = *t.scratch;
    TT_FATAL(
        scratch.device() == t.input.device(), "all_to_all_matmul: scratch_tensor must be on the input's mesh device");
    TT_FATAL(
        !shares_memory(scratch, t.input) && !shares_memory(scratch, t.weight),
        "all_to_all_matmul: scratch_tensor shares memory with an input");
}

}  // namespace

void check_support(const AllToAllMatmulParams& attrs, const Tensor& input, const Tensor& weight) {
    refuse_unless(input.dtype() == DataType::BFLOAT16, "dtype", input.dtype(), "BFLOAT16");
    refuse_unless(
        weight.dtype() == DataType::BFLOAT16 || weight.dtype() == DataType::BFLOAT8_B,
        "weight_dtype",
        weight.dtype(),
        "BFLOAT16, BFLOAT8_B");
    refuse_unless(input.layout() == Layout::TILE, "layout", input.layout(), "TILE");
    // fp32_dest_acc_en: both values supported; alignment: both tags supported.
    const BufferType out_buffer = attrs.output_memory_config.buffer_type();
    refuse_unless(
        out_buffer == BufferType::DRAM || out_buffer == BufferType::L1, "output_buffer", out_buffer, "DRAM, L1");
    refuse_unless(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "memory_layout",
        input.memory_config().memory_layout(),
        "INTERLEAVED");
    refuse_unless(attrs.cluster_axis == 1, "cluster_axis", attrs.cluster_axis, "1");
    refuse_unless(attrs.topology == tt::tt_fabric::Topology::Ring, "topology", topology_name(attrs.topology), "Ring");
    refuse_unless(attrs.num_links == 2, "num_links", attrs.num_links, "2");
}

PlanInputs plan_inputs_of(const AllToAllMatmulParams& attrs, const Tensor& input, const Tensor& weight) {
    const auto& x = input.logical_shape();
    const auto& w = weight.logical_shape();
    return PlanInputs{
        .B = x[0],
        .Hl = x[1],
        .S = x[2],
        .K = x[3],
        .H = w[1],
        .N = w[3],
        .in_tile_bytes = static_cast<uint32_t>(input.buffer()->page_size()),
        .w_tile_bytes = static_cast<uint32_t>(weight.buffer()->page_size()),
        .payload_bytes = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes()),
        .output_in_l1 = attrs.output_memory_config.buffer_type() == BufferType::L1,
        .fp32_dest_acc_en = attrs.fp32_dest_acc_en,
    };
}

AllToAllMatmulDeviceOperation::program_factory_t AllToAllMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return AllToAllMatmulProgramFactory{};
}

void AllToAllMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    // The public entry checked the caller errors and the contract before launching; a direct prim caller has not.
    check_support(attrs, t.input, t.weight);
    validate_scratch_placement(t);
    auto* mesh_device = t.input.device();
    const MeshTopology& topology = mesh_topology(mesh_device, attrs.cluster_axis, attrs.num_links);
    const Plan& plan = solve_plan(topology, plan_inputs_of(attrs, t.input, t.weight));
    const TensorSpec expected = scratch_spec(topology, plan, t.input.dtype());
    TT_FATAL(
        t.scratch->tensor_spec() == expected,
        "all_to_all_matmul: scratch_tensor spec {} does not match the plan's {}",
        t.scratch->tensor_spec(),
        expected);
}

void AllToAllMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& t) {
    validate_scratch_placement(t);
}

AllToAllMatmulDeviceOperation::spec_return_value_t AllToAllMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    const auto& x = t.input.logical_shape();
    const auto& w = t.weight.logical_shape();
    const uint32_t G = w[1] / x[1];
    return TensorSpec(
        ttnn::Shape({x[0], w[1], x[2] / G, w[3]}),
        TensorLayout(t.input.dtype(), PageConfig(Layout::TILE), attrs.output_memory_config));
}

AllToAllMatmulDeviceOperation::tensor_return_value_t AllToAllMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(attrs, t), t.input.device());
}

}  // namespace ttnn::operations::all_to_all_matmul

namespace ttnn::prim {

Tensor all_to_all_matmul(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& scratch,
    const ttnn::operations::all_to_all_matmul::AllToAllMatmulParams& params) {
    using OperationType = ttnn::operations::all_to_all_matmul::AllToAllMatmulDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.input = input, .weight = weight, .scratch = scratch});
}

}  // namespace ttnn::prim
