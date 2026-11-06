// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "distributed_rms_norm_device_operation.hpp"

#include <tt-metalium/math.hpp>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>

#include "distributed_rms_norm_planner.hpp"

namespace ttnn::operations::distributed_rms_norm {

using namespace tt::tt_metal;

namespace {

void refuse_axis_unless(bool supported, std::string_view axis, const auto& value, std::string_view allowed) {
    if (!supported) {
        throw UnsupportedAxisValue(
            fmt::format("distributed_rms_norm: {}={} not in SUPPORTED [{}]", axis, value, allowed));
    }
}

bool is_supported_layout(TensorMemoryLayout layout) {
    return layout == TensorMemoryLayout::INTERLEAVED || layout == TensorMemoryLayout::HEIGHT_SHARDED ||
           layout == TensorMemoryLayout::WIDTH_SHARDED;
}

// Same buffer type and overlapping per-bank address ranges (a view wraps the same address in a Buffer of its own).
bool shares_memory(const Tensor& a, const Tensor& b) {
    const Buffer* x = a.buffer();
    const Buffer* y = b.buffer();
    if (x->buffer_type() != y->buffer_type()) {
        return false;
    }
    const uint64_t x0 = x->address();  // smuggled-rta-ok: aliasing check in validation, never a kernel arg
    const uint64_t y0 = y->address();  // smuggled-rta-ok: aliasing check in validation, never a kernel arg
    return x0 < y0 + y->aligned_size_per_bank() && y0 < x0 + x->aligned_size_per_bank();
}

// Checked on every call: what the key cannot fix (which device each buffer is on, where it lies).
void validate_call_buffers(const DistributedRmsNormInputs& t) {
    auto* mesh = t.input.device();
    TT_FATAL(
        !t.gamma.has_value() || t.gamma->device() == mesh, "distributed_rms_norm: gamma must be on the input's mesh");
    if (t.output.has_value()) {
        TT_FATAL(t.output->device() == mesh, "distributed_rms_norm: output_tensor must be on the input's mesh");
        // In place is refused: a core's x reads of a later chunk / block and the native shard pack interleave with
        // the output writes, and the out-split reader writes tiles the writer's core has not finished reading.
        TT_FATAL(
            !shares_memory(*t.output, t.input),
            "distributed_rms_norm: output_tensor shares memory with the input (in place is not supported)");
        TT_FATAL(
            !t.gamma.has_value() || !shares_memory(*t.output, *t.gamma),
            "distributed_rms_norm: output_tensor shares memory with gamma");
    }
    if (t.scratch.has_value()) {
        TT_FATAL(t.scratch->device() == mesh, "distributed_rms_norm: scratch_tensor must be on the input's mesh");
        TT_FATAL(
            !shares_memory(*t.scratch, t.input), "distributed_rms_norm: scratch_tensor shares memory with the input");
        TT_FATAL(
            !t.gamma.has_value() || !shares_memory(*t.scratch, *t.gamma),
            "distributed_rms_norm: scratch_tensor shares memory with gamma");
        TT_FATAL(
            !t.output.has_value() || !shares_memory(*t.scratch, *t.output),
            "distributed_rms_norm: scratch_tensor shares memory with the output");
    }
}

void validate_output_spec(const DistributedRmsNormParams& attrs, const DistributedRmsNormInputs& t) {
    if (!t.output.has_value()) {
        return;
    }
    const Tensor& out = *t.output;
    const TensorSpec want = DistributedRmsNormDeviceOperation::compute_output_specs(attrs, t);
    TT_FATAL(
        out.logical_shape() == want.logical_shape(),
        "distributed_rms_norm: output_tensor shape {} differs from {}",
        out.logical_shape(),
        want.logical_shape());
    TT_FATAL(
        out.dtype() == want.data_type(),
        "distributed_rms_norm: output_tensor dtype {} differs from {}",
        out.dtype(),
        want.data_type());
    TT_FATAL(
        out.layout() == want.layout(),
        "distributed_rms_norm: output_tensor layout {} differs from {}",
        out.layout(),
        want.layout());
    TT_FATAL(
        out.memory_config() == want.memory_config(),
        "distributed_rms_norm: output_tensor memory_config {} differs from {}",
        out.memory_config(),
        want.memory_config());
    TT_FATAL(
        out.tensor_spec() == want,
        "distributed_rms_norm: output_tensor spec {} differs from {}",
        out.tensor_spec(),
        want);
}

void validate_scratch_spec(const DistributedRmsNormInputs& t, const detail::Plan& plan) {
    TT_FATAL(t.scratch.has_value(), "distributed_rms_norm: scratch_tensor (the exchange landing) is missing");
    const Tensor& scratch = *t.scratch;
    const TensorSpec want = detail::landing_spec(*t.input.device(), plan.landing_alloc_words);
    const MemoryConfig& have_mc = scratch.memory_config();
    const MemoryConfig& want_mc = want.memory_config();
    TT_FATAL(
        scratch.dtype() == want.data_type() && scratch.layout() == want.layout(),
        "distributed_rms_norm: scratch_tensor must be {} {}, got {} {}",
        want.data_type(),
        want.layout(),
        scratch.dtype(),
        scratch.layout());
    TT_FATAL(
        have_mc.buffer_type() == want_mc.buffer_type() && have_mc.memory_layout() == want_mc.memory_layout() &&
            have_mc.shard_spec().has_value() && want_mc.shard_spec().has_value() &&
            have_mc.shard_spec()->grid == want_mc.shard_spec()->grid,
        "distributed_rms_norm: scratch_tensor memory_config {} differs from {}",
        have_mc,
        want_mc);
    // Exactly the plan's landing: the plan reserves that many bytes below the CB region's end, not the held size.
    TT_FATAL(
        scratch.logical_shape() == want.logical_shape(),
        "distributed_rms_norm: scratch_tensor shape {} differs from the {} the plan needs",
        scratch.logical_shape(),
        want.logical_shape());
}

const detail::Plan& plan_of(const DistributedRmsNormParams& attrs, const DistributedRmsNormInputs& t) {
    TT_FATAL(t.output.has_value(), "distributed_rms_norm: the output tensor must be allocated before planning");
    return detail::solve_plan(detail::plan_inputs(attrs, t, *t.output));
}

}  // namespace

GroupResolution check_caller_inputs(
    const Tensor& input,
    int32_t cluster_axis,
    const std::optional<Tensor>& gamma,
    tt::tt_fabric::Topology topology,
    std::optional<int32_t> num_links) {
    const auto& shape = input.logical_shape();
    if (shape.rank() < 2) {
        throw std::invalid_argument(
            fmt::format("distributed_rms_norm: input must have rank >= 2, got rank {}", shape.rank()));
    }
    const uint32_t W = shape[-1];
    if (gamma.has_value()) {
        const ttnn::Shape want =
            gamma->layout() == Layout::TILE
                ? ttnn::Shape({1u, 1u, 1u, W})
                : ttnn::Shape({1u, 1u, tt::div_up(W, tt::constants::TILE_WIDTH), tt::constants::TILE_WIDTH});
        if (gamma->logical_shape() != want) {
            throw std::invalid_argument(fmt::format(
                "distributed_rms_norm: gamma ({}) must have per-device shape {}, got {}",
                gamma->layout(),
                want,
                gamma->logical_shape()));
        }
    }
    if (cluster_axis != 0 && cluster_axis != 1) {
        throw std::invalid_argument(
            fmt::format("distributed_rms_norm: cluster_axis must be 0 or 1, got {}", cluster_axis));
    }
    const auto axis = static_cast<uint32_t>(cluster_axis);
    auto* mesh = input.device();
    const uint32_t G = mesh->shape()[axis];
    if (G < 2) {
        throw std::invalid_argument(
            fmt::format("distributed_rms_norm: cluster_axis={} has size {}; nothing to reduce", axis, G));
    }
    const bool ring = topology == tt::tt_fabric::Topology::Ring && G >= 3;
    if (ring && !detail::fabric_wraps_axis(axis)) {
        throw std::invalid_argument(
            fmt::format("distributed_rms_norm: Ring along cluster_axis={} needs a fabric config that wraps it", axis));
    }
    const uint32_t usable = detail::usable_links(*mesh, axis, ring);
    if (usable < 1) {
        throw std::invalid_argument(fmt::format(
            "distributed_rms_norm: some hop of the group has no forwarding link (topology={}, cluster_axis={})",
            topology,
            axis));
    }
    if (num_links.has_value() && (*num_links <= 0 || static_cast<uint32_t>(*num_links) > usable)) {
        throw std::invalid_argument(
            fmt::format("distributed_rms_norm: num_links={} outside [1, {}] usable links per hop", *num_links, usable));
    }
    return GroupResolution{.group_size = G, .usable_links = usable, .ring = ring};
}

void check_support(
    const DistributedRmsNormParams& attrs,
    const DistributedRmsNormInputs& t,
    const std::optional<MemoryConfig>& requested_memory_config) {
    const Tensor& input = t.input;
    const DataType dtype = input.dtype();
    refuse_axis_unless(
        dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT16 || dtype == DataType::BFLOAT8_B,
        "dtype",
        dtype,
        "FLOAT32, BFLOAT16, BFLOAT8_B");
    refuse_axis_unless(input.layout() == Layout::TILE, "layout", input.layout(), "TILE");
    const uint32_t rank = input.logical_shape().rank();
    refuse_axis_unless(rank >= 2 && rank <= 4, "rank", rank, "2, 3, 4");
    if (t.gamma.has_value()) {
        const DataType g_dtype = t.gamma->dtype();
        refuse_axis_unless(
            g_dtype == DataType::FLOAT32 || g_dtype == DataType::BFLOAT16,
            "gamma_dtype",
            g_dtype,
            "FLOAT32, BFLOAT16, none");
        const Layout g_layout = t.gamma->layout();
        refuse_axis_unless(
            g_layout == Layout::TILE || g_layout == Layout::ROW_MAJOR,
            "gamma_layout",
            g_layout,
            "TILE, ROW_MAJOR, none");
    }
    const TensorMemoryLayout in_layout = input.memory_config().memory_layout();
    refuse_axis_unless(
        is_supported_layout(in_layout), "memory_layout", in_layout, "INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED");
    refuse_axis_unless(attrs.cluster_axis <= 1, "cluster_axis", attrs.cluster_axis, "0, 1");
    refuse_axis_unless(
        attrs.topology == tt::tt_fabric::Topology::Linear || attrs.topology == tt::tt_fabric::Topology::Ring,
        "topology",
        attrs.topology,
        "Linear, Ring");
    refuse_axis_unless(attrs.num_links == 1 || attrs.num_links == 2, "num_links", attrs.num_links, "1, 2, None");
    if (requested_memory_config.has_value() && !is_supported_layout(requested_memory_config->memory_layout())) {
        throw UnsupportedAxisValue(fmt::format(
            "distributed_rms_norm: output memory_layout={} not in SUPPORTED",
            requested_memory_config->memory_layout()));
    }
    if (dtype == DataType::FLOAT32 && !attrs.compute_config.fp32_dest_acc_en) {
        throw ExcludedCell(
            "distributed_rms_norm: unsupported combination (refinement candidate): {dtype=float32, "
            "fp32_dest_acc_en=False}");
    }
    if (requested_memory_config.has_value() &&
        requested_memory_config->memory_layout() != TensorMemoryLayout::INTERLEAVED &&
        !requested_memory_config->shard_spec().has_value()) {
        throw RegimeRefusal("distributed_rms_norm: a sharded output memory_config must carry a shard spec");
    }
    if (attrs.exchange_mode == ExchangeMode::MCAST && detail::is_2d_fabric()) {
        throw RegimeRefusal(
            "distributed_rms_norm: exchange_mode='mcast' needs a 1D fabric (FABRIC_1D / FABRIC_1D_RING)");
    }
}

DistributedRmsNormDeviceOperation::program_factory_t DistributedRmsNormDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return DistributedRmsNormProgramFactory{};
}

void DistributedRmsNormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    validate_call_buffers(t);
    validate_output_spec(attrs, t);
    // Throws RegimeRefusal where the Python planner raised NotImplementedError.
    const detail::Plan& plan = plan_of(attrs, t);
    validate_scratch_spec(t, plan);
}

void DistributedRmsNormDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& t) {
    validate_call_buffers(t);
}

ttsl::hash::hash_t DistributedRmsNormDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    // epsilon is left out: override_runtime_arguments re-applies it (compute common EPS_BITS) on every hit. Every
    // other attribute, every tensor spec and the plan solved against this call's L1 budget are in the key; tensor
    // addresses never are.
    TT_FATAL(
        t.output.has_value() && t.scratch.has_value(),
        "distributed_rms_norm: the entry passes the output and the exchange landing");
    const detail::Plan& plan = plan_of(attrs, t);
    auto hash = tt::tt_metal::operation::hash_operation<DistributedRmsNormDeviceOperation>(
        attrs.cluster_axis,
        attrs.topology,
        attrs.group_size,
        attrs.num_links,
        attrs.ring,
        attrs.exchange_mode,
        attrs.compute_config,
        attrs.output_memory_config,
        t.input.tensor_spec(),
        t.output->tensor_spec(),
        t.scratch->tensor_spec(),
        t.gamma.has_value(),
        detail::plan_hash(plan));
    if (t.gamma.has_value()) {
        hash = ttsl::hash::hash_objects(hash, t.gamma->tensor_spec());
    }
    return hash;
}

DistributedRmsNormDeviceOperation::spec_return_value_t DistributedRmsNormDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    // As the Python allocated it: the input's shape, dtype and layout, the resolved memory config.
    return TensorSpec(
        t.input.logical_shape(),
        TensorLayout(t.input.dtype(), PageConfig(t.input.layout()), attrs.output_memory_config));
}

DistributedRmsNormDeviceOperation::tensor_return_value_t DistributedRmsNormDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (t.output.has_value()) {
        return *t.output;
    }
    return create_device_tensor(compute_output_specs(attrs, t), t.input.device());
}

}  // namespace ttnn::operations::distributed_rms_norm

namespace ttnn::prim {

Tensor distributed_rms_norm(
    const Tensor& input,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& output,
    const std::optional<Tensor>& scratch,
    const ttnn::operations::distributed_rms_norm::DistributedRmsNormParams& params) {
    using OperationType = ttnn::operations::distributed_rms_norm::DistributedRmsNormDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.input = input, .gamma = gamma, .output = output, .scratch = scratch});
}

}  // namespace ttnn::prim
