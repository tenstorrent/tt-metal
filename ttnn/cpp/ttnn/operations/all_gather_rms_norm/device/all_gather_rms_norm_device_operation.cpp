// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_rms_norm_device_operation.hpp"

#include <fmt/format.h>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::all_gather_rms_norm {

using namespace tt::tt_metal;

namespace {

void refuse_axis(bool ok, std::string_view axis, const std::string& value, std::string_view allowed) {
    if (!ok) {
        throw UnsupportedAxisValue(
            fmt::format("all_gather_rms_norm: {}={} not in SUPPORTED [{}]", axis, value, allowed));
    }
}

bool is_sharded_layout(TensorMemoryLayout l) {
    return l == TensorMemoryLayout::HEIGHT_SHARDED || l == TensorMemoryLayout::WIDTH_SHARDED;
}

std::string output_placement(const MemoryConfig& mc) {
    if (mc.memory_layout() == TensorMemoryLayout::INTERLEAVED) {
        return mc.buffer_type() == BufferType::L1 ? "l1_interleaved" : "dram_interleaved";
    }
    return "l1_sharded";
}

const MemoryConfig& out_memory_config(const AllGatherRmsNormParams& attrs, const AllGatherRmsNormInputs& t) {
    return t.output.has_value() ? t.output->memory_config() : attrs.output_memory_config;
}

}  // namespace

std::vector<CoreCoord> input_shard_cores(const Tensor& input) {
    const auto& mc = input.memory_config();
    if (!mc.is_sharded() || !mc.shard_spec().has_value()) {
        return {};
    }
    return planner::ordered_shard_cores(mc.shard_spec()->grid, true);
}

const planner::MeshPlan& mesh_plan_of(const AllGatherRmsNormParams& attrs, const Tensor& input) {
    auto& mesh = *input.device();
    const uint32_t G = mesh.shape()[attrs.cluster_axis];
    const bool ring = attrs.topology == tt::tt_fabric::Topology::Ring && G > 2;
    return planner::build_mesh_plan(mesh, attrs.cluster_axis, ring, attrs.num_links, input_shard_cores(input));
}

void check_support(const AllGatherRmsNormParams& attrs, const AllGatherRmsNormInputs& t) {
    const Tensor& x = t.input;
    const DataType dtype = x.dtype();
    refuse_axis(
        dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT16 || dtype == DataType::BFLOAT8_B,
        "dtype",
        fmt::format("{}", dtype),
        "float32, bfloat16, bfloat8_b");
    refuse_axis(x.layout() == Layout::TILE, "layout", fmt::format("{}", x.layout()), "TILE");
    const uint32_t rank = x.logical_shape().rank();
    refuse_axis(rank >= 2 && rank <= 4, "rank", std::to_string(rank), "2, 3, 4");
    if (t.gamma.has_value()) {
        const DataType g = t.gamma->dtype();
        refuse_axis(
            g == DataType::FLOAT32 || g == DataType::BFLOAT16,
            "gamma_dtype",
            fmt::format("{}", g),
            "float32, bfloat16, none");
    }
    const TensorMemoryLayout in_layout = x.memory_config().memory_layout();
    refuse_axis(
        in_layout == TensorMemoryLayout::INTERLEAVED || is_sharded_layout(in_layout),
        "memory_layout",
        fmt::format("{}", in_layout),
        "INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED");
    refuse_axis(attrs.cluster_axis <= 1, "cluster_axis", std::to_string(attrs.cluster_axis), "0, 1");
    refuse_axis(
        attrs.topology == tt::tt_fabric::Topology::Linear || attrs.topology == tt::tt_fabric::Topology::Ring,
        "topology",
        std::to_string(static_cast<int>(attrs.topology)),
        "Linear, Ring");
    refuse_axis(attrs.num_links >= 1 && attrs.num_links <= 2, "num_links", std::to_string(attrs.num_links), "1, 2");

    // EXCLUSIONS
    if (dtype == DataType::FLOAT32 && !attrs.fp32_dest_acc_en) {
        throw ExcludedCell(
            "all_gather_rms_norm: unsupported combination (refinement candidate): {'dtype': float32, "
            "'fp32_dest_acc_en': False}");
    }
    if (dtype == DataType::BFLOAT8_B && x.logical_shape()[-1] % tt::constants::TILE_WIDTH != 0) {
        throw ExcludedCell(
            "all_gather_rms_norm: unsupported combination (refinement candidate): {'dtype': bfloat8_b, "
            "'alignment': 'w_non_aligned'}");
    }

    // placement
    const auto& in_mc = x.memory_config();
    if (in_layout == TensorMemoryLayout::INTERLEAVED) {
        if (in_mc.buffer_type() != BufferType::DRAM) {
            throw UnsupportedAxisValue("all_gather_rms_norm: an interleaved input must be in DRAM");
        }
    } else if (in_mc.buffer_type() != BufferType::L1) {
        throw UnsupportedAxisValue(
            fmt::format("all_gather_rms_norm: a sharded input must be L1-resident, got {}", in_mc.buffer_type()));
    }
    const MemoryConfig& out_mc = out_memory_config(attrs, t);
    if (out_mc.memory_layout() == TensorMemoryLayout::INTERLEAVED) {
        if (out_mc.buffer_type() != BufferType::DRAM && out_mc.buffer_type() != BufferType::L1) {
            throw UnsupportedAxisValue(
                fmt::format("all_gather_rms_norm: interleaved output must be in DRAM or L1, got {}", out_mc));
        }
    } else if (!is_sharded_layout(out_mc.memory_layout()) || out_mc.buffer_type() != BufferType::L1) {
        throw UnsupportedAxisValue(
            fmt::format("all_gather_rms_norm: output memory config {} is not supported", out_mc));
    } else if (!out_mc.shard_spec().has_value()) {
        throw UnsupportedAxisValue("all_gather_rms_norm: a sharded output memory config needs a shard spec");
    }
    (void)output_placement(out_mc);
    if (t.gamma.has_value() && t.gamma->memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        throw UnsupportedAxisValue("all_gather_rms_norm: gamma must be interleaved");
    }
    if (t.beta.has_value() && t.beta->memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        throw UnsupportedAxisValue("all_gather_rms_norm: beta must be interleaved");
    }
}

AllGatherRmsNormDeviceOperation::program_factory_t AllGatherRmsNormDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return AllGatherRmsNormMeshWorkloadFactory{};
}

void AllGatherRmsNormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    check_support(attrs, t);
    const planner::MeshPlan& mesh_plan = mesh_plan_of(attrs, t.input);
    const planner::ProgramPlan& plan = planner::solve_program_plan(attrs, t, mesh_plan);  // L1 fit refusal
    if (plan.is_ru) {
        TT_FATAL(t.scratch.has_value(), "all_gather_rms_norm: a W_local % 32 != 0 call needs its staging scratch");
        const TensorSpec want = planner::staging_spec(plan.geo, compute_output_specs(attrs, t).data_type());
        TT_FATAL(
            t.scratch->tensor_spec() == want,
            "all_gather_rms_norm: scratch_tensor spec {} does not match the staging this call needs {}",
            t.scratch->tensor_spec(),
            want);
        TT_FATAL(
            t.scratch->device() == t.input.device(), "all_gather_rms_norm: scratch_tensor must be on the input's mesh");
    }
}

void AllGatherRmsNormDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

ttsl::hash::hash_t AllGatherRmsNormDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    // epsilon is left out: override_runtime_arguments re-applies it (worker common EPS_BITS). Every other attribute,
    // every tensor spec and the plan solved against the free L1 (the staging layout) are in the key.
    const planner::MeshPlan& mesh_plan = mesh_plan_of(attrs, t.input);
    const planner::ProgramPlan& plan = planner::solve_program_plan(attrs, t, mesh_plan);
    auto hash = tt::tt_metal::operation::hash_operation<AllGatherRmsNormDeviceOperation>(
        attrs.cluster_axis,
        attrs.is_layer_norm,
        attrs.topology,
        attrs.num_links,
        attrs.math_fidelity,
        attrs.fp32_dest_acc_en,
        attrs.math_approx_mode,
        attrs.output_memory_config,
        t.input.tensor_spec(),
        t.gamma.has_value(),
        t.beta.has_value(),
        t.output.has_value(),
        t.scratch.has_value(),
        plan.stage_max_pages);
    for (const auto* opt : {&t.gamma, &t.beta, &t.output, &t.scratch}) {
        if (opt->has_value()) {
            hash = ttsl::hash::hash_objects(hash, (*opt)->tensor_spec());
        }
    }
    for (const auto& [coord, sp] : plan.stage_plans) {
        hash = ttsl::hash::hash_objects(hash, sp.has_value());
        if (sp.has_value()) {
            hash = ttsl::hash::hash_objects(hash, sp->max_pages, sp->nch_max, sp->unit_off);
            for (const auto& lane : sp->lanes) {
                hash = ttsl::hash::hash_objects(hash, lane.n_st, lane.tot);
            }
        }
    }
    return hash;
}

AllGatherRmsNormDeviceOperation::spec_return_value_t AllGatherRmsNormDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (t.output.has_value()) {
        return t.output->tensor_spec();
    }
    const uint32_t G = t.input.device()->shape()[attrs.cluster_axis];
    ttnn::Shape shape = t.input.logical_shape();
    shape[-1] = shape[-1] * G;
    return TensorSpec(shape, TensorLayout(t.input.dtype(), PageConfig(Layout::TILE), attrs.output_memory_config));
}

AllGatherRmsNormDeviceOperation::tensor_return_value_t AllGatherRmsNormDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (t.output.has_value()) {
        return *t.output;
    }
    return create_device_tensor(compute_output_specs(attrs, t), t.input.device());
}

}  // namespace ttnn::operations::all_gather_rms_norm

namespace ttnn::prim {

Tensor all_gather_rms_norm(
    const Tensor& input,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    const std::optional<Tensor>& output,
    const std::optional<Tensor>& scratch,
    const ttnn::operations::all_gather_rms_norm::AllGatherRmsNormParams& params) {
    using OperationType = ttnn::operations::all_gather_rms_norm::AllGatherRmsNormDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params,
        OperationType::tensor_args_t{
            .input = input, .gamma = gamma, .beta = beta, .output = output, .scratch = scratch});
}

}  // namespace ttnn::prim
