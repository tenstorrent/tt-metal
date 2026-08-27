// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_fused_distributed_groupnorm_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

void DitFusedDistributedGroupnormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Input must be allocated");
    TT_FATAL(input.layout() == Layout::TILE, "Input layout must be TILE, got {}", input.layout());
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "Input dtype must be BFLOAT16 for v1, got {}", input.dtype());
    TT_FATAL(
        input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "Input memory layout must be INTERLEAVED; sharded input is not supported. Got {}.",
        input.memory_config().memory_layout());
    TT_FATAL(
        args.output_mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "Output memory layout must be INTERLEAVED; sharded output is not supported. Got {}.",
        args.output_mem_config.memory_layout());

    const auto& shape = input.logical_shape();
    TT_FATAL(shape.rank() == 4, "Input rank must be 4 ([N, 1, H*W, C]), got {}", shape.rank());
    TT_FATAL(shape[1] == 1, "Input dim1 must be 1 (shape [N, 1, H*W, C]); got {}", shape[1]);
    // v1 folds the spatial extent as physical_volume()/C, which spans all batches — that is the
    // wrong statistic for N>1 (GroupNorm must reduce per (batch, group), not across batch). Hard
    // require N==1; per-batch looping is deferred to a later version.
    TT_FATAL(
        shape[0] == 1,
        "v1 supports batch N==1 only (shape [N, 1, H*W, C]); got N={}. GroupNorm stats must not fold across batch.",
        shape[0]);

    const uint32_t C = shape[3];
    TT_FATAL(args.num_groups >= 1, "num_groups must be >= 1");
    TT_FATAL(C % args.num_groups == 0, "C ({}) must be divisible by num_groups ({})", C, args.num_groups);
    TT_FATAL(C % TILE_WIDTH == 0, "C ({}) must be divisible by TILE_WIDTH ({})", C, TILE_WIDTH);
    // The welford compute accumulates every one of the 32 rows in each tile; the per-group count is
    // derived from the (padded) HW. Require tile-aligned H*W so the padded HW equals the logical HW
    // (padded_HW == true count) — there is no row mask.
    TT_FATAL(
        shape[2] % TILE_HEIGHT == 0,
        "H*W ({}) must be divisible by TILE_HEIGHT ({}) so there are no padded rows inflating "
        "the per-group count; got H*W={}.",
        shape[2],
        TILE_HEIGHT,
        shape[2]);
    // The reused welford GroupNorm kernels require an input_mask to zero sub-tile group padding.
    TT_FATAL(
        tensor_args.input_mask.has_value(),
        "dit_fused_distributed_groupnorm requires an input_mask (welford GroupNorm). Build it with "
        "the standard group_norm input-mask helper.");
    {
        const auto& mask = tensor_args.input_mask.value();
        TT_FATAL(mask.layout() == Layout::TILE, "input_mask layout must be TILE, got {}", mask.layout());
        TT_FATAL(mask.storage_type() == StorageType::DEVICE, "input_mask must be on device");
        TT_FATAL(mask.buffer() != nullptr, "input_mask must be allocated");
        TT_FATAL(input.device() == mask.device(), "input and input_mask must be on the same device");
        TT_FATAL(
            mask.padded_shape()[1] == args.num_groups,
            "input_mask dim1 must equal num_groups ({}), got {}",
            args.num_groups,
            mask.padded_shape()[1]);
        TT_FATAL(
            mask.padded_shape()[2] == TILE_HEIGHT,
            "input_mask height must equal TILE_HEIGHT ({}), got {}",
            TILE_HEIGHT,
            mask.padded_shape()[2]);
        TT_FATAL(
            mask.padded_shape()[3] % TILE_WIDTH == 0,
            "input_mask inner dim must be divisible by TILE_WIDTH ({}), got {}",
            TILE_WIDTH,
            mask.padded_shape()[3]);
    }

    const auto& padded = input.padded_shape();
    TT_FATAL(padded[3] == shape[3], "Input last logical dim ({}) must equal padded last dim ({})", shape[3], padded[3]);

    // Writer is the stock RM welford GN gamma/beta kernel: DRAM-packed row-major, last dim TILE_WIDTH.
    auto validate_affine = [&](const Tensor& t, const char* name) {
        TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} must be on device", name);
        TT_FATAL(t.buffer() != nullptr, "{} must be allocated", name);
        TT_FATAL(input.device() == t.device(), "{} must be on the same device as input", name);
        TT_FATAL(
            t.dtype() == DataType::BFLOAT16 || t.dtype() == DataType::FLOAT32,
            "{} dtype must be BFLOAT16 or FLOAT32, got {}",
            name,
            t.dtype());
        TT_FATAL(t.layout() == Layout::ROW_MAJOR, "{} layout must be ROW_MAJOR, got {}", name, t.layout());
        TT_FATAL(
            t.padded_shape()[3] == TILE_WIDTH,
            "{} inner dim must equal TILE_WIDTH ({}), got {}",
            name,
            TILE_WIDTH,
            t.padded_shape()[3]);
    };
    if (gamma.has_value()) {
        validate_affine(*gamma, "Weight");
    }
    if (beta.has_value()) {
        TT_FATAL(gamma.has_value(), "bias requires weight to also be provided");
        validate_affine(*beta, "Bias");
        TT_FATAL(gamma->dtype() == beta->dtype(), "Weight and bias must have the same dtype");
        TT_FATAL(gamma->layout() == beta->layout(), "Weight and bias must have the same layout");
    }

    TT_FATAL(args.ring_size >= 1, "ring_size must be >= 1");
    TT_FATAL(args.cluster_axis < 2, "cluster_axis must be 0 or 1");
    if (args.ring_size > 1) {
        TT_FATAL(
            !args.multi_device_global_semaphore.empty(),
            "multi_device_global_semaphore must be non-empty when ring_size > 1");
    }

    const auto sizing = compute_sizing(args, input);
    if (!sizing.is_local) {
        const auto expected_spec = make_stats_tensor_spec(sizing);
        const auto& e_shape = expected_spec.logical_shape();
        TT_FATAL(
            tensor_args.persistent_output_buffer.has_value(),
            "persistent_output_buffer is required for ring_size > 1. "
            "Allocate via dit_fused_distributed_groupnorm_create_stats_buffer "
            "(shape [1, 1, {}, {}], dtype=FLOAT32, layout=ROW_MAJOR, DRAM INTERLEAVED).",
            e_shape[2],
            e_shape[3]);
        const auto& buf = tensor_args.persistent_output_buffer.value();
        TT_FATAL(buf.storage_type() == StorageType::DEVICE, "persistent_output_buffer must be on device");
        TT_FATAL(buf.buffer() != nullptr, "persistent_output_buffer must be allocated");
        TT_FATAL(input.device() == buf.device(), "persistent_output_buffer must be on the same device as input");
        TT_FATAL(
            buf.tensor_spec() == expected_spec,
            "persistent_output_buffer must match the spec from "
            "dit_fused_distributed_groupnorm_create_stats_buffer: shape [1, 1, {}, {}], dtype=FLOAT32, "
            "layout=ROW_MAJOR, DRAM INTERLEAVED. Got shape {}, dtype={}, layout={}.",
            e_shape[2],
            e_shape[3],
            buf.logical_shape(),
            buf.dtype(),
            buf.layout());
    }
}

DitFusedDistributedGroupnormDeviceOperation::spec_return_value_t
DitFusedDistributedGroupnormDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& logical = input.logical_shape();

    std::vector<tt::tt_metal::TensorSpec> specs;
    specs.reserve(2);

    ttnn::Shape output_shape({logical[0], logical[1], logical[2], logical[3]});
    specs.emplace_back(output_shape, TensorLayout(input.dtype(), PageConfig(Layout::TILE), args.output_mem_config));

    const auto sizing = compute_sizing(args, input);
    if (!sizing.is_local) {
        specs.push_back(make_stats_tensor_spec(sizing));
    }
    return specs;
}

DitFusedDistributedGroupnormDeviceOperation::tensor_return_value_t
DitFusedDistributedGroupnormDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    std::vector<Tensor> tensors;
    tensors.reserve(specs.size());
    auto* mesh_device = tensor_args.input.device();
    tensors.push_back(create_device_tensor(specs[0], mesh_device));
    if (specs.size() > 1) {
        TT_FATAL(
            tensor_args.persistent_output_buffer.has_value(), "persistent_output_buffer is required for ring_size > 1");
        tensors.push_back(tensor_args.persistent_output_buffer.value());
    }
    return tensors;
}

ttsl::hash::hash_t DitFusedDistributedGroupnormDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "DitFusedDistributedGroupnormDeviceOperation::compute_program_hash");
    auto* mesh_device = tensor_args.input.device();
    auto sd_id = args.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    return tt::tt_metal::operation::hash_operation<DitFusedDistributedGroupnormDeviceOperation>(
        args.num_groups,
        args.eps,
        // The activation is a compile-time define on the compute kernel, so it MUST be hashed:
        // without it a fused and a non-fused call with identical shapes collide in the program
        // cache and the second call silently reuses the first call's compiled program.
        args.fused_activation,
        args.output_mem_config,
        args.cluster_axis,
        args.ring_size,
        args.topology,
        args.compute_kernel_config,
        subdevice_core_range_set,
        tensor_args);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor dit_fused_distributed_groupnorm(
    const Tensor& input_tensor,
    int num_groups,
    float epsilon,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    ttnn::ccl::Topology topology,
    const std::optional<Tensor>& input_mask,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<Tensor>& persistent_output_buffer,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& fused_activation) {
    using OperationType = ttnn::experimental::prim::DitFusedDistributedGroupnormDeviceOperation;

    auto kernel_config_val = init_device_compute_kernel_config(
        mesh_device.arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4, false, true, false);

    const auto& mesh_view = mesh_device.get_view();
    const std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    tt::tt_fabric::Topology topology_ = (num_devices > 1)
                                            ? ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis)
                                            : tt::tt_fabric::Topology::Linear;

    auto operation_attributes = OperationType::operation_attributes_t{
        .eps = epsilon,
        .num_groups = static_cast<uint32_t>(num_groups),
        .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
        .compute_kernel_config = kernel_config_val,
        .fused_activation = fused_activation,
        .cluster_axis = cluster_axis,
        .ring_size = static_cast<uint32_t>(num_devices),
        .topology = topology_,
        .multi_device_global_semaphore = multi_device_global_semaphore,
        .sub_device_id = subdevice_id,
    };

    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .gamma = weight,
        .beta = bias,
        .input_mask = input_mask,
        .persistent_output_buffer = persistent_output_buffer};

    auto outputs = ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
    return outputs[0];
}

}  // namespace ttnn::prim
