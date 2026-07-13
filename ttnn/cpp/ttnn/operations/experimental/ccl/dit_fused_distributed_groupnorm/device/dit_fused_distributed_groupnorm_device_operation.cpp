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
    (void)args.use_welford;

    const auto& padded = input.padded_shape();
    TT_FATAL(padded[3] == shape[3], "Input last logical dim ({}) must equal padded last dim ({})", shape[3], padded[3]);

    // gamma/beta are DRAM-packed row-major (last dim == TILE_WIDTH), exactly as prepped for
    // ttnn::group_norm, so we only validate dtype/layout here (not a channel-length last dim).
    auto validate_affine = [&](const Tensor& t, const char* name) {
        TT_FATAL(
            t.dtype() == DataType::BFLOAT16 || t.dtype() == DataType::FLOAT32,
            "{} dtype must be BFLOAT16 or FLOAT32, got {}",
            name,
            t.dtype());
        TT_FATAL(
            t.layout() == Layout::TILE || t.layout() == Layout::ROW_MAJOR,
            "{} layout must be TILE or ROW_MAJOR, got {}",
            name,
            t.layout());
    };
    if (gamma.has_value()) {
        validate_affine(*gamma, "Weight");
    }
    if (beta.has_value()) {
        TT_FATAL(gamma.has_value(), "bias requires weight to also be provided");
        validate_affine(*beta, "Bias");
    }

    TT_FATAL(args.ring_size >= 1, "ring_size must be >= 1");
    TT_FATAL(args.num_links >= 1, "num_links must be >= 1");
    TT_FATAL(args.cluster_axis < 2, "cluster_axis must be 0 or 1");
    if (args.ring_size > 1) {
        TT_FATAL(
            !args.multi_device_global_semaphore.empty(),
            "multi_device_global_semaphore must be non-empty when ring_size > 1");
    }

    const auto sizing = compute_sizing(args, input);
    if (!sizing.is_local) {
        TT_FATAL(
            tensor_args.persistent_output_buffer.has_value(),
            "persistent_output_buffer is required for ring_size > 1. "
            "Allocate via dit_fused_distributed_groupnorm_create_stats_buffer "
            "(shape [1, 1, {}, {}], dtype=FLOAT32, layout=ROW_MAJOR, DRAM INTERLEAVED).",
            sizing.total_pages,
            sizing.page_size_bytes / sizeof(float));
        const auto& buf = tensor_args.persistent_output_buffer.value();
        TT_FATAL(buf.storage_type() == StorageType::DEVICE, "persistent_output_buffer must be on device");
        TT_FATAL(buf.buffer() != nullptr, "persistent_output_buffer must be allocated");
        TT_FATAL(buf.layout() == Layout::ROW_MAJOR, "persistent_output_buffer layout must be ROW_MAJOR");
        TT_FATAL(buf.dtype() == DataType::FLOAT32, "persistent_output_buffer dtype must be FLOAT32");
        TT_FATAL(
            buf.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
            "persistent_output_buffer must be in DRAM");
        TT_FATAL(
            buf.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "persistent_output_buffer must be INTERLEAVED");
        const auto& b_shape = buf.logical_shape();
        TT_FATAL(b_shape.rank() == 4, "persistent_output_buffer rank must be 4");
        const uint32_t expected_width = sizing.page_size_bytes / sizeof(float);
        TT_FATAL(
            b_shape[0] == 1 && b_shape[1] == 1 && b_shape[2] == sizing.total_pages && b_shape[3] == expected_width,
            "persistent_output_buffer shape must be [1, 1, {}, {}], got [{}, {}, {}, {}]",
            sizing.total_pages,
            expected_width,
            b_shape[0],
            b_shape[1],
            b_shape[2],
            b_shape[3]);
    }
}

DitFusedDistributedGroupnormDeviceOperation::spec_return_value_t
DitFusedDistributedGroupnormDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& logical = input.logical_shape();

    std::vector<TensorSpec> specs;
    specs.reserve(2);

    ttnn::Shape output_shape({logical[0], logical[1], logical[2], logical[3]});
    specs.emplace_back(output_shape, TensorLayout(input.dtype(), PageConfig(Layout::TILE), args.output_mem_config));

    const auto sizing = compute_sizing(args, input);
    if (!sizing.is_local) {
        const uint32_t floats_per_page = sizing.page_size_bytes / sizeof(float);
        ttnn::Shape stats_shape({1u, 1u, sizing.total_pages, floats_per_page});
        MemoryConfig stats_mem{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
        specs.emplace_back(stats_shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), stats_mem));
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
        args.use_welford,
        args.output_mem_config,
        args.cluster_axis,
        args.num_links,
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
    bool use_welford,
    const std::optional<Tensor>& persistent_output_buffer,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    using OperationType = ttnn::experimental::prim::DitFusedDistributedGroupnormDeviceOperation;

    auto arch = is_device_tensor(input_tensor) ? input_tensor.device()->arch() : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        arch, compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4, false, true, false);

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
        .use_welford = use_welford,
        .cluster_axis = cluster_axis,
        .num_links = static_cast<uint32_t>(num_preferred_links.value_or(1)),
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
