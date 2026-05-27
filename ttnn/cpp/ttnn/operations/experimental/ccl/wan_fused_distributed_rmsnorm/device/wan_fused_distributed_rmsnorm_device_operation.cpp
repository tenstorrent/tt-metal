// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "wan_fused_distributed_rmsnorm_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/device.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

void WanFusedDistributedRmsnormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    const auto& trans_mat = tensor_args.transformation_mat;
    const auto& rope_cos = tensor_args.rope_cos;
    const auto& rope_sin = tensor_args.rope_sin;

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Input must be allocated");
    TT_FATAL(input.layout() == Layout::TILE, "Input layout must be TILE, got {}", input.layout());
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "Input dtype must be BFLOAT16, got {}", input.dtype());

    const auto& shape = input.logical_shape();
    TT_FATAL(shape.rank() == 4, "Input rank must be 4, got {}", shape.rank());
    TT_FATAL(shape[0] == 1 && shape[1] == 1, "Input must have batch=1, channel=1 (shape [1,1,N,H])");
    TT_FATAL(args.num_heads_per_device >= 1, "num_heads_per_device must be >= 1");
    TT_FATAL(
        shape[3] % args.num_heads_per_device == 0,
        "Input H ({}) must be divisible by num_heads_per_device ({})",
        shape[3],
        args.num_heads_per_device);

    const auto& padded = input.padded_shape();
    TT_FATAL(padded[3] == shape[3], "Input last logical dim ({}) must equal padded last dim ({})", shape[3], padded[3]);

    if (weight.has_value()) {
        TT_FATAL(weight->layout() == Layout::TILE, "Weight layout must be TILE");
        TT_FATAL(weight->dtype() == DataType::BFLOAT16, "Weight dtype must be BFLOAT16");
        TT_FATAL(
            weight->padded_shape()[-1] == padded[3],
            "Weight last dim ({}) must equal input H per device ({})",
            weight->padded_shape()[-1],
            padded[3]);
    }

    const bool rope_present = trans_mat.has_value() || rope_cos.has_value() || rope_sin.has_value();
    const bool rope_complete = trans_mat.has_value() && rope_cos.has_value() && rope_sin.has_value();
    TT_FATAL(
        !rope_present || rope_complete,
        "RoPE requires transformation_mat, rope_cos, and rope_sin all to be provided together");

    // RoPE cos/sin shape: [B, num_heads_dim, N, head_dim] where num_heads_dim is
    // either 1 (broadcast across heads — same cos/sin for every head) or
    // num_heads_per_device (per-head cos/sin). Both rope_cos and rope_sin must
    // match.
    if (rope_complete) {
        const auto& cos_shape = rope_cos->logical_shape();
        const auto& sin_shape = rope_sin->logical_shape();
        TT_FATAL(
            cos_shape == sin_shape, "rope_cos and rope_sin must have the same shape ({} vs {})", cos_shape, sin_shape);
        TT_FATAL(cos_shape.rank() == 4, "rope_cos must be 4D, got rank {}", cos_shape.rank());
        TT_FATAL(
            cos_shape[1] == 1 || cos_shape[1] == args.num_heads_per_device,
            "rope_cos dim 1 ({}) must be 1 (broadcast across heads) or num_heads_per_device ({})",
            cos_shape[1],
            args.num_heads_per_device);
    }

    TT_FATAL(args.ring_size >= 1, "ring_size must be >= 1");
    TT_FATAL(args.num_links >= 1, "num_links must be >= 1");
    TT_FATAL(args.cluster_axis < 2, "cluster_axis must be 0 or 1");
    if (args.ring_size > 1) {
        TT_FATAL(
            !args.multi_device_global_semaphore.empty(),
            "multi_device_global_semaphore must be non-empty when ring_size > 1");
    }
}

WanFusedDistributedRmsnormDeviceOperation::spec_return_value_t
WanFusedDistributedRmsnormDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& logical = input.logical_shape();

    std::vector<TensorSpec> specs;
    specs.reserve(2);

    // Post-allgather output reshapes to [1, num_heads_per_device, N, H/num_heads_per_device].
    ttnn::Shape output_shape({1u, args.num_heads_per_device, logical[2], logical[3] / args.num_heads_per_device});
    const auto out_dtype = args.dtype.value_or(input.dtype());
    specs.emplace_back(output_shape, TensorLayout(out_dtype, PageConfig(Layout::TILE), args.output_mem_config));

    // Persistent stats DRAM scratch for the MUX writer path (Phase 9
    // packed-page layout). Each page holds one chunk's worth of post-reduce
    // stats in row-major form: TILE_HEIGHT * window_size fp32 values =
    // TILE_HEIGHT * window_size * 4 bytes per page. There are total_pages =
    // ring_size * num_chunks_per_device pages — independent of num_workers
    // by design, so the caller need not know the worker count.
    //
    // We expose the buffer as a ROW_MAJOR fp32 tensor of shape
    // [1, 1, total_pages, TILE_HEIGHT * window_size]; ttnn defaults each
    // row to one accessor page, so TensorAccessor page_idx maps 1:1 to the
    // packed-page index the writer/reader address.
    //
    // Allocated as a regular device tensor so the framework's
    // create_device_tensor places it as a mesh-coherent MeshBuffer — every
    // chip gets the same DRAM address, which the fabric mcast relies on.
    const auto sizing = compute_sizing(args, input);
    if (sizing.use_mux) {
        ttnn::Shape stats_shape({1u, 1u, sizing.total_pages, TILE_HEIGHT * sizing.window_size});
        MemoryConfig stats_mem{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
        specs.emplace_back(stats_shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), stats_mem));
    }

    return specs;
}

WanFusedDistributedRmsnormDeviceOperation::tensor_return_value_t
WanFusedDistributedRmsnormDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    std::vector<Tensor> tensors;
    tensors.reserve(specs.size());
    auto* mesh_device = tensor_args.input.device();
    for (const auto& spec : specs) {
        tensors.push_back(create_device_tensor(spec, mesh_device));
    }
    return tensors;
}

ttsl::hash::hash_t WanFusedDistributedRmsnormDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "WanFusedDistributedRmsnormDeviceOperation::compute_program_hash");
    auto* mesh_device = tensor_args.input.device();
    auto sd_id = args.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    return tt::tt_metal::operation::hash_operation<WanFusedDistributedRmsnormDeviceOperation>(
        args.epsilon,
        args.num_heads_per_device,
        args.dtype,
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

Tensor wan_fused_distributed_rmsnorm(
    const Tensor& input_tensor,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    ttnn::ccl::Topology topology,
    float epsilon,
    uint32_t num_heads_per_device,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& transformation_mat,
    const std::optional<const Tensor>& rope_cos,
    const std::optional<const Tensor>& rope_sin,
    const std::optional<const DataType>& dtype,
    const std::optional<Tensor>& persistent_output_buffer,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::WanFusedDistributedRmsnormDeviceOperation;

    auto arch = is_device_tensor(input_tensor) ? input_tensor.device()->arch() : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        arch, compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4, true, false, false);

    const auto& mesh_view = mesh_device.get_view();
    const std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);

    auto operation_attributes = OperationType::operation_attributes_t(
        epsilon,
        num_heads_per_device,
        dtype,
        memory_config.value_or(input_tensor.memory_config()),
        cluster_axis,
        static_cast<uint32_t>(num_preferred_links.value_or(1)),
        static_cast<uint32_t>(num_devices),
        topology_,
        multi_device_global_semaphore,
        subdevice_id,
        kernel_config_val);

    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .weight = weight,
        .transformation_mat = transformation_mat,
        .rope_cos = rope_cos,
        .rope_sin = rope_sin,
        .persistent_output_buffer = persistent_output_buffer};

    auto outputs = ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
    // outputs[0] = rmsnorm output, outputs[1] (if present) = stats DRAM scratch.
    return outputs[0];
}

}  // namespace ttnn::prim
