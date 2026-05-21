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

    // Post-allgather output reshapes to [1, num_heads_per_device, N, H/num_heads_per_device].
    ttnn::Shape output_shape({1u, args.num_heads_per_device, logical[2], logical[3] / args.num_heads_per_device});

    const auto out_dtype = args.dtype.value_or(input.dtype());

    return TensorSpec(output_shape, TensorLayout(out_dtype, PageConfig(Layout::TILE), args.output_mem_config));
}

WanFusedDistributedRmsnormDeviceOperation::tensor_return_value_t
WanFusedDistributedRmsnormDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(spec, tensor_args.input.device());
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

ttnn::experimental::prim::WanFusedDistributedRmsnormDeviceOperation::tensor_return_value_t
wan_fused_distributed_rmsnorm(
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

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
