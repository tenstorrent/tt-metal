// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_rms_norm_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/device.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {

void AllGatherRMSNormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const auto& gamma = tensor_args.weight;
    const auto& beta = tensor_args.bias;

    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    // Generic regime: interleaved, TILE, arbitrary M (no single-tile-row restriction).
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to all_gather_rms_norm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to all_gather_rms_norm need to be allocated in buffers on device!");
    TT_FATAL(a.layout() == Layout::TILE, "Input tensor layout must be TILE but got {}", a.layout());
    TT_FATAL(
        a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input tensor must be INTERLEAVED but got {}",
        a.memory_config().memory_layout());
    TT_FATAL(
        a.dtype() == DataType::FLOAT32 or a.dtype() == DataType::BFLOAT16 or a.dtype() == DataType::BFLOAT8_B,
        "Input tensor dtype must be FLOAT32, BFLOAT16, or BFLOAT8_B but got {}",
        a.dtype());
    TT_FATAL(
        a.padded_shape()[-1] % tile_width == 0,
        "Input last dim ({}) must be a multiple of tile width ({})",
        a.padded_shape()[-1],
        tile_width);

    TT_FATAL(args.num_links > 0, "num_links must be greater than 0 but got {}", args.num_links);
    // ring_size == 1 (single device) is allowed: the reduce runs entirely locally with no fabric.
    TT_FATAL(args.ring_size >= 1, "ring_size must be >= 1 but got {}", args.ring_size);

    if (b.has_value()) {
        TT_FATAL(b.value().layout() == Layout::TILE, "Residual tensor layout must be TILE");
        TT_FATAL(a.padded_shape() == b.value().padded_shape(), "Residual shape must match input shape");
        TT_FATAL(a.device() == b.value().device(), "Residual tensor must be on the same device as input");
    }

    if (gamma.has_value()) {
        TT_FATAL(a.device() == gamma.value().device(), "Input tensor device must match gamma tensor device");
        TT_FATAL(
            gamma.value().dtype() == DataType::FLOAT32 or gamma.value().dtype() == DataType::BFLOAT16,
            "Gamma tensor dtype must be FLOAT32 or BFLOAT16 but got {}",
            gamma.value().dtype());
    }

    TT_FATAL(args.has_beta == beta.has_value(), "has_beta ({}) must match presence of bias tensor", args.has_beta);
    if (beta.has_value()) {
        TT_FATAL(gamma.has_value(), "beta (bias) requires weight (gamma) to be present");
        TT_FATAL(a.device() == beta.value().device(), "Input tensor device must match beta tensor device");
        TT_FATAL(
            beta.value().dtype() == DataType::FLOAT32 or beta.value().dtype() == DataType::BFLOAT16,
            "Beta tensor dtype must be FLOAT32 or BFLOAT16 but got {}",
            beta.value().dtype());
    }
}

TensorSpec AllGatherRMSNormDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Note: preallocated_stats (if provided) is the intermediate gathered-stats buffer, not the output.
    const auto& input_tensor = tensor_args.input;
    return TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout(args.dtype.value_or(input_tensor.dtype()), PageConfig(Layout::TILE), args.output_mem_config));
}

Tensor AllGatherRMSNormDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

ttsl::hash::hash_t AllGatherRMSNormDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllGatherRMSNormDeviceOperation::compute_program_hash is called");

    auto* mesh_device = tensor_args.input.device();
    auto sd_id = args.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    return tt::tt_metal::operation::hash_operation<AllGatherRMSNormDeviceOperation>(
        args.eps,
        args.output_mem_config,
        args.compute_kernel_config,
        args.dtype,
        args.topology,
        args.num_links,
        args.ring_size,
        args.cluster_axis,
        args.has_beta,
        subdevice_core_range_set,
        tensor_args);
}

ttnn::Tensor all_gather_rms_norm(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& global_semaphore,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    float epsilon,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    ttnn::ccl::Topology topology,
    std::optional<size_t> num_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::Tensor>& persistent_stats_tensor) {
    using OperationType = AllGatherRMSNormDeviceOperation;

    auto arch = is_device_tensor(input_tensor) ? input_tensor.device()->arch() : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        arch, compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4, true, false, false);

    const auto& mesh_view = mesh_device.get_view();
    uint32_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    // get_usable_topology() queries the fabric context, which is only initialized on multi-device runs.
    // For a single device (num_devices == 1) the reduce is entirely local (no fabric), so skip the query
    // and pass the requested topology through unchanged.
    ttnn::ccl::Topology topology_ =
        (num_devices > 1) ? ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis) : topology;

    auto operation_attributes = OperationType::operation_attributes_t(
        epsilon,
        memory_config.value_or(input_tensor.memory_config()),
        kernel_config_val,
        dtype,
        topology_,
        num_links.value_or(1),
        num_devices,
        cluster_axis,
        bias.has_value(),
        global_semaphore,
        subdevice_id);

    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .residual_input_tensor = residual_input_tensor,
        .weight = weight,
        .bias = bias,
        .preallocated_stats = persistent_stats_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
