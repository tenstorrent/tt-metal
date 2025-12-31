// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_allgather.hpp"
#include "device/rms_allgather_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include <ttnn/device.hpp>

namespace ttnn {

Tensor fused_rms_1_1_32_8192(
    const Tensor& input_tensor,
    const operations::normalization::LayerNormProgramConfig& program_config,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& semaphore,
    const std::optional<Tensor>& persistent_output_tensor,
    std::optional<size_t> num_preferred_links,
    ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<const DataType> dtype,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const Tensor>& residual_input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& stats,
    bool use_noc1_only) {
    using OperationType = operations::fused::normalization::RMSAllGatherDeviceOperation;

    auto arch = is_device_tensor(input_tensor) ? input_tensor.device()->arch() : GetDefaultDevice()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    const auto& mesh_view = mesh_device.get_view();
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    tt::tt_fabric::Topology topology_ = ccl::get_usable_topology(input_tensor, topology, cluster_axis);

    auto [subblock_wt, block_wt, inplace, grid_size] = std::visit(
        [](const auto& config) -> std::tuple<uint32_t, uint32_t, bool, CoreCoord> {
            using T = std::decay_t<decltype(config)>;
            if constexpr (std::is_same_v<T, operations::normalization::LayerNormShardedMultiCoreProgramConfig>) {
                return {
                    static_cast<uint32_t>(config.subblock_w),
                    static_cast<uint32_t>(config.block_w),
                    config.inplace,
                    config.compute_with_storage_grid_size};
            } else {
                TT_FATAL(false, "RMSAllGather only supports LayerNormShardedMultiCoreProgramConfig");
                return {0, 0, false, CoreCoord{0, 0}};
            }
        },
        program_config);

    auto operation_attributes = OperationType::operation_attributes_t(
        epsilon,
        memory_config.value_or(input_tensor.memory_config()),
        subblock_wt,
        block_wt,
        inplace,
        grid_size,
        kernel_config_val,
        dtype,
        topology_,
        num_preferred_links.value_or(1),
        num_devices,
        semaphore,
        subdevice_id,
        cluster_axis,
        use_noc1_only);

    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .residual_input_tensor = residual_input_tensor,
        .weight = weight,
        .stats = stats,
        .preallocated_output = persistent_output_tensor};

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn
