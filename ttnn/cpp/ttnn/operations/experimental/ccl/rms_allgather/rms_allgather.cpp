// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_allgather.hpp"

namespace ttnn {
namespace operations::fused::normalization {

ttnn::Tensor ExecuteFusedRMSNorm::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::operations::normalization::LayerNormProgramConfig& program_config,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    const std::optional<size_t> num_preferred_links,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<const DataType> dtype,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    float epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& stats,
    bool is_pre) {
    auto arch = is_tensor_on_device_or_multidevice(input_tensor)
                    ? input_tensor.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    const auto mesh_view = mesh_device.get_view();
    auto devices = input_tensor.get_workers();
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    int32_t rank = input_tensor.get_logical_shape().rank();
    std::vector<Tensor> output_tensors = {Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};
    std::vector<std::optional<Tensor>> optional_output_tensors = {persistent_output_tensor};
    const std::vector<std::optional<const Tensor>> optional_input_tensors = {residual_input_tensor, weight, stats};
    CoreCoord grid_size = devices[0]->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    tt::tt_metal::operation::launch_op(
        [num_preferred_links,
         memory_config,
         cluster_axis,
         num_devices,
         topology,
         semaphore,
         subdevice_id,
         epsilon,
         program_config,
         kernel_config_val,
         dtype,
         is_pre](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_device_tensor = input_tensors.at(0);
            const auto& input_tensor = input_tensors.at(0);
            return tt::tt_metal::operation::run(
                RMSAllGather(
                    epsilon,
                    memory_config.value_or(input_tensor.memory_config()),
                    program_config,
                    kernel_config_val,
                    dtype,
                    topology,
                    is_pre,
                    num_preferred_links.value_or(1),
                    num_devices,
                    semaphore,
                    subdevice_id,
                    cluster_axis),
                {input_tensor},
                optional_input_tensors,
                optional_output_tensors);
        },
        {input_tensor},
        output_tensors,
        optional_input_tensors,  // optional_input_tensors
        optional_output_tensors);
    return output_tensors.at(0);
}

}  // namespace operations::fused::normalization

}  // namespace ttnn
