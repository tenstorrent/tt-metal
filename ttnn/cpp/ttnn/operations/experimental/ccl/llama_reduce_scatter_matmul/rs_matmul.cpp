// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_op.hpp"
#include "ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/rs_matmul.hpp"

namespace ttnn::operations::experimental::ccl {

std::vector<ttnn::Tensor> ExecuteLlamaReduceScatterMatmul::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,               // mm0 used
    const ttnn::Tensor& weight_tensor,              // mm1 used
    ttnn::Tensor& intermediate_packet_buffer,       // rs2
    int32_t dim,                                    // rs3
    const GlobalSemaphore& cross_device_semaphore,  // rs4
    const uint32_t cluster_axis,                    // rs 5
    const MeshDevice& mesh_device,                  // rs 6
    const uint32_t num_links,                       // rs 7 default 1
    const tt::tt_metal::SubDeviceId& subdevice_id,
    const std::optional<const ttnn::Tensor>& second_weight_tensor,
    const std::optional<const ttnn::Tensor>& rs_tensor,  // rs1
    tt::tt_fabric::Topology topology,
    const std::optional<ttnn::MemoryConfig>& memory_config_rs,  // rs 8 default std::nullopt
    const std::optional<ttnn::MemoryConfig>& memory_config_mm,  // mm4 used but default std::nullopt
    const std::optional<const ttnn::DeviceComputeKernelConfig>
        compute_kernel_config,                                   // mm8 used but default std::nullopt
    const std::optional<const GlobalCircularBuffer>& global_cb,  // mm12 used but default std::nullopt
    const std::optional<const ttnn::CoreGrid> core_grid,         // mm9 may use but default std::nullopt
    const bool transpose_a,                                      // mm2 set false
    const bool transpose_b,                                      // mm3 set false
    const std::optional<const DataType> dtype,                   // mm5 set false
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,  // mm6 std::nullopt
    const std::optional<const std::string>& activation,                                  // mm7 set false
    const std::optional<const tt::tt_metal::Tile>& output_tile,                          // mm10 std::nullopt
    const std::optional<Tensor>& optional_output_tensor,                                 // mm11 std::nullopt
    bool use_noc1_only) {
    const auto& mesh_view = mesh_device.get_view();
    const uint32_t ring_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    TT_FATAL(ring_devices > 1, "reduce_scatter async op will only work for ring_devices > 1, but has {}", ring_devices);
    return ttnn::prim::llama_rs_matmul(
        input_tensor,
        weight_tensor,
        rs_tensor,
        intermediate_packet_buffer,
        dim,
        cross_device_semaphore,
        cluster_axis,
        ring_devices,
        num_links,
        subdevice_id,
        memory_config_rs,
        memory_config_mm,
        compute_kernel_config,
        global_cb,
        core_grid,
        transpose_a,
        transpose_b,
        dtype,
        program_config,
        activation,
        output_tile,
        optional_output_tensor,
        topology,
        use_noc1_only,
        second_weight_tensor);
}

}  // namespace ttnn::operations::experimental::ccl
