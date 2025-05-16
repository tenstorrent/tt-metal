// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include "ttnn/run_operation.hpp"

#include <optional>
#include <vector>
#include <algorithm>

/* Fusion includes */
#include "cpp/ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "cpp/ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn {
namespace experimental {

struct AllGatherMatmul {
    /* All Gather Params */
    const ttnn::AllGather all_gather_struct;

    /* Matmul Params */
    const operations::matmul::Matmul matmul_struct;

    /* Fusion Params */
    const CoreCoord all_gather_core_grid_offset;

    /* Physical Devices this op runs on*/
    std::vector<IDevice*> devices;

    /* General */
    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors = {std::nullopt}) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& mesh_coordinate,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
    static constexpr auto attribute_names = std::forward_as_tuple("all_gather_struct", "all_gather_core_grid_offset");
    const auto attribute_values() const {
        return std::forward_as_tuple(this->all_gather_struct, this->all_gather_core_grid_offset);
    }
};

tt::tt_metal::operation::ProgramWithCallbacks all_gather_matmul_multi_core_with_workers(

    /* General Params */
    const Tensor& input_tensor,
    Tensor& all_gather_output_tensor,
    Tensor& datacopy_output_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    chip_id_t target_device_id,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    ttnn::ccl::Topology topology,
    const CoreCoord core_grid_offset,

    /* Matmul Params */
    const std::optional<const Tensor> bias,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out);
}  // namespace experimental

namespace operations {
namespace experimental {
namespace ccl {

std::vector<Tensor> rs_matmul(
    const ttnn::Tensor& input_tensor,                           // mm0 used
    const ttnn::Tensor& weight_tensor,                          // mm1 used
    const ttnn::Tensor& rs_tensor,                              // rs1
    ttnn::Tensor& intermediate_packet_buffer,                   // rs2
    uint32_t dim,                                               // rs3
    const GlobalSemaphore& cross_device_semaphore,              // rs4
    const uint32_t cluster_axis,                                // rs 5
    const MeshDevice& mesh_device,                              // rs 6
    const uint32_t num_links,                                   // rs 7 default 1
    const std::optional<ttnn::MemoryConfig>& memory_config_rs,  // rs 8 default std::nullopt
    const std::optional<ttnn::MemoryConfig>& memory_config_mm,  // mm4 used but default std::nullopt
    const std::optional<const ttnn::DeviceComputeKernelConfig>
        compute_kernel_config,                                      // mm8 used but default std::nullopt
    const std::optional<const GlobalCircularBuffer>& global_cb,     // mm12 used but default std::nullopt
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,  // rs and mm13 used same but default std::nullopt
    const std::optional<const ttnn::CoreGrid> core_grid,            // mm9 may use but default std::nullopt
    const bool transpose_a,                                         // mm2 set false
    const bool transpose_b,                                         // mm3 set false
    const std::optional<const DataType> dtype,                      // mm5 set false
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,  // mm6 std::nullopt
    const std::optional<const std::string>& activation,                                  // mm7 set false
    const std::optional<const tt::tt_metal::Tile>& output_tile,                          // mm10 std::nullopt
    const std::optional<Tensor>& optional_output_tensor                                  // mm11 std::nullopt
);

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
