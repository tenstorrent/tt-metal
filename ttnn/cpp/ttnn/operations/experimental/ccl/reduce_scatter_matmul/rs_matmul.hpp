// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/ccl/reduce_scatter_matmul/device/rs_matmul_op.hpp"
#include "ttnn/distributed/api.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteReduceScatterMatmul {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,                                          // mm0 used
        const ttnn::Tensor& weight_tensor,                                         // mm1 used
        const ttnn::Tensor& rs_tensor,                                             // rs1
        ttnn::Tensor& intermediate_packet_buffer,                                  // rs2
        uint32_t dim,                                                              // rs3
        const GlobalSemaphore& cross_device_semaphore,                             // rs4
        const uint32_t cluster_axis,                                               // rs 5
        const MeshDevice& mesh_device,                                             // rs 6
        const uint32_t num_links,                                                  // rs 7 default 1
        const std::optional<ttnn::MemoryConfig>& memory_config_rs = std::nullopt,  // rs 8 default std::nullopt
        const std::optional<ttnn::MemoryConfig>& memory_config_mm = std::nullopt,  // mm4 used but default std::nullopt
        const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config =
            std::nullopt,  // mm8 used but default std::nullopt
        const std::optional<const GlobalCircularBuffer>& global_cb =
            std::nullopt,  // mm12 used but default std::nullopt
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id =
            std::nullopt,  // rs and mm13 used same but default std::nullopt
        const std::optional<const ttnn::CoreGrid> core_grid = std::nullopt,  // mm9 may use but default std::nullopt
        const bool transpose_a = false,                                      // mm2 set false
        const bool transpose_b = false,                                      // mm3 set false
        const std::optional<const DataType> dtype = std::nullopt,            // mm5 set false
        const std::optional<const operations::matmul::MatmulProgramConfig>& program_config =
            std::nullopt,                                                           // mm6 std::nullopt
        const std::optional<const std::string>& activation = std::nullopt,          // mm7 set false
        const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,  // mm10 std::nullopt
        const std::optional<Tensor>& optional_output_tensor = std::nullopt          // mm11 std::nullopt
    );
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto reduce_scatter_matmul = ttnn::register_operation<
    "ttnn::experimental::rs_matmul",
    ttnn::operations::experimental::ccl::ExecuteReduceScatterMatmul>();

}  // namespace experimental
}  // namespace ttnn
