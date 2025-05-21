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
#include "cpp/ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::ccl {

struct AllGatherRS {
    const operations::matmul::Matmul matmul_struct;
    const LlamaReduceScatterDeviceOperation rs_struct;

    void validate_on_program_cache_miss(
        const LlamaReduceScatterDeviceOperation::operation_attributes_t&,
        const LlamaReduceScatterDeviceOperation::tensor_args_t&,
        std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors);
    void validate_on_program_cache_hit(
        const LlamaReduceScatterDeviceOperation::operation_attributes_t&,
        const LlamaReduceScatterDeviceOperation::tensor_args_t&,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors);
};
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

}  // namespace ttnn::operations::experimental::ccl
