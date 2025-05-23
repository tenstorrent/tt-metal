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
#include "cpp/ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::ccl {

struct AllGatherRS {
    struct matmul_tensor_args_t {
        const Tensor input_tensor;
        const Tensor weight_tensor;
    };
    void validate_on_program_cache_miss(
        const LlamaReduceScatterDeviceOperation&,
        const LlamaReduceScatterDeviceOperation::operation_attributes_t&,
        const LlamaReduceScatterDeviceOperation::tensor_args_t&,
        const operations::matmul::Matmul&,
        const matmul_tensor_args_t&) const;
    void validate_on_program_cache_hit(
        const LlamaReduceScatterDeviceOperation&,
        const LlamaReduceScatterDeviceOperation::operation_attributes_t&,
        const LlamaReduceScatterDeviceOperation::tensor_args_t&,
        const operations::matmul::Matmul&,
        const matmul_tensor_args_t&) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(
        const LlamaReduceScatterDeviceOperation&,
        const LlamaReduceScatterDeviceOperation::operation_attributes_t&,
        const LlamaReduceScatterDeviceOperation::tensor_args_t&,
        const operations::matmul::Matmul&,
        const matmul_tensor_args_t&) const;
    std::vector<Tensor> create_output_tensors(
        const LlamaReduceScatterDeviceOperation&,
        const LlamaReduceScatterDeviceOperation::operation_attributes_t&,
        const LlamaReduceScatterDeviceOperation::tensor_args_t&,
        const operations::matmul::Matmul&,
        const matmul_tensor_args_t&) const;
    std::tuple<
        LlamaReduceScatterDeviceOperation,
        LlamaReduceScatterDeviceOperation::operation_attributes_t,
        LlamaReduceScatterDeviceOperation::tensor_args_t,
        operations::matmul::Matmul,
        matmul_tensor_args_t>
    invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        const ttnn::Tensor& rs_tensor,
        ttnn::Tensor& intermediate_packet_buffer,
        const int32_t dim,
        const GlobalSemaphore& semaphore,
        const tt::tt_metal::SubDeviceId subdevice_id,
        const uint32_t cluster_axis,
        const uint32_t ring_devices,
        const uint32_t num_links,
        const std::optional<ttnn::MemoryConfig>& memory_config_rs = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config_mm = std::nullopt,
        const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const GlobalCircularBuffer>& global_cb = std::nullopt,
        const std::optional<const ttnn::CoreGrid> core_grid = std::nullopt,
        const bool transpose_a = false,
        const bool transpose_b = false,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
        const std::optional<const std::string>& activation = std::nullopt,
        const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

}  // namespace ttnn::operations::experimental::ccl
