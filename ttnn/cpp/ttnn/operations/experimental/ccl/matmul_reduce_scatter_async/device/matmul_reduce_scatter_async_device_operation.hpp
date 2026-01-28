// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"

/* Fusion includes */
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_program_factory.hpp"

namespace ttnn::experimental::prim {

struct MatmulReduceScatterAsyncDeviceOperation {
    using operation_attributes_t = MatmulReduceScatterAsyncParams;
    using tensor_args_t = MatmulReduceScatterAsyncInputs;
    using spec_return_value_t = MatmulReduceScatterAsyncResultSpec;
    using tensor_return_value_t = MatmulReduceScatterAsyncResult;
    using program_factory_t = std::variant<MatmulReduceScatterAsyncProgramFactory>;
    using shared_variables_t = MatmulReduceScatterAsyncProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::MatmulReduceScatterAsyncDeviceOperation::tensor_return_value_t matmul_reduce_scatter_async(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    Tensor& persistent_intermediate_buffer,
    Tensor& persistent_output_buffer,
    uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    CoreCoord reduce_scatter_core_grid_offset,
    const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
    const std::optional<const Tensor>& bias = std::nullopt,
    uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config_rs = std::nullopt,
    const std::optional<MemoryConfig>& intermediate_memory_config_rs = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    const std::optional<MemoryConfig>& memory_config_mm = std::nullopt,
    bool transpose_a = false,
    bool transpose_b = false,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
    const std::optional<const std::string>& activation = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<const ttnn::CoreGrid> core_grid = std::nullopt);

}  // namespace ttnn::prim
