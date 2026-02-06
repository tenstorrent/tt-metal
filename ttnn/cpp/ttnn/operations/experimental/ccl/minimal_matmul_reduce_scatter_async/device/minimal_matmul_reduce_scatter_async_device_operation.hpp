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

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "minimal_matmul_reduce_scatter_async_device_operation_types.hpp"
#include "minimal_matmul_reduce_scatter_async_program_factory.hpp"

namespace ttnn::operations::experimental::ccl::minimal_matmul_reduce_scatter_async {

struct MinimalMatmulReduceScatterAsyncDeviceOperation {
    using operation_attributes_t = minimal_matmul_reduce_scatter_async::operation_attributes_t;
    using tensor_args_t = minimal_matmul_reduce_scatter_async::tensor_args_t;
    using spec_return_value_t = minimal_matmul_reduce_scatter_async::spec_return_value_t;
    using tensor_return_value_t = minimal_matmul_reduce_scatter_async::tensor_return_value_t;
    using program_factory_t = std::variant<program::MinimalMatmulReduceScatterAsyncProgramFactory>;
    using shared_variables_t = program::MinimalMatmulReduceScatterAsyncProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::ccl::minimal_matmul_reduce_scatter_async

namespace ttnn::prim {

ttnn::operations::experimental::ccl::minimal_matmul_reduce_scatter_async::
    MinimalMatmulReduceScatterAsyncDeviceOperation::tensor_return_value_t
    minimal_matmul_reduce_scatter_async(
        const Tensor& input_tensor,
        const Tensor& weight_tensor,
        Tensor& persistent_intermediate_buffer,
        std::optional<ttnn::Tensor>& persistent_output_buffer,
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
        std::optional<uint32_t> cluster_axis = std::nullopt,
        std::optional<uint32_t> num_workers_per_link = std::nullopt,
        const std::optional<MemoryConfig>& memory_config_mm = std::nullopt,
        std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const ::ttnn::experimental::prim::MinimalMatmulConfig>& program_config = std::nullopt,
        const std::optional<const operations::unary::UnaryWithParam>& activation = std::nullopt,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::prim
