// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "matmul_reduce_scatter_async_device_operation_types.hpp"
#include "matmul_reduce_scatter_async_program_factory.hpp"
#include "ttnn/decorators.hpp"
#include <variant>

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async {

struct MatmulReduceScatterAsyncDeviceOperation {
    using operation_attributes_t = matmul_reduce_scatter_async::operation_attributes_t;
    using tensor_args_t = matmul_reduce_scatter_async::tensor_args_t;
    using spec_return_value_t = matmul_reduce_scatter_async::spec_return_value_t;
    using tensor_return_value_t = matmul_reduce_scatter_async::tensor_return_value_t;
    using program_factory_t = std::variant<program::MatmulReduceScatterAsyncMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& weight_tensor,
        Tensor& persistent_intermediate_buffer,
        Tensor& persistent_output_buffer,
        uint32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        CoreCoord reduce_scatter_core_grid_offset,
        const std::optional<GlobalSemaphore>& barrier_semaphore,
        const std::optional<const Tensor>& bias,
        uint32_t num_links,
        const std::optional<MemoryConfig>& memory_config_rs,
        const std::optional<MemoryConfig>& intermediate_memory_config_rs,
        ttnn::ccl::Topology topology,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
        const std::optional<MemoryConfig>& memory_config_mm,
        bool transpose_a,
        bool transpose_b,
        std::optional<const DataType> dtype,
        const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
        const std::optional<const std::string>& activation,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
        std::optional<const ttnn::CoreGrid> core_grid);
};

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async

namespace ttnn::prim {
constexpr auto matmul_reduce_scatter_async = ttnn::register_operation<
    "ttnn::prim::matmul_reduce_scatter_async",
    ttnn::operations::experimental::ccl::matmul_reduce_scatter_async::MatmulReduceScatterAsyncDeviceOperation>();
}  // namespace ttnn::prim
