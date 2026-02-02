// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.hpp"

#include "ttnn/decorators.hpp"

#include <optional>
#include <variant>
#include <vector>

namespace ttnn::experimental::prim {

struct AllGatherMatmulAsyncDeviceOperation {
    using operation_attributes_t = AllGatherMatmulAsyncParams;
    using tensor_args_t = AllGatherMatmulAsyncInputs;
    using spec_return_value_t = AllGatherMatmulAsyncResultSpec;
    using tensor_return_value_t = AllGatherMatmulAsyncResult;
    using program_factory_t = std::variant<AllGatherMatmulAsyncMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::AllGatherMatmulAsyncDeviceOperation::tensor_return_value_t all_gather_matmul_async(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    CoreCoord all_gather_core_grid_offset,
    const std::optional<const Tensor>& bias = std::nullopt,
    uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config_ag = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    const std::optional<MemoryConfig>& memory_config_mm = std::nullopt,
    bool transpose_a = false,
    bool transpose_b = false,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
    const std::optional<const std::string>& activation = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<const ttnn::CoreGrid> core_grid = std::nullopt,
    std::optional<uint32_t> chunks_per_sync = std::nullopt,
    std::optional<uint32_t> num_workers_per_link = std::nullopt,
    std::optional<uint32_t> num_buffers_per_channel = std::nullopt);

}  // namespace ttnn::prim
