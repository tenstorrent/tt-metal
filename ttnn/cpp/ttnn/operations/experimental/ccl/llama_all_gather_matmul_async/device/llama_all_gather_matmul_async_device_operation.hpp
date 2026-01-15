// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.hpp"

namespace ttnn::experimental::prim {

struct LlamaAllGatherMatmulAsyncDeviceOperation {
    using operation_attributes_t = LlamaAllGatherMatmulAsyncParams;
    using tensor_args_t = LlamaAllGatherMatmulAsyncInputs;
    using spec_return_value_t = LlamaAllGatherMatmulAsyncResultSpec;
    using tensor_return_value_t = LlamaAllGatherMatmulAsyncResult;
    using program_factory_t = std::variant<LlamaAllGatherMatmulAsyncProgramFactory>;
    using shared_variables_t = LlamaAllGatherMatmulAsyncProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::LlamaAllGatherMatmulAsyncDeviceOperation::tensor_return_value_t llama_all_gather_matmul_async(
    const Tensor& input0,
    const Tensor& input1,
    const Tensor& intermediate_tensor,
    int32_t dim,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& global_semaphore,
    const std::optional<tt::tt_metal::MemoryConfig>& ag_memory_config,
    const std::optional<tt::tt_metal::MemoryConfig>& mm_memory_config,
    std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb = std::nullopt);

}  // namespace ttnn::prim
