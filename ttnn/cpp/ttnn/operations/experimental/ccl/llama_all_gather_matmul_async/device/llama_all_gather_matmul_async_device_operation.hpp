// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <utility>
#include <vector>
#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_1d_mm_fusion.hpp"

namespace ttnn::operations::experimental::ccl {
namespace llama_all_gather_matmul_async {

struct LlamaAllGatherMatmulAsyncDeviceOperation {
    using operation_attributes_t = llama_all_gather_matmul_async::operation_attributes_t;
    using tensor_args_t = llama_all_gather_matmul_async::tensor_args_t;
    using spec_return_value_t = llama_all_gather_matmul_async::spec_return_value_t;
    using tensor_return_value_t = llama_all_gather_matmul_async::tensor_return_value_t;
    using program_factory_t = std::variant<program::LlamaAllGatherMatmulAsyncProgramFactory>;
    using shared_variables_t = program::LlamaAllGatherMatmulAsyncProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input0,
        const Tensor& input1,
        const Tensor& intermediate_tensor,
        const std::vector<IDevice*>& devices,
        int32_t dim,
        size_t num_links,
        size_t ring_size,
        const tt::tt_metal::MemoryConfig& output_memory_config,
        ttnn::ccl::Topology topology,
        const GlobalSemaphore& global_semaphore,
        const operations::matmul::Matmul& matmul_struct,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        const std::optional<tensor_return_value_t>& preallocated_output_tensors = std::nullopt);
};

}  // namespace llama_all_gather_matmul_async
}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {

constexpr auto llama_all_gather_matmul_async = ttnn::register_operation<
    "ttnn::prim::llama_all_gather_matmul_async",
    ttnn::operations::experimental::ccl::llama_all_gather_matmul_async::LlamaAllGatherMatmulAsyncDeviceOperation>();

}  // namespace ttnn::prim
