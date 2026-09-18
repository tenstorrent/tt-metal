// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_sp_async/device/matmul_reduce_scatter_sp_async_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_sp_async/device/matmul_reduce_scatter_sp_async_program_factory.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct MatmulReduceScatterSpAsyncDeviceOperation {
    using operation_attributes_t = MatmulReduceScatterSpAsyncParams;
    using tensor_args_t = MatmulReduceScatterSpAsyncInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<MatmulReduceScatterSpAsyncProgramFactory>;
    using shared_variables_t = MatmulReduceScatterSpAsyncProgramFactory::shared_variables_t;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// All parameters already resolved by the ttnn::experimental wrapper (num_links, ring_size, usable topology,
// compute kernel configs, worker count). Returns the op's full output vector (see k*Idx in the types header).
std::vector<Tensor> matmul_reduce_scatter_sp_async(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    uint32_t cluster_axis,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool transpose_b,
    uint32_t num_links,
    uint32_t ring_size,
    ttnn::ccl::Topology topology,
    uint32_t ccl_core_rows,
    uint32_t num_workers_per_link,
    const std::optional<MemoryConfig>& memory_config,
    tt::tt_metal::DataType output_dtype,
    const ttnn::DeviceComputeKernelConfig& matmul_compute_kernel_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& reduce_scatter_compute_kernel_config,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    bool debug_serialize_reduce_scatter = false);

}  // namespace ttnn::prim
