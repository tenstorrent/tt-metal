// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_sp_async/device/all_gather_matmul_sp_async_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_sp_async/device/all_gather_matmul_sp_async_program_factory.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct AllGatherMatmulSpAsyncDeviceOperation {
    using operation_attributes_t = AllGatherMatmulSpAsyncParams;
    using tensor_args_t = AllGatherMatmulSpAsyncInputs;
    using spec_return_value_t = AllGatherMatmulSpAsyncResultSpec;
    using tensor_return_value_t = AllGatherMatmulSpAsyncResult;
    using program_factory_t = std::variant<AllGatherMatmulSpAsyncMeshWorkloadFactory>;

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// all_gather_async's default worker heuristic (default_workers in all_gather_async_default_program_factory.cpp:
// 4/2/1 workers per direction by data moved per link) capped by what fits the bottom `ccl_core_rows` rows of the
// device grid: num_links x 2 directions x (workers + 1 mux core when workers > 1) cores.
uint32_t sp_default_all_gather_workers(
    const Tensor& input, uint32_t ring_size, ttnn::ccl::Topology topology, uint32_t num_links, uint32_t ccl_core_rows);

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Fully resolved entry point (num_links, topology, workers and the matmul numerics already decided).
ttnn::experimental::prim::AllGatherMatmulSpAsyncDeviceOperation::tensor_return_value_t all_gather_matmul_sp_async(
    const Tensor& input,
    const Tensor& weight,
    uint32_t cluster_axis,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool transpose_b,
    const std::optional<const Tensor>& bias,
    uint32_t num_links,
    ttnn::ccl::Topology topology,
    uint32_t ccl_core_rows,
    uint32_t num_workers_per_link,
    const std::optional<MemoryConfig>& memory_config,
    DataType output_dtype,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    bool debug_serialize_ag = false,
    bool ag_signal_on_receive = true,
    bool in1_resident = true);

}  // namespace ttnn::prim
