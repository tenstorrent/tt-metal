// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_op_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_ring_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail {

struct StridedReduceScatterAsyncDeviceOperation {
    using operation_attributes_t = strided_reduce_scatter_async::detail::operation_attributes_t;
    using tensor_args_t = strided_reduce_scatter_async::detail::tensor_args_t;
    using spec_return_value_t = strided_reduce_scatter_async::detail::spec_return_value_t;
    using tensor_return_value_t = strided_reduce_scatter_async::detail::tensor_return_value_t;
    using program_factory_t = std::variant<RingStridedReduceScatterMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail

namespace ttnn::prim {

ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail::StridedReduceScatterAsyncDeviceOperation::
    tensor_return_value_t
    strided_reduce_scatter_async(
        const ttnn::Tensor& input_tensor,
        const std::optional<ttnn::Tensor>& optional_intermediate_tensor,
        const std::optional<ttnn::Tensor>& optional_output_tensor,
        uint32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        MemoryConfig output_mem_config,
        std::optional<MemoryConfig> optional_intermediate_mem_config,
        ttnn::ccl::Topology topology,
        std::vector<GlobalSemaphore> semaphore,
        std::optional<GlobalSemaphore> barrier_semaphore,
        bool using_persistent_buffers,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
        std::optional<uint32_t> cluster_axis,
        std::optional<uint32_t> chunks_per_sync,
        std::optional<uint32_t> num_workers_per_link,
        std::optional<uint32_t> num_buffers_per_channel,
        std::optional<uint32_t> mm_cores_y,
        std::optional<uint32_t> mm_block_ht,
        std::optional<uint32_t> mm_block_wt,
        std::optional<uint32_t> mm_M_block_ht,
        std::optional<uint32_t> mm_N_block_wt,
        std::optional<uint32_t> chunk_width_in_mm_blocks);

}  // namespace ttnn::prim
