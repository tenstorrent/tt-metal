// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "reduce_scatter_minimal_async_types.hpp"
#include "ring_reduce_scatter_minimal_async_program.hpp"
#include "line_reduce_scatter_minimal_async_program.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include <variant>

namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async {

void reduce_scatter_common_validates(
    const ttnn::Tensor& input_tensor,
    ttnn::ccl::Topology topology,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor);

struct ReduceScatterMinimalAsyncDeviceOperation {
    using operation_attributes_t = reduce_scatter_minimal_async::operation_attributes_t;
    using tensor_args_t = reduce_scatter_minimal_async::tensor_args_t;
    using spec_return_value_t = reduce_scatter_minimal_async::spec_return_value_t;
    using tensor_return_value_t = reduce_scatter_minimal_async::tensor_return_value_t;
    using program_factory_t = std::variant<
        program::ring::RingReduceScatterMinimalAsyncProgramFactory,
        program::line::LineReduceScatterMinimalAsyncProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const std::optional<std::vector<Tensor>>& persistent_output_buffers,
        uint32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        const std::optional<GlobalSemaphore>& barrier_semaphore,
        uint32_t num_links,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<MemoryConfig>& intermediate_memory_config,
        ttnn::ccl::Topology topology,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
        std::optional<uint32_t> cluster_axis,
        std::optional<uint32_t> chunks_per_sync,
        std::optional<uint32_t> num_workers_per_link,
        std::optional<uint32_t> num_buffers_per_channel);
};

}  // namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async

namespace ttnn::prim {
constexpr auto reduce_scatter_minimal_async = ttnn::register_operation<
    "ttnn::prim::reduce_scatter_minimal_async",
    ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::ReduceScatterMinimalAsyncDeviceOperation>();
}  // namespace ttnn::prim
