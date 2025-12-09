// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"  //TODO: migrate this code to use new all_gather_async API. This code relies on the old all_gather_async device_operation header

namespace ttnn::operations::experimental::ccl::all_gather_matmul_async {

struct operation_attributes_t {
    ttnn::AllGatherAsync all_gather_async;  // TODO: migrate this code to use new all_gather_async API. This code relies
                                            // on the old all_gather_async struct
    operations::matmul::Matmul
        matmul{};  // TODO: migrate this code to use new matmul API. This code relies on the old matmul struct
    CoreCoord all_gather_core_grid_offset;

    operation_attributes_t() :
        all_gather_async(make_default_all_gather_async()), matmul(), all_gather_core_grid_offset({0, 0}) {}
    operation_attributes_t(
        const ttnn::AllGatherAsync& all_gather_async,
        const operations::matmul::Matmul& matmul,
        CoreCoord all_gather_core_grid_offset) :
        all_gather_async(all_gather_async), matmul(matmul), all_gather_core_grid_offset(all_gather_core_grid_offset) {}

    static constexpr auto attribute_names = std::forward_as_tuple("matmul_struct", "all_gather_core_grid_offset");
    auto attribute_values() const { return std::forward_as_tuple(this->matmul, this->all_gather_core_grid_offset); }

private:
    static const ttnn::AllGatherAsync& make_default_all_gather_async() {
        static MemoryConfig default_mem_config{};
        static std::vector<GlobalSemaphore> default_semaphore{};
        static std::optional<tt::tt_metal::SubDeviceId> default_sub_device_id = std::nullopt;
        static std::optional<GlobalSemaphore> default_barrier = std::nullopt;
        static std::optional<uint32_t> default_cluster_axis = std::nullopt;
        static std::optional<uint32_t> default_chunks_per_sync = std::nullopt;
        static std::optional<uint32_t> default_num_workers_per_link = std::nullopt;
        static std::optional<uint32_t> default_num_buffers_per_channel = std::nullopt;
        static const ttnn::AllGatherAsync default_instance(
            /*dim=*/0,
            /*num_links=*/0,
            /*ring_size=*/0,
            default_mem_config,
            ttnn::ccl::Topology::Ring,
            default_semaphore,
            default_sub_device_id,
            default_cluster_axis,
            /*use_all_gather_async_llama_sharded=*/false,
            /*use_optimal_ccl_for_llama=*/false,
            default_barrier,
            /*using_persistent_buffers=*/false,
            default_chunks_per_sync,
            default_num_workers_per_link,
            default_num_buffers_per_channel);
        return default_instance;
    }
};

struct tensor_args_t {
    Tensor input_tensor;
    Tensor weight_tensor;
    std::optional<const Tensor> bias;
    std::optional<Tensor> persistent_output_buffer;
};

using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<Tensor>;

}  // namespace ttnn::operations::experimental::ccl::all_gather_matmul_async
