// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::experimental::all_gather_minimal_matmul_async {

struct all_gather_minimal_matmul_async_override_variables_t {
    uint32_t num_cores;
    std::vector<CoreCoord> cores;
    tt::tt_metal::KernelHandle in0_sender_kernels_id;
    tt::tt_metal::KernelHandle in0_receiver_kernels_id;
    tt::tt_metal::KernelHandle in1_sender_kernels_id;
    tt::tt_metal::KernelHandle in1_receiver_kernels_id;
    bool transpose_core_grid;
};

struct AllGatherMinimalMatmulAsyncConfig {
    AllGatherMinimalMatmulAsyncConfig(
        uint32_t M_block_size_ = 1,
        uint32_t K_block_size_ = 1,
        uint32_t N_block_size_ = 1,
        uint32_t subblock_h_ = 1,
        uint32_t subblock_w_ = 1,
        CoreCoord compute_with_storage_grid_size_ = {1, 1}) :
        M_block_size(M_block_size_),
        K_block_size(K_block_size_),
        N_block_size(N_block_size_),
        subblock_h(subblock_h_),
        subblock_w(subblock_w_),
        compute_with_storage_grid_size(compute_with_storage_grid_size_) {}

    uint32_t M_block_size;
    uint32_t K_block_size;
    uint32_t N_block_size;
    uint32_t subblock_h;
    uint32_t subblock_w;

    CoreCoord compute_with_storage_grid_size;

    static constexpr auto attribute_names = std::make_tuple(
        "M_block_size", "K_block_size", "N_block_size", "subblock_h", "subblock_w", "compute_with_storage_grid_size");

    auto attribute_values() const {
        return std::forward_as_tuple(
            this->M_block_size,
            this->K_block_size,
            this->N_block_size,
            this->subblock_h,
            this->subblock_w,
            this->compute_with_storage_grid_size);
    }
};

struct AllGatherMinimalMatmulAsyncOp {
    std::optional<const AllGatherMinimalMatmulAsyncConfig> config;
    std::optional<unary::UnaryWithParam> fused_activation;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    uint32_t num_links;
    uint32_t ring_size;
    ttnn::ccl::Topology topology;
    std::vector<GlobalSemaphore> semaphore;
    std::optional<uint32_t> cluster_axis;
    const std::optional<GlobalSemaphore>& barrier_semaphore;
    bool using_persistent_buffers;
    uint32_t chunks_per_sync;
    uint32_t num_workers_per_link;
    uint32_t num_buffers_per_channel;

    AllGatherMinimalMatmulAsyncOp(
        std::optional<const AllGatherMinimalMatmulAsyncConfig> config,
        std::optional<unary::UnaryWithParam> fused_activation,
        std::optional<tt::tt_metal::MemoryConfig> output_mem_config,
        std::optional<tt::tt_metal::DataType> output_dtype,
        DeviceComputeKernelConfig compute_kernel_config,
        uint32_t num_links,
        uint32_t ring_size,
        ttnn::ccl::Topology topology,
        std::vector<GlobalSemaphore> semaphore,
        std::optional<uint32_t> cluster_axis,
        const std::optional<GlobalSemaphore>& barrier_semaphore,
        bool using_persistent_buffers,
        uint32_t chunks_per_sync,
        uint32_t num_workers_per_link,
        uint32_t num_buffers_per_channel) :
        config(config),
        fused_activation(fused_activation),
        output_mem_config(output_mem_config),
        output_dtype(output_dtype),
        compute_kernel_config(compute_kernel_config),
        num_links(num_links),
        ring_size(ring_size),
        topology(topology),
        semaphore(std::move(semaphore)),
        cluster_axis(cluster_axis),
        barrier_semaphore(barrier_semaphore),
        using_persistent_buffers(using_persistent_buffers),
        chunks_per_sync(chunks_per_sync),
        num_workers_per_link(num_workers_per_link),
        num_buffers_per_channel(num_buffers_per_channel) {}

    static constexpr auto attribute_names = std::make_tuple(
        "num_links",
        "ring_size",
        "topology",
        "barrier_semaphore",
        "using_persistent_buffers",
        "cluster_axis",
        "semaphore",
        "chunks_per_sync",
        "num_workers_per_link",
        "num_buffers_per_channel");

    auto attribute_values() const {
        return std::forward_as_tuple(
            this->num_links,
            this->ring_size,
            this->topology,
            this->barrier_semaphore,
            this->using_persistent_buffers,
            this->cluster_axis,
            this->semaphore,
            this->chunks_per_sync,
            this->num_workers_per_link,
            this->num_buffers_per_channel);
    }

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;

    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors) const;

    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& coord,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::experimental::all_gather_minimal_matmul_async
