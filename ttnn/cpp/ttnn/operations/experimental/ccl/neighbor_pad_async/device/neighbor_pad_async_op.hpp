// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"

#include "ttnn/run_operation.hpp"

#include <optional>
#include <vector>

namespace ttnn {

struct NeighborPadAsync {
    std::vector<IDevice*> devices;
    const uint32_t dim;
    const uint32_t padding_left;
    const uint32_t padding_right;
    const std::string& padding_mode;
    const uint32_t cluster_axis;
    const GlobalSemaphore& final_semaphore;
    const GlobalSemaphore& barrier_semaphore;
    const uint32_t num_links;
    const MemoryConfig output_mem_config;
    const ccl::Topology topology;
    const uint32_t ring_size;
    std::optional<uint32_t> secondary_cluster_axis;
    std::optional<std::vector<uint32_t>> secondary_mesh_shape;

    NeighborPadAsync(
        std::vector<IDevice*> devices,
        uint32_t dim,
        uint32_t padding_left,
        uint32_t padding_right,
        const std::string& padding_mode,
        uint32_t cluster_axis,
        const GlobalSemaphore& final_semaphore,
        const GlobalSemaphore& barrier_semaphore,
        uint32_t num_links,
        MemoryConfig output_mem_config,
        ccl::Topology topology,
        uint32_t ring_size,
        std::optional<uint32_t> secondary_cluster_axis,
        std::optional<std::vector<uint32_t>> secondary_mesh_shape) :
        devices(std::move(devices)),
        dim(dim),
        padding_left(padding_left),
        padding_right(padding_right),
        padding_mode(padding_mode),
        cluster_axis(cluster_axis),
        final_semaphore(final_semaphore),
        barrier_semaphore(barrier_semaphore),
        num_links(num_links),
        output_mem_config(std::move(output_mem_config)),
        topology(topology),
        ring_size(ring_size),
        secondary_cluster_axis(secondary_cluster_axis),
        secondary_mesh_shape(std::move(secondary_mesh_shape)) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("dim", dim);
        attrs.emplace_back("padding_left", padding_left);
        attrs.emplace_back("padding_right", padding_right);
        attrs.emplace_back("padding_mode", padding_mode);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("final_semaphore", final_semaphore);
        attrs.emplace_back("barrier_semaphore", barrier_semaphore);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("secondary_cluster_axis", secondary_cluster_axis);
        attrs.emplace_back("secondary_mesh_shape", secondary_mesh_shape);
        return attrs;
    }

    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors) const;
    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

std::tuple<CoreRangeSet, std::vector<CoreCoord>> choose_worker_cores(
    size_t num_links, size_t num_workers_per_link, IDevice* device);

}  // namespace ttnn
