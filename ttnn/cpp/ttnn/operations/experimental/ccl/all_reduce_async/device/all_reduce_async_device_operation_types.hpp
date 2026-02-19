// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/sub_device_types.hpp>
#include <tt_stl/reflection.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::experimental::prim {

struct AllReduceAsyncParams {
    uint32_t num_links = 0;
    uint32_t ring_size = 0;
    DataType dtype = DataType::BFLOAT16;
    MemoryConfig output_mem_config;
    ttnn::ccl::Topology topology;
    GlobalSemaphore semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    bool use_noc1_only = false;
    bool use_optimal_ccl_for_llama = false;
    uint32_t cluster_axis = 0;
    distributed::MeshDevice* mesh_device = nullptr;

    AllReduceAsyncParams(
        uint32_t num_links,
        uint32_t ring_size,
        DataType dtype,
        MemoryConfig output_mem_config,
        ttnn::ccl::Topology topology,
        GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        bool use_noc1_only,
        bool use_optimal_ccl_for_llama,
        uint32_t cluster_axis,
        distributed::MeshDevice* mesh_device) :
        num_links(num_links),
        ring_size(ring_size),
        dtype(dtype),
        output_mem_config(std::move(output_mem_config)),
        topology(topology),
        semaphore(std::move(semaphore)),
        sub_device_id(sub_device_id),
        use_noc1_only(use_noc1_only),
        use_optimal_ccl_for_llama(use_optimal_ccl_for_llama),
        cluster_axis(cluster_axis),
        mesh_device(mesh_device) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("dtype", dtype);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("semaphore", semaphore);
        attrs.emplace_back("use_noc1_only", use_noc1_only);
        attrs.emplace_back("use_optimal_ccl_for_llama", use_optimal_ccl_for_llama);
        attrs.emplace_back("cluster_axis", cluster_axis);
        return attrs;
    }
};

struct AllReduceAsyncInputs {
    Tensor input_tensor;
    Tensor buffer_tensor;
};

}  // namespace ttnn::experimental::prim
