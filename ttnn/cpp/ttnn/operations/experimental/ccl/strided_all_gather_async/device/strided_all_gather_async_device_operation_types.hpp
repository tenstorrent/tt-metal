// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct StridedAllGatherAsyncParams {
    const std::vector<tt::tt_metal::IDevice*> devices;
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const ttnn::ccl::Topology topology;
    const std::vector<GlobalSemaphore> semaphore;
    const std::optional<uint32_t> cluster_axis;
    const std::optional<uint32_t> num_workers_per_link;
    const std::optional<uint32_t> num_buffers_per_channel;
    const std::optional<uint32_t> mm_cores_y;
    const std::optional<uint32_t> mm_block_ht;
    const std::optional<uint32_t> mm_block_wt;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "dim",
        "num_links",
        "ring_size",
        "output_mem_config",
        "topology",
        "cluster_axis",
        "num_workers_per_link",
        "num_buffers_per_channel",
        "mm_cores_y",
        "mm_block_ht",
        "mm_block_wt");
    auto attribute_values() const {
        return std::make_tuple(
            dim,
            num_links,
            ring_size,
            std::cref(output_mem_config),
            topology,
            std::cref(cluster_axis),
            std::cref(num_workers_per_link),
            std::cref(num_buffers_per_channel),
            std::cref(mm_cores_y),
            std::cref(mm_block_ht),
            std::cref(mm_block_wt));
    }
};

struct StridedAllGatherAsyncInputs {
    const Tensor input_tensor;
    const std::optional<Tensor> persistent_output_buffer;
};

}  // namespace ttnn::experimental::prim
