// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>

#include <cstdint>
#include <vector>
#include <optional>

namespace ttnn {
namespace ccl {

class RingTopology;

namespace reduce_scatter_detail {

enum class Direction {
    CLOCKWISE = 0,
    RIGHT = 0,

    COUNTER_CLOCKWISE = 1,
    LEFT = 1,

    UNASSIGNED
};

static_assert(
    Direction::CLOCKWISE == Direction::RIGHT,
    "Direction::CLOCKWISE == Direction::RIGHT not equal but expected to be for current design");
static_assert(
    Direction::COUNTER_CLOCKWISE == Direction::LEFT,
    "Direction::COUNTER_CLOCKWISE == Direction::LEFT not equal but expected to be for current design");

/*
 * Contains various attributes about a given worker
 */
struct WorkerAttributes {
    std::size_t link = std::numeric_limits<std::size_t>::max();
    std::size_t channel = std::numeric_limits<std::size_t>::max();

    // Workers cooperate to process the data in a given slice. This represents the index in that
    // list of workers. Note that for line reduce scatter, we may have n workers but only n/2 of
    // them will cooperate for a given slice (half will work on slices from one direction and the
    // other half for the other direction)
    std::size_t index_in_slice = std::numeric_limits<std::size_t>::max();
    Direction direction = Direction::UNASSIGNED;
    CoreCoord location_logical = {std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()};
    std::optional<uint32_t> send_to_edm_semaphore_id = std::nullopt;
    std::optional<uint32_t> receive_from_edm_semaphore_id = std::nullopt;
    std::optional<std::size_t> associated_worker_index = std::nullopt;
    std::optional<CoreCoord> associated_worker_core_logical = std::nullopt;
};

struct WorkerTransferInfo {
    WorkerTransferInfo(
        std::vector<uint32_t> const& pages_per_full_chunk_per_worker, uint32_t num_links, uint32_t num_workers);

    uint32_t get_num_pages_per_full_chunk(WorkerAttributes const& worker_attrs) const;

    std::vector<uint32_t> pages_per_full_chunk_per_worker;
    uint32_t num_links;
    uint32_t num_workers;
};

std::size_t get_global_worker_id(std::size_t link, std::size_t channel_id, std::size_t num_channels_per_link);
std::size_t get_global_worker_id(WorkerAttributes const& attrs, std::size_t num_channels_per_link);
std::size_t get_worker_index_in_slice(
    ttnn::ccl::RingTopology const& tc,
    std::size_t global_worker_index,
    std::size_t worker_channel_id,
    std::size_t num_edm_channels_per_link,
    std::size_t link);

std::vector<WorkerAttributes> build_worker_attributes(
    ttnn::ccl::RingTopology const& topology_config,
    std::vector<CoreCoord> const& worker_cores_list,
    std::optional<std::vector<CoreCoord>> const& second_worker_cores_list,

    uint32_t worker_sender_semaphore_id,
    uint32_t worker_receiver_semaphore_id,
    std::optional<uint32_t> worker_sender_semaphore_id_second_core_range,
    std::optional<uint32_t> worker_receiver_semaphore_id_second_core_range,

    std::size_t num_links,
    std::size_t num_channels_per_link,
    const std::function<bool(std::size_t)>& is_buffer_in_clockwise_direction_fn);

}  // namespace reduce_scatter_detail
}  // namespace ccl
}  // namespace ttnn
