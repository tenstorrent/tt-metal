// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/host/reduce_scatter_common.hpp"
// #include "tt_metal/common/base.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>

namespace ttnn {
namespace ccl {


namespace reduce_scatter_detail {

WorkerTransferInfo::WorkerTransferInfo(
    std::vector<uint32_t> const& pages_per_full_chunk_per_worker, uint32_t num_links, uint32_t num_workers) :
    pages_per_full_chunk_per_worker(pages_per_full_chunk_per_worker),
    num_links(num_links),
    num_workers(num_workers) {}

uint32_t WorkerTransferInfo::get_num_pages_per_full_chunk(WorkerAttributes const& worker_attrs) const {
    std::size_t index = worker_attrs.link * num_workers + worker_attrs.channel;
    TT_ASSERT(index < pages_per_full_chunk_per_worker.size(), "Index {} out of bounds for pages_per_full_chunk_per_worker of size {}", index, pages_per_full_chunk_per_worker.size());
    return pages_per_full_chunk_per_worker.at(index);
}

std::size_t get_global_worker_id(std::size_t link, std::size_t channel_id, std::size_t num_channels_per_link) {
    return link * num_channels_per_link + channel_id;
}
std::size_t get_global_worker_id(WorkerAttributes const& attrs, std::size_t num_channels_per_link) {
    return get_global_worker_id(attrs.link, attrs.channel, num_channels_per_link);
}


std::size_t get_worker_index_in_slice(ttnn::ccl::RingTopology const& tc, std::size_t global_worker_index, std::size_t worker_channel_id, std::size_t num_edm_channels_per_link, std::size_t link) {
    std::size_t worker_tensor_slice_index = !tc.is_linear ?
        global_worker_index :
        (worker_channel_id % (num_edm_channels_per_link / 2)) + ((num_edm_channels_per_link / 2) * link);
    return worker_tensor_slice_index;
}

/*
 * For each live worker on this chip, we specify explicitly details for it:
 * - which direction datapath it is in
 * - its location
 * - its associated worker (if it is a linear topology)
 * - its relative worker index
 */
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
    std::function<bool(std::size_t)> is_buffer_in_clockwise_direction_fn) {

    std::vector<WorkerAttributes> worker_attributes;

    std::size_t workers_per_slice = num_channels_per_link / (topology_config.is_linear ? 2 : 1);

    std::size_t worker_cores_idx = 0;
    std::size_t second_worker_cores_idx = 0;

    bool split_grids = second_worker_cores_list.has_value();
    auto const first_workers_list = split_grids && topology_config.ring_index == 0 ?
        second_worker_cores_list.value():
        worker_cores_list;
    auto const first_send_to_edm_sem_id = split_grids && topology_config.ring_index == 0 ?
        worker_sender_semaphore_id_second_core_range :
        worker_sender_semaphore_id;
    auto const first_read_from_edm_sem_id = split_grids && topology_config.ring_index == 0 ?
        worker_receiver_semaphore_id_second_core_range :
        worker_receiver_semaphore_id;

    std::optional<std::vector<CoreCoord>> second_workers_list =
        !topology_config.is_linear || !split_grids || (split_grids && topology_config.ring_index == 0) ?
            worker_cores_list :
            second_worker_cores_list.value();
    auto const second_send_to_edm_sem_id = !topology_config.is_linear || !split_grids || (split_grids && topology_config.ring_index == 0) ?
        worker_sender_semaphore_id :
        worker_sender_semaphore_id_second_core_range;
    auto const second_read_from_edm_sem_id = !topology_config.is_linear || !split_grids || (split_grids && topology_config.ring_index == 0) ?
        worker_receiver_semaphore_id :
        worker_receiver_semaphore_id_second_core_range;

    for (std::size_t l = 0; l < num_links; l++) {
        for (std::size_t i = 0; i < workers_per_slice; i++) {
            auto worker_id = get_global_worker_id(l, i, num_channels_per_link);
            TT_ASSERT(worker_cores_idx < worker_cores_list.size());

            worker_attributes.push_back(
                {
                    l,
                    i,
                    i,
                    is_buffer_in_clockwise_direction_fn(worker_id) ? Direction::CLOCKWISE : Direction::COUNTER_CLOCKWISE,
                    first_workers_list[worker_cores_idx],
                    first_send_to_edm_sem_id,
                    first_read_from_edm_sem_id
                }
            );
            log_trace(tt::LogOp, "Worker {} direction= {}", i, worker_attributes.back().direction == Direction::CLOCKWISE ? "CLOCKWISE" : "COUNTER-CLOCKWISE");
            worker_cores_idx++;
        }
        if (topology_config.is_linear) {
            auto & second_vec_index = split_grids ? second_worker_cores_idx : worker_cores_idx;
            for (std::size_t i = 0; i < workers_per_slice; i++) {
                TT_ASSERT(second_vec_index < second_workers_list.value().size());
                std::size_t my_logical_index = workers_per_slice + i;
                std::size_t my_idx = worker_attributes.size();
                worker_attributes.push_back(
                    {
                        l,
                        my_logical_index,
                        i,
                        is_buffer_in_clockwise_direction_fn(my_logical_index) ?
                            Direction::CLOCKWISE : Direction::COUNTER_CLOCKWISE,
                        second_workers_list.value()[second_vec_index],
                        second_send_to_edm_sem_id,
                        second_read_from_edm_sem_id
                    }
                );
                log_trace(tt::LogOp, "Worker {} direction= {}", my_logical_index, worker_attributes.back().direction == Direction::CLOCKWISE ? "CLOCKWISE" : "COUNTER-CLOCKWISE");
                std::size_t associated_idx = my_idx - workers_per_slice;
                worker_attributes[my_idx].associated_worker_index = associated_idx;
                worker_attributes[my_idx].associated_worker_core_logical = worker_attributes[associated_idx].location_logical;
                worker_attributes[associated_idx].associated_worker_index = my_idx;
                worker_attributes[associated_idx].associated_worker_core_logical = worker_attributes[my_idx].location_logical;
                second_vec_index++;
            }
        }
    }

    // Validate the worker attributes
    for (const auto &wa : worker_attributes) {
        TT_ASSERT(wa.send_to_edm_semaphore_id.has_value() || wa.receive_from_edm_semaphore_id.has_value(), "Internal error. Incorrectly setup worker attributes for reduce scatter");
    }

    // Log worker attributes
    log_trace(tt::LogOp, "Worker Attributes:");
    for (const auto &wa : worker_attributes) {
        log_trace(tt::LogOp, "\tAttributes: link={}, index={}, core_logical=(x={},y={}), direction={}, associated_core=(x={},y={}), associated_index={}",
            wa.link,
            wa.channel,
            wa.location_logical.x,
            wa.location_logical.y,
            wa.direction == Direction::CLOCKWISE ? "CLOCKWISE": "COUNTER-CLOCKWISE",
            wa.associated_worker_core_logical.has_value() ? std::to_string(wa.associated_worker_core_logical.value().x) : "std::nullopt",
            wa.associated_worker_core_logical.has_value() ? std::to_string(wa.associated_worker_core_logical.value().y) : "std::nullopt",
            wa.associated_worker_index.has_value() ? std::to_string(wa.associated_worker_index.value()) : "std::nullopt"
            );

    }

    TT_ASSERT(!topology_config.is_linear || std::ranges::all_of(worker_attributes, [](auto const& wa) { return wa.associated_worker_index.has_value() && wa.associated_worker_core_logical.has_value(); }));

    return worker_attributes;
}

} // namespace reduce_scatter_detail
} // namespace ccl
} // namespace ttnn
