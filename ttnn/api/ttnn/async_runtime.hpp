// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/types.hpp"
#include "ttnn/operation.hpp"
#include "types.hpp"

namespace ttnn {

void write_buffer(
    QueueId cq_id,
    Tensor& dst,
    std::vector<std::shared_ptr<void>> src,
    const std::optional<tt::tt_metal::BufferRegion>& region = std::nullopt);

void read_buffer(
    QueueId cq_id,
    Tensor& src,
    std::vector<std::shared_ptr<void>> dst,
    const std::optional<tt::tt_metal::BufferRegion>& region = std::nullopt,
    size_t src_offset = 0,
    bool blocking = true);

void queue_synchronize(tt::tt_metal::distributed::MeshCommandQueue& cq);

void event_synchronize(const tt::tt_metal::distributed::MeshEvent& event);

void wait_for_event(tt::tt_metal::distributed::MeshCommandQueue& cq, const tt::tt_metal::distributed::MeshEvent& event);

// Record an event for device to device synchronization. This event should be passed to `wait_for_event`.
tt::tt_metal::distributed::MeshEvent record_event(tt::tt_metal::distributed::MeshCommandQueue& cq);
// Record an event for device to host synchronization. This event should be passed to `event_synchronize`.
tt::tt_metal::distributed::MeshEvent record_event_to_host(tt::tt_metal::distributed::MeshCommandQueue& cq);
}  // namespace ttnn
