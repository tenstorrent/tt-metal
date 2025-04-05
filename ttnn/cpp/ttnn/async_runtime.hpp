// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <memory>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "types.hpp"

namespace tt {
namespace tt_metal {
class CommandQueue;
class Event;
}  // namespace tt_metal
}  // namespace tt

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

void queue_synchronize(tt::tt_metal::CommandQueue& cq);

void event_synchronize(const std::shared_ptr<tt::tt_metal::Event>& event);

bool event_query(const std::shared_ptr<tt::tt_metal::Event>& event);

void wait_for_event(tt::tt_metal::CommandQueue& cq, const std::shared_ptr<tt::tt_metal::Event>& event);

void record_event(tt::tt_metal::CommandQueue& cq, const std::shared_ptr<tt::tt_metal::Event>& event);
}  // namespace ttnn
