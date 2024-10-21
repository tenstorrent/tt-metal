// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/run_operation.hpp"
#include "types.hpp"

namespace ttnn {
    using queue_id = uint8_t;

    void write_buffer(queue_id cq_id, Tensor& dst, std::vector<std::shared_ptr<void>> src, const std::optional<std::size_t> transfer_size = std::nullopt);

    void read_buffer(queue_id cq_id, Tensor& src, std::vector<std::shared_ptr<void>> dst, const std::optional<std::size_t> transfer_size = std::nullopt, size_t src_offset = 0, bool blocking = true);

    void queue_synchronize(CommandQueue& cq);

    void event_synchronize(std::shared_ptr<Event> event);

    bool event_query(std::shared_ptr<Event> event);

    void wait_for_event(CommandQueue& cq, std::shared_ptr<Event> event);

    void record_event(CommandQueue& cq, std::shared_ptr<Event> event);
}
