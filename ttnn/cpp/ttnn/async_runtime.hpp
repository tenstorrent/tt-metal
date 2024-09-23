// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/run_operation.hpp"
#include "types.hpp"

namespace ttnn {
    using DeviceBuffer = std::shared_ptr<Buffer>;
    using queue_id = uint8_t;

    DeviceBuffer allocate_buffer_on_device(size_t buffer_size_bytes, types::Device* device, const Shape& shape, DataType data_type, Layout layout, const MemoryConfig& memory_config, const std::optional<ShardSpecBuffer>& shard_spec = std::nullopt, const std::optional<Tile>& tile = std::nullopt);

    void write_buffer(queue_id cq_id, Tensor& dst, std::vector<std::shared_ptr<void>> src, const std::optional<std::size_t> transfer_size = std::nullopt);

    void read_buffer(queue_id cq_id, Tensor& src, std::vector<std::shared_ptr<void>> dst, const std::optional<std::size_t> transfer_size = std::nullopt, size_t src_offset = 0, bool blocking = true);

    void queue_synchronize(CommandQueue& cq);

    void event_synchronize(std::shared_ptr<Event> event);

    bool event_query(std::shared_ptr<Event> event);

    void wait_for_event(CommandQueue& cq, std::shared_ptr<Event> event);

    void record_event(CommandQueue& cq, std::shared_ptr<Event> event);
}
