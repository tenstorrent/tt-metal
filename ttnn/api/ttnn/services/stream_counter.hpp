// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt_stl/reflection.hpp>

namespace tt::tt_metal {

// A 32-bit L1 word a stream service's kernels or workers bump, and how far it moves per transfer.
struct StreamCounter {
    CoreCoord core;  // logical
    DeviceAddr addr = 0;
    uint32_t step = 0;
};

// How a transfer moves into a device: it lands in the service's FIFO, then the consuming op copies it out.
struct StreamInputCounters {
    StreamCounter arrived;
    StreamCounter consumed;
};

// How a transfer moves out of a device: the producing op writes it, the service hands it to fabric, and the
// next stage's receiver takes it.
struct StreamOutputCounters {
    StreamCounter produced;
    StreamCounter pushed;
    StreamCounter forwarded;
};

// Every mesh coordinate's sender-side service words in row-major order (row * num_cols + col), for an op whose worker
// kernels signal the service directly (outbound_socket_service_sync). The service builds it once, since its cores and
// addresses are fixed for its lifetime, so a launch does not look them up chip by chip.
struct StreamSenderCoordTable {
    uint32_t num_cols = 0;
    std::vector<uint32_t> data_ready_addrs;
    std::vector<CoreCoord> service_cores;     // logical
    std::vector<uint32_t> metadata_addrs;     // empty when the service has no metadata
    std::vector<uint32_t> termination_addrs;  // empty when the service has no control words
    // 1 per coordinate the service covers; empty means every coordinate. A D2D service whose mapper covers a row
    // block of the mesh (a [RING,RING] seam boundary) leaves the other coordinates out, and the op skips them.
    std::vector<uint8_t> participates;
    uint64_t hash = 0;  // of all the above
};

// A service's StreamSenderCoordTable, built on first use, so every service lays it out and hashes it the same way.
class StreamSenderCoordTableCache {
public:
    // fill(row, col, table) appends one coordinate's entries; coordinates are visited row-major.
    template <typename Fill>
    std::shared_ptr<const StreamSenderCoordTable> get(uint32_t num_rows, uint32_t num_cols, const Fill& fill) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (table_ == nullptr) {
            auto table = std::make_shared<StreamSenderCoordTable>();
            table->num_cols = num_cols;
            for (uint32_t row = 0; row < num_rows; ++row) {
                for (uint32_t col = 0; col < num_cols; ++col) {
                    fill(row, col, *table);
                }
            }
            table->hash = ttsl::hash::hash_objects_with_default_seed(
                table->num_cols,
                table->data_ready_addrs,
                table->service_cores,
                table->metadata_addrs,
                table->termination_addrs,
                table->participates);
            table_ = std::move(table);
        }
        return table_;
    }

private:
    std::mutex mutex_;
    std::shared_ptr<const StreamSenderCoordTable> table_;
};

}  // namespace tt::tt_metal
