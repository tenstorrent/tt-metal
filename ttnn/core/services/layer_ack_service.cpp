// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/services/layer_ack_service.hpp"

#include <cstddef>
#include <vector>

#include <tt_stl/span.hpp>

#include <internal/service/inter_process_counter_channel.hpp>

#include "ttnn/services/d2h_socket_service.hpp"

namespace tt::tt_metal {

LayerAckService::LayerAckService(D2HStreamService& d2h_service, distributed::InterProcessCounterChannel& ack_channel) :
    d2h_service_(d2h_service), ack_channel_(ack_channel) {}

LayerAckService::~LayerAckService() { stop(); }

void LayerAckService::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return;  // already running
    }
    reader_ = std::thread([this] { reader_loop(); });
}

void LayerAckService::stop() {
    if (!running_.exchange(false)) {
        return;  // never started, or already stopped
    }
    if (reader_.joinable()) {
        reader_.join();
    }
}

void LayerAckService::reader_loop() {
    // Metadata record size is fixed for the service's lifetime; allocate once.
    std::vector<std::byte> metadata(d2h_service_.metadata_size_bytes());
    while (running_.load(std::memory_order_acquire)) {
        // Blocks until the device sends the next per-layer record.
        d2h_service_.read_metadata(ttsl::Span<std::byte>(metadata.data(), metadata.size()));

        // A record may have completed the same instant stop() flipped the flag.
        if (!running_.load(std::memory_order_acquire)) {
            break;
        }
        ack_channel_.inject(1);
    }
}

}  // namespace tt::tt_metal
