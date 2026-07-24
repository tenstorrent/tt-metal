// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/services/layer_ack_service.hpp"

#include <algorithm>
#include <cstddef>
#include <thread>
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

    const auto sockets = d2h_service_.get_sockets();
    TT_FATAL(!sockets.empty(), "LayerAckService: metadata-only D2HStreamService exposes no sockets");
    while (running_.load(std::memory_order_acquire)) {
        // read_metadata() reads one page from *every* socket, and each of those reads blocks
        // uninterruptibly until its data arrives. Only proceed once every socket has a record
        // ready, so the read below never blocks and stop() is observed promptly.
        const bool all_ready =
            std::all_of(sockets.begin(), sockets.end(), [](auto* socket) { return socket->has_data(); });
        if (!all_ready) {
            std::this_thread::yield();
            continue;
        }
        d2h_service_.read_metadata(ttsl::Span<std::byte>(metadata.data(), metadata.size()));
        ack_channel_.inject(1);
    }
}

}  // namespace tt::tt_metal
