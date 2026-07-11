// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/services/layer_ack_service.hpp"

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

#include <tt_stl/span.hpp>

#include <internal/disaggregation/layer_completion_message.hpp>
#include <internal/disaggregation/layer_completion_queue.hpp>

#include "ttnn/services/d2h_socket_service.hpp"

namespace tt::tt_metal {

LayerAckService::LayerAckService(
    D2HStreamService& d2h_service,
    const std::string& ring_shm_name,
    uint32_t source_rank,
    uint32_t num_layers,
    uint32_t first_layer_idx,
    uint32_t local_layers,
    uint32_t connect_timeout_ms) :
    d2h_service_(d2h_service),
    ring_shm_name_(ring_shm_name),
    source_rank_(source_rank),
    num_layers_(num_layers),
    first_layer_idx_(first_layer_idx),
    local_layers_(local_layers),
    connect_timeout_ms_(connect_timeout_ms) {
    TT_FATAL(local_layers_ > 0, "LayerAckService: local_layers must be > 0");
    TT_FATAL(
        first_layer_idx_ + local_layers_ <= num_layers_,
        "LayerAckService: this rank's slice [{}, {}) exceeds the global layer count {}",
        first_layer_idx_,
        first_layer_idx_ + local_layers_,
        num_layers_);
}

LayerAckService::~LayerAckService() { stop(); }

void LayerAckService::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return;  // already running
    }
    // Connect to the router-owned ring here (not in the ctor) so the router — which
    // creates the ring — is guaranteed constructed first. connect() polls up to
    // connect_timeout_ms_ to tolerate a small construction-order gap.
    producer_ = internal::LayerCompletionQueue::connect(ring_shm_name_, connect_timeout_ms_);
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
        // Drain one record per completed layer. The bytes are unused — this service only
        // counts completions — but the FIFO must still be drained or the device stalls.
        d2h_service_.read_metadata(ttsl::Span<std::byte>(metadata.data(), metadata.size()));

        // Derive a globally-dense ordering key for the k-th completion on this rank.
        const uint64_t k = record_count_++;
        const uint32_t chunk = static_cast<uint32_t>(k / local_layers_);
        const uint32_t layer = first_layer_idx_ + static_cast<uint32_t>(k % local_layers_);  // global layer
        const uint64_t seq = static_cast<uint64_t>(chunk) * num_layers_ + layer;

        // reserved stays 0: 0xFFFFFFFF is the router's end-of-stream sentinel — never a real completion.
        const internal::LayerCompletionMessage msg{seq, source_rank_, layer, /*request_id=*/chunk, /*reserved=*/0};

        // Full-ring backpressure: spin rather than drop, but stay responsive to stop().
        while (!producer_->try_push(msg)) {
            if (!running_.load(std::memory_order_acquire)) {
                return;
            }
            std::this_thread::yield();
        }
    }
}

}  // namespace tt::tt_metal
