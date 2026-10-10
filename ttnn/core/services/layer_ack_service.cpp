// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/services/layer_ack_service.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

#include <internal/disaggregation/layer_completion_message.hpp>
#include <internal/disaggregation/layer_completion_queue.hpp>
#include "ttnn/services/d2h_socket_service.hpp"

namespace tt::tt_metal {

namespace {
// The D2H ack record is the chunk's PrefillMetadata: {slot_id, actual_start, actual_end}.
constexpr std::size_t kAckIdentityBytes = sizeof(internal::AckIdentity);
}  // namespace

LayerAckService::LayerAckService(
    D2HStreamService& d2h_service,
    const std::string& ring_shm_name,
    uint32_t source_rank,
    uint32_t num_ack_layers,
    uint32_t ack_first_idx,
    uint32_t ack_local_count,
    uint32_t connect_timeout_ms,
    std::vector<uint32_t> ack_layer_ids,
    uint32_t protocol) :
    d2h_service_(d2h_service),
    ring_shm_name_(ring_shm_name),
    source_rank_(source_rank),
    ack_space_{num_ack_layers, ack_first_idx, ack_local_count, std::move(ack_layer_ids)},
    connect_timeout_ms_(connect_timeout_ms),
    protocol_(protocol) {
    TT_FATAL(ack_local_count > 0, "LayerAckService: ack_local_count must be > 0");
    TT_FATAL(
        ack_first_idx + ack_local_count <= num_ack_layers,
        "LayerAckService: this rank's ACK slice [{}, {}) exceeds the global ACK count {}",
        ack_first_idx,
        ack_first_idx + ack_local_count,
        num_ack_layers);
    TT_FATAL(protocol_ == 1 || protocol_ == 2, "LayerAckService: protocol must be 1 or 2, got {}", protocol_);
    TT_FATAL(
        protocol_ != 2 || d2h_service.metadata_size_bytes() >= kAckIdentityBytes,
        "LayerAckService: protocol 2 needs at least {} metadata bytes per D2H record (the chunk's "
        "{{slot_id, actual_start, actual_end}}), but the service is configured for {}",
        kAckIdentityBytes,
        d2h_service.metadata_size_bytes());
    TT_FATAL(
        ack_space_.layer_ids.empty() || ack_space_.layer_ids.size() == ack_local_count,
        "LayerAckService: ack_layer_ids has {} entries but this rank emits {} records per chunk",
        ack_space_.layer_ids.size(),
        ack_local_count);
}

LayerAckService::~LayerAckService() { join(); }

void LayerAckService::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return;
    }
    try {
        if (protocol_ == 2) {
            producer_v2_ = internal::LayerCompletionQueueV2::connect(ring_shm_name_, connect_timeout_ms_);
        } else {
            producer_ = internal::LayerCompletionQueue::connect(ring_shm_name_, connect_timeout_ms_);
        }
        reader_ = std::thread([this] { reader_loop(); });
    } catch (...) {
        running_.store(false, std::memory_order_release);
        throw;
    }
}

void LayerAckService::join() {
    if (!running_.exchange(false)) {
        return;
    }
    if (reader_.joinable()) {
        reader_.join();
    }
}

void LayerAckService::stop() {
    join();
    check();
}

void LayerAckService::check() {
    if (!failed_.load(std::memory_order_acquire)) {
        return;
    }
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error = std::exchange(error_, nullptr);
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

void LayerAckService::fail(std::exception_ptr error) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    if (!error_) {
        error_ = std::move(error);
    }
    failed_.store(true, std::memory_order_release);
}

void LayerAckService::reader_loop() {
    std::vector<std::byte> metadata(d2h_service_.metadata_size_bytes());
    const bool identity_available = metadata.size() >= kAckIdentityBytes;  // v1 may use narrower records
    const auto sockets = d2h_service_.get_sockets();
    internal::AckDerivation derive(ack_space_);

    // Full-ring backpressure: wait, warning on entry and then every 10 s; only stop() ends it.
    const auto push_blocking = [this](auto& queue, const auto& msg) {
        std::optional<std::chrono::steady_clock::time_point> blocked_since;
        auto next_log = std::chrono::steady_clock::time_point{};
        while (!queue->try_push(msg)) {
            if (!running_.load(std::memory_order_acquire)) {
                return false;
            }
            const auto now = std::chrono::steady_clock::now();
            if (!blocked_since) {
                blocked_since = now;
            }
            if (now >= next_log) {
                log_warning(
                    tt::LogOp,
                    "LayerAckService: rank {} ring full for {} s; waiting for the router to drain",
                    source_rank_,
                    std::chrono::duration_cast<std::chrono::seconds>(now - *blocked_since).count());
                next_log = now + std::chrono::seconds(10);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return true;
    };

    while (running_.load(std::memory_order_acquire)) {
        // read_metadata() blocks per socket, so read only once every socket holds a record.
        const bool all_ready =
            std::all_of(sockets.begin(), sockets.end(), [](auto* socket) { return socket->has_data(); });
        if (!all_ready) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        try {
            d2h_service_.read_metadata(ttsl::Span<std::byte>(metadata.data(), metadata.size()));
        } catch (...) {
            fail(std::current_exception());
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        if (failed_.load(std::memory_order_acquire)) {
            continue;  // keep the FIFO drained so the device does not stall; emit nothing more
        }

        std::optional<internal::AckIdentity> identity;
        if (identity_available) {
            internal::AckIdentity words{};
            std::memcpy(words.data(), metadata.data(), kAckIdentityBytes);  // unaligned buffer
            identity = words;
        }
        const auto result = derive.step(identity);
        if (const auto* desync = std::get_if<internal::AckDesync>(&result)) {
            fail(std::make_exception_ptr(std::runtime_error(fmt::format(
                "LayerAckService: rank {} saw a new chunk (slot={} pos=[{},{})) at ACK record {} of {} "
                "(record #{}): a D2H ack record was lost, so every later label would be wrong",
                source_rank_,
                desync->identity[0],
                desync->identity[1],
                desync->identity[2],
                desync->slice_idx,
                ack_space_.ack_local_count,
                desync->record))));
            continue;
        }
        const auto& label = std::get<internal::AckLabel>(result);
        if (protocol_ == 2) {
            const internal::LayerCompletionMessageV2 msg{
                label.seq,
                source_rank_,
                label.chunk,
                label.identity[0],
                label.identity[1],
                label.identity[2],
                label.layer,
                label.layer + 1,
                /*flags=*/0};
            if (!push_blocking(producer_v2_, msg)) {
                return;
            }
        } else {
            const internal::LayerCompletionMessage msg{
                label.seq, source_rank_, label.layer, /*request_id=*/label.chunk, /*reserved=*/0};
            if (!push_blocking(producer_, msg)) {
                return;
            }
        }
    }
}

}  // namespace tt::tt_metal
