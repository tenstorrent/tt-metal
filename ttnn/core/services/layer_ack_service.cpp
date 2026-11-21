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
#include <variant>
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
    uint32_t first_record,
    uint32_t record_count,
    uint32_t connect_timeout_ms,
    std::vector<uint32_t> layer_ids) :
    d2h_service_(d2h_service),
    ring_shm_name_(ring_shm_name),
    layout_{first_record, record_count, std::move(layer_ids)},
    connect_timeout_ms_(connect_timeout_ms) {
    TT_FATAL(record_count > 0, "LayerAckService: record_count must be > 0");
    TT_FATAL(
        d2h_service.metadata_size_bytes() >= kAckIdentityBytes,
        "LayerAckService: needs at least {} metadata bytes per D2H record (the chunk's "
        "{{slot_id, actual_start, actual_end}}), but the service is configured for {}",
        kAckIdentityBytes,
        d2h_service.metadata_size_bytes());
    TT_FATAL(
        layout_.layer_ids.empty() || layout_.layer_ids.size() == record_count,
        "LayerAckService: layer_ids has {} entries but this rank emits {} records per chunk",
        layout_.layer_ids.size(),
        record_count);
}

LayerAckService::~LayerAckService() { join(); }

void LayerAckService::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return;
    }
    try {
        producer_ = internal::LayerCompletionQueue::connect(ring_shm_name_, connect_timeout_ms_);
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
    const auto sockets = d2h_service_.get_sockets();
    internal::LayerAckDecoder decoder(layout_);

    // Full-ring backpressure: wait, warning on entry and then every 10 s; only stop() ends it.
    const auto push_blocking = [this](const internal::LayerCompletionMessage& msg) {
        std::optional<std::chrono::steady_clock::time_point> blocked_since;
        auto next_log = std::chrono::steady_clock::time_point{};
        while (!producer_->try_push(msg)) {
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
                    "LayerAckService: ring {} full for {} s; waiting for the router to drain",
                    ring_shm_name_,
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

        internal::AckIdentity identity{};
        std::memcpy(&identity, metadata.data(), kAckIdentityBytes);  // unaligned buffer
        const auto result = decoder.decode(identity);
        if (const auto* lost = std::get_if<internal::LostAckRecord>(&result)) {
            fail(std::make_exception_ptr(std::runtime_error(fmt::format(
                "LayerAckService: ring {} saw a new chunk (slot={} pos=[{},{})) at ACK record {} of {} "
                "(record #{}): a D2H ack record was lost, so every later label would be wrong",
                ring_shm_name_,
                lost->identity.slot_id,
                lost->identity.pos_start,
                lost->identity.pos_end,
                lost->slice_idx,
                layout_.record_count,
                lost->record))));
            continue;
        }
        const auto& ack = std::get<internal::DecodedAck>(result);
        const internal::LayerCompletionMessage msg{
            ack.identity.slot_id,
            ack.identity.pos_start,
            ack.identity.pos_end,
            ack.layer,
            ack.layer + 1,
            /*flags=*/0,
            internal::layer_completion_host_timestamp_ns()};
        if (!push_blocking(msg)) {
            return;
        }
    }
}

}  // namespace tt::tt_metal
