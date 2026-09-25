// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/services/layer_ack_service.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
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
    uint32_t connect_timeout_ms,
    std::vector<uint32_t> ack_layer_ids,
    uint32_t protocol) :
    d2h_service_(d2h_service),
    ring_shm_name_(ring_shm_name),
    source_rank_(source_rank),
    num_layers_(num_layers),
    first_layer_idx_(first_layer_idx),
    local_layers_(local_layers),
    connect_timeout_ms_(connect_timeout_ms),
    ack_layer_ids_(std::move(ack_layer_ids)),
    protocol_(protocol) {
    TT_FATAL(local_layers_ > 0, "LayerAckService: local_layers must be > 0");
    TT_FATAL(
        first_layer_idx_ + local_layers_ <= num_layers_,
        "LayerAckService: this rank's ACK slice [{}, {}) exceeds the global ACK count {}",
        first_layer_idx_,
        first_layer_idx_ + local_layers_,
        num_layers_);
    TT_FATAL(protocol_ == 1 || protocol_ == 2, "LayerAckService: protocol must be 1 or 2, got {}", protocol_);
    // v2's payload IS the record's {slot_id, actual_start, actual_end}, so a narrower record
    // cannot serve it. Reject here rather than on the detached reader thread.
    TT_FATAL(
        protocol_ != 2 || d2h_service.metadata_size_bytes() >= 3 * sizeof(uint32_t),
        "LayerAckService: protocol 2 needs at least {} metadata bytes per D2H record (the chunk's "
        "{{slot_id, actual_start, actual_end}}), but the service is configured for {}",
        3 * sizeof(uint32_t),
        d2h_service.metadata_size_bytes());
    TT_FATAL(
        ack_layer_ids_.empty() || ack_layer_ids_.size() == local_layers_,
        "LayerAckService: ack_layer_ids has {} entries but this rank emits {} records per chunk; the "
        "mapping must cover exactly one global layer per ACK record",
        ack_layer_ids_.size(),
        local_layers_);
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
    if (protocol_ == 2) {
        producer_v2_ = internal::LayerCompletionQueueV2::connect(ring_shm_name_, connect_timeout_ms_);
    } else {
        producer_ = internal::LayerCompletionQueue::connect(ring_shm_name_, connect_timeout_ms_);
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

// The D2H ack record is the chunk's PrefillMetadata: three uint32 words
// {slot_id, actual_start, actual_end}, forwarded verbatim by the device-side ack op.
namespace {
constexpr std::size_t kAckIdentityWords = 3;
constexpr std::size_t kAckIdentityBytes = kAckIdentityWords * sizeof(uint32_t);
// Cap the desync warnings: one lost record desyncs every completion after it, so an
// unbounded log would drown the run it is trying to explain.
constexpr uint64_t kMaxDesyncReports = 8;
}  // namespace

void LayerAckService::reader_loop() {
    // Metadata record size is fixed for the service's lifetime; allocate once.
    std::vector<std::byte> metadata(d2h_service_.metadata_size_bytes());
    // A service configured with a narrower record cannot carry the chunk identity. v1 still
    // works (it only needs the count); v2 is rejected in the ctor, where it is still catchable.
    const bool identity_available = metadata.size() >= kAckIdentityBytes;

    const auto sockets = d2h_service_.get_sockets();
    TT_FATAL(!sockets.empty(), "LayerAckService: metadata-only D2HStreamService exposes no sockets");

    // Full-ring backpressure: wait rather than drop, but stay responsive to stop().
    const auto push_blocking = [this](auto& queue, const auto& msg) {
        while (!queue->try_push(msg)) {
            if (!running_.load(std::memory_order_acquire)) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return true;
    };

    while (running_.load(std::memory_order_acquire)) {
        // read_metadata() reads one page from *every* socket, and each of those reads blocks
        // uninterruptibly until its data arrives. Only proceed once every socket has a record
        // ready, so the read below never blocks and stop() is observed promptly.
        const bool all_ready =
            std::all_of(sockets.begin(), sockets.end(), [](auto* socket) { return socket->has_data(); });
        if (!all_ready) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        d2h_service_.read_metadata(ttsl::Span<std::byte>(metadata.data(), metadata.size()));

        // The record's own identity. memcpy (not a reinterpret_cast) — the buffer has no
        // alignment guarantee and the words are little-endian uint32 on both ends.
        std::array<uint32_t, kAckIdentityWords> words{};
        if (identity_available) {
            std::memcpy(words.data(), metadata.data(), kAckIdentityBytes);
        }
        const uint32_t slot_id = words[0];
        const uint32_t pos_start = words[1];
        const uint32_t pos_end = words[2];

        // Derive this rank's k-th completion. ack_idx is ACK space (dense, what the reorder
        // buffer sequences on); layer is the GLOBAL layer the record attests to, which is what
        // the payload must carry so the router can address the KV stage.
        const uint64_t k = record_count_++;
        const uint32_t slice_idx = static_cast<uint32_t>(k % local_layers_);
        const uint32_t chunk = static_cast<uint32_t>(k / local_layers_);
        const uint32_t ack_idx = first_layer_idx_ + slice_idx;
        const uint64_t seq = static_cast<uint64_t>(chunk) * num_layers_ + ack_idx;
        const uint32_t layer = ack_layer_ids_.empty() ? ack_idx : ack_layer_ids_[slice_idx];

        // Drop detection. Every record of one chunk carries the same identity, so a chunk
        // boundary must land on slice_idx == 0. Landing anywhere else means records were lost,
        // and from there the counter mislabels the chunk and layer of everything that follows
        // (the failure this service used to have no way to see).
        if (identity_available) {
            const bool identity_changed = have_prev_identity_ && (slot_id != prev_slot_id_ ||
                                                                  pos_start != prev_pos_start_ ||
                                                                  pos_end != prev_pos_end_);
            if (identity_changed && slice_idx != 0 && desync_count_++ < kMaxDesyncReports) {
                log_warning(
                    tt::LogOp,
                    "LayerAckService: rank {} saw a new chunk (slot={} pos=[{},{})) at ACK record {} of {} "
                    "-- records were dropped, so chunk/layer labelling is now wrong for the rest of the run",
                    source_rank_,
                    slot_id,
                    pos_start,
                    pos_end,
                    slice_idx,
                    local_layers_);
            }
            prev_slot_id_ = slot_id;
            prev_pos_start_ = pos_start;
            prev_pos_end_ = pos_end;
            have_prev_identity_ = true;
        }

        if (protocol_ == 2) {
            // Self-describing: the chunk identity comes from the record, not from the counter.
            // TODO(#54632): layer_start/layer_end is the single acking layer. On a hybrid stack
            // that is sparse in global layer space, so a consumer accounting coverage against the
            // global layer count never sees a request tile. Same open span question as the host
            // sink (layer_completion_sink.py); resolve both together. Options and the dense-ack
            // proposal that would remove the question entirely:
            // models/demos/common/prefill/docs/LAYER_COMPLETION_OPENS.md
            const internal::LayerCompletionMessageV2 msg{
                seq, source_rank_, chunk, slot_id, pos_start, pos_end, layer, layer + 1, /*flags=*/0};
            if (!push_blocking(producer_v2_, msg)) {
                return;
            }
        } else {
            // reserved stays 0: 0xFFFFFFFF is the router's end-of-stream sentinel — never a real completion.
            const internal::LayerCompletionMessage msg{seq, source_rank_, layer, /*request_id=*/chunk, /*reserved=*/0};
            if (!push_blocking(producer_, msg)) {
                return;
            }
        }
    }
}

}  // namespace tt::tt_metal
