// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace tt::tt_metal {
class D2HStreamService;
}  // namespace tt::tt_metal

namespace tt::tt_metal::internal {
struct LayerCompletionMessage;
struct LayerCompletionMessageV2;
template <typename MsgT>
class LayerCompletionQueueT;
using LayerCompletionQueue = LayerCompletionQueueT<LayerCompletionMessage>;
using LayerCompletionQueueV2 = LayerCompletionQueueT<LayerCompletionMessageV2>;
}  // namespace tt::tt_metal::internal

namespace tt::tt_metal {

// LayerAckService bridges a metadata-only D2HStreamService to the pipelined-prefill
// layer-completion ring (LayerCompletionQueue). It owns a single reader thread that
// drains D2HStreamService::read_metadata() (one record per completed layer) and, for
// each record, derives a globally-dense ordering key `seq` and pushes one
// LayerCompletionMessage into the ring.
//
// The service is a PURE PRODUCER: it never touches the scheduler-facing counter
// channel. The LayerCompletionRouter owns the ring (create) and the counter channel,
// aggregates completions across ranks (MPI when world_size > 1, local-only otherwise),
// and injects into the scheduler channel. One code path serves single- and multi-host:
// the router is always present, so this service always just pushes.
//
// It does NOT own or construct the D2H service — it holds a reference and must not
// outlive it. It CONNECTS to the router-owned ring by name (so no C++ queue object
// crosses into Python), and owns that producer handle for its lifetime.
//
// seq derivation (see reader_loop): the k-th completion on THIS rank maps to
//   chunk = k / local_layers            (which pipelined-prefill chunk/request)
//   layer = first_layer_idx + k % local_layers   (GLOBAL layer index)
//   seq   = chunk * num_layers + layer  (globally dense across all ranks)
// so the router's reorder buffer sequences completions from every rank into one
// contiguous 0,1,2,… stream.
class LayerAckService {
public:
    LayerAckService(
        D2HStreamService& d2h_service,
        const std::string& ring_shm_name,  // router-owned ring; this service connects
        uint32_t source_rank,
        uint32_t num_layers,       // GLOBAL total layer count (seq stride)
        uint32_t first_layer_idx,  // this rank's slice offset into the global range
        uint32_t local_layers,
        uint32_t connect_timeout_ms = 30'000,
        std::vector<uint32_t> ack_layer_ids = {},
        uint32_t protocol = 1);
    ~LayerAckService();

    LayerAckService(const LayerAckService&) = delete;
    LayerAckService& operator=(const LayerAckService&) = delete;
    LayerAckService(LayerAckService&&) = delete;
    LayerAckService& operator=(LayerAckService&&) = delete;

    // Connect to the router-owned ring, then launch the reader thread. Idempotent:
    // a second call while running is a no-op. The router (ring owner) must be
    // constructed first; connect() polls up to connect_timeout_ms to tolerate a gap.
    void start();

    // Signal the reader thread to exit and join it. Idempotent. Must be called
    // before the router stops (the router drains the ring + fires the MPI sentinel
    // at teardown; stopping the producer first ensures the last records land).
    //
    // The reader thread calls read_metadata() only after every socket reports
    // has_data(), so it never parks inside a blocking read. stop() therefore
    // joins promptly (within one poll tick) even when no further record arrives.
    void stop();

private:
    void reader_loop();

    D2HStreamService& d2h_service_;
    std::unique_ptr<internal::LayerCompletionQueue> producer_;
    std::unique_ptr<internal::LayerCompletionQueueV2> producer_v2_;

    std::string ring_shm_name_;
    uint32_t source_rank_;
    uint32_t num_layers_;
    uint32_t first_layer_idx_;
    uint32_t local_layers_;
    uint32_t connect_timeout_ms_;
    std::vector<uint32_t> ack_layer_ids_;
    uint32_t protocol_;

    uint64_t record_count_ = 0;  // per-rank monotonic completion counter (k in seq derivation)

    bool have_prev_identity_ = false;
    uint32_t prev_slot_id_ = 0;
    uint32_t prev_pos_start_ = 0;
    uint32_t prev_pos_end_ = 0;
    uint64_t desync_count_ = 0;

    std::thread reader_;
    std::atomic<bool> running_{false};
};

}  // namespace tt::tt_metal
