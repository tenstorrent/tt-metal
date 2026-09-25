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
struct LayerCompletionMessage;    // fwd — defined in internal/disaggregation/layer_completion_message.hpp
struct LayerCompletionMessageV2;  // fwd — same header
template <typename MsgT>
class LayerCompletionQueueT;  // fwd — defined in internal/disaggregation/layer_completion_queue.hpp
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
// THE RECORD CARRIES IDENTITY. Each D2H record is the chunk's PrefillMetadata —
// three uint32 words {slot_id, actual_start, actual_end} — forwarded verbatim by the
// device-side ack op (see zero_pad_and_ack in models/.../kv_ack.py). This service used
// to read and DISCARD those bytes and reconstruct everything from a counter; it now
// reads them, emits them (v2), and uses them to detect a dropped record.
//
// ACK SPACE vs LAYER SPACE. `num_layers`, `first_layer_idx` and `local_layers` count the
// records this rank actually EMITS, not the layers it holds: a hybrid stack acks only on
// KV-writing layers (`block.py` gates the ack on `attention.writes_kv`), so a 24-layer
// Kimi-K3 rank emits 6. The reorder buffer needs that dense, so seq is derived in ACK
// space. But `layer_idx` on the wire must be the GLOBAL layer — the router addresses the
// KV stage with it, and the host-callback transport puts the global index in that same
// field. `ack_layer_ids` supplies the mapping; without it the two transports disagree on
// what `layer_idx` means for every hybrid model.
//
// seq derivation (see reader_loop): the k-th completion on THIS rank maps to
//   chunk   = k / local_layers                     (which pipelined-prefill chunk/request)
//   ack_idx = first_layer_idx + k % local_layers   (ACK space — dense, for the reorder buffer)
//   seq     = chunk * num_layers + ack_idx         (globally dense across all ranks)
//   layer   = ack_layer_ids[k % local_layers]      (GLOBAL layer index; identity when empty)
// so the router's reorder buffer sequences completions from every rank into one
// contiguous 0,1,2,… stream while the payload stays addressable.
class LayerAckService {
public:
    LayerAckService(
        D2HStreamService& d2h_service,
        const std::string& ring_shm_name,  // router-owned ring; this service connects
        uint32_t source_rank,
        uint32_t num_layers,       // GLOBAL total layer count (seq stride)
        uint32_t first_layer_idx,  // this rank's slice offset into the global range
        uint32_t local_layers,     // ACK records this rank emits per chunk
        uint32_t connect_timeout_ms = 30'000,
        // GLOBAL layer index of each of this rank's ACK records, in emission order
        // (ADAPTER.kv_slot_layer_ids restricted to this rank's slice). Empty = dense
        // model, where ack index == global layer and no mapping is needed. When
        // non-empty it must have exactly `local_layers` entries.
        std::vector<uint32_t> ack_layer_ids = {},
        // 1 = v1 count protocol (LayerCompletionMessage). 2 = v2 structured protocol
        // (LayerCompletionMessageV2), which carries the record's real slot/position
        // identity instead of discarding it. Must match the router's protocol.
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
    // Exactly one is connected in start(), per protocol_.
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

    // Previous record's chunk identity, for drop detection (see reader_loop). A chunk
    // boundary that does not land on ack_idx == 0 means records were lost, which silently
    // relabels every completion after it.
    bool have_prev_identity_ = false;
    uint32_t prev_slot_id_ = 0;
    uint32_t prev_pos_start_ = 0;
    uint32_t prev_pos_end_ = 0;
    uint64_t desync_count_ = 0;

    std::thread reader_;
    std::atomic<bool> running_{false};
};

}  // namespace tt::tt_metal
