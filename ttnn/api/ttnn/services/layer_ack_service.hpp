// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include <internal/disaggregation/layer_ack_derivation.hpp>

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

// Bridges a metadata-only D2HStreamService to the host-local layer-completion ring: one reader
// thread drains the D2H records (one per acked layer) and pushes one message per record into the
// router-owned ring it connects to. Labels come from AckDerivation (ACK space; see
// layer_ack_derivation.hpp); under protocol 2 the record's {slot_id, actual_start, actual_end}
// travel in the message. Holds a reference to the D2H service and must not outlive it.
//
// A lost record (identity change mid-chunk) or a failure on the reader thread is fatal for the
// run: the reader keeps draining the D2H FIFO so the device never stalls, but emits nothing more,
// and check() / stop() rethrow the error on the caller's thread.
class LayerAckService {
public:
    LayerAckService(
        D2HStreamService& d2h_service,
        const std::string& ring_shm_name,
        uint32_t source_rank,
        uint32_t num_ack_layers,   // records one chunk produces across all ranks (seq stride)
        uint32_t ack_first_idx,    // this rank's first ACK index
        uint32_t ack_local_count,  // records this rank emits per chunk
        uint32_t connect_timeout_ms = 30'000,
        // Global layer of each local record, in emission order (empty: dense, ack index == layer).
        std::vector<uint32_t> ack_layer_ids = {},
        uint32_t protocol = 1);  // 1: counted (LayerCompletionMessage); 2: structured (V2)
    ~LayerAckService();

    LayerAckService(const LayerAckService&) = delete;
    LayerAckService& operator=(const LayerAckService&) = delete;
    LayerAckService(LayerAckService&&) = delete;
    LayerAckService& operator=(LayerAckService&&) = delete;

    // Connect to the router-owned ring (the router must exist; connect polls up to the timeout),
    // then launch the reader thread. Idempotent.
    void start();
    // Stop and join the reader thread, then rethrow a stored failure once. Idempotent. Call it
    // before the router stops.
    void stop();
    // Rethrow a stored failure (desync or reader-thread exception) without stopping.
    void check();

private:
    void reader_loop();
    void fail(std::exception_ptr error);
    void join();

    D2HStreamService& d2h_service_;
    std::variant<
        std::monostate,
        std::unique_ptr<internal::LayerCompletionQueue>,
        std::unique_ptr<internal::LayerCompletionQueueV2>>
        producer_;  // connected in start(), per protocol_

    std::string ring_shm_name_;
    uint32_t source_rank_;
    internal::AckSpace ack_space_;
    uint32_t connect_timeout_ms_;
    uint32_t protocol_;

    std::thread reader_;
    std::atomic<bool> running_{false};
    std::atomic<bool> failed_{false};
    std::mutex error_mutex_;
    std::exception_ptr error_;
};

}  // namespace tt::tt_metal
