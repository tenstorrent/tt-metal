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
#include <vector>

#include <internal/disaggregation/layer_ack_decoder.hpp>

namespace tt::tt_metal {
class D2HStreamService;
}  // namespace tt::tt_metal

namespace tt::tt_metal::internal {
class LayerCompletionQueue;
}  // namespace tt::tt_metal::internal

namespace tt::tt_metal {

// Bridges a metadata-only D2HStreamService to the host-local layer-completion ring: a reader thread drains the
// D2H records (one per acked layer) and pushes one message per record, decoded by LayerAckDecoder, into the ring
// it connects to. Holds a reference to the D2H service and must not outlive it.
// A lost record or a reader-thread failure is fatal for the run: the reader keeps draining the FIFO so the
// device never stalls but emits nothing more; check() and stop() rethrow on the caller's thread.
class LayerAckService {
public:
    LayerAckService(
        D2HStreamService& d2h_service,
        const std::string& ring_shm_name,
        uint32_t first_record,  // index of this rank's first record in a chunk's record sequence
        uint32_t record_count,  // records this rank emits per chunk
        uint32_t connect_timeout_ms = 30'000,
        // Global layer of each local record, in emission order (empty: dense, record index == layer).
        std::vector<uint32_t> layer_ids = {});
    ~LayerAckService();

    LayerAckService(const LayerAckService&) = delete;
    LayerAckService& operator=(const LayerAckService&) = delete;
    LayerAckService(LayerAckService&&) = delete;
    LayerAckService& operator=(LayerAckService&&) = delete;

    // Connect to the ring (polls up to the timeout) and start the reader. Idempotent.
    void start();
    // Stop and join the reader, then rethrow a stored failure once. Idempotent; call before the router stops.
    void stop();
    // Rethrow a stored failure without stopping.
    void check();

private:
    void reader_loop();
    void fail(std::exception_ptr error);
    void join();

    D2HStreamService& d2h_service_;
    std::unique_ptr<internal::LayerCompletionQueue> producer_;  // connected in start()

    std::string ring_shm_name_;
    internal::LayerAckLayout layout_;
    uint32_t connect_timeout_ms_;

    std::thread reader_;
    std::atomic<bool> running_{false};
    std::atomic<bool> failed_{false};
    std::mutex error_mutex_;
    std::exception_ptr error_;
};

}  // namespace tt::tt_metal
