// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <thread>

namespace tt::tt_metal {
class D2HStreamService;
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {
class InterProcessCounterChannel;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal {

// LayerAckService bridges a metadata-only D2HStreamService to a scheduler-facing
// InterProcessCounterChannel. It owns a single reader thread that blocks on
// D2HStreamService::read_metadata() (one record per completed layer) and injects
// one ack into the counter channel per record.
//
// It does NOT own or construct the D2H service or the counter channel — it only
// holds references and must not outlive either. The runner constructs both (the
// D2H service in metadata-only mode, the channel as the ack producer) and hands
// them here.
class LayerAckService {
public:
    LayerAckService(D2HStreamService& d2h_service, distributed::InterProcessCounterChannel& ack_channel);
    ~LayerAckService();

    LayerAckService(const LayerAckService&) = delete;
    LayerAckService& operator=(const LayerAckService&) = delete;
    LayerAckService(LayerAckService&&) = delete;
    LayerAckService& operator=(LayerAckService&&) = delete;

    // Launch the reader thread. Idempotent: a second call while running is a no-op.
    void start();

    // Signal the reader thread to exit and join it. Idempotent.
    //
    // The reader thread calls read_metadata() only after every socket reports
    // has_data(), so it never parks inside a blocking read. stop() therefore
    // joins promptly (within one poll tick) even when no further record arrives.
    void stop();

private:
    void reader_loop();

    D2HStreamService& d2h_service_;
    distributed::InterProcessCounterChannel& ack_channel_;

    std::thread reader_;
    std::atomic<bool> running_{false};
};

}  // namespace tt::tt_metal
