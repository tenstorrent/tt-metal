// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-side transport for D2DStreamService's host mode. A sender pump drains each transfer from the sender
// mesh (D2HStreamService) and ships it to the peer host; a receiver pump takes it from the peer and feeds
// the receiver mesh (H2DStreamService). A transfer is an ordered list of host buffers (every coord's shard,
// then the metadata blob) that both sides lay out identically. There are no acknowledgements: flow control
// is back-pressure. The sender's MPI send completes only once the receiver pump posted its receive, and it
// does that only after the receiver mesh took the previous transfer. The device side is injected as
// callbacks, so everything here runs without a device. The loopback channel holds one transfer at most, for the
// same effect. Stopping a pump while an MPI transfer is in flight needs the data path quiesced.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <tt-metalium/distributed_context.hpp>

namespace ttnn::d2d_host {

// A contiguous host buffer that is part of a transfer.
struct Part {
    std::byte* data = nullptr;
    size_t bytes = 0;
};

// One direction of data, sender -> receiver. Every call blocks until it completes or `stop` is set; it
// returns false when stopped. A transfer is its sequence number followed by its parts, in order.
class Channel {
public:
    virtual ~Channel() = default;
    virtual bool send_transfer(uint32_t seq, const std::vector<Part>& parts, const std::atomic<bool>& stop) = 0;
    virtual bool recv_transfer(uint32_t& seq, const std::vector<Part>& parts, const std::atomic<bool>& stop) = 0;
};

// Both endpoints in one process, e.g. create_pair in host mode: transfers go through a queue.
// Returns {sender endpoint, receiver endpoint}.
std::pair<std::unique_ptr<Channel>, std::unique_ptr<Channel>> make_loopback_channels();

// Transfers over MPI point-to-point to/from `peer` in `ctx`, on message tag `tag`. One service owns its tag;
// MPI orders messages per (source, destination, tag).
std::unique_ptr<Channel> make_mpi_channel(
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> ctx,
    tt::tt_metal::distributed::multihost::Rank peer,
    int tag);

// Cumulative and last-transfer timings, in nanoseconds.
struct PumpStats {
    uint64_t transfers = 0;
    uint64_t bytes = 0;
    // Sender: wait for the device to have data, device -> host read, host -> peer send. Receiver: wait for and
    // receive from the peer, host -> device write.
    uint64_t wait_ns = 0;
    uint64_t device_ns = 0;
    uint64_t channel_ns = 0;
    uint64_t last_wait_ns = 0;
    uint64_t last_device_ns = 0;
    uint64_t last_channel_ns = 0;
};

struct SenderOps {
    // The buffers `read` fills, in transfer order. Fixed for the pump's life.
    std::vector<Part> parts;
    // True once the producer workers have made the next transfer available. Polled while idle, so a
    // stop request is honoured; `read` may block until the device has streamed the data out.
    std::function<bool()> ready;
    // Drain one transfer from the sender device into `parts`.
    std::function<void()> read;
};

struct ReceiverOps {
    // The buffers the peer's transfer lands in, in transfer order. Fixed for the pump's life.
    std::vector<Part> parts;
    // Feed `parts` to the receiver device. Blocks while the device is still busy with the previous transfer.
    std::function<void()> write;
};

// A pump owns one thread. Destruction stops it and joins; an exception thrown by a callback or the channel
// is kept and rethrown by rethrow_if_failed().
class PumpBase {
public:
    virtual ~PumpBase();
    PumpBase(const PumpBase&) = delete;
    PumpBase& operator=(const PumpBase&) = delete;

    PumpStats stats() const;
    bool failed() const;
    void rethrow_if_failed() const;
    // Stops and joins the thread. Idempotent.
    void stop();

protected:
    PumpBase() = default;
    void start(std::function<void()> body);
    void record_failure(std::exception_ptr e);

    std::atomic<bool> stop_{false};
    mutable std::mutex mutex_;
    PumpStats stats_;
    std::exception_ptr error_;
    std::thread thread_;
};

class SenderPump : public PumpBase {
public:
    SenderPump(SenderOps ops, std::shared_ptr<Channel> channel);
    ~SenderPump() override;

private:
    void run();
    SenderOps ops_;
    std::shared_ptr<Channel> channel_;
};

class ReceiverPump : public PumpBase {
public:
    ReceiverPump(ReceiverOps ops, std::shared_ptr<Channel> channel);
    ~ReceiverPump() override;

private:
    void run();
    ReceiverOps ops_;
    std::shared_ptr<Channel> channel_;
};

}  // namespace ttnn::d2d_host
