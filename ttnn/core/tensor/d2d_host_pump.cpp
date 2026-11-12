// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/d2d_host_pump.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

namespace ttnn::d2d_host {

namespace {

using Clock = std::chrono::steady_clock;
namespace multihost = tt::tt_metal::distributed::multihost;

uint64_t ns_since(Clock::time_point start) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

// Idle polling backs off from a yield to a short sleep, so a parked service costs little CPU while a busy
// one still reacts within microseconds.
class Backoff {
public:
    // `may_sleep` is false while a transfer is moving: MPI progresses only inside the polling calls, so sleeping
    // between polls starves it.
    void pause(bool may_sleep) {
        if (!may_sleep || spins_ < kSpinsBeforeSleep) {
            ++spins_;
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(kSleepMicros));
        }
    }
    void reset() { spins_ = 0; }

private:
    static constexpr uint32_t kSpinsBeforeSleep = 2000;
    static constexpr int64_t kSleepMicros = 50;
    uint32_t spins_ = 0;
};

template <typename Pred>
bool wait_until(Pred&& pred, const std::atomic<bool>& stop) {
    Backoff backoff;
    while (!pred()) {
        if (stop.load(std::memory_order_acquire)) {
            return false;
        }
        backoff.pause(/*may_sleep=*/true);
    }
    return true;
}

struct LoopbackTransfer {
    uint32_t seq = 0;
    std::vector<std::vector<std::byte>> parts;
};

struct LoopbackState {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<LoopbackTransfer> transfers;
};

class LoopbackChannel : public Channel {
public:
    explicit LoopbackChannel(std::shared_ptr<LoopbackState> state) : state_(std::move(state)) {}

    bool send_transfer(uint32_t seq, const std::vector<Part>& parts, const std::atomic<bool>& stop) override {
        // One transfer in the queue at most, like an MPI rendezvous send that waits for the receiver.
        std::unique_lock lock(state_->mutex);
        while (!state_->transfers.empty()) {
            if (stop.load(std::memory_order_acquire)) {
                return false;
            }
            state_->cv.wait_for(lock, std::chrono::milliseconds(1));
        }
        LoopbackTransfer t{.seq = seq};
        for (const auto& p : parts) {
            t.parts.emplace_back(p.data, p.data + p.bytes);
        }
        state_->transfers.push_back(std::move(t));
        state_->cv.notify_all();
        return true;
    }

    bool recv_transfer(uint32_t& seq, const std::vector<Part>& parts, const std::atomic<bool>& stop) override {
        std::unique_lock lock(state_->mutex);
        while (state_->transfers.empty()) {
            if (stop.load(std::memory_order_acquire)) {
                return false;
            }
            state_->cv.wait_for(lock, std::chrono::milliseconds(1));
        }
        const LoopbackTransfer& t = state_->transfers.front();
        TT_FATAL(
            t.parts.size() == parts.size(),
            "D2D host loopback: transfer has {} parts, receiver expects {}",
            t.parts.size(),
            parts.size());
        for (size_t i = 0; i < parts.size(); ++i) {
            TT_FATAL(
                t.parts[i].size() == parts[i].bytes,
                "D2D host loopback: part {} is {} bytes, receiver expects {}",
                i,
                t.parts[i].size(),
                parts[i].bytes);
            std::memcpy(parts[i].data, t.parts[i].data(), parts[i].bytes);
        }
        seq = t.seq;
        state_->transfers.pop_front();
        return true;
    }

private:
    std::shared_ptr<LoopbackState> state_;
};

class MpiChannel : public Channel {
public:
    MpiChannel(std::shared_ptr<multihost::DistributedContext> ctx, multihost::Rank peer, int tag) :
        ctx_(std::move(ctx)), peer_(peer), tag_(tag) {}

    bool send_transfer(uint32_t seq, const std::vector<Part>& parts, const std::atomic<bool>& stop) override {
        seq_ = seq;
        std::vector<multihost::RequestPtr> requests;
        requests.reserve(parts.size() + 1);
        for (const auto& p : parts) {
            TT_FATAL(
                p.bytes <= static_cast<size_t>(std::numeric_limits<int>::max()),
                "D2D host mode: a {} byte part exceeds the MPI message size limit",
                p.bytes);
        }
        requests.push_back(ctx_->isend(
            ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&seq_), sizeof(seq_)), peer_, multihost::Tag{tag_}));
        for (const auto& p : parts) {
            requests.push_back(ctx_->isend(ttsl::Span<std::byte>(p.data, p.bytes), peer_, multihost::Tag{tag_}));
        }
        return complete_all(requests, stop, /*idle=*/false);
    }

    bool recv_transfer(uint32_t& seq, const std::vector<Part>& parts, const std::atomic<bool>& stop) override {
        // The sequence number goes first: wait for it alone so an idle receiver posts one small receive, not
        // one per part.
        std::vector<multihost::RequestPtr> head;
        head.push_back(ctx_->irecv(
            ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&seq_), sizeof(seq_)), peer_, multihost::Tag{tag_}));
        if (!complete_all(head, stop, /*idle=*/true)) {
            return false;
        }
        seq = seq_;
        std::vector<multihost::RequestPtr> requests;
        requests.reserve(parts.size());
        for (const auto& p : parts) {
            requests.push_back(ctx_->irecv(ttsl::Span<std::byte>(p.data, p.bytes), peer_, multihost::Tag{tag_}));
        }
        return complete_all(requests, stop, /*idle=*/false);
    }

private:
    // Polls the requests so a stop request is honoured; on stop the unfinished ones are cancelled.
    static bool complete_all(
        std::vector<multihost::RequestPtr>& requests, const std::atomic<bool>& stop, bool idle) {
        Backoff backoff;
        std::vector<bool> done(requests.size(), false);
        size_t remaining = requests.size();
        while (remaining > 0) {
            for (size_t i = 0; i < requests.size(); ++i) {
                if (!done[i] && requests[i]->test().has_value()) {
                    done[i] = true;
                    --remaining;
                }
            }
            if (remaining == 0) {
                break;
            }
            if (stop.load(std::memory_order_acquire)) {
                for (size_t i = 0; i < requests.size(); ++i) {
                    if (!done[i]) {
                        requests[i]->cancel();
                    }
                }
                return false;
            }
            backoff.pause(idle);
        }
        return true;
    }

    std::shared_ptr<multihost::DistributedContext> ctx_;
    multihost::Rank peer_;
    int tag_;
    uint32_t seq_ = 0;
};

}  // namespace

std::pair<std::unique_ptr<Channel>, std::unique_ptr<Channel>> make_loopback_channels() {
    auto state = std::make_shared<LoopbackState>();
    return {std::make_unique<LoopbackChannel>(state), std::make_unique<LoopbackChannel>(state)};
}

std::unique_ptr<Channel> make_mpi_channel(
    std::shared_ptr<multihost::DistributedContext> ctx, multihost::Rank peer, int tag) {
    return std::make_unique<MpiChannel>(std::move(ctx), peer, tag);
}

PumpBase::~PumpBase() { stop(); }

void PumpBase::start(std::function<void()> body) {
    thread_ = std::thread([this, body = std::move(body)]() {
        try {
            body();
        } catch (...) {
            record_failure(std::current_exception());
        }
    });
}

void PumpBase::stop() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void PumpBase::record_failure(std::exception_ptr e) {
    std::lock_guard lock(mutex_);
    if (!error_) {
        error_ = std::move(e);
    }
}

PumpStats PumpBase::stats() const {
    std::lock_guard lock(mutex_);
    return stats_;
}

bool PumpBase::failed() const {
    std::lock_guard lock(mutex_);
    return error_ != nullptr;
}

void PumpBase::rethrow_if_failed() const {
    std::exception_ptr e;
    {
        std::lock_guard lock(mutex_);
        e = error_;
    }
    if (e) {
        std::rethrow_exception(e);
    }
}

SenderPump::SenderPump(SenderOps ops, std::shared_ptr<Channel> channel) :
    ops_(std::move(ops)), channel_(std::move(channel)) {
    TT_FATAL(ops_.ready && ops_.read, "D2D host sender: callbacks must be set");
    start([this] { run(); });
}

SenderPump::~SenderPump() { stop(); }

void SenderPump::run() {
    const uint64_t transfer_bytes = std::accumulate(
        ops_.parts.begin(), ops_.parts.end(), uint64_t{0}, [](uint64_t sum, const Part& p) { return sum + p.bytes; });
    uint32_t seq = 0;
    while (true) {
        const auto wait_start = Clock::now();
        if (!wait_until(ops_.ready, stop_)) {
            return;
        }
        const uint64_t wait_ns = ns_since(wait_start);

        const auto read_start = Clock::now();
        ops_.read();
        const uint64_t device_ns = ns_since(read_start);

        const auto send_start = Clock::now();
        if (!channel_->send_transfer(seq, ops_.parts, stop_)) {
            return;
        }
        const uint64_t channel_ns = ns_since(send_start);
        ++seq;

        std::lock_guard lock(mutex_);
        ++stats_.transfers;
        stats_.bytes += transfer_bytes;
        stats_.wait_ns += wait_ns;
        stats_.device_ns += device_ns;
        stats_.channel_ns += channel_ns;
        stats_.last_wait_ns = wait_ns;
        stats_.last_device_ns = device_ns;
        stats_.last_channel_ns = channel_ns;
    }
}

ReceiverPump::ReceiverPump(ReceiverOps ops, std::shared_ptr<Channel> channel) :
    ops_(std::move(ops)), channel_(std::move(channel)) {
    TT_FATAL(ops_.write, "D2D host receiver: callbacks must be set");
    start([this] { run(); });
}

ReceiverPump::~ReceiverPump() { stop(); }

void ReceiverPump::run() {
    const uint64_t transfer_bytes = std::accumulate(
        ops_.parts.begin(), ops_.parts.end(), uint64_t{0}, [](uint64_t sum, const Part& p) { return sum + p.bytes; });
    uint32_t seq = 0;
    while (true) {
        const auto recv_start = Clock::now();
        uint32_t got = 0;
        if (!channel_->recv_transfer(got, ops_.parts, stop_)) {
            return;
        }
        const uint64_t channel_ns = ns_since(recv_start);
        TT_FATAL(got == seq, "D2D host receiver: transfer {} arrived, expected {} (out of order or lost)", got, seq);

        const auto write_start = Clock::now();
        ops_.write();
        const uint64_t device_ns = ns_since(write_start);
        ++seq;

        std::lock_guard lock(mutex_);
        ++stats_.transfers;
        stats_.bytes += transfer_bytes;
        stats_.device_ns += device_ns;
        stats_.channel_ns += channel_ns;
        stats_.last_device_ns = device_ns;
        stats_.last_channel_ns = channel_ns;
    }
}

}  // namespace ttnn::d2d_host
