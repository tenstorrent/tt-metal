// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/up_front_compile.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

#include <tt_stl/assert.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>  // tt::tt_metal::detail::CompileProgram

#include "ttnn/graph/graph_processor.hpp"
#include <tt-metalium/graph_tracking.hpp>  // GraphTracker (for real-alloc hook tweak)

namespace ttnn::up_front_compile {

namespace {
// Per-thread "currently collecting" flag, matching GraphTracker's per-thread
// capture semantics: the collect forward pass runs on one thread, and the
// device-op funnel that consults active() runs on that same thread.
thread_local bool t_collecting = false;
}  // namespace

ProgramCollector& ProgramCollector::instance() {
    static ProgramCollector inst;
    return inst;
}

ProgramCollector* ProgramCollector::active() { return t_collecting ? &instance() : nullptr; }

void ProgramCollector::set_active(bool active) { t_collecting = active; }

void ProgramCollector::collect(std::uint64_t program_hash, tt::tt_metal::distributed::MeshWorkload&& workload) {
    std::lock_guard<std::mutex> lk(mutex_);
    ++total_collected_;
    // hash==0 means the device program cache was disabled, so no real structural
    // hash was computed. Fall back to a unique synthetic key so distinct programs
    // are NOT collapsed into one (we lose dedup, but stay correct). Synthetic keys
    // count down from the top of the range to avoid colliding with real hashes.
    std::uint64_t key = program_hash != 0 ? program_hash : (UINT64_MAX - synthetic_key_++);
    // First entry per key wins; later structural duplicates are dropped.
    programs_.try_emplace(key, std::move(workload));
}

std::size_t ProgramCollector::num_unique() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return programs_.size();
}

std::size_t ProgramCollector::num_collected() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return total_collected_;
}

void ProgramCollector::clear() {
    std::lock_guard<std::mutex> lk(mutex_);
    programs_.clear();
    claimed_.clear();
    total_collected_ = 0;
    synthetic_key_ = 0;
}

std::vector<tt::tt_metal::Program*> ProgramCollector::program_pointers() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<tt::tt_metal::Program*> out;
    for (auto& [hash, workload] : programs_) {
        for (auto& [range, program] : workload.get_programs()) {
            out.push_back(&program);
        }
    }
    return out;
}

std::vector<tt::tt_metal::Program*> ProgramCollector::claim_uncompiled() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<tt::tt_metal::Program*> out;
    for (auto& [hash, workload] : programs_) {
        if (claimed_.insert(hash).second) {  // first time we see this hash -> claim it now
            for (auto& [range, program] : workload.get_programs()) {
                out.push_back(&program);
            }
        }
    }
    return out;
}

void begin_collect(bool clear, bool real_alloc) {
    if (clear) {
        ProgramCollector::instance().clear();
    }
    ProgramCollector::set_active(true);
    // NO_DISPATCH: buffer allocations are mocked (addr 0) and nothing dispatches,
    // so the collect pass uses no real device memory. The funnel's stash + early
    // return (device_operation.hpp) handles program collection.
    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    if (real_alloc) {
        // Real-alloc collect: unblock the allocator so buffers get REAL (deterministic)
        // addresses — so address-baked kernels (e.g. pool reader_indices) and
        // address-branched program selection (e.g. move forward/backward) build the SAME
        // program the real run will, and thus warm. Dispatch/write stay blocked (no compute).
        // Costs real device memory (~the real run's peak) and requires the dealloc lifecycle
        // to run (deallocate is host-side, so it does). See up_front_compile.hpp.
        if (auto hook = tt::tt_metal::GraphTracker::instance().get_hook()) {
            if (auto* ph = dynamic_cast<ttnn::graph::ProcessorHooks*>(hook.get())) {
                ph->set_block_alloc(false);
            }
        }
    }
}

void end_collect() {
    // Discard the JSON graph; we only used NO_DISPATCH for allocation blocking.
    ttnn::graph::GraphProcessor::end_graph_capture();
    ProgramCollector::set_active(false);
}

namespace {
// JIT-compile a batch of programs across up to max_workers threads. Returns the error count.
// Compile-only: builds kernels into the on-disk JIT cache. No command queue / program-cache
// state is touched, so distinct programs compile concurrently and safely; buffer addresses are
// irrelevant to compile. (Phase-0 spike confirmed this is also safe to run concurrently with a
// collect pass building OTHER programs on the same device.)
std::size_t compile_batch(
    tt::tt_metal::IDevice* dev, const std::vector<tt::tt_metal::Program*>& progs, int max_workers) {
    if (progs.empty()) {
        return 0;
    }
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> errors{0};
    auto worker = [&]() {
        std::size_t i;
        while ((i = next.fetch_add(1)) < progs.size()) {
            try {
                tt::tt_metal::detail::CompileProgram(dev, *progs[i]);
            } catch (...) {
                errors.fetch_add(1);
            }
        }
    };
    int n = std::min<int>(max_workers, static_cast<int>(progs.size()));
    std::vector<std::thread> pool;
    pool.reserve(n);
    for (int w = 0; w < n; ++w) {
        pool.emplace_back(worker);
    }
    for (auto& t : pool) {
        t.join();
    }
    return errors.load();
}

tt::tt_metal::IDevice* resolve_device(tt::tt_metal::distributed::MeshDevice* device, const char* who) {
    TT_FATAL(device != nullptr, "{}: device must not be null", who);
    auto devices = device->get_devices();
    TT_FATAL(!devices.empty(), "{}: mesh device has no devices", who);
    return devices.front();
}

int resolve_workers(int max_workers) {
    return max_workers > 0 ? max_workers : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}
}  // namespace

CompileStats parallel_compile(tt::tt_metal::distributed::MeshDevice* device, int max_workers, bool clear) {
    using clock = std::chrono::steady_clock;

    tt::tt_metal::IDevice* dev = resolve_device(device, "up_front_compile");
    auto& store = ProgramCollector::instance();
    std::vector<tt::tt_metal::Program*> progs = store.program_pointers();
    max_workers = resolve_workers(max_workers);

    CompileStats stats;
    stats.num_programs = progs.size();
    stats.max_workers = max_workers;

    auto t0 = clock::now();
    stats.num_errors = compile_batch(dev, progs, max_workers);
    stats.wall_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - t0).count();

    if (clear) {
        store.clear();
    }
    return stats;
}

namespace {
// One process-wide streaming session (matches the single process-wide ProgramCollector).
struct StreamingState {
    std::thread manager;
    std::atomic<bool> stop{false};
    CompileStats stats;
    bool running = false;
};
StreamingState& stream_state() {
    static StreamingState s;
    return s;
}
}  // namespace

void start_streaming_compile(tt::tt_metal::distributed::MeshDevice* device, int max_workers) {
    tt::tt_metal::IDevice* dev = resolve_device(device, "start_streaming_compile");
    auto& s = stream_state();
    TT_FATAL(!s.running, "start_streaming_compile: a streaming session is already running");
    max_workers = resolve_workers(max_workers);

    s.stop.store(false);
    s.stats = CompileStats{};
    s.stats.max_workers = max_workers;
    s.running = true;

    auto t0 = std::chrono::steady_clock::now();
    s.manager = std::thread([dev, max_workers, t0]() {
        auto& st = stream_state();
        std::size_t total = 0;
        std::size_t errs = 0;
        // Loop: claim whatever has been collected-but-not-yet-compiled and compile it, running
        // concurrently with the collect thread that keeps adding programs. Poll when idle; exit
        // once stop is set AND there is nothing left to claim.
        while (true) {
            auto progs = ProgramCollector::instance().claim_uncompiled();
            if (!progs.empty()) {
                errs += compile_batch(dev, progs, max_workers);
                total += progs.size();
                continue;
            }
            if (st.stop.load()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        // Final drain: catch any programs collected between the last claim and stop.
        auto progs = ProgramCollector::instance().claim_uncompiled();
        errs += compile_batch(dev, progs, max_workers);
        total += progs.size();

        st.stats.num_programs = total;
        st.stats.num_errors = errs;
        st.stats.wall_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t0).count();
    });
}

CompileStats finish_streaming_compile() {
    auto& s = stream_state();
    if (!s.running) {
        return CompileStats{};
    }
    s.stop.store(true);
    if (s.manager.joinable()) {
        s.manager.join();
    }
    s.running = false;
    return s.stats;
}

}  // namespace ttnn::up_front_compile
