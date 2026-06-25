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

CompileStats parallel_compile(tt::tt_metal::distributed::MeshDevice* device, int max_workers, bool clear) {
    using clock = std::chrono::steady_clock;

    TT_FATAL(device != nullptr, "up_front_compile: device must not be null");
    auto devices = device->get_devices();
    TT_FATAL(!devices.empty(), "up_front_compile: mesh device has no devices");
    tt::tt_metal::IDevice* dev = devices.front();

    auto& store = ProgramCollector::instance();
    std::vector<tt::tt_metal::Program*> progs = store.program_pointers();

    if (max_workers <= 0) {
        max_workers = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }

    CompileStats stats;
    stats.num_programs = progs.size();
    stats.max_workers = max_workers;

    auto t0 = clock::now();
    if (!progs.empty()) {
        std::atomic<std::size_t> next{0};
        std::atomic<std::size_t> errors{0};
        auto worker = [&]() {
            std::size_t i;
            while ((i = next.fetch_add(1)) < progs.size()) {
                try {
                    // Compile-only: builds kernels into the on-disk JIT cache. No command
                    // queue / program-cache state is touched, so distinct programs compile
                    // concurrently and safely. Buffer addresses are irrelevant to compile.
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
        stats.num_errors = errors.load();
    }
    stats.wall_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - t0).count();

    if (clear) {
        store.clear();
    }
    return stats;
}

}  // namespace ttnn::up_front_compile
