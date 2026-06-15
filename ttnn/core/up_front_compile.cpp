// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/up_front_compile.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>  // tt::tt_metal::detail::CompileProgram

#include "ttnn/graph/graph_processor.hpp"
#include <tt-metalium/graph_tracking.hpp>  // GraphTracker

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
    // hash==0 means the program cache was disabled (no real hash). Use a synthetic key
    // counting down from UINT64_MAX so distinct programs aren't collapsed into one and we
    // don't collide with real hashes.
    std::uint64_t key = program_hash != 0 ? program_hash : (UINT64_MAX - synthetic_key_++);
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

std::unordered_map<std::uint64_t, tt::tt_metal::distributed::MeshWorkload> ProgramCollector::take_workloads() {
    std::lock_guard<std::mutex> lk(mutex_);
    auto out = std::move(programs_);
    programs_.clear();  // a moved-from map is valid-but-unspecified; force it empty
    total_collected_ = 0;
    synthetic_key_ = 0;
    return out;
}

void begin_collect(bool clear, bool real_alloc) {
    // Nesting under an existing capture would skip our blocking hook (begin_capture only
    // installs one when none is present), leaving the collector active but dispatch unblocked.
    TT_FATAL(
        tt::tt_metal::GraphTracker::instance().get_hook() == nullptr,
        "up_front_compile: begin_collect cannot run while a graph capture / hook is already active");
    if (clear) {
        ProgramCollector::instance().clear();
    }
    // NO_DISPATCH: buffer allocations are mocked (addr 0) and nothing dispatches,
    // so the collect pass uses no real device memory. The funnel's stash + early
    // return (device_operation.hpp) handles program collection.
    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    ProgramCollector::set_active(true);
    if (real_alloc) {
        // Block dispatch only, letting the allocator hand out live addresses: this better
        // captures ops that rely on live allocator state, ensuring higher JIT hit rates.
        auto* ph = dynamic_cast<ttnn::graph::ProcessorHooks*>(tt::tt_metal::GraphTracker::instance().get_hook().get());
        TT_ASSERT(ph != nullptr, "up_front_compile: real_alloc requires the NO_DISPATCH ProcessorHooks to be active");
        ph->set_capture_block(ttnn::graph::CaptureBlock::DispatchOnly);
    }
}

void end_collect() {
    ProgramCollector::set_active(false);
    // Discard the JSON graph; we only used NO_DISPATCH for allocation blocking.
    ttnn::graph::GraphProcessor::end_graph_capture();
}

CompileStats parallel_compile(tt::tt_metal::distributed::MeshDevice* device, int max_workers) {
    using clock = std::chrono::steady_clock;

    TT_FATAL(device != nullptr, "up_front_compile: device must not be null");
    auto devices = device->get_devices();
    TT_FATAL(!devices.empty(), "up_front_compile: mesh device has no devices");
    tt::tt_metal::IDevice* dev = devices.front();

    auto& store = ProgramCollector::instance();
    // Take ownership of the collected workloads up front so the worker threads
    // compile from a snapshot whose lifetime we control. up_front_compile releases
    // the GIL, so without this a concurrent up_front_clear() / begin_collect(clear=true)
    // could destroy the MeshWorkloads — and the Program* borrowed into them — while
    // workers are still compiling (use-after-free).
    auto workloads = store.take_workloads();
    std::vector<tt::tt_metal::Program*> progs;
    for (auto& [hash, workload] : workloads) {
        for (auto& [range, program] : workload.get_programs()) {
            progs.push_back(&program);
        }
    }

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
                } catch (const std::exception& e) {
                    // Non-fatal: a failed warm-up compile just leaves that program cold for the
                    // real run. Log the reason so the failure isn't silent behind the error count.
                    errors.fetch_add(1);
                    log_warning(tt::LogAlways, "up_front_compile: program {} failed to compile: {}", i, e.what());
                } catch (...) {
                    errors.fetch_add(1);
                    log_warning(tt::LogAlways, "up_front_compile: program {} failed to compile (unknown exception)", i);
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

    // take_workloads() already emptied the store; the local snapshot drops here.
    return stats;
}

}  // namespace ttnn::up_front_compile
