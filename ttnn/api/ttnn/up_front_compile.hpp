// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace tt::tt_metal {
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {
class MeshWorkload;
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace ttnn::up_front_compile {

// Result of a parallel_compile pass.
struct CompileStats {
    std::size_t num_programs = 0;  // distinct programs JIT-compiled
    std::size_t num_errors = 0;    // programs whose compile threw
    int max_workers = 0;
    double wall_seconds = 0.0;
};

// ---------------------------------------------------------------------------
// Up-front parallel precompile: JIT-compile a distinct set of programs in parallel
// to warm the on-disk kernel cache (TT_METAL_CACHE), so the subsequent real run is
// warm. Works for any op dispatched through the device-op adapter.
//
// Usage:
//     ttnn.graph.up_front_begin_collect()           # NO_DISPATCH: nothing runs,
//     model(dummy_input, device)                    #   each op stashes its workload
//     ttnn.graph.up_front_end_collect()
//     ttnn.graph.up_front_compile(device, workers)  # parallel JIT, warms the cache
//
// Run on a COLD program cache with the cache ENABLED, so each op is a miss that
// reaches the collector with a distinct hash.
//
// Mechanism: a graph-capture pass blocks dispatch while letting the allocator hand out
// REAL buffer addresses (nothing executes); the device-op funnel moves each built-but-
// uncompiled MeshWorkload into the collector (keyed by hash, skipping cache + enqueue),
// then parallel_compile JIT-compiles the distinct set. Real addresses mean kernels that
// bake or branch on a buffer address build the same program the real run will, so they warm.
// ---------------------------------------------------------------------------
class ProgramCollector {
public:
    // Process-wide store of collected programs.
    static ProgramCollector& instance();

    // The active collector for the current (capturing) thread, or nullptr. The
    // device-op dispatch funnel checks this on every op.
    static ProgramCollector* active();

    // Move a freshly-built workload in, keyed by its program-cache hash. The
    // first entry per hash wins; later structural duplicates are dropped (same
    // dedup the device program cache would do). Thread-safe.
    void collect(std::uint64_t program_hash, tt::tt_metal::distributed::MeshWorkload&& workload);

    std::size_t num_unique() const;     // distinct program hashes collected
    std::size_t num_collected() const;  // total ops that stashed (incl. dropped dups)
    void clear();

    // Toggle the per-thread active flag. Used by begin/end_collect.
    static void set_active(bool active);

    // Move the collected set out under the mutex and reset the counters; the caller
    // owns it for the whole compile. Leaves the store empty.
    std::unordered_map<std::uint64_t, tt::tt_metal::distributed::MeshWorkload> take_workloads();

private:
    ProgramCollector() = default;

    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, tt::tt_metal::distributed::MeshWorkload> programs_;
    std::size_t total_collected_ = 0;
    std::uint64_t synthetic_key_ = 0;  // for hash==0 (cache-disabled) fallback
};

// Begin a collect pass: blocks dispatch (nothing executes) while the allocator hands out
// REAL buffer addresses, and marks the collector active on this thread.
//
// clear=true (default) drops any previously collected programs first — the
// model-forward usage (one begin/end around a single run). clear=false
// ACCUMULATES into the existing set — the cross-test usage: a pytest plugin
// wraps each test body in begin_collect(clear=false)/end_collect(), so programs
// from every test pile into one deduped set, then a single parallel_compile at
// session end.
//
// Real addresses mean address-baked / address-branched kernels (e.g. pool reader_indices,
// move forward/backward) build the same program the real run will, so they warm too. The
// allocator is deterministic across processes, so a fresh-device collect and a fresh-device
// real run land buffers at identical addresses. COST: real device memory (~the real run's
// peak) and the alloc/free sequence must match the real run (it does if collect replays the
// same forward). A workload that would OOM the real run OOMs here too; that body is skipped.
void begin_collect(bool clear = true);

// End the collect pass: stops NO_DISPATCH capture and deactivates the collector.
// Collected programs remain in ProgramCollector::instance() until parallel_compile
// (or ProgramCollector::clear) is called.
void end_collect();

// JIT-compile every distinct collected program in parallel, warming the on-disk
// kernel cache. Compiles for devices.front() of the mesh — a homogeneous mesh
// shares one build key, so one compile warms the whole mesh. max_workers<=0 uses
// the hardware concurrency. Consumes the collected set (the store is left empty).
CompileStats parallel_compile(tt::tt_metal::distributed::MeshDevice* device, int max_workers = 0);

}  // namespace ttnn::up_front_compile
