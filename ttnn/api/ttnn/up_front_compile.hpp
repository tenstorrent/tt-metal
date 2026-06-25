// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

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
// Up-front parallel precompile.
//
// The expensive part of a cold model run is JIT kernel compilation, done one
// program at a time behind sequential op dispatch. This warms the on-disk
// kernel cache (TT_METAL_CACHE) for a model's whole distinct program set, up
// front and in parallel, so the subsequent real run / trace capture runs warm.
//
// Usage (op-agnostic — works for any op that dispatches through the device-op
// adapter, i.e. generic_op and every C++ ProgramDescriptor-migrated op):
//
//     ttnn.graph.up_front_begin_collect()           # NO_DISPATCH: nothing runs,
//     model(dummy_input, device)                    #   each op stashes its workload
//     ttnn.graph.up_front_end_collect()
//     ttnn.graph.up_front_compile(device, workers)  # parallel JIT, warms the cache
//     # real run / trace capture is now warm
//
// Requirements:
//   * Run on a COLD device program cache with the cache ENABLED, so each op is
//     a cache miss (reaches the collector) and carries a distinct program hash.
//     (If the hash is 0 — cache disabled — the collector keeps every program
//     distinct via a synthetic key, trading dedup for correctness.)
//
// Mechanism: a NO_DISPATCH graph capture mocks all buffer allocations (addr 0)
// and blocks dispatch, so the collect pass uses no real device memory. The
// dispatch funnel (device_operation.hpp::create_and_cache_mesh_workload) moves
// the freshly-built-but-uncompiled MeshWorkload into the collector, keyed by
// program hash, and skips caching + enqueue. parallel_compile then JIT-compiles
// the distinct set. Address-independence of compilation makes addr-0 fine.
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

    // Borrowed pointers to every collected program (valid until clear()).
    std::vector<tt::tt_metal::Program*> program_pointers();

private:
    ProgramCollector() = default;

    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, tt::tt_metal::distributed::MeshWorkload> programs_;
    std::size_t total_collected_ = 0;
    std::uint64_t synthetic_key_ = 0;  // for hash==0 (cache-disabled) fallback
};

// Begin a collect pass: enables NO_DISPATCH graph capture (buffers mocked,
// nothing dispatched) and marks the collector active on this thread.
//
// clear=true (default) drops any previously collected programs first — the
// model-forward usage (one begin/end around a single run). clear=false
// ACCUMULATES into the existing set — the cross-test usage: a pytest plugin
// wraps each test body in begin_collect(clear=false)/end_collect(), so programs
// from every test pile into one deduped set, then a single parallel_compile at
// session end. (begin/end_collect push/pop a NO_DISPATCH graph-capture frame
// per call and the hook is cleanly removed on end, so per-test wrapping is safe;
// only the clear must be suppressed to accumulate — clear the collector once up
// front via ProgramCollector::clear() / up_front_clear instead.)
//
// real_alloc=false (default): NO_DISPATCH mocks every buffer at address 0 → zero device
// memory, scales to any model, but kernels that bake a buffer address into compile-time
// args (e.g. pool reader_indices) or branch on addresses (e.g. move forward/backward)
// collect the addr-0 variant and MISS on the real run. real_alloc=true: let the allocator
// assign REAL addresses during collect (dispatch still blocked) so those build the same
// program the real run will → they warm. The L1 allocator is deterministic across processes,
// so a fresh-device collect and a fresh-device real run land buffers at identical addresses.
// COST: real device memory (~the real run's peak; a collect OOM is a faithful signal the real
// run would OOM too) + the alloc/free sequence must match the real run (it does if collect
// replays the same forward). Use real_alloc for models that fit; addr-0 for the giant ones.
void begin_collect(bool clear = true, bool real_alloc = false);

// End the collect pass: stops NO_DISPATCH capture and deactivates the collector.
// Collected programs remain in ProgramCollector::instance() until parallel_compile
// (or ProgramCollector::clear) is called.
void end_collect();

// JIT-compile every distinct collected program in parallel, warming the on-disk
// kernel cache. Compiles for devices.front() of the mesh — a homogeneous mesh
// shares one build key, so one compile warms the whole mesh. max_workers<=0 uses
// the hardware concurrency. If clear, the collected programs are released after.
CompileStats parallel_compile(tt::tt_metal::distributed::MeshDevice* device, int max_workers = 0, bool clear = true);

}  // namespace ttnn::up_front_compile
