// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>

namespace tt::tt_metal::host_dispatch_microbench {

enum class Slot : int {
    // Python nanobind entry (mesh descriptor assembly only; excludes prim::patchable_generic_op).
    PatchableNanobindMeshSetup = 0,
    // ttnn::prim::patchable_generic_op (C++ entry through device_operation::launch return).
    PatchablePrimThroughLaunch,
    // PatchableGenericOpDeviceOperation::compute_program_hash (subset of mesh workload hash).
    PatchableComputeProgramHashOnly,
    // Program cache miss: PatchableGenericMeshProgramFactory::create_at excluding discover.
    PatchableCreateAtProgramBuild,
    // Program cache miss: discover_address_slots only.
    PatchableDiscoverAddressSlots,
    // Program cache hit: collect_io_tensor_addresses inside patch_program_from_io_tensors.
    PatchableCollectIoTensorAddresses,
    // Program cache hit: patch per-core + common RT args + dynamic CB addresses.
    PatchableApplySlotPatches,
    // Shared mesh launch path (all device ops using MeshDeviceOperationAdapter).
    MeshComputeWorkloadHash,
    MeshProgramCacheContains,
    MeshCacheHitOverrideRuntimeArgs,
    MeshCacheHitEnqueueWorkload,
    MeshCacheMissCreateAndCacheWorkload,
    // device_operation::launch before launch_operation_with_adapter (tensor visits, topology, etc.).
    DeviceOpLaunchPreamble,

    Count
};

inline const char* slot_name(Slot s) {
    static constexpr const char* kNames[] = {
        "patchable_nanobind_mesh_setup",
        "patchable_prim_through_launch",
        "patchable_compute_program_hash_only",
        "patchable_create_at_program_build",
        "patchable_discover_address_slots",
        "patchable_collect_io_tensor_addresses",
        "patchable_apply_slot_patches",
        "mesh_compute_workload_hash",
        "mesh_program_cache_contains",
        "mesh_cache_hit_override_runtime_args",
        "mesh_cache_hit_enqueue_workload",
        "mesh_cache_miss_create_and_cache_workload",
        "device_op_launch_preamble",
    };
    static_assert(sizeof(kNames) / sizeof(kNames[0]) == static_cast<size_t>(Slot::Count), "slot name table mismatch");
    return kNames[static_cast<int>(s)];
}

inline bool is_enabled() {
    static const bool kOn = [] {
        const char* v = std::getenv("TTNN_HOST_DISPATCH_MICROBENCH");
        return v != nullptr && v[0] == '1';
    }();
    return kOn;
}

struct Stats {
    std::atomic<uint64_t> total_ns{0};
    std::atomic<uint64_t> samples{0};
};

inline std::array<Stats, static_cast<size_t>(Slot::Count)>& stats_table() {
    static std::array<Stats, static_cast<size_t>(Slot::Count)> table{};
    return table;
}

inline void record(Slot slot, int64_t nanoseconds) {
    if (nanoseconds < 0) {
        nanoseconds = 0;
    }
    auto& e = stats_table()[static_cast<size_t>(slot)];
    e.total_ns.fetch_add(static_cast<uint64_t>(nanoseconds), std::memory_order_relaxed);
    e.samples.fetch_add(1, std::memory_order_relaxed);
}

class ScopedTimer {
    Slot slot_;
    bool on_;
    std::chrono::steady_clock::time_point t0_{};

public:
    explicit ScopedTimer(Slot slot) : slot_(slot), on_(is_enabled()) {
        if (on_) {
            t0_ = std::chrono::steady_clock::now();
        }
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

    ~ScopedTimer() {
        if (!on_) {
            return;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0_).count();
        record(slot_, ns);
    }
};

inline void reset_stats() {
    for (auto& e : stats_table()) {
        e.total_ns.store(0, std::memory_order_relaxed);
        e.samples.store(0, std::memory_order_relaxed);
    }
}

inline std::string format_report() {
    std::string out;
    out.append(
        "TTNN_HOST_DISPATCH_MICROBENCH report (TTNN_HOST_DISPATCH_MICROBENCH=1). "
        "Note: some slots nest (e.g. mesh_compute_workload_hash includes patchable_compute_program_hash_only); "
        "do not sum all rows for a total.\n");
    uint64_t any_samples = 0;
    for (int i = 0; i < static_cast<int>(Slot::Count); ++i) {
        const auto& e = stats_table()[static_cast<size_t>(i)];
        const uint64_t n = e.samples.load(std::memory_order_relaxed);
        if (n == 0) {
            continue;
        }
        any_samples += n;
        const uint64_t total = e.total_ns.load(std::memory_order_relaxed);
        const double total_ms = static_cast<double>(total) / 1e6;
        const double avg_us = static_cast<double>(total) / static_cast<double>(n) / 1e3;
        out += "  ";
        out += slot_name(static_cast<Slot>(i));
        out += ": samples=" + std::to_string(n);
        out += " total_ms=" + std::to_string(total_ms);
        out += " avg_us=" + std::to_string(avg_us);
        out += '\n';
    }
    if (any_samples == 0) {
        out += "  (no samples — run workload with env var set)\n";
    }
    return out;
}

}  // namespace tt::tt_metal::host_dispatch_microbench
