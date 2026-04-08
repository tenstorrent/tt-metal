// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::generic {

using OptionalAddr = std::optional<std::uint32_t>;

// ── Low-level address-matching helpers (shared with discover_address_slots) ──

/// Collect buffer addresses from IO tensors (nullopt for tensors without a device buffer).
inline std::vector<OptionalAddr> collect_io_tensor_addresses(const std::vector<Tensor>& io_tensors) {
    std::vector<OptionalAddr> addrs;
    addrs.reserve(io_tensors.size());
    for (const auto& t : io_tensors) {
        auto* buf = t.buffer();
        addrs.push_back(buf ? std::optional{buf->address()} : std::nullopt);
    }
    return addrs;
}

/// Find the first IO tensor whose address matches *value*, or nullopt.
inline std::optional<std::uint32_t> find_io_tensor_index(std::uint32_t value, const std::vector<OptionalAddr>& addrs) {
    for (size_t i = 0; i < addrs.size(); ++i) {
        if (addrs[i].has_value() && addrs[i].value() == value) {
            return static_cast<std::uint32_t>(i);
        }
    }
    return std::nullopt;
}

/// For each CBDescriptor with a buffer, match its address against *tensor_addrs*.
inline std::vector<std::pair<uint32_t, uint32_t>> compute_cb_io_tensor_map(
    const tt::tt_metal::ProgramDescriptor& desc, const std::vector<OptionalAddr>& tensor_addrs) {
    std::vector<std::pair<uint32_t, uint32_t>> result;
    for (size_t ci = 0; ci < desc.cbs.size(); ++ci) {
        const auto* buf = desc.cbs[ci].buffer;
        if (buf != nullptr) {
            if (auto ti = find_io_tensor_index(buf->address(), tensor_addrs)) {
                result.emplace_back(static_cast<uint32_t>(ci), *ti);
            }
        }
    }
    return result;
}

// ── AddressSlots: opaque mapping for refreshing stale ProgramDescriptors ────

/// Slot: a per-core runtime arg that holds an IO tensor address.
struct PerCoreRTArgSlot {
    std::uint32_t kernel_idx;
    CoreCoord core;
    std::uint32_t arg_idx;
    std::uint32_t io_tensor_index;
};

/// Slot: a common runtime arg that holds an IO tensor address.
struct CommonRTArgSlot {
    std::uint32_t kernel_idx;
    std::uint32_t arg_idx;
    std::uint32_t io_tensor_index;
};

/// Slot: a CB whose buffer comes from an IO tensor.
struct CBSlot {
    std::uint32_t cb_idx;
    std::uint32_t io_tensor_index;
};

/// Complete mapping of every position in a ProgramDescriptor that references
/// an IO tensor address.  Computed once at build time via ``compute_address_slots``
/// (when addresses are valid), held opaquely by Python, and passed back to
/// ``patchable_generic_op`` on each launch to refresh stale values.
struct AddressSlots {
    std::vector<PerCoreRTArgSlot> per_core_rt_arg_slots;
    std::vector<CommonRTArgSlot> common_rt_arg_slots;
    std::vector<CBSlot> cb_slots;
    std::vector<OptionalAddr> io_addrs_at_build;
};

/// Compute the full address-slot mapping.  Must be called while buffer pointers
/// and runtime arg addresses are valid (at build time, before tensors are freed).
/// Same address-matching logic as ``discover_address_slots`` in the program factory.
inline AddressSlots compute_address_slots(
    const tt::tt_metal::ProgramDescriptor& desc, const std::vector<Tensor>& io_tensors) {
    AddressSlots slots;
    auto tensor_addrs = collect_io_tensor_addresses(io_tensors);

    for (size_t ki = 0; ki < desc.kernels.size(); ++ki) {
        const auto& kd = desc.kernels[ki];
        for (const auto& [coord, args] : kd.runtime_args) {
            for (size_t ai = 0; ai < args.size(); ++ai) {
                if (auto ti = find_io_tensor_index(args[ai], tensor_addrs)) {
                    slots.per_core_rt_arg_slots.push_back(
                        {static_cast<uint32_t>(ki), coord, static_cast<uint32_t>(ai), *ti});
                }
            }
        }
        for (size_t ai = 0; ai < kd.common_runtime_args.size(); ++ai) {
            if (auto ti = find_io_tensor_index(kd.common_runtime_args[ai], tensor_addrs)) {
                slots.common_rt_arg_slots.push_back({static_cast<uint32_t>(ki), static_cast<uint32_t>(ai), *ti});
            }
        }
    }

    for (const auto& [cb_idx, io_idx] : compute_cb_io_tensor_map(desc, tensor_addrs)) {
        slots.cb_slots.push_back({cb_idx, io_idx});
    }

    slots.io_addrs_at_build = std::move(tensor_addrs);
    return slots;
}

// ── Descriptor patching ─────────────────────────────────────────────────────

/// Patch a ProgramDescriptor in place so all IO-tensor-derived addresses and
/// buffer pointers reflect the current ``io_tensors``.  Returns immediately
/// when all addresses match the build-time snapshot (zero-cost hot path).
inline void patch_stale_descriptor(
    tt::tt_metal::ProgramDescriptor& desc, const std::vector<Tensor>& io_tensors, const AddressSlots& slots) {
    // Quick check: if all IO tensor addresses match build time, nothing is stale.
    bool needs_patch = false;
    for (size_t i = 0; i < slots.io_addrs_at_build.size() && i < io_tensors.size(); ++i) {
        if (slots.io_addrs_at_build[i].has_value()) {
            auto* buf = io_tensors[i].buffer();
            if (!buf || buf->address() != *slots.io_addrs_at_build[i]) {
                needs_patch = true;
                break;
            }
        }
    }
    if (!needs_patch) {
        return;
    }

    for (const auto& slot : slots.per_core_rt_arg_slots) {
        auto* buf = io_tensors[slot.io_tensor_index].buffer();
        if (buf) {
            auto& rt_args = desc.kernels[slot.kernel_idx].runtime_args;
            for (auto& [core, args] : rt_args) {
                if (core == slot.core) {
                    args[slot.arg_idx] = buf->address();
                    break;
                }
            }
        }
    }
    for (const auto& slot : slots.common_rt_arg_slots) {
        auto* buf = io_tensors[slot.io_tensor_index].buffer();
        if (buf) {
            desc.kernels[slot.kernel_idx].common_runtime_args[slot.arg_idx] = buf->address();
        }
    }
    for (const auto& slot : slots.cb_slots) {
        desc.cbs[slot.cb_idx].buffer = io_tensors[slot.io_tensor_index].buffer();
    }
}

}  // namespace ttnn::operations::experimental::generic
