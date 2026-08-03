// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

// Binding a sharded L1 tensor to a circular buffer under per-core L1 allocation.
//
// A CB that aliases a tensor's L1 (rather than staging a copy) carries ONE address for its whole
// core range, and `Buffer::address()` reports only the FIRST core's. A per-core allocated buffer
// (experimental_set_per_core_allocation) has an independent address on every core, so binding one
// with a single wide CB points every core but the first at the first core's address -- silently
// wrong, and the hazard ttnn/api/ttnn/operation_concepts.hpp warns about under
// SupportsPerCoreAllocation.
//
// So a per-core allocated tensor is bound as one single-core CB per core, each pinned to that
// core's own address via CBDescriptor::absolute_address. Program dispatch already emits one CB
// config payload per (non-overlapping) core range, so single-core ranges give each core its own
// address with no changes below the descriptor. Lockstep tensors keep the single wide descriptor
// (one multicast config write instead of N unicast ones).
//
// Ops that use this get their cache-hit patching for free: resolve_bindings bails on any descriptor
// carrying an absolute_address, which routes the op to the slow-path rebuild that re-derives the
// descriptor and re-applies each CB's own address. That matters because a per-core allocated
// tensor is typically freed and re-fetched around every use and lands at different addresses.
namespace per_core_allocation_cb::detail {
namespace pca = tt::tt_metal::experimental::per_core_allocation;

// Address to bind on `core`. Cores outside the tensor's shard grid hold no shard of their own and
// get the reference (first-core) address -- exactly what a single lockstep CB gave them before.
// They never read valid data there, but the CB must still be declared on them when the kernels
// running there reference the index.
inline uint32_t address_for_core(
    const tt::tt_metal::Buffer& buffer,
    const std::unordered_map<CoreCoord, tt::tt_metal::DeviceAddr>& per_core_addresses,
    const CoreCoord& core) {
    const auto it = per_core_addresses.find(core);
    return static_cast<uint32_t>(it != per_core_addresses.end() ? it->second : buffer.address());
}

inline void validate_single_device(const Tensor& tensor) {
    // Per-core addresses are allocated independently on each physical device, but a
    // ProgramDescriptor is built once for the whole mesh, so one descriptor cannot name the right
    // address on more than one device. The reference buffer only answers for the first.
    TT_FATAL(
        tensor.device()->num_devices() == 1,
        "Per-core L1 allocation is only supported on a single-device mesh (got {} devices). Per-core addresses "
        "differ per device, and the program descriptor carries one address per core for the whole mesh.",
        tensor.device()->num_devices());
}
}  // namespace per_core_allocation_cb::detail

// Fan `base` out into one single-core descriptor per core in its core range when `tensor` is
// per-core allocated; otherwise return it unchanged. `base` must already be fully populated
// (total_size, core_ranges, format_descriptors, buffer = tensor.buffer()); `core_ranges` is the set
// the kernels expect the CB on, which may be wider than the tensor's shard grid.
inline std::vector<tt::tt_metal::CBDescriptor> make_per_core_cb_descriptors(
    const Tensor& tensor, tt::tt_metal::CBDescriptor base) {
    namespace detail = per_core_allocation_cb::detail;
    if (!detail::pca::is_per_core_allocation(*tensor.buffer())) {
        return {std::move(base)};
    }
    detail::validate_single_device(tensor);

    const auto& per_core_addresses = detail::pca::get_per_core_addresses(*tensor.buffer());
    std::vector<tt::tt_metal::CBDescriptor> descriptors;
    for (const auto& core : corerange_to_cores(base.core_ranges, std::nullopt, /*row_wise=*/true)) {
        tt::tt_metal::CBDescriptor descriptor = base;
        descriptor.core_ranges = CoreRangeSet(CoreRange(core, core));
        descriptor.absolute_address = detail::address_for_core(*tensor.buffer(), per_core_addresses, core);
        descriptors.push_back(std::move(descriptor));
    }
    return descriptors;
}

}  // namespace ttnn
