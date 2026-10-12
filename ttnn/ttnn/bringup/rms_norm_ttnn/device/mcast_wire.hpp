// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The positional multicast wire of one host Mcast channel.
//
// Mcast::attach(descriptor, prefix, kernels, first_semaphore_id) appends the channel's argument blocks to a kernel's
// CURRENT argument lists. This program factory places the wire in the MIDDLE of its argument lists (fixed CT / RT
// bases that the kernels decode with dataflow_kernel_lib::McastArgs<CT, RT>), so it attaches the channel to a scratch
// data-movement kernel with empty argument lists, placed on the real kernel's cores, and reads the blocks back. The
// blocks are exactly what attach() would have appended; nothing is re-encoded here.

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/kernel_lib/mcast/host/mcast.hpp"

namespace ttnn::operations::bringup::rms_norm_ttnn::mcast_wire {

struct McastWire {
    std::vector<uint32_t> ct;
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> rt;
    std::vector<tt::tt_metal::SemaphoreDescriptor> semaphores;  // the channel's own (none when it adopts ids)
    uint32_t next_semaphore_id = 0;

    const std::vector<uint32_t>& compile_time_args() const { return ct; }

    const std::vector<uint32_t>& runtime_args(const tt::tt_metal::CoreCoord& core) const {
        const auto it = rt.find({static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)});
        TT_FATAL(it != rt.end(), "mcast wire: core ({},{}) is outside the attached kernel", core.x, core.y);
        return it->second;
    }

    // Whether `core` holds a sender role (any phase) in this channel.
    bool is_sender(const tt::tt_metal::CoreCoord& core) const {
        namespace w = dataflow_kernel_lib::mcast_wire;
        const auto metadata = w::decode_compile_time_metadata(ct);
        const w::RuntimeLayout layout(metadata);
        if (layout.roles == w::OMITTED) {
            return (metadata.kernel.roles & w::CAN_SEND) != 0u;
        }
        const auto& args = runtime_args(core);
        TT_FATAL(layout.roles < args.size(), "mcast wire: runtime block too short for its role word");
        return (args[layout.roles] & w::CAN_SEND) != 0u;
    }
};

// `cores`: every core the real kernel runs on (cores outside the channel get a role-less block); `noc`: the real
// kernel's NoC (attach checks it against the channel's); `first_semaphore_id`: the channel's semaphores are
// consecutive from it. With `adopt`, the program already owns those ids (`existing` must hold each on the channel's
// cores, initial value 0) and the channel adds none of its own.
inline McastWire extract(
    const ttnn::kernel_lib::host::Mcast& mcast,
    const tt::tt_metal::CoreRangeSet& cores,
    tt::tt_metal::NOC noc,
    uint32_t first_semaphore_id,
    bool adopt = false,
    const tt::tt_metal::ProgramDescriptor::SemaphoreDescriptors& existing = {}) {
    tt::tt_metal::ProgramDescriptor scratch;
    tt::tt_metal::KernelDescriptor kernel;
    kernel.kernel_source = "mcast_wire_scratch";
    kernel.core_ranges = cores;
    kernel.config = tt::tt_metal::DataMovementConfigDescriptor{.noc = noc};
    const std::array kernels{std::ref(kernel)};
    mcast.attach(scratch, "mcast_wire", kernels, first_semaphore_id);
    McastWire wire;
    wire.ct = kernel.compile_time_args;
    for (auto& [core, args] : kernel.runtime_args) {
        wire.rt[{static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)}] = std::move(args);
    }
    wire.next_semaphore_id = mcast.next_semaphore_id();
    if (!adopt) {
        wire.semaphores.assign(scratch.semaphores.begin(), scratch.semaphores.end());
        return wire;
    }
    for (const auto& s : scratch.semaphores) {
        const bool owned = std::any_of(existing.begin(), existing.end(), [&](const auto& e) {
            return e.id == s.id && e.core_type == s.core_type && e.initial_value == s.initial_value &&
                   e.core_ranges.contains(s.core_ranges);
        });
        TT_FATAL(owned, "mcast wire: adopted semaphore {} is not owned by the program on the channel's cores", s.id);
    }
    return wire;
}

// Row-major index of `core` within the rectangle `rect` (the in-group index of a fixed sender).
inline uint32_t row_major_index(const tt::tt_metal::CoreRange& rect, const tt::tt_metal::CoreCoord& core) {
    TT_FATAL(rect.contains(core), "mcast wire: core ({},{}) is outside the group", core.x, core.y);
    return static_cast<uint32_t>((core.y - rect.start_coord.y) * rect.grid_size().x + (core.x - rect.start_coord.x));
}

inline std::vector<uint32_t> absent_compile_time_args() {
    std::vector<uint32_t> ct;
    ttnn::kernel_lib::host::append_absent_mcast_compile_time_args_to(ct);
    return ct;
}

}  // namespace ttnn::operations::bringup::rms_norm_ttnn::mcast_wire
