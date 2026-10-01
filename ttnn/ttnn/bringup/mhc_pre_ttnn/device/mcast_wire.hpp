// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The positional multicast wire of one host multicast family, on #57547's attachment API.
//
// #57547 replaced the old helpers' compile_time_args() / runtime_args(core) / owned_semaphores() queries with
// attach(descriptor, prefix, kernels), which appends the family's argument blocks to a kernel's CURRENT argument
// lists. This program factory places the wire in the MIDDLE of its argument lists (fixed CT / RT bases that the
// kernels decode with dataflow_kernel_lib::McastArgs<CT, RT>, which is still positional), so it attaches the family
// to a scratch data-movement kernel with empty argument lists and reads the blocks back. The blocks are exactly what
// attach() would have appended; nothing is re-encoded here. The old helpers' per-kernel pre_handshake override has no
// equivalent in the new API, so compile_time_args(pre_handshake) clears or sets the wire's PRE_HANDSHAKE flag the way
// the old override did: the family keeps its consumer-ready semaphore either way, so the program's semaphore set (and
// every id after it) is unchanged.

#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host.hpp"

namespace ttnn::operations::bringup::mhc_pre_ttnn::mcast_wire {

struct McastWire {
    std::vector<uint32_t> ct;
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> rt;
    std::vector<tt::tt_metal::SemaphoreDescriptor> semaphores;  // the family's own (none when it adopts ids)

    std::vector<uint32_t> compile_time_args(std::optional<bool> pre_handshake = std::nullopt) const {
        namespace w = dataflow_kernel_lib::mcast_wire;
        auto out = ct;
        if (pre_handshake.has_value() && out.size() > w::FLAGS && out[w::TAG] == w::FAMILY) {
            out[w::FLAGS] = *pre_handshake ? (out[w::FLAGS] | w::PRE_HANDSHAKE) : (out[w::FLAGS] & ~w::PRE_HANDSHAKE);
        }
        return out;
    }

    const std::vector<uint32_t>& runtime_args(const tt::tt_metal::CoreCoord& core) const {
        const auto it = rt.find({static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)});
        TT_FATAL(it != rt.end(), "mcast wire: core ({},{}) is outside the attached kernel", core.x, core.y);
        return it->second;
    }

    // Whether `core` holds a sender role (any phase) in this family.
    bool is_sender(const tt::tt_metal::CoreCoord& core) const {
        namespace w = dataflow_kernel_lib::mcast_wire;
        const auto& args = runtime_args(core);
        const uint32_t at =
            w::roles_offset(ct[w::ROTATING_SPAN], ct[w::RECTANGLE_CAPACITY], w::transfer_mode(ct[w::FLAGS])) + w::ROLES;
        TT_FATAL(at < args.size(), "mcast wire: runtime block too short for its role word");
        return (args[at] & w::CAN_SEND) != 0u;
    }

    uint32_t num_semaphores() const { return static_cast<uint32_t>(semaphores.size()); }
};

// `cores`: every core the real kernel runs on (cores outside the family get a role-less block); `noc`: the real
// kernel's NoC (attach checks it against the family's); `existing`: semaphores the family adopts by id.
template <typename Family>
McastWire extract(
    const Family& family,
    const tt::tt_metal::CoreRangeSet& cores,
    tt::tt_metal::NOC noc,
    const tt::tt_metal::ProgramDescriptor::SemaphoreDescriptors& existing = {}) {
    tt::tt_metal::ProgramDescriptor scratch;
    scratch.semaphores = existing;
    tt::tt_metal::KernelDescriptor kernel;
    kernel.kernel_source = "mcast_wire_scratch";
    kernel.core_ranges = cores;
    kernel.config = tt::tt_metal::DataMovementConfigDescriptor{.noc = noc};
    family.attach(scratch, "mcast_wire", kernel);
    McastWire wire;
    wire.ct = kernel.compile_time_args;
    for (auto& [core, args] : kernel.runtime_args) {
        wire.rt[{static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)}] = std::move(args);
    }
    wire.semaphores.assign(scratch.semaphores.begin() + existing.size(), scratch.semaphores.end());
    return wire;
}

inline std::vector<uint32_t> absent_compile_time_args() {
    std::vector<uint32_t> ct;
    ttnn::kernel_lib::host::append_absent_mcast_compile_time_args_to(ct);
    return ct;
}

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn::mcast_wire
