// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Attaches multicast arguments to direct-Program kernels, allocating the channel's semaphores in the Program
// through the public host API on the first attachment.

#include "ttnn/kernel_lib/mcast/host/mcast_impl.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tt_stl/assert.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::kernel_lib::host {
using namespace tt::tt_metal;
using dataflow_kernel_lib::TransferMode;
namespace wire = dataflow_kernel_lib::mcast_wire;

namespace {

void insert_offsets(DataMovementConfig& config, std::string_view prefix, uint32_t rt_offset) {
    TT_FATAL(!prefix.empty(), "Multicast attachment requires a nonempty prefix");
    TT_FATAL(config.compile_args.size() <= std::numeric_limits<uint32_t>::max(), "Multicast CT offset overflow");
    const auto ct_name = std::string(prefix) + "_ct_offset";
    const auto rt_name = std::string(prefix) + "_rt_offset";
    TT_FATAL(
        !config.named_compile_args.contains(ct_name) && !config.named_compile_args.contains(rt_name),
        "Multicast attachment prefix is already in use");
    config.named_compile_args.emplace(ct_name, static_cast<uint32_t>(config.compile_args.size()));
    config.named_compile_args.emplace(rt_name, rt_offset);
}

}  // namespace

void McastImpl::require_unbound_() const {
    TT_FATAL(
        bound_program_ == nullptr, "A Program-bound Mcast cannot also attach to a ProgramDescriptor or ProgramSpec");
}

void McastImpl::attach(
    Program& program,
    const CoreRangeSet& cores,
    DataMovementConfig& config,
    std::string_view prefix,
    std::vector<std::pair<CoreCoord, std::vector<uint32_t>>>& runtime_args) {
    prepare_arguments_();
    TT_FATAL(
        bound_program_ == nullptr || bound_program_ == &program,
        "This Mcast is already attached to a different Program");

    std::map<CoreCoord, size_t> entry_for_core;
    std::optional<size_t> rt_offset;
    for (size_t entry = 0; entry < runtime_args.size(); ++entry) {
        const auto& [core, args] = runtime_args[entry];
        TT_FATAL(cores.contains(core), "Multicast runtime arguments target unplaced core {}", core);
        TT_FATAL(entry_for_core.emplace(core, entry).second, "Duplicate multicast runtime arguments for core {}", core);
        TT_FATAL(
            !rt_offset || *rt_offset == args.size(),
            "Multicast attachment requires the same operation runtime argument count on every core, but core {} has "
            "{} and another core has {}",
            core,
            args.size(),
            *rt_offset);
        rt_offset = args.size();
    }
    TT_FATAL(rt_offset.value_or(0) <= std::numeric_limits<uint32_t>::max(), "Multicast RT offset overflow");
    const auto placed_cores = corerange_to_cores(cores);
    for (const auto& core : placed_cores) {
        check_kernel_noc_(core, config.noc);
        TT_FATAL(
            rt_offset.value_or(0) == 0 || entry_for_core.contains(core),
            "Missing operation runtime arguments for multicast core {}",
            core);
    }
    auto staged_config = config;
    insert_offsets(staged_config, prefix, static_cast<uint32_t>(rt_offset.value_or(0)));

    // The first attachment binds the channel to `program` and creates its semaphores there.
    if (bound_program_ == nullptr) {
        std::array<uint32_t, 3> ids{UNUSED_SEM_ID, UNUSED_SEM_ID, UNUSED_SEM_ID};
        for (uint32_t role = 0; role < required_semaphores_(); ++role) {
            ids[role] = CreateSemaphore(program, participating_, 0);
        }
        program_semaphore_ids_ = ids;
        bound_program_ = &program;
    }

    const auto metadata = argument_metadata_();
    const auto ct = compile_time_args_(program_semaphore_ids_, metadata);
    staged_config.compile_args.insert(staged_config.compile_args.end(), ct.begin(), ct.end());

    // Placed cores without operation arguments get entries holding only the multicast arguments.
    auto staged_args = runtime_args;
    for (const auto& core : placed_cores) {
        auto found = entry_for_core.find(core);
        if (found == entry_for_core.end()) {
            found = entry_for_core.emplace(core, staged_args.size()).first;
            staged_args.emplace_back(core, std::vector<uint32_t>{});
        }
        const auto rt = runtime_args_(core, metadata);
        auto& args = staged_args[found->second].second;
        args.insert(args.end(), rt.begin(), rt.end());
    }
    config = std::move(staged_config);
    runtime_args = std::move(staged_args);
}

void McastImpl::check_kernel_noc_(const CoreCoord& core, NOC noc) const {
    const auto* group = group_for_core_(core);
    if (group == nullptr) {
        return;
    }
    const bool chain = wire::transfer_mode(layout_.flags) == TransferMode::ChainUnicast;
    const bool sender = std::find(group->senders_.begin(), group->senders_.end(), core) != group->senders_.end();
    TT_FATAL(
        !(sender || chain) || noc == cfg_.noc,
        "Multicast sender/forwarder NoC differs from the configured multicast NoC");
}

void attach_absent_mcast(DataMovementConfig& config, std::string_view prefix) {
    auto staged = config;
    insert_offsets(staged, prefix, 0);
    staged.compile_args.push_back(wire::ABSENT);
    config = std::move(staged);
}

}  // namespace ttnn::kernel_lib::host
