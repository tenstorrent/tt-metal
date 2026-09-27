// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>
#include "tt_metal/impl/buffers/semaphore.hpp"

namespace ttnn::kernel_lib::host {
using namespace tt::tt_metal;
using dataflow_kernel_lib::TransferMode;
namespace wire = dataflow_kernel_lib::mcast_wire;

namespace {

NOC kernel_noc(const KernelDescriptor& kernel, tt::ARCH arch) {
    return std::visit(
        ttsl::overloaded{
            [arch](const ReaderConfigDescriptor&) { return tt::tt_metal::detail::preferred_noc_for_dram_read(arch); },
            [arch](const WriterConfigDescriptor&) { return tt::tt_metal::detail::preferred_noc_for_dram_write(arch); },
            [](const DataMovementConfigDescriptor& config) { return config.noc; },
            [](const ComputeConfigDescriptor&) -> NOC {
                TT_THROW("Multicast attachment requires a data-movement kernel");
            }},
        kernel.config);
}

void append_offsets(KernelDescriptor& kernel, std::string_view prefix, uint32_t ct_offset, uint32_t rt_offset) {
    TT_FATAL(!prefix.empty(), "Multicast attachment requires a nonempty prefix");
    const auto ct_name = std::string(prefix) + "_ct_offset";
    const auto rt_name = std::string(prefix) + "_rt_offset";
    for (const auto& [name, value] : kernel.named_compile_time_args) {
        TT_FATAL(name != ct_name && name != rt_name, "Multicast attachment prefix is already in use");
    }
    TT_FATAL(kernel.compile_time_args.size() <= std::numeric_limits<uint32_t>::max(), "Multicast CT offset overflow");
    kernel.named_compile_time_args.emplace_back(ct_name, ct_offset);
    kernel.named_compile_time_args.emplace_back(rt_name, rt_offset);
}

}  // namespace

void McastFamily::attach(
    ProgramDescriptor& descriptor,
    std::string_view prefix,
    std::span<const std::reference_wrapper<KernelDescriptor>> targets) const {
    require_arguments_prepared_();
    require_unbound_();
    TT_FATAL(!targets.empty(), "Multicast attachment requires at least one kernel");

    // Validate staged copies so a failure preserves both caller resources and kernels.
    auto semaphores = descriptor.semaphores;
    const auto ids = resolve_semaphore_ids_(semaphores);
    if (!cfg_.sem_ids) {
        for (uint32_t role = 0; role < required_semaphores_(); ++role) {
            semaphores.push_back({.id = ids[role], .core_ranges = participating_, .initial_value = 0});
        }
    }
    const auto ct = compile_time_args_(ids);
    const bool chain = wire::transfer_mode(layout_.flags) == TransferMode::ChainUnicast;
    std::set<const KernelDescriptor*> selected;
    std::vector<KernelDescriptor> kernels;
    kernels.reserve(targets.size());
    for (auto target : targets) {
        TT_FATAL(selected.insert(&target.get()).second, "Duplicate multicast attachment kernel");
        auto kernel = target.get();
        const auto noc = kernel_noc(kernel, prepared_arch_);
        size_t rt_offset = 0;
        std::set<CoreCoord> runtime_cores;
        for (const auto& [core, args] : kernel.runtime_args) {
            TT_FATAL(kernel.core_ranges.contains(core), "Multicast runtime arguments target an unplaced core");
            TT_FATAL(runtime_cores.insert(core).second, "Duplicate per-core multicast runtime arguments");
            rt_offset = std::max(rt_offset, args.size());
        }
        TT_FATAL(rt_offset <= std::numeric_limits<uint32_t>::max(), "Multicast RT offset overflow");
        TT_FATAL(
            kernel.compile_time_args.size() <= std::numeric_limits<uint32_t>::max() - ct.size(),
            "Multicast CT offset overflow");
        append_offsets(
            kernel, prefix, static_cast<uint32_t>(kernel.compile_time_args.size()), static_cast<uint32_t>(rt_offset));
        // Check against original prefixes, before padding could hide an invalid binding.
        for (const auto& binding : kernel.buffer_bindings) {
            const auto entry =
                std::find_if(kernel.runtime_args.begin(), kernel.runtime_args.end(), [&](const auto& item) {
                    return item.first == binding.core;
                });
            TT_FATAL(
                entry != kernel.runtime_args.end() && binding.arg_idx < entry->second.size(),
                "Multicast attachment encountered an invalid buffer binding");
        }
        for (const auto& core : corerange_to_cores(kernel.core_ranges)) {
            if (const auto* group = group_for_core_(core)) {
                const bool sender =
                    std::find(group->senders_.begin(), group->senders_.end(), core) != group->senders_.end();
                TT_FATAL(
                    !(sender || chain) || noc == cfg_.noc, "Multicast sender/forwarder NoC differs from its family");
            }
            auto entry = std::find_if(kernel.runtime_args.begin(), kernel.runtime_args.end(), [&](const auto& item) {
                return item.first == core;
            });
            if (entry == kernel.runtime_args.end()) {
                kernel.runtime_args.emplace_back(core, std::vector<uint32_t>{});
                entry = std::prev(kernel.runtime_args.end());
            }
            const auto payload = runtime_args_(core);
            TT_FATAL(
                rt_offset <= std::numeric_limits<uint32_t>::max() - payload.size(),
                "Multicast runtime argument positions overflow");
            auto& args = entry->second;
            args.resize(rt_offset, 0);
            args.insert(args.end(), payload.begin(), payload.end());
        }
        kernel.compile_time_args.insert(kernel.compile_time_args.end(), ct.begin(), ct.end());
        kernels.push_back(std::move(kernel));
    }
    descriptor.semaphores = std::move(semaphores);
    for (size_t i = 0; i < targets.size(); ++i) {
        targets[i].get() = std::move(kernels[i]);
    }
}

void attach_absent(KernelDescriptor& kernel, std::string_view prefix) {
    TT_FATAL(
        !std::holds_alternative<ComputeConfigDescriptor>(kernel.config),
        "Multicast attachment requires a data-movement kernel");
    auto staged = kernel;
    append_offsets(staged, prefix, static_cast<uint32_t>(staged.compile_time_args.size()), 0);
    const auto ct = detail::absent_mcast_compile_time_args();
    staged.compile_time_args.insert(staged.compile_time_args.end(), ct.begin(), ct.end());
    kernel = std::move(staged);
}

void Mcast1D::attach(
    ProgramDescriptor& descriptor,
    std::string_view prefix,
    std::span<const std::reference_wrapper<KernelDescriptor>> kernels) const {
    family_->attach(descriptor, prefix, kernels);
}

void Mcast2D::attach(
    ProgramDescriptor& descriptor,
    std::string_view prefix,
    std::span<const std::reference_wrapper<KernelDescriptor>> kernels) const {
    family_->attach(descriptor, prefix, kernels);
}

}  // namespace ttnn::kernel_lib::host
