// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_handoff.hpp"

#include <algorithm>
#include <set>
#include <string>

#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/combine_fabric2d_device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/combine_fabric2d_kernel_interface.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine::overlap {

namespace {

namespace cf = ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d;

template <typename Descriptor>
auto find_writer(Descriptor& desc) {
    const auto is_writer = [](const tt::tt_metal::KernelDescriptor& k) {
        return k.kernel_source.ends_with("wave_routed_expert_writer.cpp");
    };
    auto writer = desc.kernels.end();
    for (auto it = desc.kernels.begin(); it != desc.kernels.end(); ++it) {
        if (is_writer(*it)) {
            TT_FATAL(writer == desc.kernels.end(), "routed expert: the program carries two writer kernels");
            writer = it;
        }
    }
    TT_FATAL(writer != desc.kernels.end(), "routed expert: the program carries no writer kernel");
    return writer;
}

// The unified reader's view of where combine's walks open: expert_offsets' row for the ring chip diametrically
// opposite this one, which both of combine's walk directions take first. Appended past every other argument.
void append_far_run(tt::tt_metal::ProgramDescriptor& desc, tt::tt_metal::Buffer* expert_offsets, uint32_t dg_far) {
    auto* reader = std::find_if(desc.kernels.begin(), desc.kernels.end(), [](const auto& k) {
        return k.kernel_source.ends_with("wave_routed_expert_reader.cpp");
    });
    TT_FATAL(reader != desc.kernels.end(), "routed expert: the program carries no reader kernel");
    uint32_t base = 0;
    for (const auto& [core, args] : reader->runtime_args) {
        base = std::max(base, static_cast<uint32_t>(args.size()));
    }
    for (auto& [core, args] : reader->runtime_args) {
        args.resize(base, 0);
        args.push_back(static_cast<uint32_t>(expert_offsets->address()));
        args.push_back(static_cast<uint32_t>(expert_offsets->aligned_page_size()));
        args.push_back(dg_far);
        // The address is re-patched on every program-cache hit: the tensor may live elsewhere next call.
        reader->buffer_bindings.push_back(
            tt::tt_metal::BufferBinding{.core = core, .arg_idx = base, .buffer = expert_offsets});
    }
    reader->defines.emplace_back("URE_FAR_RT_BASE", std::to_string(base));
}

// The writers keep the tile-rows combine takes last in combine's L1 slots, which no kernel of theirs may sit on.
void append_l1_homes(
    tt::tt_metal::ProgramDescriptor& desc, const std::vector<uint32_t>& l1_homes, tt::tt_metal::IDevice* device) {
    std::set<tt::tt_metal::CoreCoord> homes;
    for (std::size_t w = 0; w < l1_homes.size(); w += 2) {
        homes.insert(tt::tt_metal::CoreCoord{l1_homes[w] & 0xFFFF, l1_homes[w] >> 16});
    }
    for (auto& kernel : desc.kernels) {
        for (const auto& range : kernel.core_ranges.ranges()) {
            for (const auto& core : range) {
                TT_FATAL(
                    !homes.contains(device->worker_core_from_logical_core(core)),
                    "routed expert: an L1 slot for the routed expert's output sits on its own core");
            }
        }
        if (kernel.kernel_source.ends_with("wave_routed_expert_writer.cpp")) {
            kernel.defines.emplace_back("URE_L1H_CT_BASE", std::to_string(kernel.compile_time_args.size()));
            kernel.defines.emplace_back("URE_L1H_SLOTS", std::to_string(l1_homes.size() / 2));
            kernel.compile_time_args.insert(kernel.compile_time_args.end(), l1_homes.begin(), l1_homes.end());
        }
    }
}

}  // namespace

uint32_t expert_done_writer_count(const tt::tt_metal::ProgramDescriptor& desc) {
    return static_cast<uint32_t>(find_writer(desc)->core_ranges.num_cores());
}

void append_expert_done_signal(tt::tt_metal::ProgramDescriptor& desc, const ExpertDoneSignal& signal) {
    auto* writer = find_writer(desc);

    // One base for every core, so it can be a define: the widest core's list decides it.
    uint32_t base = 0;
    for (const auto& [core, args] : writer->runtime_args) {
        base = std::max(base, static_cast<uint32_t>(args.size()));
    }
    // A writer core with no arguments of its own still walks every expert and must still report.
    std::set<CoreCoord> carried;
    for (const auto& [core, args] : writer->runtime_args) {
        carried.insert(core);
    }
    for (const auto& range : writer->core_ranges.ranges()) {
        for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; y++) {
            for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; x++) {
                if (!carried.contains(CoreCoord{x, y})) {
                    writer->runtime_args.emplace_back(
                        CoreCoord{x, y}, tt::tt_metal::KernelDescriptor::CoreRuntimeArgs{});
                }
            }
        }
    }
    for (auto& [core, args] : writer->runtime_args) {
        args.resize(base, 0);
        args.push_back(signal.collector_noc_x);
        args.push_back(signal.collector_noc_y);
        args.push_back(signal.collector_counts_addr);
        args.push_back(signal.go_addr);
    }
    writer->defines.emplace_back("HYB_EXPERT_DONE_RT_BASE", std::to_string(base));
}

void merge_routed_expert(
    tt::tt_metal::WorkloadDescriptor& workload,
    const std::map<ttnn::MeshCoordinate, CollectorTarget>& collectors,
    const tt::tt_metal::ProgramDescriptor& re_descriptor,
    const CombineFabric2dParams& combine_args,
    tt::tt_metal::Buffer* expert_offsets,
    uint32_t go_addr) {
    const uint32_t extent = cf::ring_extent(combine_args);
    for (auto& program : workload.programs) {
        TT_FATAL(
            program.range.shape().mesh_size() == 1,
            "routed expert overlapped with combine: combine built one program for {} chips; each chip needs its own "
            "collector",
            program.range.shape().mesh_size());
        const auto coord = program.range.start_coord();
        const auto& collector = collectors.at(coord);
        // NOC coordinates, because the writers address the collector with get_noc_addr.
        const auto collector_noc = combine_args.device->worker_core_from_logical_core(collector.logical);

        auto re = re_descriptor;
        append_expert_done_signal(
            re,
            ExpertDoneSignal{
                .collector_noc_x = static_cast<uint32_t>(collector_noc.x),
                .collector_noc_y = static_cast<uint32_t>(collector_noc.y),
                .collector_counts_addr = collector.counts_addr,
                .go_addr = go_addr});
        append_far_run(re, expert_offsets, (cf::my_dg_index(combine_args, coord) + extent / 2) % extent);
        if (!collector.l1_homes.empty()) {
            append_l1_homes(re, collector.l1_homes, combine_args.device);
        }
        program.descriptor = tt::tt_metal::merge_program_descriptors({program.descriptor, re});
    }
}

void validate_overlap_semaphores(
    const char* op_name,
    ttnn::MeshDevice* device,
    uint32_t combine_axis,
    const std::optional<GlobalSemaphore>& fwd_arrived,
    const std::optional<GlobalSemaphore>& final_arrived,
    const std::optional<GlobalSemaphore>& expert_go,
    const std::vector<GlobalSemaphore>& channel_semaphores,
    uint32_t fwd_arrived_addr,
    uint32_t final_arrived_addr,
    uint32_t expert_go_addr,
    const std::vector<uint32_t>& channel_semaphore_addrs) {
    TT_FATAL(
        fwd_arrived.has_value() && final_arrived.has_value() && expert_go.has_value(),
        "{} overlapped with combine needs the caller's fwd_arrived, final_arrived and expert_go global semaphores, "
        "created once and passed on every call",
        op_name);
    TT_FATAL(
        fwd_arrived->device() == device && final_arrived->device() == device && expert_go->device() == device,
        "{} overlapped with combine: fwd_arrived, final_arrived and expert_go must belong to the same mesh device as "
        "dispatched_buffer",
        op_name);
    TT_FATAL(
        fwd_arrived->address() != final_arrived->address() && fwd_arrived->address() != expert_go->address() &&
            final_arrived->address() != expert_go->address(),
        "{} overlapped with combine: fwd_arrived (0x{:x}), final_arrived (0x{:x}) and expert_go (0x{:x}) count "
        "independently, so they must be three distinct semaphores",
        op_name,
        fwd_arrived->address(),
        final_arrived->address(),
        expert_go->address());
    // Combine bakes these addresses into its compile-time arguments, so the hashed copies must be the live ones.
    TT_FATAL(
        fwd_arrived_addr == fwd_arrived->address() && final_arrived_addr == final_arrived->address() &&
            expert_go_addr == expert_go->address(),
        "{} overlapped with combine: the hashed semaphore addresses do not match the semaphores",
        op_name);
    const uint32_t channels_needed =
        channel_semaphore_count(static_cast<uint32_t>(device->shape()[static_cast<int32_t>(combine_axis)]));
    TT_FATAL(
        channel_semaphores.size() == channels_needed,
        "{} overlapped with combine: {} channel semaphores passed, combine needs {} on this mesh",
        op_name,
        channel_semaphores.size(),
        channels_needed);
    TT_FATAL(
        channel_semaphore_addrs.size() == channel_semaphores.size(),
        "{} overlapped with combine: {} hashed channel addresses for {} semaphores",
        op_name,
        channel_semaphore_addrs.size(),
        channel_semaphores.size());
    // Each counts something different, so a repeat is one counter serving two purposes: not an error anywhere,
    // just wrong numbers. The trio above is already checked pairwise; this folds the channels in with it.
    std::set<uint32_t> distinct{fwd_arrived_addr, final_arrived_addr, expert_go_addr};
    for (uint32_t i = 0; i < channels_needed; i++) {
        TT_FATAL(
            channel_semaphores[i].device() == device,
            "{} overlapped with combine: channel semaphore {} belongs to another mesh device",
            op_name,
            i);
        TT_FATAL(
            channel_semaphore_addrs[i] == channel_semaphores[i].address(),
            "{} overlapped with combine: hashed address of channel semaphore {} does not match the semaphore",
            op_name,
            i);
        TT_FATAL(
            distinct.insert(channel_semaphore_addrs[i]).second,
            "{} overlapped with combine: channel semaphore {} repeats an address (0x{:x}) already in use",
            op_name,
            i,
            channel_semaphore_addrs[i]);
    }
}

void validate_forwarding_buffer(
    const char* op_name,
    const std::optional<ttnn::Tensor>& forwarding_buffer,
    const CombineFabric2dParams& combine_args,
    const CombineFabric2dInputs& combine_inputs) {
    if (!forwarding_buffer.has_value()) {
        return;
    }
    const auto expected = forwarding_buffer_spec(combine_args, cf::token_size_bytes(combine_inputs));
    TT_FATAL(
        forwarding_buffer->tensor_spec() == expected,
        "{} overlapped with combine: forwarding_buffer is {} {} {}, combine needs {} {} {}",
        op_name,
        forwarding_buffer->logical_shape(),
        forwarding_buffer->dtype(),
        forwarding_buffer->memory_config(),
        expected.logical_shape(),
        expected.data_type(),
        expected.memory_config());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine::overlap
