// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d_program_factory.hpp"

#include <algorithm>
#include <memory>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt_stl/assert.hpp>

#include "dispatch_fabric2d_assignments.hpp"
#include "dispatch_fabric2d_placement.hpp"
#include "kernels/dataflow/dispatch_fabric2d_reader_ct_args.hpp"
#include "kernels/dataflow/dispatch_fabric2d_sender_ct_args.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

namespace {

// The parts of L1 that have to sit at the same address on every chip, because a sender on one chip writes
// them on another.
constexpr uint32_t PKT_HDR_DRAIN_OFF = 0x0000;
constexpr uint32_t DRAIN_SINK_OFF = 0x0400;
constexpr uint32_t RING_OFF = 0x1000;
static_assert(PKT_HDR_DRAIN_OFF < DRAIN_SINK_OFF, "drain sink overlaps the drain packet header");
static_assert(DRAIN_SINK_OFF < RING_OFF, "drain sink overlaps the token ring");

uint32_t ring_extent_of(const DispatchFabric2dParams& args) {
    return static_cast<uint32_t>(args.device->shape()[static_cast<int32_t>(args.axis)]);
}

uint32_t token_size_bytes(const DispatchFabric2dInputs& t) {
    return static_cast<uint32_t>(t.input_tensor.buffer()->aligned_page_size());
}

std::vector<uint32_t> ring_chip_ids(ttnn::MeshDevice* mesh, const ttnn::MeshCoordinate& coord, uint32_t axis) {
    const uint32_t extent = static_cast<uint32_t>(mesh->shape()[static_cast<int32_t>(axis)]);
    std::vector<uint32_t> ids(extent);
    for (uint32_t row = 0; row < extent; row++) {
        ttnn::MeshCoordinate c = coord;
        c[static_cast<int32_t>(axis)] = row;
        ids[row] = static_cast<uint32_t>(mesh->get_fabric_node_id(c).chip_id);
    }
    return ids;
}

// The reader's L1 working set: its copy of the control tensors, the 64-byte-padded indices, and the
// routing index it builds from them. Sized for the worst case, which is every token routed to experts
// this chip actually sends.
uint32_t control_region_bytes(const DispatchFabric2dParams& args, uint32_t extent) {
    const uint32_t w = args.num_routed_experts;
    const uint32_t pad_stride =
        dspf2d::META_PAD_STRIDE *
        ((args.num_experts_per_tok * 2 + dspf2d::META_PAD_STRIDE - 1) / dspf2d::META_PAD_STRIDE);
    const uint32_t words = (extent + 2) * w  // expert_offsets rows, counts, region offsets
                           + (w + 1)         // dispatch table, with its trailing sentinel column
                           + w               // the running per-expert allocator
                           +
                           3 * extent * args.experts_per_chip  // chip -> experts inverse, bucket lengths, bucket starts
                           + 2 * args.seq_len_per_chip * args.num_experts_per_tok;  // (token, page) per entry
    return args.seq_len_per_chip * pad_stride + words * static_cast<uint32_t>(sizeof(uint32_t));
}

L1Layout compute_l1_layout(ttnn::MeshDevice* mesh, uint32_t token_bytes, uint32_t control_bytes, uint32_t sem_floor) {
    const uint32_t base =
        static_cast<uint32_t>(mesh->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    L1Layout l;
    l.pkt_hdr_drain = base + PKT_HDR_DRAIN_OFF;
    l.drain_sink = base + DRAIN_SINK_OFF;
    l.ring = base + RING_OFF;
    l.pkt_hdr_ring = l.ring + dspf2d::NUM_L1_SLOTS * (token_bytes + dspf2d::FORWARDING_METADATA_SIZE);
    const uint32_t hdr_ring_bytes =
        2 * dspf2d::NUM_L1_SLOTS * static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    // 64-byte aligned: a DRAM read needs a 64-byte-aligned L1 destination on Blackhole, and the control
    // region is read straight out of DRAM.
    l.control = (l.pkt_hdr_ring + hdr_ring_bytes + 63u) & ~63u;
    const uint32_t end = l.control + control_bytes;
    TT_FATAL(
        end <= sem_floor,
        "dispatch_fabric2d: L1 layout needs {} B (ends at 0x{:x}) but the global-semaphore region starts at "
        "0x{:x}. Reduce seq_len_per_chip ({}) or the token page ({} B).",
        end - base,
        end,
        sem_floor,
        0,
        token_bytes);
    return l;
}

// The reader/sender ring handshake is two monotonic single-writer counters, plus one counter the upstream
// chip's sender bumps as it fills this stream's forwarding region.
//
// GlobalSemaphores rather than the op's own L1 region so they sit at an address uniform across the mesh:
// `fwd_arrived` is bumped by the upstream chip, which has to know where it lives.
//
// Nothing zeroes them between launches -- they outlive the cached workload -- so the kernels reset all
// three at end of stream.
struct RingSemaphores {
    tt::tt_metal::GlobalSemaphore filled;
    tt::tt_metal::GlobalSemaphore freed;
    tt::tt_metal::GlobalSemaphore fwd_arrived;

    uint32_t lowest_address() const {
        return static_cast<uint32_t>(std::min({filled.address(), freed.address(), fwd_arrived.address()}));
    }
};

RingSemaphores allocate_ring_semaphores(ttnn::MeshDevice* mesh) {
    const auto grid = mesh->compute_with_storage_grid_size();
    const CoreRangeSet all_workers(CoreRange(CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}));
    RingSemaphores sems{
        ttnn::global_semaphore::create_global_semaphore(mesh, all_workers, 0, tt::tt_metal::BufferType::L1),
        ttnn::global_semaphore::create_global_semaphore(mesh, all_workers, 0, tt::tt_metal::BufferType::L1),
        ttnn::global_semaphore::create_global_semaphore(mesh, all_workers, 0, tt::tt_metal::BufferType::L1)};
    tt::tt_metal::distributed::Synchronize(mesh, std::nullopt, {});
    return sems;
}

// Never initialised and never read back: pure staging for tokens passing through a chip. One page per
// token, and the page is token + routing tail so a single fabric write lands both.
struct ForwardingBuffer {
    std::shared_ptr<ttnn::Tensor> owner;
    tt::tt_metal::Buffer* buffer = nullptr;
    uint32_t pages_per_stream = 0;
};

ForwardingBuffer allocate_forwarding_buffer(
    ttnn::MeshDevice* mesh, const DispatchFabric2dParams& args, uint32_t token_bytes, uint32_t extent) {
    ForwardingBuffer fwd;
    fwd.pages_per_stream = fwd_pages_per_stream(
        extent, args.num_links, args.seq_len_per_chip, args.num_experts_per_tok, args.experts_per_chip);
    const uint32_t page_bytes = token_bytes + dspf2d::FORWARDING_METADATA_SIZE;
    TT_FATAL(
        page_bytes % 64 == 0, "dispatch_fabric2d: forwarding page {} B must be 64-byte aligned for DRAM", page_bytes);
    const uint32_t pages = fwd.pages_per_stream * stream_count(args.num_links);
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape({pages, page_bytes / static_cast<uint32_t>(sizeof(uint32_t))}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
    // Throws if it does not fit DRAM, which IS the "verify it fits" check.
    fwd.owner = std::make_shared<ttnn::Tensor>(create_device_tensor(spec, mesh));
    fwd.buffer = fwd.owner->buffer();
    TT_FATAL(
        fwd.buffer->aligned_page_size() == page_bytes,
        "dispatch_fabric2d: forwarding page is {} B after alignment but the op addresses it as {} B",
        fwd.buffer->aligned_page_size(),
        page_bytes);
    return fwd;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor DispatchFabric2dProgramFactory::create_workload_descriptor(
    const DispatchFabric2dParams& args,
    const DispatchFabric2dInputs& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/) {
    auto* mesh = args.device;
    const uint32_t extent = ring_extent_of(args);
    const uint32_t token_bytes = token_size_bytes(tensor_args);
    const uint32_t meta_bytes = static_cast<uint32_t>(tensor_return_value[1].buffer()->aligned_page_size());

    validate_chunk_agreement(extent, args.num_links);
    const auto placement = decide_placement(mesh, args.axis, args.num_links);
    const auto sems = allocate_ring_semaphores(mesh);
    const auto fwd = allocate_forwarding_buffer(mesh, args, token_bytes, extent);
    const L1Layout l1 = compute_l1_layout(mesh, token_bytes, control_region_bytes(args, extent), sems.lowest_address());

    tt::tt_metal::Buffer* dram[dspf2d::ReaderRtArg::kCount] = {};
    dram[dspf2d::ReaderRtArg::kInputAddr] = tensor_args.input_tensor.buffer();
    dram[dspf2d::ReaderRtArg::kIndicesAddr] = tensor_args.indices_tensor.buffer();
    dram[dspf2d::ReaderRtArg::kExpertOffsetsAddr] = tensor_args.expert_offsets_tensor.buffer();
    dram[dspf2d::ReaderRtArg::kDispatchTableAddr] = tensor_args.expert_dispatch_table_tensor.buffer();
    dram[dspf2d::ReaderRtArg::kCountsAddr] = tensor_args.expert_token_counts.buffer();
    dram[dspf2d::ReaderRtArg::kRegionOffsetsAddr] = tensor_args.expert_region_offsets.buffer();
    dram[dspf2d::ReaderRtArg::kOutPayloadAddr] = tensor_return_value[0].buffer();
    dram[dspf2d::ReaderRtArg::kOutMetaAddr] = tensor_return_value[1].buffer();
    dram[dspf2d::ReaderRtArg::kFwdAddr] = fwd.buffer;
    for (uint32_t i = 0; i < dspf2d::ReaderRtArg::kCount; i++) {
        TT_FATAL(dram[i] != nullptr, "dispatch_fabric2d: buffer for runtime arg {} is not allocated", i);
    }

    tt::tt_metal::WorkloadDescriptor workload;
    workload.semaphores.push_back(sems.filled);
    workload.semaphores.push_back(sems.freed);
    workload.semaphores.push_back(sems.fwd_arrived);
    workload.buffers.push_back({fwd.owner, fwd.buffer});

    for (const auto& coord : ttnn::MeshCoordinateRange(mesh->shape())) {
        const uint32_t row = static_cast<uint32_t>(coord[static_cast<int32_t>(args.axis)]);
        const auto chip_ids = ring_chip_ids(mesh, coord, args.axis);
        const auto work_by_stream = generate_assignments(chip_ids, row, args.num_links);
        const uint32_t linearized = ccl::common::get_linearized_index(coord, mesh->get_view());

        tt::tt_metal::ProgramDescriptor desc;
        for (const auto& [stream, self] : placement.at(coord)) {
            const auto& downstream = placement.at(self.downstream_coord).at(stream);
            const auto& work = work_by_stream.at(stream);

            KernelPlan plan;
            plan.stream = stream;
            plan.extent = extent;
            plan.fwd_pages_per_stream = fwd.pages_per_stream;
            plan.ring_filled_addr = static_cast<uint32_t>(sems.filled.address());
            plan.ring_freed_addr = static_cast<uint32_t>(sems.freed.address());
            plan.fwd_arrived_addr = static_cast<uint32_t>(sems.fwd_arrived.address());

            tt::tt_metal::KernelDescriptor snd;
            snd.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch_fabric2d/device/kernels/dataflow/"
                "sender_dispatch_fabric2d.cpp";
            snd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
            snd.core_ranges = CoreRangeSet(CoreRange(self.worker_logical));
            snd.compile_time_args =
                dspf2d::SenderCtArgs(token_bytes, meta_bytes, self, downstream, l1, plan).to_ct_word_arr();
            snd.config = tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                // NOC_1 routes -Y first, so a worker one row from its eth core reaches it in a single hop.
                .noc = tt::tt_metal::NOC::NOC_1,
            };
            auto snd_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
            desc.kernels.push_back(std::move(snd));

            uint32_t own_count = 0;
            std::vector<uint32_t> assignment_words;
            std::vector<uint32_t> schedule;
            for (const auto& a : work) {
                if (a.is_relay) {
                    schedule.push_back(dspf2d::SCHED_FWD | a.relay_chunk);
                    continue;
                }
                schedule.push_back(own_count++);
                assignment_words.push_back(a.dst_chip_id);
                assignment_words.push_back(a.dst_row);
                assignment_words.push_back(a.split_idx);
                assignment_words.push_back(a.split_count);
            }

            tt::tt_metal::KernelDescriptor rdr;
            rdr.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch_fabric2d/device/kernels/dataflow/"
                "reader_dispatch_fabric2d.cpp";
            rdr.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
            rdr.core_ranges = CoreRangeSet(CoreRange(self.worker_logical));
            rdr.compile_time_args = dspf2d::ReaderCtArgs(
                                        args,
                                        token_bytes,
                                        meta_bytes,
                                        linearized,
                                        row,
                                        static_cast<uint32_t>(mesh->get_fabric_node_id(coord).chip_id),
                                        static_cast<uint32_t>(self.downstream_node.chip_id),
                                        l1,
                                        plan,
                                        own_count,
                                        static_cast<uint32_t>(schedule.size()) - own_count)
                                        .to_ct_word_arr(chip_ids, assignment_words, schedule);
            for (uint32_t i = 0; i < dspf2d::ReaderRtArg::kCount; i++) {
                tt::tt_metal::TensorAccessorArgs(dram[i]).append_to(rdr.compile_time_args);
            }
            rdr.config = tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::NOC_0,
            };

            // Buffer* so the framework records BufferBindings and rewrites them per dispatch: an address
            // describes an allocation, not a program, and a cached program must not carry a stale one.
            tt::tt_metal::KernelDescriptor::RTArgList rdr_rt;
            for (uint32_t i = 0; i < dspf2d::ReaderRtArg::kCount; i++) {
                rdr_rt.push_back(dram[i]);
            }
            rdr.emplace_runtime_args(self.worker_logical, rdr_rt);
            desc.kernels.push_back(std::move(rdr));

            std::vector<uint32_t> snd_rt{1u};  // num_connections
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                mesh->get_fabric_node_id(coord),
                std::vector<tt::tt_fabric::FabricNodeId>{self.downstream_node},
                std::vector<uint32_t>{self.link_idx},
                desc,
                snd_id,
                self.worker_logical,
                snd_rt);
            // That call fills the vector; it does not attach it. Without this the sender reads an empty
            // runtime-arg list, takes garbage as its connection count and blocks forever opening.
            tt::tt_metal::KernelDescriptor::RTArgList snd_rt_list;
            snd_rt_list.append(snd_rt);
            desc.kernels[snd_id].emplace_runtime_args(self.worker_logical, snd_rt_list);
        }
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(desc)});
    }
    return workload;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
