// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d_program_factory.hpp"

#include <algorithm>
#include <memory>
#include <string_view>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
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
#include "dispatch_fabric2d_untilize.hpp"
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

// One token, in the ring, in a fabric packet, in staging, and in a forwarding page. Taken from the
// output payload buffer because that is the one tensor whose page IS a token in both input layouts --
// a TILE input is paged by tile, not by token.
uint32_t token_size_bytes(const ttnn::Tensor& out_payload) {
    return static_cast<uint32_t>(out_payload.buffer()->aligned_page_size());
}

bool input_is_tiled(const DispatchFabric2dInputs& t) {
    return t.input_tensor.layout() == tt::tt_metal::Layout::TILE;
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
//
// The block sizes live in the kernel interface, because the kernel's carve reads the same list. This
// used to be an independent sum here, and twice it fell behind the carve -- overrunning the control
// region into the global semaphores, with a green build both times.
dspf2d::ControlGeometry control_geometry(const DispatchFabric2dParams& args, uint32_t extent) {
    return dspf2d::ControlGeometry{
        .seq_len = args.seq_len_per_chip,
        .indices_pad_stride = dspf2d::META_PAD_STRIDE *
                              ((args.num_experts_per_tok * 2 + dspf2d::META_PAD_STRIDE - 1) / dspf2d::META_PAD_STRIDE),
        .extent = extent,
        .num_routed_experts = args.num_routed_experts,
        .experts_per_chip = args.experts_per_chip,
        .topk = args.num_experts_per_tok,
        .num_relay = relay_chunks_per_stream(extent),
        .fanout = args.fanout ? 1u : 0u};
}

L1Layout compute_l1_layout(
    ttnn::MeshDevice* mesh, uint32_t token_bytes, const dspf2d::ControlGeometry& g, uint32_t sem_floor) {
    const bool fanout = g.fanout != 0u;
    const uint32_t control_bytes = dspf2d::control_region_bytes(g);
    const uint32_t base =
        static_cast<uint32_t>(mesh->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    L1Layout l;
    l.pkt_hdr_drain = base + PKT_HDR_DRAIN_OFF;
    l.drain_sink = base + DRAIN_SINK_OFF;
    l.ring = base + RING_OFF;
    l.pkt_hdr_ring = l.ring + dspf2d::NUM_L1_SLOTS * (token_bytes + dspf2d::FORWARDING_METADATA_SIZE);
    // The same expression the sender indexes the pool with; the reason they must agree is stated there.
    const uint32_t hdr_ring_bytes = dspf2d::headers_per_slot(fanout) * dspf2d::NUM_L1_SLOTS *
                                    static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    // Both RISCs address these, so they cannot live in the reader's control carve. Sized to nothing
    // under unicast so its layout is untouched.
    l.mc_delivery = (l.pkt_hdr_ring + hdr_ring_bytes + 63u) & ~63u;
    const uint32_t delivery_bytes =
        fanout ? dspf2d::NUM_L1_SLOTS * dspf2d::FO_MAX_DESTS * static_cast<uint32_t>(sizeof(dspf2d::FanoutDelivery))
               : 0u;
    l.mc_meta = (l.mc_delivery + delivery_bytes + 63u) & ~63u;
    const uint32_t meta_bytes = fanout ? dspf2d::NUM_L1_SLOTS * dspf2d::FO_MAX_DESTS * dspf2d::MC_META_SLOT_BYTES : 0u;
    // 64-byte aligned: a DRAM read needs a 64-byte-aligned L1 destination on Blackhole, and the control
    // region is read straight out of DRAM.
    l.control = (l.mc_meta + meta_bytes + 63u) & ~63u;
    const uint32_t end = l.control + control_bytes;
    // Naming every driver rather than one: the ring scales with the token page, while the control
    // region's blocks divide between those that scale with the sequence and those that scale with the
    // expert count and the ring extent, and which of them overflowed is not something the caller can
    // infer from a single total.
    TT_FATAL(
        end <= sem_floor,
        "dispatch_fabric2d: the L1 layout needs {} B, ending at 0x{:x}, but the global semaphores start at 0x{:x}. "
        "The token ring is {} B of it at a {} B token page; the control region is {} B and grows with "
        "seq_len_per_chip={}, num_routed_experts={}, extent={}, experts_per_chip={} and topk={}.",
        end - base,
        end,
        sem_floor,
        dspf2d::NUM_L1_SLOTS * (token_bytes + dspf2d::FORWARDING_METADATA_SIZE),
        token_bytes,
        control_bytes,
        g.seq_len,
        g.num_routed_experts,
        g.extent,
        g.experts_per_chip,
        g.topk);
    return l;
}

// Four counters: the reader/sender ring handshake is two monotonic single-writer ones, `fwd_arrived`
// is bumped by the upstream chip's sender as it fills this stream's forwarding region, and
// `untilized` is bumped by this chip's untilizer writers as they land stripes in staging.
//
// GlobalSemaphores rather than the op's own L1 region so they sit at an address uniform across the mesh:
// `fwd_arrived` is bumped by the upstream chip, which has to know where it lives.
//
// Nothing zeroes them between launches -- they outlive the cached workload -- so the kernels reset all
// four at end of stream, `untilized` only where there is a pool to have bumped it.
struct RingSemaphores {
    tt::tt_metal::GlobalSemaphore filled;
    tt::tt_metal::GlobalSemaphore freed;
    tt::tt_metal::GlobalSemaphore fwd_arrived;
    // Stripes the untilizer pool has landed in staging. Allocated whatever the input layout, so the
    // reader's argument list does not depend on it -- the row-major path never reads it.
    tt::tt_metal::GlobalSemaphore untilized;

    uint32_t lowest_address() const {
        return static_cast<uint32_t>(
            std::min({filled.address(), freed.address(), fwd_arrived.address(), untilized.address()}));
    }
};

RingSemaphores allocate_ring_semaphores(ttnn::MeshDevice* mesh, const CoreRangeSet& universe) {
    const auto make = [&] {
        return ttnn::global_semaphore::create_global_semaphore(mesh, universe, 0, tt::tt_metal::BufferType::L1);
    };
    RingSemaphores sems{make(), make(), make(), make()};
    tt::tt_metal::distributed::Synchronize(mesh, std::nullopt, {});
    return sems;
}

// A device buffer the op allocates for itself, never initialises and never reads back on the host. The
// workload holds the owner so it survives a program-cache hit, which is what lets the kernels address
// it by a runtime argument the framework rewrites per dispatch.
struct OwnedScratch {
    std::shared_ptr<ttnn::Tensor> owner;
    tt::tt_metal::Buffer* buffer = nullptr;
};

// Typed UINT32 rather than by what the pages hold, so that a page is EXACTLY page_bytes rather than
// that rounded up to an alignment: every one of these buffers is addressed by page index from a kernel
// that computed the index itself, and a page wider than it thinks would shear the whole buffer.
OwnedScratch allocate_scratch(
    ttnn::MeshDevice* mesh, uint32_t num_pages, uint32_t page_bytes, std::string_view what) {
    TT_FATAL(page_bytes % 64 == 0, "dispatch_fabric2d: {} page {} B must be 64-byte aligned for DRAM", what, page_bytes);
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape({num_pages, page_bytes / static_cast<uint32_t>(sizeof(uint32_t))}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
    OwnedScratch scratch;
    // Throws if it does not fit DRAM, which IS the "verify it fits" check.
    scratch.owner = std::make_shared<ttnn::Tensor>(create_device_tensor(spec, mesh));
    scratch.buffer = scratch.owner->buffer();
    TT_FATAL(
        scratch.buffer->aligned_page_size() == page_bytes,
        "dispatch_fabric2d: {} page is {} B after alignment but the op addresses it as {} B",
        what,
        scratch.buffer->aligned_page_size(),
        page_bytes);
    return scratch;
}

// Pages one stream may put through its slice of the forwarding region.
//
// Fan-out puts at most one page per token per direction through a region rather than one per
// (token, expert) pair per destination, so its bound is a different expression, not a scaling of the
// other. Loose by a whole chunk under the terminal rule, since the last of a stream's m chunks is
// identically empty -- deliberately not tightened: this bound is the only thing standing between a
// stream and its neighbour's slice of a shared tensor, and the kernel's own check of it is an ASSERT
// that is compiled out on this hardware.
uint32_t fwd_pages_for(const DispatchFabric2dParams& args, uint32_t extent) {
    return args.fanout ? mc_fwd_pages_per_stream(extent, args.num_links, args.seq_len_per_chip)
                       : fwd_pages_per_stream(
                             extent, args.num_links, args.seq_len_per_chip, args.num_experts_per_tok,
                             args.experts_per_chip);
}

}  // namespace

tt::tt_metal::WorkloadDescriptor DispatchFabric2dProgramFactory::create_workload_descriptor(
    const DispatchFabric2dParams& args,
    const DispatchFabric2dInputs& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/) {
    auto* mesh = args.device;
    const uint32_t extent = ring_extent_of(args);
    const uint32_t token_bytes = token_size_bytes(tensor_return_value[0]);
    const uint32_t meta_bytes = static_cast<uint32_t>(tensor_return_value[1].buffer()->aligned_page_size());
    TT_FATAL(
        meta_bytes >= dspf2d::METADATA_WIRE_BYTES,
        "dispatch_fabric2d: metadata page is {} B but the last hop writes {} B into it",
        meta_bytes,
        dspf2d::METADATA_WIRE_BYTES);
    const bool tiled = input_is_tiled(tensor_args);
    if (!tiled) {
        // The row-major path reads tokens straight out of the input, so the two pages have to be the
        // same size; they are both one row of the same emb_dim in the same dtype.
        TT_FATAL(
            tensor_args.input_tensor.buffer()->aligned_page_size() == token_bytes,
            "dispatch_fabric2d: a ROW_MAJOR input pages a token at {} B but the output pages it at {} B",
            tensor_args.input_tensor.buffer()->aligned_page_size(),
            token_bytes);
    }

    validate_chunk_agreement(extent, args.num_links);
    const auto placement = decide_placement(mesh, args.axis, args.num_links, args.worker_core_range_set);
    const auto sems = allocate_ring_semaphores(mesh, args.worker_core_range_set);
    // One page per token passing through a chip, and the page is token + routing tail so a single
    // fabric write lands both.
    const uint32_t fwd_pages = fwd_pages_for(args, extent);
    const OwnedScratch fwd = allocate_scratch(
        mesh,
        fwd_pages * stream_count(args.num_links),
        token_bytes + dspf2d::FORWARDING_METADATA_SIZE,
        "forwarding");
    // Only under TILE: where a tiled input's tokens end up, one row-major page each, so the stream
    // cores address a token by page index exactly as they do a row-major input. The row-major path
    // allocates nothing and runs the program it always has.
    const OwnedScratch staging =
        tiled ? allocate_scratch(mesh, args.seq_len_per_chip, token_bytes, "staging") : OwnedScratch{};
    const auto untilize = plan_untilize(
        tensor_args.input_tensor,
        tensor_return_value[0],
        args.seq_len_per_chip,
        token_bytes,
        static_cast<uint32_t>(sems.untilized.address()),
        staging.buffer);
    const L1Layout l1 = compute_l1_layout(mesh, token_bytes, control_geometry(args, extent), sems.lowest_address());

    tt::tt_metal::Buffer* dram[dspf2d::ReaderRtArg::kCount] = {};
    // Under TILE the tokens reach the stream cores through staging, and the accessor arguments are
    // chained off this same table -- so pointing it at staging is the whole of what the transport
    // needs to know about the input layout.
    dram[dspf2d::ReaderRtArg::kInputAddr] = tiled ? staging.buffer : tensor_args.input_tensor.buffer();
    dram[dspf2d::ReaderRtArg::kIndicesAddr] = tensor_args.indices_tensor.buffer();
    dram[dspf2d::ReaderRtArg::kExpertOffsetsAddr] = tensor_args.expert_offsets_tensor.buffer();
    dram[dspf2d::ReaderRtArg::kDispatchTableAddr] = tensor_args.expert_dispatch_table_tensor.buffer();
    dram[dspf2d::ReaderRtArg::kCountsAddr] = tensor_args.expert_token_counts.buffer();
    dram[dspf2d::ReaderRtArg::kRegionOffsetsAddr] = tensor_args.expert_region_offsets.buffer();
    dram[dspf2d::ReaderRtArg::kOutPayloadAddr] = tensor_return_value[0].buffer();
    dram[dspf2d::ReaderRtArg::kOutMetaAddr] = tensor_return_value[1].buffer();
    dram[dspf2d::ReaderRtArg::kFwdAddr] = fwd.buffer;
    // Always bound. Non-fanout mode has no reach table, and leaving the slot null would mean a
    // different runtime-arg layout per mode -- the host/kernel drift that is the hardest class of bug
    // here. The stand-in is never read: the reader only touches it under the fanout compile-time arg.
    dram[dspf2d::ReaderRtArg::kFanoutReachAddr] = tensor_args.fanout_reach.has_value()
                                                      ? tensor_args.fanout_reach->buffer()
                                                      : tensor_args.expert_offsets_tensor.buffer();
    for (uint32_t i = 0; i < dspf2d::ReaderRtArg::kCount; i++) {
        TT_FATAL(dram[i] != nullptr, "dispatch_fabric2d: buffer for runtime arg {} is not allocated", i);
    }

    tt::tt_metal::WorkloadDescriptor workload;
    workload.semaphores.push_back(sems.filled);
    workload.semaphores.push_back(sems.freed);
    workload.semaphores.push_back(sems.fwd_arrived);
    workload.semaphores.push_back(sems.untilized);
    workload.buffers.push_back({fwd.owner, fwd.buffer});
    if (tiled) {
        workload.buffers.push_back({staging.owner, staging.buffer});
    }

    // Stripes the stream readers wait for before their first token read; zero is the row-major path,
    // where the wait compiles out and there is no pool.
    const uint32_t untilize_stripes = untilize.has_value() ? untilize->num_stripes : 0u;

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
            plan.fwd_pages_per_stream = fwd_pages;
            plan.ring_filled_addr = static_cast<uint32_t>(sems.filled.address());
            plan.ring_freed_addr = static_cast<uint32_t>(sems.freed.address());
            plan.fwd_arrived_addr = static_cast<uint32_t>(sems.fwd_arrived.address());
            plan.untilize_sem_addr = static_cast<uint32_t>(sems.untilized.address());
            plan.untilize_stripes = untilize_stripes;

            tt::tt_metal::KernelDescriptor snd;
            snd.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch_fabric2d/device/kernels/dataflow/"
                "sender_dispatch_fabric2d.cpp";
            snd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
            snd.core_ranges = CoreRangeSet(CoreRange(self.worker_logical));
            snd.compile_time_args =
                dspf2d::SenderCtArgs(token_bytes, meta_bytes, self, downstream, l1, plan, args.fanout).to_ct_word_arr();
            snd.config = tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                // NOC_1 routes -Y first, so a worker one row from its eth core reaches it in a single hop.
                .noc = tt::tt_metal::NOC::NOC_1,
            };
            auto snd_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
            desc.kernels.push_back(std::move(snd));

            uint32_t own_count = 0;
            std::vector<uint32_t> assignment_words;
            for (const auto& a : work) {
                if (a.is_relay) {
                    continue;
                }
                own_count++;
                assignment_words.push_back(a.dst_chip_id);
                assignment_words.push_back(a.dst_row);
                assignment_words.push_back(a.split_idx);
                assignment_words.push_back(a.split_count);
            }

            const auto to_words = [](const std::vector<dspf2d::ChunkDescriptor>& cs) {
                std::vector<uint32_t> w;
                w.reserve(cs.size() * dspf2d::ASSIGNMENT_WORDS);
                for (const auto& d : cs) {
                    w.push_back(d.origin_row);
                    w.push_back(d.dst_row);
                    w.push_back(d.split_idx);
                    w.push_back(d.split_count);
                }
                return w;
            };
            const auto in_words = to_words(forwarding_chunks(stream, row, extent, args.num_links));
            const auto out_words = to_words(outgoing_chunks(stream, row, extent, args.num_links));

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
                                        static_cast<uint32_t>(work.size()) - own_count)
                                        .to_ct_word_arr(chip_ids, assignment_words, in_words, out_words);
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
        if (untilize.has_value()) {
            add_untilizer_pool(desc, placement.at(coord), args.worker_core_range_set, *untilize);
        }
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(desc)});
    }
    return workload;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
