// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d_program_factory.hpp"

#include <algorithm>
#include <array>
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
#include <tt-metalium/math.hpp>
#include <tt-logger/tt-logger.hpp>
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
// them on another. The signal header needs one fabric packet header (checked in compute_l1_layout) and the
// drain sink one word; the gaps are slack.
constexpr uint32_t PKT_HDR_SIGNAL_OFF = 0x0000;
// The courier's own header. A credit can be posted from wherever the sender happens to be, including from
// inside the wait for its reader, so it must not share a header with an arrival bump that may be in flight.
constexpr uint32_t PKT_HDR_CREDIT_OFF = 0x0200;
constexpr uint32_t DRAIN_SINK_OFF = 0x0400;
constexpr uint32_t QUEUE_OFF = 0x1000;
static_assert(PKT_HDR_SIGNAL_OFF < PKT_HDR_CREDIT_OFF, "the credit header overlaps the signal header");
static_assert(PKT_HDR_CREDIT_OFF < DRAIN_SINK_OFF, "drain sink overlaps the credit packet header");
static_assert(DRAIN_SINK_OFF < QUEUE_OFF, "drain sink overlaps the token queue");

uint32_t ring_extent_of(const DispatchFabric2dParams& args) {
    return static_cast<uint32_t>(args.device->shape()[static_cast<int32_t>(args.axis)]);
}

// Bytes of one token wherever the op moves it. Taken from the output payload, whose page is one token in
// both input layouts; a TILE input is paged by tile.
uint32_t token_size_bytes(const ttnn::Tensor& out_payload) {
    return static_cast<uint32_t>(out_payload.buffer()->aligned_page_size());
}

bool input_is_tiled(const DispatchFabric2dInputs& t) { return t.input_tensor.layout() == tt::tt_metal::Layout::TILE; }

std::vector<uint32_t> ring_chip_ids(ttnn::MeshDevice* mesh, const ttnn::MeshCoordinate& coord, uint32_t axis) {
    const uint32_t extent = static_cast<uint32_t>(mesh->shape()[static_cast<int32_t>(axis)]);
    std::vector<uint32_t> ids(extent);
    for (uint32_t pos = 0; pos < extent; pos++) {
        ttnn::MeshCoordinate c = coord;
        c[static_cast<int32_t>(axis)] = pos;
        ids[pos] = static_cast<uint32_t>(mesh->get_fabric_node_id(c).chip_id);
    }
    return ids;
}

// The reader's scratch: its copy of the control tensors, the 64-byte-padded indices, and the routing index.
// Sized for the worst case, every token routed to experts this chip sends. The block sizes come from the
// kernel interface so the host and the kernel lay out the scratch the same way.
dspf2d::ScratchGeometry scratch_geometry(const DispatchFabric2dParams& args, uint32_t extent) {
    return dspf2d::ScratchGeometry{
        .seq_len = args.seq_len_per_chip,
        .indices_pad_stride = dspf2d::META_PAD_STRIDE *
                              ((args.num_experts_per_tok * 2 + dspf2d::META_PAD_STRIDE - 1) / dspf2d::META_PAD_STRIDE),
        .extent = extent,
        .num_routed_experts = args.num_routed_experts,
        .experts_per_chip = args.experts_per_chip,
        .topk = args.num_experts_per_tok,
        .num_forward = forward_descriptors_per_stream(extent)};
}

L1Layout compute_l1_layout(
    ttnn::MeshDevice* mesh, uint32_t token_bytes, const dspf2d::ScratchGeometry& g, uint32_t sem_floor) {
    const uint32_t scratch_bytes = dspf2d::scratch_bytes(g);
    const uint32_t base =
        static_cast<uint32_t>(mesh->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    L1Layout l;
    l.pkt_hdr_signal = base + PKT_HDR_SIGNAL_OFF;
    l.pkt_hdr_credit = base + PKT_HDR_CREDIT_OFF;
    l.drain_sink = base + DRAIN_SINK_OFF;
    l.queue = base + QUEUE_OFF;
    l.pkt_hdr_queue = l.queue + dspf2d::QUEUE_DEPTH * (token_bytes + dspf2d::FORWARDING_METADATA_SIZE);
    const uint32_t hdr_bytes = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    TT_FATAL(
        PKT_HDR_SIGNAL_OFF + hdr_bytes <= PKT_HDR_CREDIT_OFF && PKT_HDR_CREDIT_OFF + hdr_bytes <= DRAIN_SINK_OFF,
        "dispatch_fabric2d: a {} B fabric packet header does not fit between the signal, credit and drain slots",
        hdr_bytes);
    // One header per entry, the same stride the sender indexes the pool with.
    const uint32_t hdr_queue_bytes = dspf2d::QUEUE_DEPTH * hdr_bytes;
    // 64-byte aligned: a DRAM read needs a 64-byte-aligned L1 destination on Blackhole, and DRAM reads
    // land directly in the scratch.
    l.scratch = (l.pkt_hdr_queue + hdr_queue_bytes + 63u) & ~63u;
    const uint32_t end = l.scratch + scratch_bytes;
    TT_FATAL(
        end <= sem_floor,
        "dispatch_fabric2d: the L1 layout needs {} B, ending at 0x{:x}, but the global semaphores start at 0x{:x}. "
        "The token queue is {} B of it at a {} B token page; the scratch is {} B and grows with "
        "seq_len_per_chip={}, num_routed_experts={}, extent={}, experts_per_chip={} and topk={}.",
        end - base,
        end,
        sem_floor,
        dspf2d::QUEUE_DEPTH * (token_bytes + dspf2d::FORWARDING_METADATA_SIZE),
        token_bytes,
        scratch_bytes,
        g.seq_len,
        g.num_routed_experts,
        g.extent,
        g.experts_per_chip,
        g.topk);
    return l;
}

// The fixed counters: `filled` and `freed` are the reader/sender queue handshake, `final_arrived` is raised by
// the upstream chip's sender for every token it writes straight into this chip's output, `launch_credit` is
// raised by the downstream chip when it starts a launch, and `untilized` is raised by this chip's untilizer
// writers as they write tile rows to staging. The per-channel counters are described below.
//
// The launch credit crosses direction: our stream s sender grants it to the downstream chip's reverse stream
// core, and our own `launch_credit` is granted by that core's sender. Our sender takes one before its first
// send, so nothing it writes reaches the downstream chip before that chip is in the same launch.
//
// `final_arrived` is the receive count for the last hop. Nothing else on this chip sees those tokens land, so
// without it this chip's program can finish -- and the next op on it read the output -- while the upstream
// chip is still delivering. The reader waits for the count it expects before exiting.
//
// GlobalSemaphores so they sit at the same address on every chip: a neighbouring chip signals the arrival
// and credit counters and has to know their address.
//
// Nothing resets them between launches, so the kernels undo each launch's count. `filled` and `freed` are
// set to zero at end of run, and `untilized` too for a TILE input. Each channel's arrival count is lowered
// by what this reader consumed and its credit count by what the reader placed downstream, and
// `final_arrived` by the count this launch expected, which stays right without relying on the launch
// credit. `launch_credit` is lowered by the one credit taken at the start of the send loop, because the
// downstream chip can grant the next launch's credit before this launch ends.
// The forwarding counters are per channel, because a reader serves each channel separately -- that is what
// lets it drain one while another is blocked:
//
//   `vc_arrived[v]`  the upstream chip saying it put pages in our channel v. Travels with the data.
//   `vc_freed[v]`    the DOWNSTREAM chip saying it took pages out of ITS channel v, so we may write there
//                    again. It cannot tell us directly -- a stream's fabric connection only points
//                    downstream -- so the credit rides the opposite direction's sender on this chip, which
//                    already points that way.
//   `vc_outbox[v]`   that courier's queue, filled across this chip by the opposite stream's reader.
//
// `credit_done` is that reader telling the courier it will queue nothing more, so the courier knows when it
// may stop.
struct StreamSemaphores {
    tt::tt_metal::GlobalSemaphore filled;
    tt::tt_metal::GlobalSemaphore freed;
    tt::tt_metal::GlobalSemaphore final_arrived;
    tt::tt_metal::GlobalSemaphore launch_credit;
    // Allocated for both input layouts so the reader's argument list is the same; a row-major input never
    // uses it.
    tt::tt_metal::GlobalSemaphore untilized;
    tt::tt_metal::GlobalSemaphore credit_done;
    std::vector<tt::tt_metal::GlobalSemaphore> vc_arrived;
    std::vector<tt::tt_metal::GlobalSemaphore> vc_freed;
    std::vector<tt::tt_metal::GlobalSemaphore> vc_outbox;

    uint32_t lowest_address() const {
        uint32_t lowest = static_cast<uint32_t>(std::min(
            {filled.address(),
             freed.address(),
             final_arrived.address(),
             launch_credit.address(),
             untilized.address(),
             credit_done.address()}));
        for (const auto* group : {&vc_arrived, &vc_freed, &vc_outbox}) {
            for (const auto& sem : *group) {
                lowest = std::min(lowest, static_cast<uint32_t>(sem.address()));
            }
        }
        return lowest;
    }
};

StreamSemaphores allocate_stream_semaphores(ttnn::MeshDevice* mesh, const CoreRangeSet& allowed_cores, uint32_t vcs) {
    const auto make = [&] {
        return ttnn::global_semaphore::create_global_semaphore(mesh, allowed_cores, 0, tt::tt_metal::BufferType::L1);
    };
    // filled, freed, final_arrived, launch_credit, untilized, credit_done; the three vectors fill in below.
    StreamSemaphores sems{make(), make(), make(), make(), make(), make(), {}, {}, {}};
    for (uint32_t v = 0; v < vcs; v++) {
        sems.vc_arrived.push_back(make());
        sems.vc_freed.push_back(make());
        sems.vc_outbox.push_back(make());
    }
    tt::tt_metal::distributed::Synchronize(*mesh, std::nullopt, {});
    return sems;
}

// Typed UINT32 so a page is exactly page_bytes, with no alignment rounding. Kernels address these buffers
// by a page index they compute themselves, so a wider page would misplace every page after the first.
tt::tt_metal::TensorSpec dram_scratch_spec(uint32_t num_pages, uint32_t page_bytes, std::string_view what) {
    TT_FATAL(
        page_bytes % 64 == 0, "dispatch_fabric2d: {} page {} B must be 64-byte aligned for DRAM", what, page_bytes);
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({num_pages, page_bytes / static_cast<uint32_t>(sizeof(uint32_t))}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

// DRAM the op keeps tokens in between reads and writes: the caller's buffer if it passed one (validated with the
// other inputs), else one the op allocates for itself, never initialises and never reads back on the host. The
// workload holds an owned buffer so it lives as long as the cached program; the kernels get either kind's address
// as a runtime argument the framework updates on each dispatch.
struct DramScratch {
    std::shared_ptr<ttnn::Tensor> owner;  // null for the caller's buffer
    tt::tt_metal::Buffer* buffer = nullptr;
};

DramScratch dram_scratch(
    ttnn::MeshDevice* mesh, const tt::tt_metal::TensorSpec& spec, const std::optional<ttnn::Tensor>& provided) {
    DramScratch scratch;
    if (provided.has_value()) {
        scratch.buffer = provided->buffer();
    } else {
        // Throws if it does not fit in DRAM.
        scratch.owner = std::make_shared<ttnn::Tensor>(create_device_tensor(spec, mesh));
        scratch.buffer = scratch.owner->buffer();
        TT_FATAL(
            scratch.buffer->aligned_page_size() == spec.compute_page_size_bytes(),
            "dispatch_fabric2d: scratch page is {} B after alignment but the op addresses it as {} B",
            scratch.buffer->aligned_page_size(),
            spec.compute_page_size_bytes());
    }
    return scratch;
}

}  // namespace

uint32_t token_page_bytes(ttnn::MeshDevice* mesh, uint32_t emb_dim) {
    return tt::round_up(
        emb_dim * static_cast<uint32_t>(sizeof(uint16_t)),
        mesh->allocator()->get_alignment(tt::tt_metal::BufferType::DRAM));
}

tt::tt_metal::TensorSpec forwarding_buffer_spec(const DispatchFabric2dParams& args, uint32_t token_size_bytes) {
    // A page is a token plus its fwd_meta, so one fabric write carries both.
    const uint32_t pages_per_stream = fwd_pages_per_stream(ring_extent_of(args), args.fwd_vc_pages);
    return dram_scratch_spec(
        pages_per_stream * stream_count(args.num_links),
        token_size_bytes + dspf2d::FORWARDING_METADATA_SIZE,
        "forwarding");
}

tt::tt_metal::TensorSpec staging_buffer_spec(const DispatchFabric2dParams& args, uint32_t token_size_bytes) {
    // One row-major page per untilized token, so the stream cores address a token by page index as they do for a
    // row-major input. Rounded up to whole tile rows because the untilizer packs all 32 rows of a tile; the extra
    // pages hold padding rows and are never read.
    return dram_scratch_spec(
        tt::round_up(args.seq_len_per_chip, static_cast<uint32_t>(tt::constants::TILE_HEIGHT)),
        token_size_bytes,
        "staging");
}

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
    // A forwarded packet (token plus fwd_meta) is the largest packet the op sends. The fabric does
    // not reject a larger payload: the bytes past its max overwrite the next channel slot.
    TT_FATAL(
        token_bytes + dspf2d::FWD_EXTRA_BYTES <= tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(),
        "dispatch_fabric2d: token page {} B + {} B forwarding metadata exceeds the fabric max payload {}. Increase "
        "max_packet_payload_size_bytes in FabricRouterConfig.",
        token_bytes,
        dspf2d::FWD_EXTRA_BYTES,
        tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
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

    validate_descriptor_agreement(extent, args.num_links);
    const auto placement = decide_placement(mesh, args.axis, args.num_links, args.worker_core_range_set);
    const uint32_t vcs = num_vcs(extent);
    TT_FATAL(
        vcs <= dspf2d::MAX_VCS,
        "dispatch_fabric2d: a {}-chip ring needs {} forwarding channels but the kernels carry arrays for {}",
        extent,
        vcs,
        dspf2d::MAX_VCS);
    const auto sems = allocate_stream_semaphores(mesh, args.worker_core_range_set, vcs);
    // Every channel is the same depth, so a stream's section is vcs channels of fwd_vc_pages.
    const uint32_t vc_pages = args.fwd_vc_pages;
    const uint32_t fwd_pages = fwd_pages_per_stream(extent, vc_pages);
    TT_FATAL(
        token_bytes == token_page_bytes(mesh, static_cast<uint32_t>(tensor_args.input_tensor.logical_shape()[-1])),
        "dispatch_fabric2d: the output pages a token at {} B, not the {} B the staging buffers were checked against",
        token_bytes,
        token_page_bytes(mesh, static_cast<uint32_t>(tensor_args.input_tensor.logical_shape()[-1])));
    const DramScratch fwd =
        dram_scratch(mesh, forwarding_buffer_spec(args, token_bytes), tensor_args.forwarding_buffer);
    // Only for a TILE input: the untilized tokens.
    const DramScratch staging =
        tiled ? dram_scratch(mesh, staging_buffer_spec(args, token_bytes), tensor_args.staging_buffer) : DramScratch{};
    const auto untilize = plan_untilize(
        tensor_args.input_tensor,
        tensor_return_value[0],
        args.seq_len_per_chip,
        token_bytes,
        static_cast<uint32_t>(sems.untilized.address()),
        staging.buffer);
    const L1Layout l1 = compute_l1_layout(mesh, token_bytes, scratch_geometry(args, extent), sems.lowest_address());

    tt::tt_metal::Buffer* dram[dspf2d::ReaderRtArg::kCount] = {};
    // For a TILE input the stream cores read tokens from staging. The accessor arguments are built from
    // this table too, so this is the only place the stream kernels see the input layout.
    dram[dspf2d::ReaderRtArg::kInputAddr] = tiled ? staging.buffer : tensor_args.input_tensor.buffer();
    dram[dspf2d::ReaderRtArg::kIndicesAddr] = tensor_args.indices_tensor.buffer();
    dram[dspf2d::ReaderRtArg::kExpertOffsetsAddr] = tensor_args.expert_offsets_tensor.buffer();
    dram[dspf2d::ReaderRtArg::kDispatchTableAddr] = tensor_args.expert_dispatch_table_tensor.buffer();
    dram[dspf2d::ReaderRtArg::kCountsAddr] = tensor_args.expert_token_counts.buffer();
    dram[dspf2d::ReaderRtArg::kRegionOffsetsAddr] = tensor_args.expert_region_offsets.buffer();
    dram[dspf2d::ReaderRtArg::kOutPayloadAddr] = tensor_return_value[0].buffer();
    dram[dspf2d::ReaderRtArg::kOutMetaAddr] = tensor_return_value[1].buffer();
    dram[dspf2d::ReaderRtArg::kFwdAddr] = fwd.buffer;
    // Always bound, so the runtime-arg layout is the same with or without padding_config. The stand-in is
    // never read: the reader reads this buffer only when a compile-time arg says padding_config is given.
    dram[dspf2d::ReaderRtArg::kPaddingConfigAddr] = tensor_args.padding_config.has_value()
                                                        ? tensor_args.padding_config->buffer()
                                                        : tensor_args.expert_offsets_tensor.buffer();
    for (uint32_t i = 0; i < dspf2d::ReaderRtArg::kCount; i++) {
        TT_FATAL(dram[i] != nullptr, "dispatch_fabric2d: buffer for runtime arg {} is not allocated", i);
    }

    tt::tt_metal::WorkloadDescriptor workload;
    workload.semaphores.push_back(sems.filled);
    workload.semaphores.push_back(sems.freed);
    workload.semaphores.push_back(sems.final_arrived);
    workload.semaphores.push_back(sems.launch_credit);
    workload.semaphores.push_back(sems.untilized);
    workload.semaphores.push_back(sems.credit_done);
    for (const auto* group : {&sems.vc_arrived, &sems.vc_freed, &sems.vc_outbox}) {
        for (const auto& sem : *group) {
            workload.semaphores.push_back(sem);
        }
    }
    // A caller's buffer is its own to keep alive; only an owned one has to outlive the call with the workload.
    for (const DramScratch* scratch : {&fwd, &staging}) {
        if (scratch->owner) {
            workload.buffers.push_back({scratch->owner, scratch->buffer});
        }
    }

    // Tile rows the stream readers wait for before their first token read; zero for a row-major input,
    // where the wait compiles out.
    const uint32_t untilize_tile_rows = untilize.has_value() ? untilize->num_tile_rows : 0u;

    // Chips whose untilizer pool did not fit in the core row under the streams; reported once per build.
    uint32_t narrow_pools = 0;
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh->shape())) {
        const uint32_t pos = static_cast<uint32_t>(coord[static_cast<int32_t>(args.axis)]);
        const auto chip_ids = ring_chip_ids(mesh, coord, args.axis);
        const auto work_by_stream = generate_assignments(chip_ids, pos, args.num_links);
        const uint32_t linearized = ccl::common::get_linearized_index(coord, mesh->get_view());

        tt::tt_metal::ProgramDescriptor desc;
        for (const auto& [stream, self] : placement.at(coord)) {
            const auto& downstream = placement.at(self.downstream_coord).at(stream);
            // The downstream chip's stream back toward us: its sender writes to this chip, so it is the one our
            // launch credit releases.
            const auto& returning = placement.at(self.downstream_coord).at(reverse_stream(stream));
            TT_FATAL(
                returning.downstream_coord == coord,
                "dispatch_fabric2d: stream {} on the downstream chip does not send back to this chip",
                reverse_stream(stream));
            // Same link, other direction, on THIS chip: its sender points upstream of us, which is where
            // our reader's credits must go for the chip that wrote our pages to hear about them.
            const auto& opposite = placement.at(coord).at(reverse_stream(stream));
            const auto& work = work_by_stream.at(stream);

            // Channel topology. A channel IS a forward descriptor's movement class, so these are indexed
            // the same way the descriptor blocks are and the reader needs no mapping between them.
            std::vector<uint32_t> vc_next_words;
            vc_next_words.reserve(vcs);
            for (uint32_t v = 0; v < vcs; v++) {
                vc_next_words.push_back(vc_after_hop(v, extent));
            }

            KernelPlan plan;
            plan.stream = stream;
            plan.extent = extent;
            plan.fwd_pages_per_stream = fwd_pages;
            plan.vc_pages = vc_pages;
            plan.queue_filled_addr = static_cast<uint32_t>(sems.filled.address());
            plan.queue_freed_addr = static_cast<uint32_t>(sems.freed.address());
            plan.final_arrived_addr = static_cast<uint32_t>(sems.final_arrived.address());
            plan.launch_credit_addr = static_cast<uint32_t>(sems.launch_credit.address());
            plan.credit_done_addr = static_cast<uint32_t>(sems.credit_done.address());
            plan.vc_arrived_addr.clear();
            plan.vc_freed_addr.clear();
            plan.vc_outbox_addr.clear();
            for (uint32_t v = 0; v < vcs; v++) {
                plan.vc_arrived_addr.push_back(static_cast<uint32_t>(sems.vc_arrived[v].address()));
                plan.vc_freed_addr.push_back(static_cast<uint32_t>(sems.vc_freed[v].address()));
                plan.vc_outbox_addr.push_back(static_cast<uint32_t>(sems.vc_outbox[v].address()));
            }
            plan.untilize_sem_addr = static_cast<uint32_t>(sems.untilized.address());
            plan.untilize_tile_rows = untilize_tile_rows;

            tt::tt_metal::KernelDescriptor snd;
            snd.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch_fabric2d/device/kernels/dataflow/"
                "sender_dispatch_fabric2d.cpp";
            snd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
            snd.core_ranges = CoreRangeSet(CoreRange(self.worker_logical));
            snd.compile_time_args = dspf2d::SenderCtArgs(token_bytes, self, downstream, returning, l1, plan)
                                        .to_ct_word_arr(plan.vc_arrived_addr, plan.vc_freed_addr, plan.vc_outbox_addr);
            snd.config = tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                // NOC_1 routes -Y first, so a worker one core row from its eth core reaches it in a single hop.
                .noc = tt::tt_metal::NOC::NOC_1,
            };
            auto snd_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
            desc.kernels.push_back(std::move(snd));

            uint32_t own_count = 0;
            std::vector<uint32_t> assignment_words;
            std::vector<uint32_t> own_vc_words;
            for (const auto& a : work) {
                if (a.is_forward) {
                    continue;
                }
                own_count++;
                assignment_words.push_back(a.dst_chip_id);
                assignment_words.push_back(a.dst_pos);
                assignment_words.push_back(a.split_idx);
                assignment_words.push_back(a.split_count);
                // Hops from the downstream chip to this destination, which is what names the channel the
                // tokens land in over there. Distance 1 is the downstream chip itself: written straight
                // into its output, so no channel at all.
                const uint32_t distance = (a.dst_pos + extent - pos) % extent;
                const uint32_t mirrored = (stream % 2 == 0) ? distance : extent - distance;
                own_vc_words.push_back(vc_for_own(mirrored, extent));
            }

            const auto to_words = [](const std::vector<dspf2d::ChunkDescriptor>& cs) {
                std::vector<uint32_t> w;
                w.reserve(cs.size() * dspf2d::CHUNK_DESCRIPTOR_WORDS);
                for (const auto& d : cs) {
                    w.push_back(d.origin_pos);
                    w.push_back(d.dst_pos);
                    w.push_back(d.split_idx);
                    w.push_back(d.split_count);
                }
                return w;
            };
            const auto in_words = to_words(forwarding_descriptors(stream, pos, extent, args.num_links));
            const auto out_words = to_words(outgoing_descriptors(stream, pos, extent, args.num_links));
            const auto finals = incoming_final_descriptors(chip_ids, stream, pos, args.num_links);
            const auto final_words = to_words(finals);

            tt::tt_metal::KernelDescriptor rdr;
            rdr.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch_fabric2d/device/kernels/dataflow/"
                "reader_dispatch_fabric2d.cpp";
            rdr.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
            rdr.core_ranges = CoreRangeSet(CoreRange(self.worker_logical));
            rdr.compile_time_args = dspf2d::ReaderCtArgs(
                                        args,
                                        token_bytes,
                                        linearized,
                                        pos,
                                        static_cast<uint32_t>(self.downstream_node.chip_id),
                                        l1,
                                        plan,
                                        own_count,
                                        static_cast<uint32_t>(work.size()) - own_count,
                                        opposite,
                                        static_cast<uint32_t>(finals.size()))
                                        .to_ct_word_arr(
                                            chip_ids,
                                            assignment_words,
                                            in_words,
                                            out_words,
                                            plan.vc_arrived_addr,
                                            plan.vc_freed_addr,
                                            plan.vc_outbox_addr,
                                            vc_next_words,
                                            own_vc_words,
                                            final_words);
            for (tt::tt_metal::Buffer* buffer : dram) {
                tt::tt_metal::TensorAccessorArgs(buffer).append_to(rdr.compile_time_args);
            }
            rdr.config = tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::NOC_0,
            };

            // Buffer* so the framework records BufferBindings and rewrites them per dispatch: an address
            // describes an allocation, not a program, and a cached program must not carry a stale one.
            tt::tt_metal::KernelDescriptor::RTArgList rdr_rt;
            for (tt::tt_metal::Buffer* buffer : dram) {
                rdr_rt.push_back(buffer);
            }
            rdr.emplace_runtime_args(self.worker_logical, rdr_rt);

            // The core's three compute RISCs build three of the four slices of the routing index. They take
            // the reader's compile-time args unchanged because they run the reader's code over the same
            // scratch layout, so this builds one TRISC binary per stream core: the args carry per-core words.
            // No runtime args: nothing they touch is an allocation.
            tt::tt_metal::KernelDescriptor index_kernel;
            index_kernel.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch_fabric2d/device/kernels/compute/"
                "routing_index_dispatch_fabric2d.cpp";
            index_kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
            index_kernel.core_ranges = CoreRangeSet(CoreRange(self.worker_logical));
            index_kernel.compile_time_args = rdr.compile_time_args;
            index_kernel.config = tt::tt_metal::ComputeConfigDescriptor{};
            desc.kernels.push_back(std::move(rdr));
            desc.kernels.push_back(std::move(index_kernel));

            // The four RISCs hand off through these. Program semaphores, because the runtime writes the
            // initial value on every launch, so a RISC never reads a value left by an earlier or aborted
            // launch as a signal.
            for (uint32_t id = 0; id < dspf2d::INDEX_SEMAPHORES; id++) {
                desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
                    .id = id,
                    .core_type = tt::CoreType::WORKER,
                    .core_ranges = CoreRangeSet(CoreRange(self.worker_logical)),
                    .initial_value = 0,
                });
            }

            std::vector<uint32_t> snd_rt{1u};  // num_connections
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                mesh->get_fabric_node_id(coord),
                std::vector<tt::tt_fabric::FabricNodeId>{self.downstream_node},
                std::vector<uint32_t>{self.link_idx},
                desc,
                snd_id,
                self.worker_logical,
                snd_rt);
            // The call above fills the vector but does not attach it to the kernel.
            tt::tt_metal::KernelDescriptor::RTArgList snd_rt_list;
            snd_rt_list.append(snd_rt);
            desc.kernels[snd_id].emplace_runtime_args(self.worker_logical, snd_rt_list);
        }
        if (untilize.has_value() &&
            add_untilizer_pool(desc, placement.at(coord), args.worker_core_range_set, *untilize) ==
                UntilizerPoolFallback::kRowTooNarrow) {
            narrow_pools++;
        }
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(desc)});
    }
    if (narrow_pools > 0) {
        log_warning(
            tt::LogOp,
            "dispatch_fabric2d: the row under the streams has too few spare cores for the untilizer pool on {} "
            "of {} chips, so part of the pool runs on other rows.",
            narrow_pools,
            mesh->shape().mesh_size());
    }
    return workload;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
