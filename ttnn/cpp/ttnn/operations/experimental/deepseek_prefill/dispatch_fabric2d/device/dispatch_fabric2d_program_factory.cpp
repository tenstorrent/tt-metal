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
// them on another.
constexpr uint32_t PKT_HDR_SIGNAL_OFF = 0x0000;
constexpr uint32_t DRAIN_SINK_OFF = 0x0400;
constexpr uint32_t QUEUE_OFF = 0x1000;
static_assert(PKT_HDR_SIGNAL_OFF < DRAIN_SINK_OFF, "drain sink overlaps the signal packet header");
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
    for (uint32_t row = 0; row < extent; row++) {
        ttnn::MeshCoordinate c = coord;
        c[static_cast<int32_t>(axis)] = row;
        ids[row] = static_cast<uint32_t>(mesh->get_fabric_node_id(c).chip_id);
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
        .num_forward = forward_chunks_per_stream(extent)};
}

L1Layout compute_l1_layout(
    ttnn::MeshDevice* mesh, uint32_t token_bytes, const dspf2d::ScratchGeometry& g, uint32_t sem_floor) {
    const uint32_t scratch_bytes = dspf2d::scratch_bytes(g);
    const uint32_t base =
        static_cast<uint32_t>(mesh->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    L1Layout l;
    l.pkt_hdr_signal = base + PKT_HDR_SIGNAL_OFF;
    l.drain_sink = base + DRAIN_SINK_OFF;
    l.queue = base + QUEUE_OFF;
    l.pkt_hdr_queue = l.queue + dspf2d::QUEUE_DEPTH * (token_bytes + dspf2d::FORWARDING_METADATA_SIZE);
    // One header per entry, the same stride the sender indexes the pool with.
    const uint32_t hdr_queue_bytes =
        dspf2d::QUEUE_DEPTH * static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    // 64-byte aligned: a DRAM read needs a 64-byte-aligned L1 destination on Blackhole, and the
    // scratch is read straight out of DRAM.
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

// Four counters: `filled` and `freed` are the reader/sender queue handshake, `fwd_arrived` is raised by
// the upstream chip's sender as it fills this stream's fwd_section, and `untilized` is raised by this
// chip's untilizer writers as they write tile rows to staging.
//
// GlobalSemaphores so they sit at the same address on every chip: the upstream chip signals
// `fwd_arrived` and has to know its address.
//
// Nothing resets them between launches, so the kernels do it at the end of each run. `filled` and
// `freed` are set to zero, and `untilized` too for a TILE input. `fwd_arrived` is lowered by the count
// consumed, so a signal from the next launch that arrives early is kept.
struct StreamSemaphores {
    tt::tt_metal::GlobalSemaphore filled;
    tt::tt_metal::GlobalSemaphore freed;
    tt::tt_metal::GlobalSemaphore fwd_arrived;
    // Allocated for both input layouts so the reader's argument list is the same; a row-major input never
    // uses it.
    tt::tt_metal::GlobalSemaphore untilized;

    uint32_t lowest_address() const {
        return static_cast<uint32_t>(
            std::min({filled.address(), freed.address(), fwd_arrived.address(), untilized.address()}));
    }
};

StreamSemaphores allocate_stream_semaphores(ttnn::MeshDevice* mesh, const CoreRangeSet& allowed_cores) {
    const auto make = [&] {
        return ttnn::global_semaphore::create_global_semaphore(mesh, allowed_cores, 0, tt::tt_metal::BufferType::L1);
    };
    StreamSemaphores sems{make(), make(), make(), make()};
    tt::tt_metal::distributed::Synchronize(mesh, std::nullopt, {});
    return sems;
}

// A device buffer the op allocates for itself, never initialises and never reads back on the host. The
// workload holds the owner so the buffer lives as long as the cached program; the kernels get its address
// as a runtime argument the framework updates on each dispatch.
struct OwnedBuffer {
    std::shared_ptr<ttnn::Tensor> owner;
    tt::tt_metal::Buffer* buffer = nullptr;
};

// Typed UINT32 so a page is exactly page_bytes, with no alignment rounding. Kernels address these buffers
// by a page index they compute themselves, so a wider page would misplace every page after the first.
OwnedBuffer allocate_buffer(ttnn::MeshDevice* mesh, uint32_t num_pages, uint32_t page_bytes, std::string_view what) {
    TT_FATAL(
        page_bytes % 64 == 0, "dispatch_fabric2d: {} page {} B must be 64-byte aligned for DRAM", what, page_bytes);
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape({num_pages, page_bytes / static_cast<uint32_t>(sizeof(uint32_t))}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
    OwnedBuffer owned;
    // Throws if it does not fit in DRAM.
    owned.owner = std::make_shared<ttnn::Tensor>(create_device_tensor(spec, mesh));
    owned.buffer = owned.owner->buffer();
    TT_FATAL(
        owned.buffer->aligned_page_size() == page_bytes,
        "dispatch_fabric2d: {} page is {} B after alignment but the op addresses it as {} B",
        what,
        owned.buffer->aligned_page_size(),
        page_bytes);
    return owned;
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
    // A forwarded packet (token plus routing tail) is the largest packet the op sends. The fabric does
    // not reject a larger payload: the bytes past its max overwrite the next channel slot.
    TT_FATAL(
        token_bytes + dspf2d::FWD_EXTRA_BYTES <= tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(),
        "dispatch_fabric2d: token page {} B + {} B routing tail exceeds the fabric max payload {}. Increase "
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

    validate_chunk_agreement(extent, args.num_links);
    const auto placement = decide_placement(mesh, args.axis, args.num_links, args.worker_core_range_set);
    const auto sems = allocate_stream_semaphores(mesh, args.worker_core_range_set);
    // A page is a token plus its routing tail, so one fabric write carries both.
    const uint32_t fwd_pages = fwd_pages_per_stream(
        extent, args.num_links, args.seq_len_per_chip, args.num_experts_per_tok, args.experts_per_chip);
    const OwnedBuffer fwd = allocate_buffer(
        mesh, fwd_pages * stream_count(args.num_links), token_bytes + dspf2d::FORWARDING_METADATA_SIZE, "forwarding");
    // Only for a TILE input: the untilized tokens, one row-major page each, so the stream cores address a
    // token by page index as they do for a row-major input. Rounded up to whole tile rows because the
    // untilizer packs all 32 rows of a tile; the extra pages hold padding rows and are never read.
    const uint32_t staging_pages =
        tt::round_up(args.seq_len_per_chip, static_cast<uint32_t>(tt::constants::TILE_HEIGHT));
    const OwnedBuffer staging = tiled ? allocate_buffer(mesh, staging_pages, token_bytes, "staging") : OwnedBuffer{};
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
    workload.semaphores.push_back(sems.fwd_arrived);
    workload.semaphores.push_back(sems.untilized);
    workload.buffers.push_back({fwd.owner, fwd.buffer});
    if (tiled) {
        workload.buffers.push_back({staging.owner, staging.buffer});
    }

    // Tile rows the stream readers wait for before their first token read; zero for a row-major input,
    // where the wait compiles out.
    const uint32_t untilize_tile_rows = untilize.has_value() ? untilize->num_tile_rows : 0u;

    // Chips whose untilizer pool did not fit in the row under the streams; reported once per build.
    uint32_t narrow_pools = 0;
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
            plan.queue_filled_addr = static_cast<uint32_t>(sems.filled.address());
            plan.queue_freed_addr = static_cast<uint32_t>(sems.freed.address());
            plan.fwd_arrived_addr = static_cast<uint32_t>(sems.fwd_arrived.address());
            plan.untilize_sem_addr = static_cast<uint32_t>(sems.untilized.address());
            plan.untilize_tile_rows = untilize_tile_rows;

            tt::tt_metal::KernelDescriptor snd;
            snd.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch_fabric2d/device/kernels/dataflow/"
                "sender_dispatch_fabric2d.cpp";
            snd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
            snd.core_ranges = CoreRangeSet(CoreRange(self.worker_logical));
            snd.compile_time_args = dspf2d::SenderCtArgs(token_bytes, self, downstream, l1, plan).to_ct_word_arr();
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
                if (a.is_forward) {
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
                w.reserve(cs.size() * dspf2d::CHUNK_DESCRIPTOR_WORDS);
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
                                        linearized,
                                        row,
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

            // The core's three compute RISCs build three of the four slices of the routing index. They take
            // the reader's compile-time args unchanged because they run the reader's code over the same
            // scratch layout. No runtime args: nothing they touch is an allocation.
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
