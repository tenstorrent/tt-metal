// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_program_factory.hpp"
#include "combine_fabric2d_placement.hpp"
#include "combine_fabric2d_assignments.hpp"
#include "kernels/dataflow/combine_fabric2d_reader_ct_args.hpp"
#include "kernels/dataflow/combine_fabric2d_sender_ct_args.hpp"
#include "kernels/dataflow/combine_fabric2d_untilizer_ct_args.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt_stl/assert.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

namespace {

// L1 layout, that is the parts that have to be in same places on every chip (e.g. semaphores)
constexpr uint32_t PKT_HDR_DRAIN_OFF = 0x0000;
constexpr uint32_t DRAIN_SINK_OFF = 0x0400;
constexpr uint32_t PROD_BUF_OFF = 0x1000;
static_assert(PKT_HDR_DRAIN_OFF < DRAIN_SINK_OFF, "drain sink overlaps the drain packet header");
static_assert(DRAIN_SINK_OFF < PROD_BUF_OFF, "drain sink overlaps the token ring");

// Forwarding buffer: Sending data farther than to the immediate neighboring chip (e.g. 1 -> 2 -> 3)
// is not left to fabric to do. Op manages it. Multihop data movement is broken down to single hops.
// Fabric doesn't have enough context to properly orchestrate traffic nor deep enough buffers to not
// block '1 -> 2' data movements if both '1 -> 2 -> 3' and '2 -> 3' movements are fighting for the
// 2-3 link. Thus DRAM buffer for forwarding is introduced. Chip 1 sends everything going to chips
// 2, 3, 4 and 5 to chip 2. What is destined for chip 2 is written to its final DRAM address by the
// receiver eRisc. Everything else goes to forwardign buffer. Op manages when will sender cores send
// "home-made" tokens and when will they send tokens from the forwarding buffer.
//
// Pages one stream's region has to hold. The kernels pack their chunks densely, computing each chunk's
// length from expert_offsets, so this only has to bound the region's TOTAL — which is why splitting the same
// tokens into more chunks does not grow it.
//
// A destination chip's whole return volume is seq_len_per_chip * num_experts_per_tok copies — that is just
// its output buffer, so it holds however the router spread the copies over chips and experts, and whether or
// not the top-k picks are distinct. Chunks heading to the same destination therefore SHARE that volume
// rather than each reaching it. A stream relays half_extent - 1 distinct destinations: the nearer ones can
// have their whole volume on an origin that splits the run num_links ways, while the farthest is reachable
// only from the diametrically opposite chip, whose run is split across every stream. One page per chunk
// covers the integer slicing of each share.
uint32_t fwd_pages_per_stream(const CombineFabric2dParams& args) {
    const uint32_t per_destination = args.seq_len_per_chip * args.num_experts_per_tok;
    const uint32_t half_extent = ring_extent(args) / 2;
    return (half_extent - 2) * (per_destination / args.num_links) + per_destination / stream_count(args.num_links) +
           relay_chunks_per_stream(ring_extent(args)) * args.experts_per_chip;
}

constexpr uint32_t align_l1(uint32_t addr) { return (addr + 63u) & ~63u; }

// Where an untilizer core's circular buffers end. The framework lays a program's circular buffers out from
// the L1 allocator base in declaration order, DRAM-aligned, so this is both what cb_out's address is and
// what the hand-placed control tables have to clear.
uint32_t untilizer_cb_end(uint32_t base, uint32_t token_size_bytes, const CombineFabric2dInputs& tensor_args) {
    uint32_t end = align_l1(base) + cmbf2d::UNT_RING_BATCHES * cmbf2d::UNT_BATCH_ROWS * token_size_bytes;
    end = align_l1(end) + tile_size_bytes(tensor_args);  // the batch count
    end = align_l1(end) + 2 * untilize_block_tiles(tensor_args) * tile_size_bytes(tensor_args);
    return align_l1(end);
}

L1Layout compute_l1_layout(
    ttnn::MeshDevice* mesh,
    const CombineFabric2dInputs& tensor_args,
    uint32_t num_l1_slots,
    uint32_t token_size_bytes,
    uint32_t control_bytes,
    uint32_t sem_floor) {
    const uint32_t base =
        static_cast<uint32_t>(mesh->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    L1Layout l;
    l.pkt_hdr_drain = base + PKT_HDR_DRAIN_OFF;
    l.drain_sink = base + DRAIN_SINK_OFF;
    l.ring = base + PROD_BUF_OFF;
    // One prebuilt header per ring slot, past the ring itself. A slot is the token plus its metadata tail.
    l.pkt_hdr_ring = l.ring + num_l1_slots * (token_size_bytes + cmbf2d::FORWARDING_METADATA_SIZE);
    const uint32_t hdr_ring_bytes =
        num_l1_slots * static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    // 64-byte aligned: DRAM reads need a DRAM_ALIGNMENT-aligned L1 destination on Blackhole
    // (LOG_BASE_2_OF_DRAM_ALIGNMENT = 6), and the control region is read straight out of DRAM.
    l.control = align_l1(l.pkt_hdr_ring + hdr_ring_bytes);
    uint32_t end = l.control + control_bytes;
    if (dispatched_is_tiled(tensor_args)) {
        // Untilizer cores share no hand-placed memory with the cores above, so their layout starts over at
        // the base -- it has to, because that is where the framework puts cb_out and cb_out IS the batch ring.
        l.unt_ring = align_l1(base);
        l.unt_control = untilizer_cb_end(base, token_size_bytes, tensor_args);
        end = std::max(end, l.unt_control + control_bytes);
    }
    TT_FATAL(
        end <= sem_floor,
        "combine_fabric2d: L1 layout needs {} B (ends at 0x{:x}) but the global-semaphore region starts at "
        "0x{:x}. Reduce num_l1_slots ({}) or the token page ({} B).",
        end - base,
        end,
        sem_floor,
        num_l1_slots,
        token_size_bytes);
    return l;
}

// The buffers the kernels will address must exist, and the output this op allocated must match the token
// geometry it was derived from. Not caller validation: the output comes from our own compute_output_specs,
// and validate_on_program_cache_miss cannot see it.
void validate_allocations(
    const CombineFabric2dParams& args, const CombineFabric2dInputs& tensor_args, const ttnn::Tensor& output) {
    for (const auto& [tensor, name] :
         {std::pair{&tensor_args.dispatched_buffer, "dispatched_buffer"},
          std::pair{&tensor_args.dispatched_metadata, "dispatched_metadata"},
          std::pair{&tensor_args.expert_token_counts, "expert_token_counts"},
          std::pair{&tensor_args.expert_region_offsets, "expert_region_offsets"},
          std::pair{&tensor_args.expert_offsets, "expert_offsets"},
          std::pair{&output, "output"}}) {
        TT_FATAL(tensor->buffer() != nullptr, "combine_fabric2d: {} has no device buffer", name);
    }
    TT_FATAL(
        output.buffer()->aligned_page_size() == token_size_bytes(tensor_args),
        "combine_fabric2d: output page size {} must equal the token page size {} — the op moves whole tokens "
        "between the two",
        output.buffer()->aligned_page_size(),
        token_size_bytes(tensor_args));
    TT_FATAL(
        output.buffer()->num_pages() >= args.seq_len_per_chip * args.num_experts_per_tok,
        "combine_fabric2d: output holds {} pages but seq_len_per_chip x num_experts_per_tok = {} are needed",
        output.buffer()->num_pages(),
        args.seq_len_per_chip * args.num_experts_per_tok);
}

std::vector<uint32_t> ring_chip_ids(ttnn::MeshDevice* mesh, const ttnn::MeshCoordinate& coord, uint32_t axis) {
    const uint32_t extent = mesh->shape()[static_cast<int32_t>(axis)];
    std::vector<uint32_t> ids(extent);
    for (uint32_t row = 0; row < extent; row++) {
        ttnn::MeshCoordinate c = coord;
        c[static_cast<int32_t>(axis)] = row;
        ids[row] = static_cast<uint32_t>(mesh->get_fabric_node_id(c).chip_id);
    }
    return ids;
}

// The reader/sender ring handshake is two monotonic single-writer counters, plus one counter the upstream
// chip's sender bumps as it fills this stream's forwarding region.
//
// GlobalSemaphores rather than the op's own L1 region so they sit at an address uniform across the mesh:
// `fwd_arrived` is bumped by the upstream chip, which has to know where it lives.
//
// Nothing zeroes them between launches — they outlive the cached workload — so the kernels reset all three at
// end of stream. Skipping that leaves the next launch reading this one's totals, and a stale `freed`
// underflows the reader's free-slot arithmetic into a silent buffer overwrite rather than a clean failure.
// The untilizer handshake needs the same treatment for the same reason. `untilized[j]` lives on a reader's
// core and is bumped by its group's j-th untilizer; `unt_freed[c]` lives on an untilizer's core and is
// bumped by the reader on link c. One semaphore per INDEX serves the whole mesh, because a core only ever
// sees the copies it owns and the per-core copy at a uniform offset already separates them.
//
// `unt_freed` is per consumer rather than one counter they share: a group's senders take alternating halves
// of each run and so run far apart, and a shared count would let the leading one's credit release a slot the
// trailing one is still reading.
struct RingSemaphores {
    tt::tt_metal::GlobalSemaphore filled;
    tt::tt_metal::GlobalSemaphore freed;
    tt::tt_metal::GlobalSemaphore fwd_arrived;
    std::vector<tt::tt_metal::GlobalSemaphore> untilized;
    std::vector<tt::tt_metal::GlobalSemaphore> unt_freed;

    uint32_t lowest_address() const {
        uint32_t lowest = static_cast<uint32_t>(std::min({filled.address(), freed.address(), fwd_arrived.address()}));
        for (const auto* group : {&untilized, &unt_freed}) {
            for (const auto& sem : *group) {
                lowest = std::min(lowest, static_cast<uint32_t>(sem.address()));
            }
        }
        return lowest;
    }
};

RingSemaphores allocate_ring_semaphores(ttnn::MeshDevice* mesh, uint32_t num_links, uint32_t untilizers_per_group) {
    // Allocated on the full worker grid so the addresses are uniform across the mesh. One fwd_arrived
    // semaphore serves every stream: each stream is drained by a different worker core, so the per-core copy
    // at this uniform L1 offset already separates them, and the sender simply targets the right core.
    const auto grid = mesh->compute_with_storage_grid_size();
    const CoreRangeSet all_workers(CoreRange(CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}));
    auto make = [&] {
        return ttnn::global_semaphore::create_global_semaphore(mesh, all_workers, 0, tt::tt_metal::BufferType::L1);
    };
    RingSemaphores sems{make(), make(), make(), {}, {}};
    if (untilizers_per_group != 0) {
        for (uint32_t j = 0; j < untilizers_per_group; j++) {
            sems.untilized.push_back(make());
        }
        for (uint32_t c = 0; c < num_links; c++) {
            sems.unt_freed.push_back(make());
        }
    }
    tt::tt_metal::distributed::Synchronize(mesh, std::nullopt, {});
    return sems;
}

struct ForwardingBuffer {
    std::shared_ptr<ttnn::Tensor> owner;
    tt::tt_metal::Buffer* buffer = nullptr;
    uint32_t pages_per_stream = 0;
};

// Never initialised and never read back: pure staging for tokens passing through a chip. One page per token,
// and the page is token + forwarding metadata so a single fabric write lands both. Fused into ONE page rather
// than split across payload and metadata regions precisely because nothing outside the op reads it — so the
// "one page = one token" property the caller's regions must keep does not apply here, and it saves a DRAM
// read and a DRAM write per forwarded token.
//
ForwardingBuffer allocate_forwarding_buffer(
    ttnn::MeshDevice* mesh, const CombineFabric2dParams& args, const CombineFabric2dInputs& tensor_args) {
    ForwardingBuffer fwd;
    fwd.pages_per_stream = fwd_pages_per_stream(args);
    const uint32_t page_bytes = token_size_bytes(tensor_args) + cmbf2d::FORWARDING_METADATA_SIZE;
    TT_FATAL(
        page_bytes % 64 == 0, "combine_fabric2d: forwarding page {} B must be 64-byte aligned for DRAM", page_bytes);
    const uint32_t pages = stream_count(args.num_links) * fwd.pages_per_stream;
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape({pages, page_bytes / static_cast<uint32_t>(sizeof(uint32_t))}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
    // Throws if it does not fit DRAM, which IS the "verify it fits" check.
    fwd.owner = std::make_shared<ttnn::Tensor>(create_device_tensor(spec, mesh));
    fwd.buffer = fwd.owner->buffer();
    TT_FATAL(fwd.buffer != nullptr, "combine_fabric2d: forwarding buffer has no device buffer");
    TT_FATAL(
        fwd.buffer->aligned_page_size() == page_bytes,
        "combine_fabric2d: forwarding page size is {} B after alignment but the op addresses it as {} B. "
        "The token page + {} must be a multiple of the DRAM alignment.",
        fwd.buffer->aligned_page_size(),
        page_bytes,
        cmbf2d::FORWARDING_METADATA_SIZE);
    return fwd;
}

KernelPlan make_kernel_plan(
    const CombineFabric2dParams& args,
    const CombineFabric2dInputs& tensor_args,
    const ttnn::MeshCoordinate& coord,
    const RingSemaphores& sems,
    uint32_t pages_per_stream) {
    KernelPlan plan;
    plan.pages_per_stream = pages_per_stream;
    plan.ring_filled_addr = static_cast<uint32_t>(sems.filled.address());
    plan.ring_freed_addr = static_cast<uint32_t>(sems.freed.address());
    plan.fwd_arrived_addr = static_cast<uint32_t>(sems.fwd_arrived.address());
    // Which of the `num_routed_experts` columns this chip hosts. The dispatch group is this device's position
    // on the OTHER mesh axis; with one group per column of a 2D mesh that is just the other coordinate. Same
    // derivation as the production reader's compile-time `offset`.
    const uint32_t experts_per_group = args.experts_per_chip * ring_extent(args);
    const uint32_t my_group = args.device->shape().dims() > 1 ? coord[static_cast<int32_t>(args.axis == 0 ? 1 : 0)] %
                                                                    num_dispatch_groups(args, tensor_args)
                                                              : 0u;
    plan.my_expert_base = my_group * experts_per_group + my_dg_index(args, coord) * args.experts_per_chip;
    return plan;
}

// The reader's L1 copy of the control tensors: the expert_offsets slice (one row per origin chip) plus the
// counts and region offsets. A few kB, read once at startup and indexed from L1 thereafter. Plus the
// metadata prefetch pads at the front, 64-byte aligned each: a DRAM read needs a 64-byte-aligned L1
// destination on Blackhole, which no offset inside a ring slot's tail can give (the tail starts at
// token_size, and its free half is only 32-byte aligned).
uint32_t control_region_bytes(const CombineFabric2dParams& args, const CombineFabric2dInputs& tensor_args) {
    return cmbf2d::META_PREFETCH * cmbf2d::META_PAD_STRIDE +
           static_cast<uint32_t>(sizeof(uint32_t)) * num_routed_experts(tensor_args) * (ring_extent(args) + 2);
}

// cb_out FIRST: the framework lays these out from the L1 allocator base in declaration order, which is what
// makes cb_out the same memory as L1Layout::unt_ring and so lets a reader address a row by core and offset.
void add_untilizer_cbs(
    tt::tt_metal::ProgramDescriptor& desc, const CombineFabric2dInputs& tensor_args, const CoreRangeSet& core) {
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = cmbf2d::UNT_RING_BATCHES * cmbf2d::UNT_BATCH_ROWS * token_size_bytes(tensor_args),
        .core_ranges = core,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = cmbf2d::UNT_CB_OUT,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = token_size_bytes(tensor_args),
        }}}});
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = tile_size_bytes(tensor_args),
        .core_ranges = core,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = cmbf2d::UNT_CB_BATCHES,
            .data_format = tt::DataFormat::UInt32,
            .page_size = tile_size_bytes(tensor_args),
        }}}});
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = 2 * untilize_block_tiles(tensor_args) * tile_size_bytes(tensor_args),
        .core_ranges = core,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = cmbf2d::UNT_CB_IN,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = tile_size_bytes(tensor_args),
        }}}});
}

ReaderUntilizers untilizers_for_stream(
    const UntilizerGroups& groups, StreamId stream, const RingSemaphores& sems, const L1Layout& l1) {
    if (sems.unt_freed.empty()) {
        return {};
    }
    ReaderUntilizers r{l1.unt_ring, static_cast<uint32_t>(sems.unt_freed.at(stream / 2).address()), {}};
    for (uint32_t j = 0; j < groups[untilizer_group_of(stream)].size(); j++) {
        r.peers.push_back(HandshakePeer{
            groups[untilizer_group_of(stream)][j].worker_virtual,
            static_cast<uint32_t>(sems.untilized.at(j).address())});
    }
    return r;
}

tt::tt_metal::ProgramDescriptor build_program_for_coord(
    const CombineFabric2dParams& args,
    const CombineFabric2dInputs& tensor_args,
    const ttnn::MeshCoordinate& coord,
    const MeshPlacement& placement,
    const L1Layout& l1,
    const KernelPlan& chip_plan,
    const DramBuffers& dram,
    const RingSemaphores& sems) {
    tt::tt_metal::ProgramDescriptor desc;
    const auto work_by_stream =
        generate_assignments(ring_chip_ids(args.device, coord, args.axis), my_dg_index(args, coord), args.num_links);
    const auto& groups = placement.at(coord).untilizers;

    for (const auto& [stream, self] : placement.at(coord).streams) {
        KernelPlan plan = chip_plan;
        plan.stream = stream;
        const auto& downstream = placement.at(self.downstream_coord).streams.at(stream);
        const auto& work = work_by_stream.at(stream);

        tt::tt_metal::KernelDescriptor snd;
        snd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/"
            "sender_combine_fabric2d.cpp";
        snd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        snd.core_ranges = CoreRangeSet(CoreRange(self.worker_logical));
        snd.compile_time_args = cmbf2d::SenderCtArgs(tensor_args, self, downstream, l1, plan).to_ct_word_arr();
        snd.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            // NOC_1 routes -Y first, so worker (eth row + 1) -> eth core is a single hop.
            .noc = tt::tt_metal::NOC::NOC_1,
        };
        auto snd_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
        desc.kernels.push_back(std::move(snd));

        tt::tt_metal::KernelDescriptor rdr;
        rdr.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/"
            "reader_combine_fabric2d.cpp";
        rdr.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        rdr.core_ranges = CoreRangeSet(CoreRange(self.worker_logical));
        rdr.defines.emplace_back("TILE", dispatched_is_tiled(tensor_args) ? "1" : "0");
        rdr.compile_time_args =
            cmbf2d::ReaderCtArgs(
                args, tensor_args, coord, self, work, l1, plan, dram, untilizers_for_stream(groups, stream, sems, l1))
                .to_ct_word_arr();
        for (auto* buf : {dram.in, dram.out, dram.fwd, dram.meta, dram.counts, dram.region, dram.expert_offsets}) {
            tt::tt_metal::TensorAccessorArgs(buf).append_to(rdr.compile_time_args);
        }
        rdr.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
        };
        desc.kernels.push_back(std::move(rdr));  // no fabric connection => no rt args

        std::vector<uint32_t> rt_raw{1u};  // num_connections
        tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
            args.device->get_fabric_node_id(coord),
            std::vector<tt::tt_fabric::FabricNodeId>{self.downstream_node},
            std::vector<uint32_t>{stream / 2},
            desc,
            snd_id,
            self.worker_logical,
            rt_raw);
        tt::tt_metal::KernelDescriptor::RTArgList rt;
        rt.append(rt_raw);
        desc.kernels[snd_id].emplace_runtime_args(self.worker_logical, rt);
    }

    // One kernel per untilizer core rather than one over a core range: they differ by their index in the
    // group, which is what deals the group's batches out between them.
    for (uint32_t g = 0; g < UNTILIZER_GROUPS; g++) {
        for (uint32_t j = 0; j < groups[g].size(); j++) {
            const CoreRangeSet core(CoreRange(groups[g][j].logical));
            add_untilizer_cbs(desc, tensor_args, core);

            UntilizerPlan plan;
            plan.my_expert_base = chip_plan.my_expert_base;
            plan.my_index = j;
            plan.num_peers = static_cast<uint32_t>(groups[g].size());
            plan.control_addr = l1.unt_control;
            plan.produced_addr = static_cast<uint32_t>(sems.untilized.at(j).address());
            // A group serves one ring direction: group 0 is clockwise, matching untilizer_group_of.
            const StreamId first = make_stream_id(0, g == 0);
            plan.walks_down = stream_is_cw(first);
            for (uint32_t link = 0; link < args.num_links; link++) {
                plan.consumers.push_back(HandshakePeer{
                    placement.at(coord).streams.at(make_stream_id(link, g == 0)).worker_virtual,
                    static_cast<uint32_t>(sems.unt_freed.at(link).address())});
            }

            tt::tt_metal::KernelDescriptor kernel;
            kernel.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/"
                "untilizer_combine_fabric2d.cpp";
            kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
            kernel.core_ranges = core;
            kernel.compile_time_args =
                cmbf2d::UntilizerCtArgs(args, tensor_args, coord, work_by_stream.at(first), plan, dram)
                    .to_ct_word_arr();
            for (auto* buf : {dram.in, dram.counts, dram.region, dram.expert_offsets}) {
                tt::tt_metal::TensorAccessorArgs(buf).append_to(kernel.compile_time_args);
            }
            kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::NOC_0,
            };
            desc.kernels.push_back(std::move(kernel));

            tt::tt_metal::KernelDescriptor untilize;
            untilize.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/compute/"
                "untilize_combine_fabric2d.cpp";
            untilize.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
            untilize.core_ranges = core;
            untilize.compile_time_args = {
                cmbf2d::UNT_CB_IN,
                cmbf2d::UNT_CB_OUT,
                cmbf2d::UNT_CB_BATCHES,
                tiles_per_token_row(tensor_args),
                untilize_block_tiles(tensor_args),
                cmbf2d::UNT_BATCH_ROWS};
            untilize.config = tt::tt_metal::ComputeConfigDescriptor{};
            desc.kernels.push_back(std::move(untilize));
        }
    }

    return desc;
}
}  // namespace

tt::tt_metal::WorkloadDescriptor CombineFabric2dProgramFactory::create_workload_descriptor(
    const CombineFabric2dParams& operation_attributes,
    const CombineFabric2dInputs& tensor_args,
    ttnn::Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    auto* mesh_device = operation_attributes.device;
    validate_allocations(operation_attributes, tensor_args, tensor_return_value);

    const uint32_t per_group = dispatched_is_tiled(tensor_args) ? untilizers_per_group() : 0;
    const auto sems = allocate_ring_semaphores(mesh_device, operation_attributes.num_links, per_group);
    const auto l1 = compute_l1_layout(
        mesh_device,
        tensor_args,
        cmbf2d::NUM_L1_SLOTS,
        token_size_bytes(tensor_args),
        control_region_bytes(operation_attributes, tensor_args),
        sems.lowest_address());
    const auto placement =
        decide_placement(mesh_device, operation_attributes.axis, operation_attributes.num_links, per_group);
    const auto fwd = allocate_forwarding_buffer(mesh_device, operation_attributes, tensor_args);

    // Every buffer here is interleaved DRAM whose base address is uniform across the mesh, so a sender can
    // address the same buffer on any chip by page index. That is what lets a token carry a final destination
    // address computed on the chip it started from.
    const DramBuffers dram{
        tensor_args.dispatched_buffer.buffer(),
        tensor_return_value.buffer(),
        fwd.buffer,
        tensor_args.dispatched_metadata.buffer(),
        tensor_args.expert_token_counts.buffer(),
        tensor_args.expert_region_offsets.buffer(),
        tensor_args.expert_offsets.buffer()};

    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    workload_descriptor.semaphores.push_back(sems.filled);
    workload_descriptor.semaphores.push_back(sems.freed);
    workload_descriptor.semaphores.push_back(sems.fwd_arrived);
    for (const auto* group : {&sems.untilized, &sems.unt_freed}) {
        for (const auto& sem : *group) {
            workload_descriptor.semaphores.push_back(sem);
        }
    }
    workload_descriptor.buffers.push_back({fwd.owner, fwd.buffer});

    for (const auto& coord : tensor_coords.coords()) {
        workload_descriptor.programs.push_back(
            {ttnn::MeshCoordinateRange(coord),
             build_program_for_coord(
                 operation_attributes,
                 tensor_args,
                 coord,
                 placement,
                 l1,
                 make_kernel_plan(operation_attributes, tensor_args, coord, sems, fwd.pages_per_stream),
                 dram,
                 sems)});
    }
    return workload_descriptor;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
