// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_program_factory.hpp"
#include "combine_fabric2d_placement.hpp"
#include "combine_fabric2d_assignments.hpp"
#include "kernels/dataflow/combine_fabric2d_kernel_protocol.hpp"

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

// L1 layout, identical on every tensix core of every device in the mesh. Uniformity is load-bearing: a
// producer on chip A addresses the drain sink of the peer worker on chip B without knowing anything about
// B's harvesting or eth positions. Offsets are relative to the L1 unreserved base.
constexpr uint32_t PKT_HDR_DRAIN_OFF = 0x0000;
// Target of the drain's value-0 atomic increments: a remote atomic inc needs a real L1 address on the far
// chip, and nothing ever reads this word.
constexpr uint32_t DRAIN_SINK_OFF = 0x0400;
constexpr uint32_t PROD_BUF_OFF = 0x1000;
constexpr uint32_t L1_SLACK = 0x1000;
// Depth of the reader -> producer L1 ring, in tokens. Slots move in half-ring batches so one half can be
// refilled while the other drains.
constexpr uint32_t CMBF2D_NUM_L1_SLOTS = 8;
constexpr uint32_t CMBF2D_BATCH = CMBF2D_NUM_L1_SLOTS / 2;
static_assert(PKT_HDR_DRAIN_OFF < DRAIN_SINK_OFF, "drain sink overlaps the drain packet header");
static_assert(DRAIN_SINK_OFF < PROD_BUF_OFF, "drain sink overlaps the token ring");

// Forwarding buffer: DRAM staging for tokens passing THROUGH this chip. The op forwards multi-hop traffic
// itself rather than asking the fabric to, so a packet only ever travels one hop; anything bound further
// lands in the next chip's forwarding buffer and is re-sent from there.
//
// One region per stream, each holding relay_chunks_per_stream chunks of fwd_pages_per_chunk pages. A chunk's
// last used page is a sentinel marking its end, so a chunk need not be filled to capacity.
// Tokens whose routing metadata the reader prefetches in one batch. Each pad is 64 B (a DRAM read needs a
// 64-byte-aligned L1 destination on Blackhole), so this costs CMBF2D_META_PREFETCH * 64 B of L1. Big enough
// that one DRAM latency is amortised over many tokens, small enough to bound L1 regardless of how long a
// data-dependent run turns out to be.
constexpr uint32_t CMBF2D_META_PREFETCH = 64;
constexpr uint32_t CMBF2D_META_PAD_STRIDE = 64;

// Pages a chunk can hold. Chunk lengths are data-dependent — a chunk is one producer's share of one
// (origin chip -> this chip) run merged over the experts it serves — so only the expectation is known here:
//
//     avg tokens per chunk = seq_len_per_chip * num_experts_per_tok
//                            / (num_dispatch_groups * dispatch_group_size * num_links)
//
// each token picks num_experts_per_tok experts, each expert lives on one chip of one dispatch group, so an
// (origin, destination) pair carries 1/(NDG * dgs) of the traffic, split again across the planes.
//
// WARNING: provisioning 4x the mean is a heuristic, NOT a bound. It suits the roughly balanced traffic
// random routing produces (measured spread of tokens per (origin, destination) pair is max/mean 1.43 at seq
// 640, tighter as seq grows), but a chunk that does not fit would silently overwrite the next one. Nothing
// detects that yet.
uint32_t fwd_pages_per_chunk(
    uint32_t seq_len_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t num_dispatch_groups,
    uint32_t dispatch_group_size,
    uint32_t num_links) {
    const uint32_t denom = num_dispatch_groups * dispatch_group_size * num_links;
    const uint32_t mean = (seq_len_per_chip * num_experts_per_tok + denom - 1) / denom;
    return 4 * mean + 1;
}

struct L1Layout {
    uint32_t pkt_hdr_drain;
    uint32_t drain_sink;
    uint32_t ring;          // num_l1_slots tokens, filled by the reader and drained by the producer
    uint32_t pkt_hdr_ring;  // one prebuilt payload header per ring slot
    // The reader's copy of the control tensors, read once at startup and then indexed from L1.
    // Laid out as [expert_offsets: dispatch_group_size x num_routed_experts][counts: num_routed_experts]
    // [region_offsets: num_routed_experts], all uint32. Unlike everything above it, nothing on another chip
    // addresses this, so it can sit at the end — but it is still computed identically everywhere.
    uint32_t control;
};

L1Layout compute_l1_layout(
    ttnn::MeshDevice* mesh,
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
    l.pkt_hdr_ring = l.ring + num_l1_slots * (token_size_bytes + cmbf2d::SLOT_TAIL_BYTES);
    const uint32_t hdr_ring_bytes =
        num_l1_slots * static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    // 64-byte aligned: DRAM reads need a DRAM_ALIGNMENT-aligned L1 destination on Blackhole
    // (LOG_BASE_2_OF_DRAM_ALIGNMENT = 6), and the control region is read straight out of DRAM.
    l.control = (l.pkt_hdr_ring + hdr_ring_bytes + 63u) & ~63u;
    const uint32_t end = l.control + control_bytes;
    TT_FATAL(
        end + L1_SLACK <= sem_floor,
        "combine_fabric2d: L1 layout needs {} B (ends at 0x{:x}) but the global-semaphore region starts at "
        "0x{:x}. Reduce num_l1_slots ({}) or the token page ({} B).",
        end - base,
        end,
        sem_floor,
        num_l1_slots,
        token_size_bytes);
    return l;
}

struct DramBuffers {
    tt::tt_metal::Buffer* in = nullptr;
    tt::tt_metal::Buffer* out = nullptr;
    tt::tt_metal::Buffer* fwd = nullptr;
    tt::tt_metal::Buffer* meta = nullptr;
    tt::tt_metal::Buffer* counts = nullptr;
    tt::tt_metal::Buffer* region = nullptr;
    tt::tt_metal::Buffer* expert_offsets = nullptr;
};

// Everything a kernel needs that is the same for every stream on one chip, plus the stream itself.
struct KernelPlan {
    StreamId stream = 0;
    uint32_t my_row = 0;
    uint32_t my_expert_base = 0;
    uint32_t num_routed_experts = 0;
    uint32_t token_size_bytes = 0;
    uint32_t pages_per_chunk = 0;
    uint32_t ring_extent = 0;
    uint32_t ring_filled_addr = 0;
    uint32_t ring_freed_addr = 0;
    uint32_t fwd_arrived_addr = 0;
};

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

std::vector<uint32_t> pack_producer_args(
    const StreamPlacement& self, const StreamPlacement& downstream, const L1Layout& l1, const KernelPlan& plan) {
    cmbf2d::ProducerCtArgPacker a;
    a[cmbf2d::ProducerCtArg::NumL1Slots] = CMBF2D_NUM_L1_SLOTS;
    a[cmbf2d::ProducerCtArg::TokenSizeBytes] = plan.token_size_bytes;
    a[cmbf2d::ProducerCtArg::SlotTailBytes] = cmbf2d::SLOT_TAIL_BYTES;
    a[cmbf2d::ProducerCtArg::PeerChipId] = static_cast<uint32_t>(self.downstream_node.chip_id);
    a[cmbf2d::ProducerCtArg::PeerMeshId] = *self.downstream_node.mesh_id;
    a[cmbf2d::ProducerCtArg::RingAddr] = l1.ring;
    a[cmbf2d::ProducerCtArg::PktHdrRingAddr] = l1.pkt_hdr_ring;
    a[cmbf2d::ProducerCtArg::PktHdrDrainAddr] = l1.pkt_hdr_drain;
    a[cmbf2d::ProducerCtArg::DrainSinkAddr] = l1.drain_sink;
    a[cmbf2d::ProducerCtArg::Batch] = CMBF2D_BATCH;
    a[cmbf2d::ProducerCtArg::FilledAddr] = plan.ring_filled_addr;
    a[cmbf2d::ProducerCtArg::FreedAddr] = plan.ring_freed_addr;
    // The kernel needs the placement of its peer downstream worker because it sends the
    // forwarded-token-count semaphore bumps to it, addressed in the fabric packet header. This is also why
    // every worker placement on the mesh is decided before any kernel is built.
    a[cmbf2d::ProducerCtArg::FwdSemNocX] = static_cast<uint32_t>(downstream.worker_virtual.x);
    a[cmbf2d::ProducerCtArg::FwdSemNocY] = static_cast<uint32_t>(downstream.worker_virtual.y);
    a[cmbf2d::ProducerCtArg::FwdSemAddr] = plan.fwd_arrived_addr;
    std::vector<uint32_t> ct;
    a.append_to(ct);
    return ct;
}

std::vector<uint32_t> pack_reader_args(
    const CombineFabric2dParams& args,
    const StreamPlacement& self,
    const std::vector<Assignment>& work,
    const L1Layout& l1,
    const KernelPlan& plan,
    const DramBuffers& dram) {
    cmbf2d::ReaderCtArgPacker a;
    a[cmbf2d::ReaderCtArg::NumL1Slots] = CMBF2D_NUM_L1_SLOTS;
    a[cmbf2d::ReaderCtArg::TokenSizeBytes] = plan.token_size_bytes;
    a[cmbf2d::ReaderCtArg::SlotTailBytes] = cmbf2d::SLOT_TAIL_BYTES;
    a[cmbf2d::ReaderCtArg::Batch] = CMBF2D_BATCH;
    a[cmbf2d::ReaderCtArg::RingAddr] = l1.ring;
    a[cmbf2d::ReaderCtArg::FilledAddr] = plan.ring_filled_addr;
    a[cmbf2d::ReaderCtArg::FreedAddr] = plan.ring_freed_addr;
    a[cmbf2d::ReaderCtArg::DramInBaseAddr] = static_cast<uint32_t>(dram.in->address());
    a[cmbf2d::ReaderCtArg::DramOutBaseAddr] = static_cast<uint32_t>(dram.out->address());
    a[cmbf2d::ReaderCtArg::DramFwdBaseAddr] = static_cast<uint32_t>(dram.fwd->address());
    a[cmbf2d::ReaderCtArg::FwdChunksPerQuarter] = relay_chunks_per_stream(plan.ring_extent);
    a[cmbf2d::ReaderCtArg::FwdPagesPerChunk] = plan.pages_per_chunk;
    // Doubles as this stream's share of the same-chip run, which it copies after the fabric work.
    a[cmbf2d::ReaderCtArg::MyQuarter] = plan.stream;
    a[cmbf2d::ReaderCtArg::NumIncomingChunks] = relay_chunks_per_stream(plan.ring_extent);
    a[cmbf2d::ReaderCtArg::FwdSemAddr] = plan.fwd_arrived_addr;
    a[cmbf2d::ReaderCtArg::NbrChipId] = static_cast<uint32_t>(self.downstream_node.chip_id);
    a[cmbf2d::ReaderCtArg::ScheduleLen] = static_cast<uint32_t>(work.size());
    a[cmbf2d::ReaderCtArg::DramMetaBaseAddr] = static_cast<uint32_t>(dram.meta->address());
    a[cmbf2d::ReaderCtArg::DramCountsBaseAddr] = static_cast<uint32_t>(dram.counts->address());
    a[cmbf2d::ReaderCtArg::DramRegionBaseAddr] = static_cast<uint32_t>(dram.region->address());
    a[cmbf2d::ReaderCtArg::DramExpertOffsetsBaseAddr] = static_cast<uint32_t>(dram.expert_offsets->address());
    a[cmbf2d::ReaderCtArg::NumRoutedExperts] = plan.num_routed_experts;
    a[cmbf2d::ReaderCtArg::ExpertsPerChip] = args.experts_per_chip;
    a[cmbf2d::ReaderCtArg::MyExpertBase] = plan.my_expert_base;
    a[cmbf2d::ReaderCtArg::NumExpertsPerTok] = args.num_experts_per_tok;
    a[cmbf2d::ReaderCtArg::DispatchGroupSize] = args.dispatch_group_size;
    a[cmbf2d::ReaderCtArg::LocalSplitCount] = stream_count(args.num_links);
    a[cmbf2d::ReaderCtArg::MyRow] = plan.my_row;
    a[cmbf2d::ReaderCtArg::ControlAddr] = l1.control;
    a[cmbf2d::ReaderCtArg::MetaPrefetchCap] = CMBF2D_META_PREFETCH;

    uint32_t num_own = 0;
    for (const auto& w : work) {
        num_own += w.is_relay ? 0u : 1u;
    }
    a[cmbf2d::ReaderCtArg::NumAssignments] = num_own;

    std::vector<uint32_t> ct;
    a.append_to(ct);
    // Schedule: the work order, relays tagged. An own entry carries its index into the table that follows.
    uint32_t own_idx = 0;
    for (const auto& w : work) {
        ct.push_back(w.is_relay ? (cmbf2d::SCHED_FWD | w.relay_chunk) : own_idx++);
    }
    for (const auto& w : work) {
        if (w.is_relay) {
            continue;
        }
        ct.push_back(w.dst_chip_id);
        ct.push_back(w.dst_row);
        ct.push_back(w.split_idx);
        ct.push_back(w.split_count);
    }
    return ct;
}

// Facts about this call, established while checking it.
struct OpShape {
    uint32_t ring_extent = 0;
    uint32_t token_size_bytes = 0;
    uint32_t num_routed_experts = 0;
    uint32_t num_dispatch_groups = 0;
};

OpShape validate_and_derive(
    const CombineFabric2dParams& args, const CombineFabric2dInputs& tensor_args, const ttnn::Tensor& output) {
    const auto mesh_shape = args.device->shape();
    TT_FATAL(
        args.axis < mesh_shape.dims(),
        "combine_fabric2d: axis {} out of range for mesh shape {}",
        args.axis,
        mesh_shape);

    OpShape shape;
    shape.ring_extent = mesh_shape[static_cast<int32_t>(args.axis)];
    TT_FATAL(
        shape.ring_extent > 1,
        "combine_fabric2d: mesh axis {} has extent {}; need at least 2 chips to send anywhere",
        args.axis,
        shape.ring_extent);
    TT_FATAL(
        shape.ring_extent % 2 == 0,
        "combine_fabric2d: axis {} extent {} must be even — the all-destinations pattern relies on a "
        "diametrically-opposite chip at ring offset extent/2",
        args.axis,
        shape.ring_extent);

    auto* in_buf = tensor_args.dispatched_buffer.buffer();
    auto* out_buf = output.buffer();
    auto* meta_buf = tensor_args.dispatched_metadata.buffer();
    TT_FATAL(in_buf != nullptr, "combine_fabric2d: dispatched_buffer has no device buffer");
    TT_FATAL(out_buf != nullptr, "combine_fabric2d: output tensor has no device buffer");
    TT_FATAL(meta_buf != nullptr, "combine_fabric2d: dispatched_metadata has no device buffer");
    TT_FATAL(
        tensor_args.expert_token_counts.buffer() != nullptr,
        "combine_fabric2d: expert_token_counts has no device buffer");
    TT_FATAL(
        tensor_args.expert_region_offsets.buffer() != nullptr,
        "combine_fabric2d: expert_region_offsets has no device buffer");
    TT_FATAL(tensor_args.expert_offsets.buffer() != nullptr, "combine_fabric2d: expert_offsets has no device buffer");

    // A token is one page of the dispatched buffer — the op does not take a token size, it reads it off the
    // tensor the caller staged, exactly as the production op does.
    shape.token_size_bytes = static_cast<uint32_t>(in_buf->aligned_page_size());
    TT_FATAL(
        shape.token_size_bytes % sizeof(uint32_t) == 0,
        "combine_fabric2d: token page {} B must be a multiple of 4",
        shape.token_size_bytes);
    // A forwarded packet carries the token PLUS its routing tail, so the payload the fabric must accept is
    // token + tail, not just token.
    TT_FATAL(
        shape.token_size_bytes + cmbf2d::SLOT_TAIL_BYTES <= tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(),
        "combine_fabric2d: token page {} B + {} B routing tail exceeds the fabric max payload {}. Raise "
        "max_payload_size in the device's fabric_router_config.",
        shape.token_size_bytes,
        cmbf2d::SLOT_TAIL_BYTES,
        tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    TT_FATAL(
        out_buf->aligned_page_size() == shape.token_size_bytes,
        "combine_fabric2d: output page size {} must equal the token page size {} — the op moves whole tokens "
        "between the two",
        out_buf->aligned_page_size(),
        shape.token_size_bytes);
    // The output holds one page per (token, top-k slot). Checked against the real buffer because only it
    // knows the per-device page count.
    TT_FATAL(
        out_buf->num_pages() >= args.seq_len_per_chip * args.num_experts_per_tok,
        "combine_fabric2d: output holds {} pages but seq_len_per_chip x num_experts_per_tok = {} are needed",
        out_buf->num_pages(),
        args.seq_len_per_chip * args.num_experts_per_tok);
    TT_FATAL(
        meta_buf->num_pages() == in_buf->num_pages(),
        "combine_fabric2d: metadata has {} pages but the dispatched buffer has {}; they index the same flat "
        "buffer",
        meta_buf->num_pages(),
        in_buf->num_pages());

    shape.num_routed_experts = static_cast<uint32_t>(tensor_args.expert_token_counts.logical_shape()[-1]);
    shape.num_dispatch_groups = shape.num_routed_experts / (args.experts_per_chip * args.dispatch_group_size);
    TT_FATAL(shape.num_dispatch_groups >= 1, "combine_fabric2d: derived num_dispatch_groups is 0");
    return shape;
}

// The reader/producer ring handshake is two monotonic single-writer counters, plus one counter the upstream
// chip's producer bumps as it fills this stream's forwarding region.
//
// GlobalSemaphores rather than the op's own L1 region for one reason: the framework ZEROES them before
// launch. Raw L1 keeps whatever the previous program left there, and a stale `freed` underflows the reader's
// free-slot arithmetic — a silent buffer overwrite, not a clean failure.
struct RingSemaphores {
    tt::tt_metal::GlobalSemaphore filled;
    tt::tt_metal::GlobalSemaphore freed;
    tt::tt_metal::GlobalSemaphore fwd_arrived;

    uint32_t lowest_address() const {
        return static_cast<uint32_t>(std::min({filled.address(), freed.address(), fwd_arrived.address()}));
    }
};

RingSemaphores allocate_ring_semaphores(ttnn::MeshDevice* mesh) {
    // Allocated on the full worker grid so the addresses are uniform across the mesh. One fwd_arrived
    // semaphore serves every stream: each stream is drained by a different worker core, so the per-core copy
    // at this uniform L1 offset already separates them, and the producer simply targets the right core.
    const auto grid = mesh->compute_with_storage_grid_size();
    const CoreRangeSet all_workers(CoreRange(CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}));
    RingSemaphores sems{
        ttnn::global_semaphore::create_global_semaphore(mesh, all_workers, 0, tt::tt_metal::BufferType::L1),
        ttnn::global_semaphore::create_global_semaphore(mesh, all_workers, 0, tt::tt_metal::BufferType::L1),
        ttnn::global_semaphore::create_global_semaphore(mesh, all_workers, 0, tt::tt_metal::BufferType::L1)};
    tt::tt_metal::distributed::Synchronize(mesh, std::nullopt, {});
    return sems;
}

struct ForwardingBuffer {
    std::shared_ptr<ttnn::Tensor> owner;
    tt::tt_metal::Buffer* buffer = nullptr;
    uint32_t pages_per_chunk = 0;
};

// Never initialised and never read back: pure staging for tokens passing through a chip. One page per token,
// and the page is token + tail so a single fabric write lands both. Fused into ONE page rather than split
// across payload and metadata regions precisely because nothing outside the op reads it — so the
// "one page = one token" property the caller's regions must keep does not apply here, and it saves a DRAM
// read and a DRAM write per forwarded token.
//
// Parked on WorkloadDescriptor::buffers wrapped in a shared_ptr<Tensor>: holding only a
// shared_ptr<MeshBuffer> would let DeviceStorage::deallocate free the memory when the local Tensor dies
// (workload_descriptor.hpp:19-36). Being a mesh tensor, its device-local address is uniform across the mesh
// by construction, which is what lets a producer address the NEXT chip's buffer.
ForwardingBuffer allocate_forwarding_buffer(
    ttnn::MeshDevice* mesh, const CombineFabric2dParams& args, const OpShape& shape) {
    ForwardingBuffer fwd;
    fwd.pages_per_chunk = fwd_pages_per_chunk(
        args.seq_len_per_chip,
        args.num_experts_per_tok,
        shape.num_dispatch_groups,
        args.dispatch_group_size,
        args.num_links);
    const uint32_t page_bytes = shape.token_size_bytes + cmbf2d::SLOT_TAIL_BYTES;
    TT_FATAL(
        page_bytes % sizeof(uint32_t) == 0,
        "combine_fabric2d: forwarding page {} B must be a multiple of 4",
        page_bytes);
    const uint32_t pages = relay_chunks_per_mesh(shape.ring_extent, args.num_links) * fwd.pages_per_chunk;
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
        cmbf2d::SLOT_TAIL_BYTES);
    return fwd;
}

KernelPlan make_kernel_plan(
    const CombineFabric2dParams& args,
    const ttnn::MeshCoordinate& coord,
    const OpShape& shape,
    const RingSemaphores& sems,
    uint32_t pages_per_chunk) {
    KernelPlan plan;
    plan.my_row = coord[static_cast<int32_t>(args.axis)];
    plan.num_routed_experts = shape.num_routed_experts;
    plan.token_size_bytes = shape.token_size_bytes;
    plan.pages_per_chunk = pages_per_chunk;
    plan.ring_extent = shape.ring_extent;
    plan.ring_filled_addr = static_cast<uint32_t>(sems.filled.address());
    plan.ring_freed_addr = static_cast<uint32_t>(sems.freed.address());
    plan.fwd_arrived_addr = static_cast<uint32_t>(sems.fwd_arrived.address());
    // Which of the `num_routed_experts` columns this chip hosts. The dispatch group is this device's position
    // on the OTHER mesh axis; with one group per column of a 2D mesh that is just the other coordinate. Same
    // derivation as the production reader's compile-time `offset`.
    const uint32_t experts_per_group = args.experts_per_chip * args.dispatch_group_size;
    const uint32_t my_group = args.device->shape().dims() > 1
                                  ? coord[static_cast<int32_t>(args.axis == 0 ? 1 : 0)] % shape.num_dispatch_groups
                                  : 0u;
    plan.my_expert_base = my_group * experts_per_group + plan.my_row * args.experts_per_chip;
    return plan;
}

// The reader's L1 copy of the control tensors: the expert_offsets slice (one row per origin chip) plus the
// counts and region offsets. A few kB, read once at startup and indexed from L1 thereafter. Plus the
// metadata prefetch pads at the front, 64-byte aligned each: a DRAM read needs a 64-byte-aligned L1
// destination on Blackhole, which no offset inside a ring slot's tail can give (the tail starts at
// token_size, and its free half is only 32-byte aligned).
uint32_t control_region_bytes(const CombineFabric2dParams& args, const OpShape& shape) {
    return CMBF2D_META_PREFETCH * CMBF2D_META_PAD_STRIDE +
           static_cast<uint32_t>(sizeof(uint32_t)) * shape.num_routed_experts * (args.dispatch_group_size + 2);
}

tt::tt_metal::ProgramDescriptor build_program_for_coord(
    const CombineFabric2dParams& args,
    const ttnn::MeshCoordinate& coord,
    const MeshPlacement& placement,
    const L1Layout& l1,
    const KernelPlan& chip_plan,
    const DramBuffers& dram) {
    tt::tt_metal::ProgramDescriptor desc;
    const auto work_by_stream =
        generate_assignments(ring_chip_ids(args.device, coord, args.axis), chip_plan.my_row, args.num_links);

    for (const auto& [stream, self] : placement.at(coord)) {
        KernelPlan plan = chip_plan;
        plan.stream = stream;
        const auto& downstream = placement.at(self.downstream_coord).at(stream);
        const auto& work = work_by_stream.at(stream);

        tt::tt_metal::KernelDescriptor prod;
        prod.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/"
            "producer_combine_fabric2d.cpp";
        prod.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        prod.core_ranges = CoreRangeSet(CoreRange(self.worker_logical));
        prod.compile_time_args = pack_producer_args(self, downstream, l1, plan);
        prod.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            // NOC_1 routes -Y first, so worker (eth row + 1) -> eth core is a single hop.
            .noc = tt::tt_metal::NOC::NOC_1,
        };
        auto prod_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
        desc.kernels.push_back(std::move(prod));

        tt::tt_metal::KernelDescriptor rdr;
        rdr.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/"
            "reader_combine_fabric2d.cpp";
        rdr.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        rdr.core_ranges = CoreRangeSet(CoreRange(self.worker_logical));
        rdr.compile_time_args = pack_reader_args(args, self, work, l1, plan, dram);
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
            prod_id,
            self.worker_logical,
            rt_raw);
        tt::tt_metal::KernelDescriptor::RTArgList rt;
        rt.append(rt_raw);
        desc.kernels[prod_id].emplace_runtime_args(self.worker_logical, rt);
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
    const auto shape = validate_and_derive(operation_attributes, tensor_args, tensor_return_value);

    const auto sems = allocate_ring_semaphores(mesh_device);
    const auto l1 = compute_l1_layout(
        mesh_device,
        CMBF2D_NUM_L1_SLOTS,
        shape.token_size_bytes,
        control_region_bytes(operation_attributes, shape),
        sems.lowest_address());
    const auto placement = decide_placement(mesh_device, operation_attributes.axis, operation_attributes.num_links);
    const auto fwd = allocate_forwarding_buffer(mesh_device, operation_attributes, shape);

    // Every buffer here is interleaved DRAM whose base address is uniform across the mesh, so a producer can
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
    workload_descriptor.buffers.push_back({fwd.owner, fwd.buffer});

    for (const auto& coord : tensor_coords.coords()) {
        workload_descriptor.programs.push_back(
            {ttnn::MeshCoordinateRange(coord),
             build_program_for_coord(
                 operation_attributes,
                 coord,
                 placement,
                 l1,
                 make_kernel_plan(operation_attributes, coord, shape, sems, fwd.pages_per_chunk),
                 dram)});
    }
    return workload_descriptor;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
