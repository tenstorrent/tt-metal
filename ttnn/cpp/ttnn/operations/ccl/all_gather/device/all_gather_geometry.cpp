// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_geometry.hpp"

#include <algorithm>
#include <fmt/format.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::ccl {

namespace {

uint32_t ag_read_alignment(bool is_dram) {
    return is_dram ? tt::tt_metal::hal::get_dram_alignment() : tt::tt_metal::hal::get_l1_alignment();
}

// Every local device must report the mesh allocator's bank count for `type`. The host derives an interleaved
// side's page stride once (contiguous_page_stride: the mesh allocator's count) and hands the same CT value to every
// device's kernels. The walk order -- and so unicast's seqno order -- comes from that one CT value and is the same
// on every device either way; what a differing NUM_*_BANKS would break is cross-page runs, which are linear only at
// the device's own contiguity (no release fallback; the debug stride ASSERT would fire). So WalkGeometry::make
// checks only interleaved sides whose runs cross pages. Only DRAM and L1 have a kernel NUM_*_BANKS define.
// Cost: one allocator query per local device per call.
void require_uniform_bank_counts(tt::tt_metal::distributed::MeshDevice* mesh_device, tt::tt_metal::BufferType type) {
    if (mesh_device == nullptr || (type != tt::tt_metal::BufferType::DRAM && type != tt::tt_metal::BufferType::L1)) {
        return;
    }
    const uint32_t mesh_banks = mesh_device->allocator()->get_num_banks(type);
    for (const auto* device : mesh_device->get_devices()) {
        const uint32_t banks = device->allocator()->get_num_banks(type);
        TT_FATAL(
            banks == mesh_banks,
            "all_gather: device {} has {} {} banks but the mesh allocator (which sets the walk stride every device "
            "uses) has {}. all_gather needs the same bank count on every device of the mesh.",
            device->id(),
            banks,
            type == tt::tt_metal::BufferType::DRAM ? "DRAM" : "L1",
            mesh_banks);
    }
}

// The user-facing explanation of a failed relay_read_ok.
std::string relay_problem(const GatherGeometry& g) {
    if (g.out_walk_pages_per_page > 1 && g.out_aligned_page != g.out_walk_pages_per_page * g.walk_page_bytes) {
        return fmt::format(
            "the output page ({} B of content, {} B aligned) is shared by {} chunks of {} B and is padded, so a "
            "chunk-stride CB cannot mirror it (a padded slot would clobber the other devices' chunks of the page)",
            g.out_page,
            g.out_aligned_page,
            g.out_walk_pages_per_page,
            g.walk_page_bytes);
    }
    return fmt::format(
        "the output page ({} B) holds {} chunks of {} B, and a {} B chunk offset is not a multiple of the output "
        "memory's {} B NoC read alignment, so reading a chunk back out of the output would be misaligned",
        g.out_aligned_page,
        g.out_walk_pages_per_page,
        g.walk_page_bytes,
        g.walk_page_bytes,
        g.out_read_alignment);
}

}  // namespace

GatherGeometry GatherGeometry::make(
    const Tensor& input, const tt::tt_metal::TensorSpec& output_spec, int32_t dim_from_end) {
    const auto& in_buf = *input.buffer();
    GatherGeometry g;

    // --- Pages ---
    // The output side comes from the spec only, so select_program_factory (no buffer yet) and the factories
    // (which pass the output tensor's spec) derive the same values. The aligned page is the content page
    // aligned to its memory's hal alignment: what Buffer::aligned_page_size() gives (allocator alignment ==
    // hal's; WalkGeometry::make checks it against the buffer).
    const bool out_is_dram = output_spec.memory_config().is_dram();
    g.in_page = in_buf.page_size();
    g.in_aligned_page = in_buf.aligned_page_size();
    g.out_page = output_spec.compute_page_size_bytes();
    g.out_read_alignment = ag_read_alignment(out_is_dram);
    g.out_aligned_page = tt::align(g.out_page, g.out_read_alignment);
    g.in_read_alignment = ag_read_alignment(in_buf.is_dram());

    // --- Copy mode ---
    // The kernel always reads whole *aligned* input pages into L1 (required by the input's NoC read alignment,
    // DRAM or L1) but writes at output *content* (unaligned) granularity -- which is why the walk page differs
    // by copy mode. matched/concat write a whole aligned input page (== L1 read stride) into an output slot;
    // split writes output-content-sized pieces to separate output page bases.
    const uint32_t larger = std::max(g.in_page, g.out_page);
    const uint32_t smaller = std::min(g.in_page, g.out_page);
    TT_FATAL(
        smaller > 0 && larger % smaller == 0,
        "all_gather: input and output pages are {} B and {} B; one must be a whole multiple of the other",
        g.in_page,
        g.out_page);
    const bool is_split = g.in_page > g.out_page;
    g.walk_page_bytes = is_split ? g.out_page : g.in_aligned_page;
    g.out_walk_pages_per_page = is_split ? 1u : g.out_page / g.in_page;
    g.in_walk_pages_per_page = is_split ? g.in_page / g.out_page : 1u;
    g.mode = is_split ? CopyMode::Split : (g.out_walk_pages_per_page > 1 ? CopyMode::Concat : CopyMode::Matched);
    TT_FATAL(
        g.out_walk_pages_per_page == 1 || g.in_aligned_page == g.in_page,
        "concat requires an unpadded input page");  // so slots align to content

    // --- Predicates ---
    g.relay_ok = relay_read_ok(g.walk_page_bytes, g.out_walk_pages_per_page, g.out_aligned_page, g.out_read_alignment);
    g.relay_needs_padded_slot =
        ::relay_needs_padded_slot(g.walk_page_bytes, g.out_walk_pages_per_page, g.out_aligned_page);

    // --- Walk-page count ---
    g.num_input_pages = in_buf.num_pages();
    g.num_walk_pages = g.num_input_pages * g.in_walk_pages_per_page;
    TT_FATAL(
        g.num_walk_pages / g.in_walk_pages_per_page == g.num_input_pages,
        "all_gather output walk-page count overflowed uint32: {} input pages x split factor {}",
        g.num_input_pages,
        g.in_walk_pages_per_page);

    // --- Stripe geometry ---
    // input_pages_per_stripe = num input pages along [gather dim .. last dim] this device contributes per
    // stripe. For a last-dim RM gather this is the *page* count, which handles sharded RM input (> 1 input page
    // per row).
    const auto& input_shape = input.padded_shape();
    const auto tile_spec = input.layout() == Layout::TILE ? input.tensor_spec().tile() : tt::tt_metal::Tile();
    uint32_t input_pages_per_stripe = 1;
    for (int32_t i = dim_from_end; i < 0; i++) {
        uint32_t extent;
        if (i == -1) {
            if (input.layout() == ttnn::TILE_LAYOUT) {
                extent = input_shape[i] / tile_spec.get_width();
            } else {
                // This is a page count, so divide by the unaligned page size, not aligned
                extent = (input_shape[i] * input.element_size()) / g.in_page;
            }
        } else if (input.layout() == ttnn::TILE_LAYOUT && i == -2) {
            extent = input_shape[i] / tile_spec.get_height();
        } else {
            extent = input_shape[i];
        }
        input_pages_per_stripe *= extent;
    }
    // Stripe = this device's contiguous run of walk pages per row. Measured in walk pages (not output pages) so
    // multi-shard concat works: a stripe's walk pages are laid across output pages and may straddle pages.
    g.walk_pages_per_stripe = input_pages_per_stripe * g.in_walk_pages_per_page;
    TT_FATAL(g.walk_pages_per_stripe > 0, "all_gather: a stripe must hold at least one walk page");
    return g;
}

void GatherGeometry::require_relay_ok(bool relays) const {
    // route_all_gather never picks unicast for !relay_ok on a relaying topology (it routes to multicast, or
    // TT_FATALs where the topology leaves no alternative), so reaching this means the factory was chosen by
    // something other than route_all_gather.
    TT_FATAL(
        relay_ok || !relays,
        "all_gather unicast cannot relay this output: {}. Use a row size that is a multiple of the output memory's "
        "alignment, or the same buffer type on both sides.",
        relay_problem(*this));
}

WalkPageSlice GatherGeometry::slice_of(uint32_t slice_idx, uint32_t num_slices) const {
    // [first, end) of the input pages, the first `num_input_pages % num_slices` slices one page longer; then
    // scaled by the split ratio (1 unless Split).
    const uint32_t per_slice = num_input_pages / num_slices;
    const uint32_t remainder = num_input_pages % num_slices;
    const uint32_t first = (slice_idx * per_slice) + std::min(slice_idx, remainder);
    const uint32_t end = ((slice_idx + 1) * per_slice) + std::min(slice_idx + 1, remainder);
    return {.first = first * in_walk_pages_per_page, .count = (end - first) * in_walk_pages_per_page};
}

uint64_t per_link_bytes(const tt::tt_metal::TensorSpec& output_spec, uint32_t num_links) {
    return output_spec.compute_packed_buffer_size_bytes() / std::max(1u, num_links);
}

uint32_t contiguous_page_stride(const tt::tt_metal::Buffer& buffer) {
    const auto& dspec = buffer.buffer_distribution_spec();
    // TensorAccessorArgs picks the sharded accessor exactly when the buffer has a distribution spec
    // (tensor_accessor_args.cpp update_args_config), so this branch matches the kernel's accessor type.
    if (!dspec.has_value()) {
        // Interleaved: the kernel returns NUM_DRAM_BANKS / NUM_L1_BANKS, JIT defines taken from the whole
        // device (jit_device_config.cpp create_jit_device_config). The device allocator's bank counts come
        // from the same sources, so they are equal by construction:
        //   DRAM: both are soc_desc.get_num_dram_views() (l1_banking_allocator.cpp: num_dram_channels).
        //   L1:   both are get_logical_compute_cores(...).size() with the same dispatch config -- the
        //         allocator makes exactly those cores ComputeAndStore, one L1 bank each.
        // Use the *device's* allocator, not buffer.allocator(), which may be a sub-device allocator with fewer
        // L1 banks. No other buffer type has a NUM_*_BANKS define, so nothing else can be mirrored. The kernels
        // trust this value (cross-page runs are linear only if it is right; they ASSERT it in debug builds and
        // have no release fallback), hence the hard checks.
        const auto type = buffer.buffer_type();
        TT_FATAL(
            type == tt::tt_metal::BufferType::DRAM || type == tt::tt_metal::BufferType::L1,
            "all_gather: interleaved buffer type {} has no kernel bank-count define; cannot mirror its page stride",
            static_cast<int>(type));
        const uint32_t num_banks = buffer.device()->allocator()->get_num_banks(type);
        TT_FATAL(
            num_banks > 0, "all_gather: device allocator reports no banks for buffer type {}", static_cast<int>(type));
        return num_banks;
    }
    // Sharded: the dspec is already rank-squeezed at construction (BufferDistributionSpec), and
    // TensorAccessorArgs sends exactly these shapes, so the kernel sees the same ones.
    return sharded_contiguous_page_stride(dspec->tensor_shape_in_pages(), dspec->shard_shape_in_pages());
}

uint32_t sharded_contiguous_page_stride(
    const tt::tt_metal::Shape& tensor_shape, const tt::tt_metal::Shape& shard_shape) {
    const int rank = static_cast<int>(tensor_shape.rank());
    TT_FATAL(rank > 0 && shard_shape.rank() == tensor_shape.rank(), "dspec shapes must share a nonzero rank");
    // contiguous_dim(): innermost dim the shard spans more than one page of (TensorAccessor::contiguous_dim).
    int d = rank - 1;
    for (int i = rank - 1; i >= 0; --i) {
        if (shard_shape[i] > 1) {
            d = i;
            break;
        }
    }
    // tensor_strides[d]: row-major page strides of the tensor shape (dspec.h precompute_strides_ct).
    uint32_t stride = 1;
    for (int i = rank - 1; i > d; --i) {
        stride *= tensor_shape[i];
    }
    TT_FATAL(stride > 0, "all_gather: sharded page stride derived as 0 (empty tensor dim in the distribution spec)");
    return stride;
}

WalkGeometry WalkGeometry::make(const GatherGeometry& geom, const WalkPolicy& policy, const WalkInputs& in) {
    // The kernels' output accessor steps by the buffer's aligned page; every slot, packed and relay decision here
    // uses the spec-derived one. They must be the same number.
    TT_FATAL(
        in.out_buffer_aligned_page == geom.out_aligned_page,
        "all_gather: the output buffer's aligned page is {} B but its spec gives {} B (content {} B aligned to {} B)",
        in.out_buffer_aligned_page,
        geom.out_aligned_page,
        geom.out_page,
        geom.out_read_alignment);
    TT_FATAL(
        policy.k == 1 || policy.k == WalkPolicy::kAutoK, "all_gather walk policy: k={} (want 1 or auto)", policy.k);
    WalkGeometry w;
    const uint32_t page = geom.walk_page_bytes;

    // --- Slot (padded-slot policy) ---
    // One chunk per output page whose aligned page is padded: give each chunk an output page's worth of L1, so
    // the CB steps like the output and output runs may cross pages (pad included). A run may merge n chunks only
    // where the L1 slot stride equals the memory stride: true for this output by construction, and for the input
    // only when its aligned page is the slot (input_run_source).
    const bool padded_slot = policy.padded_slot && geom.relay_needs_padded_slot;

    // --- k ---
    // Only Matched with unpadded pages on both sides: a padded page cannot be cut into aligned chunks that tile
    // it, and Concat / Split already give sub-page walk pages (Concat: k would only shrink the chunk).
    const bool k_eligible = geom.mode == CopyMode::Matched && geom.in_aligned_page == geom.in_page &&
                            geom.out_aligned_page == geom.out_page && !padded_slot;
    const uint32_t align = std::max(geom.in_read_alignment, geom.out_read_alignment);
    // choose_k fills the transfer budget: a run cap is a deliberate run-length limit, so it bounds the budget.
    const uint32_t budget = in.run_cap_bytes != 0 ? std::min(in.packet_size, in.run_cap_bytes) : in.packet_size;
    const uint32_t k = (policy.k == WalkPolicy::kAutoK && k_eligible) ? choose_k(page, budget, in.burst, align) : 1u;
    TT_FATAL(k == 1 || k_is_valid(k, page, align), "all_gather: k={} does not split a {} B walk page", k, page);
    w.k = k;
    w.chunk_size = page / k;
    w.slot_bytes = padded_slot ? geom.out_aligned_page : w.chunk_size;
    w.in_cpp = geom.in_walk_pages_per_page * k;
    w.out_cpp = geom.out_walk_pages_per_page * k;
    w.cps = geom.walk_pages_per_stripe * k;
    TT_FATAL(
        slot_is_valid(w.chunk_size, w.slot_bytes, w.out_cpp, w.k),
        "all_gather: CB slot {} B for {} B chunks, {} per output page, k={}",
        w.slot_bytes,
        w.chunk_size,
        w.out_cpp,
        w.k);

    // --- Stride and run sources (gather_rules.hpp) ---
    w.stride = walk_stride_for(w.out_cpp, k, in.out_page_stride);
    w.out_runs = output_run_source(geom.out_aligned_page, w.slot_bytes, k, w.out_cpp, in.out_page_stride, w.stride);
    w.in_runs =
        input_run_source(geom.in_aligned_page, w.chunk_size, w.slot_bytes, k, w.in_cpp, in.in_page_stride, w.stride);

    // --- xfer ---
    // A packed output (its aligned page is exactly its chunks' slots) walks in columns of up to one transfer; an
    // unpacked one in plain ascending order (xfer 1): its runs cannot cross pages anyway.
    const bool out_packed_slot = geom.out_aligned_page == w.out_cpp * w.slot_bytes;
    w.xfer = out_packed_slot ? max_xfer(in.packet_size, w.slot_bytes, in.run_cap_bytes, in.burst) : 1u;

    // --- CB entry ---
    // A one-packet entry only where the stripe is no longer than the walk stride (WalkPolicy::one_packet_short_stripe).
    const bool one_packet = policy.one_packet_short_stripe && geom.walk_pages_per_stripe <= w.stride;
    if (!one_packet) {
        // A whole number of packet loads, rounded to whole input (split) or output (concat) pages --
        // walk_pages_per_group of the k = 1 geometry, which is 1 whenever k > 1 (Matched only), so k's sub-page
        // chunks are not rounded back up to a page -- times packets_per_cb_entry, clamped by L1.
        // An entry that cuts a walk page is safe: the CB is packed, and reader and writer agree on which seqno
        // sits in which slot whatever their run grouping (gather_walk.hpp; host gtest ReaderWriterGrouping).
        const uint32_t group = geom.walk_pages_per_group();
        uint32_t chunks_per_packet = std::max(1u, in.packet_size / w.slot_bytes);
        chunks_per_packet = std::max(group, (chunks_per_packet / group) * group);
        const uint32_t entry_bytes = chunks_per_packet * w.slot_bytes;
        // An integer multiplier preserves the whole-packet and whole-page properties above. The clamp is
        // defensive: it holds whatever packets_per_cb_entry is tuned to, including 1.
        const uint32_t multiplier =
            std::clamp(in.max_l1_space / (in.cb_depth * entry_bytes), 1u, std::max(1u, in.packets_per_cb_entry));
        if (multiplier < in.packets_per_cb_entry) {
            log_warning(
                tt::LogOp,
                "CircularBuffer depth reduced due to L1 pressure (only {} B available), performance may regress.",
                in.max_l1_space);
        }
        w.chunks_per_entry = chunks_per_packet * multiplier;
    } else {
        // One packet: one column, but at most gather_scatter_max_chunks of the shortest output run -- a packet
        // carries at most that many scatter segments, so a longer entry only waits on segments it cannot send.
        // Never rounded to pages.
        uint32_t min_output_run = 1;
        if (!out_packed_slot) {
            min_output_run = 1;
        } else if (w.stride == 1 && w.out_runs.cross_page) {
            min_output_run = w.cps;
        } else {
            min_output_run = k * std::max(1u, geom.walk_pages_per_stripe / w.stride);
        }
        w.chunks_per_entry = std::max(1u, std::min(w.xfer, gather_scatter_max_chunks * min_output_run));
    }
    w.cb_entry_bytes = w.chunks_per_entry * w.slot_bytes;
    return w;
}

WalkGeometry WalkGeometry::make(
    const GatherGeometry& geom,
    const SharedPolicy& policy,
    const Tensor& input,
    const Tensor& output,
    uint32_t packet_size) {
    const auto& in_buf = *input.buffer();
    const auto& out_buf = *output.buffer();
    const WalkInputs in{
        .in_page_stride = contiguous_page_stride(in_buf),
        .out_page_stride = contiguous_page_stride(out_buf),
        .out_buffer_aligned_page = static_cast<uint32_t>(out_buf.aligned_page_size()),
        .packet_size = packet_size,
        .burst = tt::tt_metal::hal::get_noc_max_burst_size_bytes(),
        .run_cap_bytes = policy.run_cap_bytes,
        .packets_per_cb_entry = policy.packets_per_cb_entry,
        .cb_depth = policy.cb_depth,
        // Only the packet-load entry's clamp reads it.
        .max_l1_space = ttnn::operations::data_movement::get_max_l1_space(input),
    };
    const auto w = make(geom, policy.walk, in);
    if (!in_buf.buffer_distribution_spec().has_value() && w.in_runs.cross_page) {
        require_uniform_bank_counts(input.device(), in_buf.buffer_type());
    }
    if (!out_buf.buffer_distribution_spec().has_value() && w.out_runs.cross_page) {
        require_uniform_bank_counts(input.device(), out_buf.buffer_type());
    }
    log_debug(
        tt::LogOp,
        "all_gather walk: k {} chunk {} slot {} stride {} xfer {} entry {} in_runs {}/{} out_runs {}/{}",
        w.k,
        w.chunk_size,
        w.slot_bytes,
        w.stride,
        w.xfer,
        w.chunks_per_entry,
        w.in_runs.whole_page,
        w.in_runs.cross_page,
        w.out_runs.whole_page,
        w.out_runs.cross_page);
    return w;
}

bool WalkGeometry::relay_ok(const GatherGeometry& geom) const {
    if (!relay_read_ok(chunk_size, out_cpp, geom.out_aligned_page, geom.out_read_alignment)) {
        return false;
    }
    return !::relay_needs_padded_slot(chunk_size, out_cpp, geom.out_aligned_page) ||
           slot_bytes == geom.out_aligned_page;
}

AvailableCores available_worker_cores(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    const auto resolved = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, resolved);
    if (sub_core_grid.has_value()) {
        cores = cores.intersection(sub_core_grid.value());
    }
    return {resolved, cores};
}

std::vector<tt::tt_metal::GlobalSemaphore> create_semaphores_and_sync(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const CoreRangeSet& cores,
    tt::tt_metal::SubDeviceId subdevice_id,
    uint32_t n) {
    // Kernel needs to wait to receive all remote data before exiting, and in some cases needs to wait for all
    // remote devices to be ready before beginning operation. Since Fabric doesn't provide such capability within
    // kernels, we need to manually sync using global semaphores.
    // Allocate the semaphores in L1_SMALL to avoid fragmenting the larger L1 memory pool.
    const bool has_l1_small = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL) > 0;
    const auto sem_buffer_type = has_l1_small ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
    if (sem_buffer_type != tt::tt_metal::BufferType::L1_SMALL) {
        log_warning(
            tt::LogOp,
            "Allocating semaphores in L1, which may fragment L1 and reduce headroom for subsequent op "
            "allocations. Configure an L1_SMALL region to mitigate this.");
    }
    std::vector<tt::tt_metal::GlobalSemaphore> out;
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        out.push_back(ttnn::global_semaphore::create_global_semaphore(mesh_device, cores, 0, sem_buffer_type));
    }
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");
    return out;
}

////////////////////////////////////////////////////////////////
// Routing
////////////////////////////////////////////////////////////////

std::optional<std::string> gather_dim_padding_reason(const Tensor& input, int32_t dim_from_end) {
    // Gather-dim padding would need the output's gather-dim extent to be num_devices * the input's padded
    // extent, but it is num_devices * the logical one. dim_from_end indexes both shapes (see
    // AllGatherParams::dim_from_end).
    const auto& logical_shape = input.logical_shape();
    const auto& padded_shape = input.padded_shape();
    if (logical_shape[dim_from_end] != padded_shape[dim_from_end]) {
        return fmt::format(
            "gather dim {} is padded from {} to {}; size must be a multiple of the tile/shard extent",
            logical_shape.get_normalized_index(dim_from_end),
            logical_shape[dim_from_end],
            padded_shape[dim_from_end]);
    }
    return std::nullopt;
}

namespace {

// Native implementation only handles cases where every output write is an aligned NoC write into the output
// buffer, computed with index math alone -- the bytes are never rearranged on-device first. If it needs a
// transpose, untilize, re-pad, or re-shard ("massaged op"), it goes to composite.
// `shape_spec` is AllGatherParams::output_spec (the shape check: a persistent output of the wrong shape is
// validate's error, not a composite route); `page_spec` is the spec the factories will actually see (the
// persistent output's own, whose sharding can differ), for every page check. num_devices is
// AllGatherParams::num_devices, the count shape_spec was built with.
std::optional<std::string> layout_reason(
    const Tensor& input,
    int32_t dim_from_end,
    uint32_t num_devices,
    const tt::tt_metal::TensorSpec& shape_spec,
    const tt::tt_metal::TensorSpec& page_spec) {
    if (auto padded = gather_dim_padding_reason(input, dim_from_end)) {
        return padded;
    }

    // The kernel walks the output page grid as the input's with only the gather dim scaled, so any other
    // difference breaks it -- e.g. an output shard width that doesn't divide the row, which pads the last dim.
    auto gathered_padded_shape = input.padded_shape();
    gathered_padded_shape[dim_from_end] *= num_devices;
    if (shape_spec.padded_shape() != gathered_padded_shape) {
        return fmt::format(
            "output stored as {} instead of {}; its shard shape must divide the gathered shape evenly",
            shape_spec.padded_shape(),
            gathered_padded_shape);
    }
    if (page_spec.padded_shape() != gathered_padded_shape) {
        return std::nullopt;  // a persistent output of the wrong shape: validate reports it
    }

    // Page sizes are UNALIGNED (content) sizes, so a matched-but-padded gather reads as matched, not concat.
    // The page checks below are no-ops for tile, whose pages are always aligned and equally sized on both sides.
    const uint32_t input_page_size = input.buffer()->page_size();
    const uint32_t input_aligned_page_size = input.buffer()->aligned_page_size();
    const uint32_t output_page_size = page_spec.compute_page_size_bytes();

    // GatherGeometry derives the walk pages per page (concat) and per input page (split) by division, and
    // TT_FATALs on a non-integer ratio, which would otherwise mis-place every write.
    if (std::max(input_page_size, output_page_size) % std::min(input_page_size, output_page_size) != 0) {
        return fmt::format(
            "input and output rows are {} B and {} B; one shard width must be a whole multiple of the other",
            input_page_size,
            output_page_size);
    }
    // concat and split move data at content granularity, so the input page must have no padding.
    if (input_page_size != output_page_size && input_page_size != input_aligned_page_size) {
        return fmt::format(
            "input rows ({} B) are padded to the {} B memory alignment; resharding needs unpadded rows",
            input_page_size,
            input.buffer()->alignment());
    }
    // Split moves output-row sized chunks, written to the output and read at those offsets inside an input
    // page, so the row must suit the NoC write alignment and the input's own (DRAM reads need 32/64 B). With that,
    // a padded CB slot (relay, out_cpp == 1) is also congruent with every input chunk offset: a slot is
    // padded only when the output alignment exceeds the chunk's, so it is a multiple of the input's.
    const uint32_t split_chunk_alignment = std::max(gather_noc_write_alignment, input.buffer()->alignment());
    if (input_page_size > output_page_size && output_page_size % split_chunk_alignment != 0) {
        return fmt::format(
            "output is sharded finer than the input; its rows ({} B) must be a multiple of {} B",
            output_page_size,
            split_chunk_alignment);
    }
    // matched and concat write a whole *aligned* input page into each output page slot, so the slot must be at
    // least that large. They differ only across memories with unequal alignments.
    const uint32_t output_aligned_page_size =
        tt::align(output_page_size, ag_read_alignment(page_spec.memory_config().is_dram()));
    if (input_page_size <= output_page_size && output_aligned_page_size < input_aligned_page_size) {
        return fmt::format(
            "input rows pad to {} B but the output reserves {} B; use the same buffer type, or a row size that is a "
            "multiple of both DRAM and L1 alignments",
            input_aligned_page_size,
            output_aligned_page_size);
    }
    return std::nullopt;
}

}  // namespace

bool unicast_relays(const AllGatherParams& args) {
    const uint32_t axis = args.get_1d_axis();
    return ::unicast_relays(args.num_devices, tt::tt_fabric::is_ring_or_torus(args.axis_topology[axis]));
}

RouteDecision route_all_gather(
    const AllGatherParams& args, const Tensor& input, const tt::tt_metal::TensorSpec& output_spec) {
    // 1. Layout: nothing native can address it. The shape is judged on args.output_spec (built from the persistent
    // output's memory config when there is one), NOT on the persistent tensor's own spec: a persistent output of
    // the wrong shape/dtype must reach validate_on_program_cache_miss and its specific errors, not be diverted to
    // composite. The pages are judged on the spec the factories will see.
    if (auto reason = layout_reason(input, args.dim_from_end, args.num_devices, args.output_spec, output_spec)) {
        return {GatherRoute::Composite, std::move(*reason)};
    }

    // From here the native kernels can address the tensors; the remaining question is unicast vs multicast.
    // The relay rule needs the geometry, built only when unicast is still a candidate.
    auto relay_fine = [&]() -> std::pair<bool, std::string> {
        if (!unicast_relays(args)) {
            return {true, {}};  // no device reads its output back
        }
        const auto geom = GatherGeometry::make(input, output_spec, args.dim_from_end);
        return {geom.relay_ok, geom.relay_ok ? std::string{} : relay_problem(geom)};
    };

    // 2. Topology limitations.
    if (args.is_true_2d()) {
        // Limitation: Unicast algorithm currently does not support true 2D topologies
        return {GatherRoute::Multicast, "true 2D topology (unicast unsupported)"};
    }
    const char* unicast_only = nullptr;
    if (tt::tt_fabric::is_2d_fabric_config(args.fabric_config) && !args.axis_is_straight[args.get_1d_axis()]) {
        // Limitation: Multicast cannot handle 2D topology reshapes/views that bend an axis
        unicast_only = "a 2D topology view that bends the gather axis (multicast cannot follow it)";
    } else if (args.fabric_config == tt::tt_fabric::FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE) {
        // Limitation: NeighborExchange only permits 1-hop unicast
        unicast_only = "FABRIC_1D_NEIGHBOR_EXCHANGE (only 1-hop unicast is permitted)";
    }
    if (unicast_only != nullptr) {
        const auto [ok, problem] = relay_fine();
        // No fallback exists here. The composite path is all_broadcast + concat, and all_broadcast line-
        // multicasts every device's input over N-1 hops on a line (ceil((N-1)/2) on a ring) -- exactly what
        // these topologies forbid -- while !ok implies the unicast relay is in play, i.e. N >= 3 on a line or
        // N >= 4 on a ring, so the composite would need multi-hop multicast too. Fail clearly instead of
        // handing composite a route it cannot take.
        TT_FATAL(
            ok,
            "all_gather cannot run on {}: the only algorithm it permits is unicast, and unicast cannot relay this "
            "output: {}. The composite fallback needs multi-hop multicast, which this topology does not permit "
            "either. Use a row size that is a multiple of the output memory's alignment, or the same buffer type "
            "on both sides.",
            unicast_only,
            problem);
        return {GatherRoute::Unicast, unicast_only};
    }

    // 3. Heuristic, then the relay rule: unicast relays by reading its own output back into the CB, and that
    // read must be NoC-read aligned (relay_read_ok). Multicast never reads the output.
    const uint32_t axis = args.get_1d_axis();
    PolicyInputs policy_in;
    policy_in.arch = input.device()->arch();
    policy_in.is_ring = tt::tt_fabric::is_ring_or_torus(args.axis_topology[axis]);
    policy_in.num_devices = args.num_devices;
    policy_in.num_links = args.axis_num_links[axis];
    policy_in.per_link_bytes = per_link_bytes(output_spec, args.axis_num_links[axis]);
    if (!prefers_unicast(policy_in)) {
        return {GatherRoute::Multicast, "heuristic"};
    }
    const auto [ok, problem] = relay_fine();
    if (!ok) {
        return {GatherRoute::Multicast, fmt::format("heuristic chose unicast, but {}", problem)};
    }
    return {GatherRoute::Unicast, "heuristic"};
}

}  // namespace ttnn::operations::ccl
