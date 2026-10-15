// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_geometry.hpp"

#include <algorithm>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::ccl {

namespace {

uint32_t ag_read_alignment(bool is_dram) {
    return is_dram ? tt::tt_metal::hal::get_dram_alignment() : tt::tt_metal::hal::get_l1_alignment();
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
    // hal's).
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
    const bool is_split = g.in_page > g.out_page;
    g.chunk_size = is_split ? g.out_page : g.in_aligned_page;
    g.out_chunks_per_page = is_split ? 1u : g.out_page / g.in_page;
    g.in_chunks_per_page = is_split ? g.in_page / g.out_page : 1u;
    g.mode = is_split ? CopyMode::Split : (g.out_chunks_per_page > 1 ? CopyMode::Concat : CopyMode::Matched);
    TT_FATAL(
        g.out_chunks_per_page == 1 || g.in_aligned_page == g.in_page,
        "concat requires an unpadded input page");  // so slots align to content

    // --- Predicates ---
    g.out_packed = g.out_aligned_page == g.out_chunks_per_page * g.chunk_size;
    g.relay_ok = g.out_packed && (!out_is_dram || g.chunk_size % g.out_read_alignment == 0);

    // --- Walk-page count ---
    g.num_input_pages = in_buf.num_pages();
    g.num_chunks = g.num_input_pages * g.in_chunks_per_page;
    TT_FATAL(
        g.num_chunks / g.in_chunks_per_page == g.num_input_pages,
        "all_gather output chunk count overflowed uint32: {} input pages x split factor {}",
        g.num_input_pages,
        g.in_chunks_per_page);

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
    // Stripe = this device's contiguous run of chunks per row. Measured in chunks (not output pages) so
    // multi-shard concat works: a stripe's chunks are laid across output pages and may straddle pages.
    g.chunks_per_stripe = input_pages_per_stripe * g.in_chunks_per_page;
    TT_FATAL(g.chunks_per_stripe > 0, "output_chunks_per_stripe must be > 0");
    return g;
}

void GatherGeometry::require_relay_packed() const {
    // Relay reads our own output back into the CB, which is packed at chunk stride, so the output's aligned page
    // must be exactly tiled by chunks. select_program_factory routes these away; reaching here means unicast was
    // forced (2D bent axis, neighbour-exchange fabric, or TT_AG_TUNING factory=unicast).
    // step 3: check relay_ok instead (also rejects a DRAM output whose chunk misses the DRAM read alignment), and
    // accept a padded output under the padded-slot policy (slot == out_aligned_page tiles it exactly).
    TT_FATAL(
        out_packed,
        "all_gather unicast cannot relay a padded output page: {} B of content in a {} B page holding {} chunk(s) of "
        "{} B. Use a row size that is a multiple of the output's memory alignment, or the same buffer type on both "
        "sides.",
        out_page,
        out_aligned_page,
        out_chunks_per_page,
        chunk_size);
}

Slice GatherGeometry::slice_of(uint32_t slice_idx, uint32_t num_slices) const {
    // [first, end) of the input pages, the first `num_input_pages % num_slices` slices one page longer; then
    // scaled by split_factor (1 unless Split).
    const uint32_t per_slice = num_input_pages / num_slices;
    const uint32_t remainder = num_input_pages % num_slices;
    const uint32_t first = (slice_idx * per_slice) + std::min(slice_idx, remainder);
    const uint32_t end = ((slice_idx + 1) * per_slice) + std::min(slice_idx + 1, remainder);
    return {.first_chunk = first * in_chunks_per_page, .num_chunks = (end - first) * in_chunks_per_page};
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
    const auto& tensor_shape = dspec->tensor_shape_in_pages();
    const auto& shard_shape = dspec->shard_shape_in_pages();
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

WalkPolicy WalkPolicy::from(const Tuning& tuning) {
    return {
        .k = tuning.walk_k,
        .xfer = tuning.xfer_policy,
        .entry = tuning.entry_policy,
        .padded_slot = tuning.padded_slot,
        .rotation = tuning.rotation,
    };
}

WalkGeometry WalkGeometry::make(
    const GatherGeometry& geom,
    const tt::tt_metal::Buffer& in,
    const tt::tt_metal::Buffer& out,
    uint32_t packet_size,
    uint32_t run_cap_bytes,
    const WalkPolicy& policy,
    uint32_t packets_per_cb_entry,
    uint32_t cb_depth,
    uint32_t max_l1_space) {
    WalkGeometry w;
    w.rotation = policy.rotation;
    const uint32_t burst = tt::tt_metal::hal::get_noc_max_burst_size_bytes();
    const uint32_t page = geom.chunk_size;

    // --- Slot (padded-slot policy) ---
    // One chunk per output page whose aligned page is padded: give each chunk an output page's worth of L1, so
    // the CB steps like the output and output runs may cross pages (pad included). A run may merge n chunks only
    // where the L1 slot stride equals the memory stride: true for this output by construction, and for the input
    // only when its aligned page is the slot (see in_runs below).
    const bool padded_slot =
        policy.padded_slot && geom.out_chunks_per_page == 1 && geom.out_aligned_page > geom.chunk_size;

    // --- k ---
    // Only Matched with unpadded pages on both sides: a padded page cannot be cut into aligned chunks that tile
    // it, and Concat / Split already give sub-page walk pages (Concat: no lanes, k would only shrink the chunk).
    const bool k_eligible = geom.mode == CopyMode::Matched && geom.in_aligned_page == geom.in_page &&
                            geom.out_aligned_page == geom.out_page && !padded_slot;
    const uint32_t align = std::max(geom.in_read_alignment, geom.out_read_alignment);
    // choose_k fills the transfer budget: a run cap is a deliberate run-length limit, so it bounds the budget.
    const uint32_t budget = run_cap_bytes != 0 ? std::min(packet_size, run_cap_bytes) : packet_size;
    uint32_t k = 1;
    if (policy.k == Tuning::kAutoK) {
        k = k_eligible ? choose_k(page, budget, burst, align) : 1u;
    } else if (policy.k > 1) {
        if (!k_eligible) {
            log_warning(
                tt::LogOp,
                "TT_AG_TUNING k={}: not applicable (copy mode {}, pages {}/{} -> {}/{}); using k=1",
                policy.k,
                static_cast<int>(geom.mode),
                geom.in_page,
                geom.in_aligned_page,
                geom.out_page,
                geom.out_aligned_page);
        } else {
            TT_FATAL(
                k_is_valid(policy.k, page, align),
                "TT_AG_TUNING k={}: a {} B page does not split into chunks aligned to {} B",
                policy.k,
                page,
                align);
            k = policy.k;
        }
    }
    w.k = k;
    w.chunk_size = page / k;
    w.slot_bytes = padded_slot ? geom.out_aligned_page : w.chunk_size;
    w.in_cpp = geom.in_chunks_per_page * k;
    w.out_cpp = geom.out_chunks_per_page * k;
    w.cps = geom.chunks_per_stripe * k;

    // --- Stride and run sources (gather_rules.hpp run_source) ---
    const uint32_t out_page_stride = contiguous_page_stride(out);
    const uint32_t in_page_stride = contiguous_page_stride(in);
    // Concat packs walk pages inside one output page, so there neighbours are one walk page apart.
    w.stride = geom.out_chunks_per_page > 1 ? 1u : out_page_stride;
    // `packed`: the accessor's aligned page is exactly its chunks' slots, so a run stepping by the aligned page
    // and the CB stepping by the slot agree. At slot == chunk this is step 1's predicate.
    const bool out_packed_slot = geom.out_aligned_page == w.out_cpp * w.slot_bytes;
    w.out_runs = run_source(out_packed_slot, k, w.out_cpp, out_page_stride, w.stride);
    if (w.slot_bytes != w.chunk_size && w.in_cpp != 1) {
        // Chunks inside an input page sit at chunk stride, the CB at slot stride: one read per chunk.
        w.in_runs = {false, false};
    } else {
        w.in_runs = run_source(geom.in_aligned_page == w.in_cpp * w.slot_bytes, k, w.in_cpp, in_page_stride, w.stride);
        w.in_runs.whole_page = w.in_runs.whole_page && w.slot_bytes == w.chunk_size;
    }

    // --- xfer ---
    const uint32_t xfer_max = chunks_per_transfer(packet_size, w.slot_bytes, run_cap_bytes, burst);
    if (policy.xfer == XferPolicy::Legacy) {
        // Step 1: an unpacked output walks in plain ascending order.
        w.xfer = out_packed_slot ? xfer_max : 1u;
    } else {
        // At most gather_scatter_max_chunks runs per column, given the shortest lane run the output guarantees:
        // one walk page without cross-page runs (an output page's walk pages when whole_page), else the lane
        // pages in one stripe row (runs stop at the row edge). Shard edges are not modelled.
        uint32_t contig = 1;
        if (!w.out_runs.cross_page) {
            contig = w.out_runs.whole_page ? w.out_cpp / k : 1u;
        } else {
            contig = std::max(1u, geom.chunks_per_stripe / w.stride);
        }
        w.xfer = xfer_for_segments(xfer_max, k, contig);
    }

    // --- CB entry ---
    if (policy.entry == EntryPolicy::Legacy) {
        // Step 1's formula: a whole number of packet loads, rounded to whole input (split) or output (concat)
        // pages -- chunks_per_group of the k = 1 geometry, which is 1 whenever k > 1 (Matched only), so k's
        // sub-page chunks are not rounded back up to a page -- times packets_per_cb_entry, clamped by L1.
        // An entry that cuts a walk page is safe: the CB is packed, and reader and writer agree on which seqno
        // sits in which slot whatever their run grouping (gather_walk.hpp; host gtest ReaderWriterGrouping).
        const uint32_t chunks_per_group = geom.chunks_per_group();
        uint32_t chunks_per_packet = std::max(1u, packet_size / w.slot_bytes);
        chunks_per_packet = std::max(chunks_per_group, (chunks_per_packet / chunks_per_group) * chunks_per_group);
        const uint32_t entry_bytes = chunks_per_packet * w.slot_bytes;
        // An integer multiplier preserves the whole-packet and whole-page properties above. The clamp is
        // defensive: it holds whatever packets_per_cb_entry is tuned to, including 1.
        const uint32_t multiplier = std::clamp(max_l1_space / (cb_depth * entry_bytes), 1u, packets_per_cb_entry);
        if (multiplier < packets_per_cb_entry) {
            log_warning(
                tt::LogOp,
                "CircularBuffer depth reduced due to L1 pressure (only {} B available), performance may regress.",
                max_l1_space);
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
            min_output_run = k * std::max(1u, geom.chunks_per_stripe / w.stride);
        }
        w.chunks_per_entry = std::max(1u, std::min(w.xfer, gather_scatter_max_chunks * min_output_run));
    }
    w.cb_entry_bytes = w.chunks_per_entry * w.slot_bytes;
    return w;
}

uint32_t WalkGeometry::rot(const Slice& s, uint32_t slice_idx, uint32_t num_slices, uint32_t dir) const {
    if (rotation == RotationMode::None || stride <= 1) {
        return 0;
    }
    const uint32_t target =
        rot_target(slice_idx, num_slices, stride, rotation == RotationMode::AbsoluteDirStagger, dir);
    // s.first_chunk is in walk pages: the slice's first lane is that page mod stride.
    return rot_for_target(target, s.first_chunk, stride);
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

}  // namespace ttnn::operations::ccl
