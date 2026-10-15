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

#if __has_include("tt_metal/fabric/fabric_edm_packet_header.hpp")
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
static_assert(
    ttnn::operations::ccl::kNocScatterWriteMaxChunks == NOC_SCATTER_WRITE_MAX_CHUNKS,
    "kNocScatterWriteMaxChunks must mirror NOC_SCATTER_WRITE_MAX_CHUNKS");
#endif

namespace ttnn::operations::ccl {

namespace {

// Shared tail of both make() overloads. `out_page`/`out_aligned_page` are the only inputs that differ between
// "real output buffer" and "output spec".
GatherGeometry make_gather_geometry_impl(
    const Tensor& input,
    uint32_t out_page,
    uint32_t out_aligned_page,
    bool out_is_dram,
    int32_t dim_from_end,
    uint32_t num_devices) {
    const auto& in_buf = *input.buffer();
    GatherGeometry g;
    g.num_devices = num_devices;
    g.out_is_dram = out_is_dram;
    g.out_read_alignment =
        out_is_dram ? tt::tt_metal::hal::get_dram_alignment() : tt::tt_metal::hal::get_l1_alignment();

    // --- Copy mode ---
    // The kernel always reads whole *aligned* input pages into L1 (required by the input's NoC read alignment,
    // DRAM or L1) but writes at output *content* (unaligned) granularity -- which is why chunk sizing differs by
    // copy mode.
    g.in_aligned_page = in_buf.aligned_page_size();
    g.in_page = in_buf.page_size();
    g.out_page = out_page;
    g.out_aligned_page = out_aligned_page;
    // matched/concat write a whole aligned input page (== L1 read stride) into an output slot;
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
    g.relay_ok = g.out_packed && (!g.out_is_dram || g.chunk_size % g.out_read_alignment == 0);

    // --- Chunk count ---
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
    g.rows = g.num_chunks / g.chunks_per_stripe;
    return g;
}

}  // namespace

GatherGeometry GatherGeometry::make(
    const Tensor& input, const Tensor& output, int32_t dim_from_end, uint32_t num_devices) {
    const auto& out_buf = *output.buffer();
    return make_gather_geometry_impl(
        input,
        static_cast<uint32_t>(out_buf.page_size()),
        static_cast<uint32_t>(out_buf.aligned_page_size()),
        out_buf.is_dram(),
        dim_from_end,
        num_devices);
}

GatherGeometry GatherGeometry::make(
    const Tensor& input, const tt::tt_metal::TensorSpec& output_spec, int32_t dim_from_end, uint32_t num_devices) {
    // Content page from the spec, aligned with the hal alignment of its memory --
    // the same value Buffer::aligned_page_size() gives once the buffer exists (allocator alignment == hal's).
    const uint32_t out_page = output_spec.compute_page_size_bytes();
    const bool out_is_dram = output_spec.memory_config().is_dram();
    const uint32_t out_align =
        out_is_dram ? tt::tt_metal::hal::get_dram_alignment() : tt::tt_metal::hal::get_l1_alignment();
    return make_gather_geometry_impl(
        input, out_page, tt::align(out_page, out_align), out_is_dram, dim_from_end, num_devices);
}

void GatherGeometry::require_relay_packed() const {
    // Relay reads our own output back into the CB, which is packed at chunk stride, so the output's aligned page
    // must be exactly tiled by chunks. select_program_factory routes these away; reaching here means unicast was
    // forced (2D bent axis or neighbour-exchange fabric).
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

UnitRange split_evenly(uint32_t num_units, uint32_t slice_idx, uint32_t num_slices) {
    const uint32_t per_slice = num_units / num_slices;
    const uint32_t remainder = num_units % num_slices;
    return {
        .first = (slice_idx * per_slice) + std::min(slice_idx, remainder),
        .end = ((slice_idx + 1) * per_slice) + std::min(slice_idx + 1, remainder),
    };
}

Slice GatherGeometry::slice_of(uint32_t slice_idx, uint32_t num_slices) const {
    // Split input pages, then scale by split_factor (1 unless Split).
    const auto pages = split_evenly(num_input_pages, slice_idx, num_slices);
    const uint32_t first_chunk = pages.first * in_chunks_per_page;
    const uint32_t end_chunk = pages.end * in_chunks_per_page;
    return {
        .first_input_page = pages.first,
        .end_input_page = pages.end,
        .first_chunk = first_chunk,
        .num_chunks = end_chunk - first_chunk,
    };
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
        // device (jit_device_config.cpp: DRAM views, logical compute cores). Use the *device's* allocator, not
        // buffer.allocator(), which may be a sub-device allocator with fewer L1 banks.
        return buffer.device()->allocator()->get_num_banks(buffer.buffer_type());
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
    return stride;
}

HostRunSource run_source(bool packed, uint32_t chunks_per_page, uint32_t page_stride, uint32_t stride) {
    if (!packed) {
        return {false, false};
    }
    const bool sub_page = chunks_per_page > 1 && stride == 1;
    return {sub_page, chunks_per_page > 1 ? (sub_page && page_stride == 1) : (page_stride == stride)};
}

uint32_t chunks_per_transfer(uint32_t packet_size, uint32_t chunk_size, uint32_t run_cap_bytes) {
    const uint32_t burst = tt::tt_metal::hal::get_noc_max_burst_size_bytes();
    uint32_t cap = packet_size < burst ? packet_size : burst;
    if (run_cap_bytes != 0 && run_cap_bytes < cap) {
        cap = run_cap_bytes;
    }
    return cap < chunk_size ? 1u : cap / chunk_size;
}

WalkGeometry WalkGeometry::make(
    const GatherGeometry& geom,
    const tt::tt_metal::Buffer& in,
    const tt::tt_metal::Buffer& out,
    uint32_t packet_size,
    uint32_t run_cap_bytes,
    EntryPolicy policy,
    uint32_t packets_per_cb_entry,
    uint32_t cb_depth,
    uint32_t max_l1_space) {
    WalkGeometry w;
    w.cb_depth = cb_depth;

    // --- Walk order: gather_walk.hpp walk_plan() ---
    w.out_page_stride = contiguous_page_stride(out);
    w.in_page_stride = contiguous_page_stride(in);
    // Concat packs chunks inside one output page, so there neighbours are one chunk apart.
    w.stride = geom.out_chunks_per_page > 1 ? 1u : w.out_page_stride;
    w.xfer_max = chunks_per_transfer(packet_size, geom.chunk_size, run_cap_bytes);
    w.xfer = geom.out_packed ? w.xfer_max : 1u;
    w.out_runs = run_source(geom.out_packed, geom.out_chunks_per_page, w.out_page_stride, w.stride);
    // The readers' input source (`in_src` in unicast_reader.cpp / multicast_reader.cpp). The kernel compares the
    // accessor's aligned page size, which TensorAccessorArgs sets to Buffer::aligned_page_size().
    w.in_runs = run_source(
        geom.in_aligned_page == geom.in_chunks_per_page * geom.chunk_size,
        geom.in_chunks_per_page,
        w.in_page_stride,
        w.stride);

    // --- Shortest output run (ScatterAware) ---
    if (!geom.out_packed) {
        w.min_output_run = 1;
    } else if (w.stride == 1 && w.out_runs.cross_page) {
        w.min_output_run = geom.chunks_per_stripe;
    } else {
        w.min_output_run = std::max(1u, geom.chunks_per_stripe / w.stride);
    }

    if (policy == EntryPolicy::Legacy) {
        // A CB entry holds a whole number of packet loads, and of input pages (split) or output pages (concat), so
        // the entry boundary never cuts a page; one of those two counts is always 1. A packet can still end early:
        // broken contiguity can fill its scatter chunks before its payload.
        const uint32_t chunks_per_group = geom.chunks_per_group();
        uint32_t chunks_per_packet = std::max(1u, packet_size / geom.chunk_size);
        chunks_per_packet = std::max(chunks_per_group, (chunks_per_packet / chunks_per_group) * chunks_per_group);
        uint32_t cb_page_size = chunks_per_packet * geom.chunk_size;
        // An integer multiplier preserves the whole-packet and whole-page properties above. The clamp is
        // defensive: it holds whatever packets_per_cb_entry is tuned to, including 1.
        const uint32_t multiplier = std::clamp(max_l1_space / (cb_depth * cb_page_size), 1u, packets_per_cb_entry);
        if (multiplier < packets_per_cb_entry) {
            log_warning(
                tt::LogOp,
                "CircularBuffer depth reduced due to L1 pressure (only {} B available), performance may regress.",
                max_l1_space);
        }
        cb_page_size *= multiplier;
        w.cb_entry_bytes = cb_page_size;
        w.chunks_per_entry = cb_page_size / geom.chunk_size;
    } else {
        // One transfer, but at most kNocScatterWriteMaxChunks of the shortest output run: a packet carries at
        // most that many scatter segments, so a longer entry only waits on segments it cannot send.
        // NOTE: unlike Legacy this does not round to chunks_per_group, so an entry may cut a page.
        w.chunks_per_entry = std::max(1u, std::min(w.xfer, kNocScatterWriteMaxChunks * w.min_output_run));
        w.cb_entry_bytes = w.chunks_per_entry * geom.chunk_size;
    }
    return w;
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
