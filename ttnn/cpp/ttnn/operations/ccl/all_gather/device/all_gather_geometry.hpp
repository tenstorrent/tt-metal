// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/tensor/tensor.hpp"

// NOTE: deliberately does not include all_gather_device_operation_types.hpp, so that header can include
// all_gather_tuning.hpp (which needs EntryPolicy from here) without a cycle.

////////////////////////////////////////////////////////////////
// Host-side all_gather geometry, computed once and shared by select_program_factory and both program
// factories. Nothing here picks a tuning value: every arch-specific knob stays in the factory that owns
// it and is passed in.
//
// Glossary (same as the factories and kernels/gather_walk.hpp):
//   input/output page -- one page of the input/output tensor buffer.
//   chunk             -- the transfer unit, min(input page, output page). An input page is
//                        in_chunks_per_page chunks; an output page is out_chunks_per_page chunks.
//   stripe            -- the chunks this device contributes per row of the output.
//   slice             -- the chunk range one (link, worker) owns.
////////////////////////////////////////////////////////////////

namespace ttnn::operations::ccl {

// Picked by input vs output *content* (unaligned) page sizes.
//   Matched (in == out): 1 chunk per input page, 1 chunk per output page.
//   Concat  (out > in) : 1 chunk per input page, out_chunks_per_page > 1; each chunk lands at a byte offset
//                        within a shared output page.
//   Split   (in > out) : in_chunks_per_page chunks per input page, 1 chunk per output page.
enum class CopyMode : uint8_t { Matched, Concat, Split };

// [first, end) of `num_units` divided into `num_slices` parts, the first `num_units % num_slices` parts one
// unit longer. Identical arithmetic to the unicast factory's slice split (all_gather_unicast_factory.cpp
// 566-570) and the multicast factory's per-link split (all_gather_multicast_factory.cpp 426-429).
struct UnitRange {
    uint32_t first = 0;
    uint32_t end = 0;
};
UnitRange split_evenly(uint32_t num_units, uint32_t slice_idx, uint32_t num_slices);

// One (link, worker)'s share of this device's contribution.
struct Slice {
    uint32_t first_input_page = 0;  // input_tile_id_start in the factories
    uint32_t end_input_page = 0;    // input_tile_id_end
    uint32_t first_chunk = 0;       // local_output_start
    uint32_t num_chunks = 0;        // num_worker_output_chunks
};

struct GatherGeometry {
    CopyMode mode = CopyMode::Matched;

    // --- Pages ---
    uint32_t in_page = 0;          // input content page size (Buffer::page_size)
    uint32_t in_aligned_page = 0;  // input page as laid out in memory (Buffer::aligned_page_size)
    uint32_t out_page = 0;         // output content page size
    uint32_t out_aligned_page = 0;
    bool out_is_dram = false;
    // NoC read alignment of the output's memory: hal DRAM alignment (64 B Blackhole, 32 B Wormhole) or hal L1
    // alignment (16 B). The unicast relay reads the output back, so this is the alignment that read must meet.
    uint32_t out_read_alignment = 0;

    // --- Chunks ---
    uint32_t chunk_size = 0;           // output_chunk_size in the factories
    uint32_t in_chunks_per_page = 1;   // split_factor
    uint32_t out_chunks_per_page = 1;  // output_chunks_per_page
    uint32_t num_input_pages = 0;
    uint32_t num_chunks = 0;         // this device's contribution in chunks (num_output_chunks)
    uint32_t chunks_per_stripe = 0;  // output_chunks_per_stripe
    uint32_t rows = 0;               // num_chunks / chunks_per_stripe
    uint32_t num_devices = 0;

    // --- Predicates ---
    // The output's aligned page is exactly tiled by chunks. This is the kernels' `packed` (gather_walk.hpp
    // walk_plan) and the predicate both select_program_factory (device_operation.cpp:325) and the unicast relay
    // TT_FATAL (unicast_factory.cpp:198) check today.
    bool out_packed = false;
    // The TRUE precondition for unicast's relay read-back: out_packed, AND the read of a chunk out of the output
    // is NoC-read aligned. out_packed alone is insufficient: in Concat the chunk is the *input's* aligned page,
    // which is only L1-aligned (16 B) when the input is in L1, so chunk k of a DRAM output page starts at
    // k * chunk -- not a multiple of the DRAM read alignment (32/64 B) -- and the relay's NoC read of it is
    // misaligned even though the page itself is exactly tiled. L1 outputs are always fine: every chunk is a
    // multiple of 16 B (the L1 alignment, or for Split enforced by use_composite_all_gather).
    bool relay_ok = false;

    // Built from the real output buffer: the factories' derivation (unicast 182-214 + 223-249, multicast
    // 242-261 + 269-294), same arithmetic, same TT_FATAL messages.
    static GatherGeometry make(const Tensor& input, const Tensor& output, int32_t dim_from_end, uint32_t num_devices);
    // Built from the output spec, before any output buffer exists: select_program_factory's derivation
    // (device_operation.cpp 316-326). Equal to the Tensor overload whenever the output buffer was created from
    // this spec; a persistent output with different sharding can differ (see WIRING.md).
    static GatherGeometry make(
        const Tensor& input, const tt::tt_metal::TensorSpec& output_spec, int32_t dim_from_end, uint32_t num_devices);

    // Unicast-only: today's relay TT_FATAL (unicast_factory.cpp 198-206), message unchanged. Checks out_packed,
    // NOT relay_ok, so wiring it in changes nothing; switching to relay_ok is a separate, behaviour-changing fix.
    void require_relay_packed() const;

    // Slice `slice_idx` of `num_slices`. The split is over *input pages*, then scaled by in_chunks_per_page, so
    // a slice never cuts an input page -- splitting num_chunks directly would differ in Split mode.
    Slice slice_of(uint32_t slice_idx, uint32_t num_slices) const;

    CopyMode copy_mode() const { return mode; }
    uint32_t chunks_per_group() const {
        return in_chunks_per_page > out_chunks_per_page ? in_chunks_per_page : out_chunks_per_page;
    }
};

// Packed bytes of the gathered output crossing one link (device_operation.cpp 261-262, unicast 268-269).
uint64_t per_link_bytes(const tt::tt_metal::TensorSpec& output_spec, uint32_t num_links);

// Host mirror of TensorAccessor::contiguous_page_stride() (tt_metal/hw/inc/api/tensor/tensor_accessor.h:142,499):
//   interleaved -> bank count (the kernel's NUM_DRAM_BANKS / NUM_L1_BANKS JIT define)
//   sharded     -> tensor_strides[contiguous_dim()] over the dspec the kernel receives.
uint32_t contiguous_page_stride(const tt::tt_metal::Buffer& buffer);

// Host mirror of gather_walk.hpp RunSource / run_source().
struct HostRunSource {
    bool sub_page = false;
    bool cross_page = false;
};
HostRunSource run_source(bool packed, uint32_t chunks_per_page, uint32_t page_stride, uint32_t stride);

// How a CB entry is sized.
//   Legacy       -- today: a whole number of packet loads, rounded to whole pages (chunks_per_group), times
//                   packets_per_cb_entry, clamped by L1 (unicast 428-447, multicast 340-358).
//   ScatterAware -- one transfer's worth, but no more than NOC_SCATTER_WRITE_MAX_CHUNKS of the shortest output
//                   run fit: min(xfer, 4 * min_output_run).
enum class EntryPolicy : uint8_t { Legacy, ScatterAware };

// Mirror of NOC_SCATTER_WRITE_MAX_CHUNKS (tt_metal/fabric/fabric_edm_packet_header.hpp:196); the .cpp
// static_asserts the two agree when that header is reachable.
inline constexpr uint32_t kNocScatterWriteMaxChunks = 4;

// Host mirror of gather_walk.hpp chunks_per_transfer(); burst = hal::get_noc_max_burst_size_bytes().
uint32_t chunks_per_transfer(uint32_t packet_size, uint32_t chunk_size, uint32_t run_cap_bytes);

struct WalkGeometry {
    uint32_t out_page_stride = 1;  // output accessor contiguous_page_stride()
    uint32_t in_page_stride = 1;   // input accessor contiguous_page_stride()
    uint32_t stride = 1;           // walk stride: 1 for Concat, else out_page_stride (walk_plan)
    uint32_t xfer_max = 1;         // chunks_per_transfer(packet, chunk, run_cap) -- the kernels' xfer_max
    uint32_t xfer = 1;             // walk_plan's xfer: xfer_max if out_packed, else 1
    HostRunSource out_runs;        // walk_plan's `out`
    HostRunSource in_runs;         // reader's `in_src` (unicast_reader.cpp 75-79, multicast_reader.cpp 189-193)
    uint32_t min_output_run = 1;   // shortest run the output can offer (ScatterAware sizing)
    uint32_t chunks_per_entry = 1;
    uint32_t cb_entry_bytes = 0;  // cb_page_size in the factories
    uint32_t cb_depth = 2;

    // `packets_per_cb_entry` and `max_l1_space` only matter for Legacy (and are the factories' existing tuned
    // value / get_max_l1_space(input)). Legacy logs the same L1-pressure warning as today.
    static WalkGeometry make(
        const GatherGeometry& geom,
        const tt::tt_metal::Buffer& in,
        const tt::tt_metal::Buffer& out,
        uint32_t packet_size,
        uint32_t run_cap_bytes,
        EntryPolicy policy,
        uint32_t packets_per_cb_entry,
        uint32_t cb_depth,
        uint32_t max_l1_space);
};

// Worker cores the op may use: subdevice's TENSIX cores, intersected with sub_core_grid (unicast 38-42,
// multicast 25-29). Pass AllGatherParams::subdevice_id / ::sub_core_grid.
struct AvailableCores {
    tt::tt_metal::SubDeviceId subdevice_id;
    CoreRangeSet cores;
};
AvailableCores available_worker_cores(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid);

// Owns the op's global semaphores; keep it (or copies of its elements) in shared_variables_t so the L1 stays
// allocated for the cached program's lifetime.
struct GatherSemaphores {
    std::vector<tt::tt_metal::GlobalSemaphore> sems;
    const tt::tt_metal::GlobalSemaphore& operator[](size_t i) const { return sems.at(i); }
    size_t size() const { return sems.size(); }
};
// `n` semaphores, initial value 0, in L1_SMALL when the device has an L1_SMALL region (else L1, with today's
// warning), created in order, then Synchronize over `subdevice_id` (unicast 52-66, multicast 36-48).
GatherSemaphores allocate_gather_semaphores(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const CoreRangeSet& cores,
    tt::tt_metal::SubDeviceId subdevice_id,
    uint32_t n);

}  // namespace ttnn::operations::ccl
