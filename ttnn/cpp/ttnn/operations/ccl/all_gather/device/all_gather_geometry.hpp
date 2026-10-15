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
#include "all_gather_tuning.hpp"     // EntryPolicy, XferPolicy, RotationMode
#include "kernels/gather_rules.hpp"  // RunSource and the walk's integer rules, shared with the kernels

// Internal to the all_gather op: include from .cpp files only (it is not an installed header).

////////////////////////////////////////////////////////////////
// Host-side all_gather geometry, computed once and shared by select_program_factory and both program
// factories. Two layers:
//
//   GatherGeometry -- what the tensors are: copy mode, pages, and counts in *walk pages* (k = 1 chunks). It
//                     does not depend on k or on any tuning value, so the factories' per-arch heuristics key on
//                     it and k never perturbs them.
//   WalkGeometry   -- how the kernels walk it: k, the chunk and CB slot, stride, xfer, run sources, CB entry.
//                     Every value is in chunks (k included) and goes to the kernels as a CT arg.
//
// Glossary (same as kernels/gather_walk.hpp):
//   input/output page -- one page of the input/output tensor buffer.
//   walk page         -- min(input page, output page). An input page is in_chunks_per_page walk pages; an
//                        output page is out_chunks_per_page walk pages.
//   chunk             -- a walk page / k: the transfer unit.
//   stripe            -- the walk pages (chunks) this device contributes per row of the output.
//   slice             -- the range one (link, worker) owns, cut in whole input pages.
////////////////////////////////////////////////////////////////

namespace ttnn::operations::ccl {

// Picked by input vs output *content* (unaligned) page sizes.
//   Matched (in == out): 1 walk page per input page, 1 per output page.
//   Concat  (out > in) : 1 walk page per input page, out_chunks_per_page > 1; each lands at a byte offset
//                        within a shared output page.
//   Split   (in > out) : in_chunks_per_page walk pages per input page, 1 per output page.
enum class CopyMode : uint8_t { Matched, Concat, Split };

// One (link, worker)'s share of this device's contribution, in walk pages (k = 1 chunks).
struct Slice {
    uint32_t first_chunk = 0;
    uint32_t num_chunks = 0;
};

struct GatherGeometry {
    CopyMode mode = CopyMode::Matched;

    // --- Pages ---
    uint32_t in_page = 0;          // input content page size (Buffer::page_size)
    uint32_t in_aligned_page = 0;  // input page as laid out in memory (Buffer::aligned_page_size)
    uint32_t out_page = 0;         // output content page size
    uint32_t out_aligned_page = 0;
    // NoC read alignment of each side's memory: hal DRAM alignment (64 B Blackhole, 32 B Wormhole) or hal L1
    // alignment (16 B). A chunk must stay a multiple of both (choose_k); the relay reads the output back.
    uint32_t in_read_alignment = 0;
    uint32_t out_read_alignment = 0;

    // --- Walk pages (k = 1 chunks) ---
    uint32_t chunk_size = 0;           // walk page bytes
    uint32_t in_chunks_per_page = 1;   // split_factor
    uint32_t out_chunks_per_page = 1;  // concat ratio
    uint32_t num_input_pages = 0;
    uint32_t num_chunks = 0;         // this device's contribution, in walk pages
    uint32_t chunks_per_stripe = 0;  // walk pages per stripe row

    // --- Predicates ---
    // The output's aligned page is exactly tiled by walk pages: the relay precondition today, and what
    // select_program_factory routes on.
    bool out_packed = false;
    // The TRUE precondition for unicast's relay read-back: out_packed, AND the read of a chunk out of the output
    // is NoC-read aligned. out_packed alone is insufficient: in Concat the chunk is the *input's* aligned page,
    // which is only L1-aligned (16 B) when the input is in L1, so chunk j of a DRAM output page starts at
    // j * chunk -- not a multiple of the DRAM read alignment -- and the relay's NoC read of it is misaligned.
    // No consumer yet: step 3 switches select_program_factory and require_relay_packed() to it.
    bool relay_ok = false;

    // One derivation for everyone: select_program_factory passes the persistent output's spec when there is
    // one, else AllGatherParams::output_spec; the factories pass the output tensor's spec (the same tensor).
    // The aligned output page is the spec's page aligned to its memory's hal alignment, which is what
    // Buffer::aligned_page_size() gives once the buffer exists.
    static GatherGeometry make(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec, int32_t dim_from_end);

    // Unicast-only relay precondition. Checks out_packed, NOT relay_ok (step 3).
    void require_relay_packed() const;

    // Slice `slice_idx` of `num_slices`. The split is over *input pages*, then scaled by in_chunks_per_page, so
    // a slice never cuts an input page.
    Slice slice_of(uint32_t slice_idx, uint32_t num_slices) const;

    // Walk pages per whole input page (split) or output page (concat); one of the two is always 1.
    uint32_t chunks_per_group() const {
        return in_chunks_per_page > out_chunks_per_page ? in_chunks_per_page : out_chunks_per_page;
    }
};

// Packed bytes of the gathered output crossing one link. Pass the same spec as GatherGeometry::make.
uint64_t per_link_bytes(const tt::tt_metal::TensorSpec& output_spec, uint32_t num_links);

// Host mirror of TensorAccessor::contiguous_page_stride() (tt_metal/hw/inc/api/tensor/tensor_accessor.h):
//   interleaved -> bank count (the kernel's NUM_DRAM_BANKS / NUM_L1_BANKS JIT define)
//   sharded     -> tensor_strides[contiguous_dim()] over the dspec the kernel receives.
// The kernels trust this value: cross-page runs are linear only if it matches their accessor. They check it
// with a debug ASSERT only (gather_walk.hpp checked_run_source); this function TT_FATALs where it cannot
// mirror the kernel's value (see the .cpp for why the bank counts agree).
uint32_t contiguous_page_stride(const tt::tt_metal::Buffer& buffer);

// The walk policies the tuning knobs select (defaults: step-1 behaviour).
struct WalkPolicy {
    uint32_t k = 1;  // Tuning::kAutoK = choose_k
    XferPolicy xfer = XferPolicy::Legacy;
    EntryPolicy entry = EntryPolicy::Legacy;
    bool padded_slot = false;
    RotationMode rotation = RotationMode::None;

    static WalkPolicy from(const Tuning& tuning);
};

struct WalkGeometry {
    // Everything below is a kernel CT arg (gather_args.hpp), in chunks.
    uint32_t k = 1;           // chunks per walk page
    uint32_t chunk_size = 0;  // walk page / k
    uint32_t slot_bytes = 0;  // CB slot: chunk_size, or the output's aligned page (padded slot)
    uint32_t in_cpp = 1;      // input chunks per input page
    uint32_t out_cpp = 1;     // output chunks per output page
    uint32_t cps = 0;         // chunks per stripe row
    uint32_t stride = 1;      // walk stride: 1 for Concat, else the output's contiguous_page_stride
    uint32_t xfer = 1;        // chunks per column
    RunSource in_runs{};      // the input's run source
    RunSource out_runs{};     // the output's run source
    uint32_t chunks_per_entry = 1;
    uint32_t cb_entry_bytes = 0;  // chunks_per_entry * slot_bytes
    RotationMode rotation = RotationMode::None;

    // `packets_per_cb_entry` and `max_l1_space` only matter for EntryPolicy::Legacy (the factories' tuned value
    // and get_max_l1_space(input)); Legacy logs the same L1-pressure warning as step 1. `run_cap_bytes` bounds
    // xfer and is the transfer budget choose_k fills.
    static WalkGeometry make(
        const GatherGeometry& geom,
        const tt::tt_metal::Buffer& in,
        const tt::tt_metal::Buffer& out,
        uint32_t packet_size,
        uint32_t run_cap_bytes,
        const WalkPolicy& policy,
        uint32_t packets_per_cb_entry,
        uint32_t cb_depth,
        uint32_t max_l1_space);

    // A walk-page slice in chunks.
    Slice in_chunks(const Slice& s) const { return {s.first_chunk * k, s.num_chunks * k}; }

    // The rot of slice `slice_idx` of `num_slices` (walk-page slice `s`) in direction `dir` (0 = forward;
    // multicast passes 0: its reader and writer share one CB, hence one walk). Device-independent, so the
    // upstream writer and the downstream relaying reader of a stripe always agree.
    uint32_t rot(const Slice& s, uint32_t slice_idx, uint32_t num_slices, uint32_t dir) const;
};

// Worker cores the op may use: subdevice's TENSIX cores, intersected with sub_core_grid. Pass
// AllGatherParams::subdevice_id / ::sub_core_grid.
struct AvailableCores {
    tt::tt_metal::SubDeviceId subdevice_id;
    CoreRangeSet cores;
};
AvailableCores available_worker_cores(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid);

// Creates `n` global semaphores, initial value 0, in L1_SMALL when the device has an L1_SMALL region (else L1,
// with a warning), in order -- then runs a blocking Synchronize over `subdevice_id`, so every device holds them
// before any program that uses them is enqueued. Keep the semaphores (or copies) in shared_variables_t so the
// L1 stays allocated for the cached program's lifetime.
std::vector<tt::tt_metal::GlobalSemaphore> create_semaphores_and_sync(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const CoreRangeSet& cores,
    tt::tt_metal::SubDeviceId subdevice_id,
    uint32_t n);

}  // namespace ttnn::operations::ccl
