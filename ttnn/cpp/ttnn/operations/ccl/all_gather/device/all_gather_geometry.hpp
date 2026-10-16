// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "all_gather_policy.hpp"                  // WalkPolicy, SharedPolicy
#include "all_gather_device_operation_types.hpp"  // AllGatherParams (route_all_gather)
#include "kernels/gather_rules.hpp"               // RunSource and the walk's integer rules, shared with the kernels

// Internal to the all_gather op: include from .cpp files only (it is not an installed header).

////////////////////////////////////////////////////////////////
// Host-side all_gather geometry, computed once and shared by select_program_factory and both program
// factories. Two layers:
//
//   GatherGeometry -- what the tensors are: copy mode, pages, and counts in *walk pages* (the k = 1 unit). It
//                     does not depend on k or on any policy value, so the per-arch policies (all_gather_policy.hpp)
//                     key on it and k never perturbs them.
//   WalkGeometry   -- how the kernels walk it: k, the chunk and CB slot, stride, xfer, run sources, CB entry.
//                     Every value is in chunks (k included) and goes to the kernels as a CT arg.
//
// Glossary (same as kernels/gather_walk.hpp):
//   input/output page -- one page of the input/output tensor buffer.
//   walk page         -- the k = 1 transfer unit, by copy mode: Matched and Concat walk the input's *aligned* page
//                        (the kernels read whole aligned input pages; an output page holds
//                        out_walk_pages_per_page of them), Split walks the output's *content* page (an input page
//                        holds in_walk_pages_per_page of them).
//   chunk             -- a walk page / k: the transfer unit (WalkGeometry and the kernels only).
//   stripe            -- the walk pages (chunks) this device contributes per row of the output.
//   slice             -- the range one (link, worker) owns, cut in whole input pages.
////////////////////////////////////////////////////////////////

namespace ttnn::operations::ccl {

// Picked by input vs output *content* (unaligned) page sizes.
//   Matched (in == out): 1 walk page per input page, 1 per output page.
//   Concat  (out > in) : 1 walk page per input page, out_walk_pages_per_page > 1; each lands at a byte offset
//                        within a shared output page.
//   Split   (in > out) : in_walk_pages_per_page walk pages per input page, 1 per output page.
enum class CopyMode : uint8_t { Matched, Concat, Split };

// One (link, worker)'s share of this device's contribution, in walk pages.
struct WalkPageSlice {
    uint32_t first = 0;
    uint32_t count = 0;
};

// The same share in chunks (walk pages * k): what the kernels walk.
struct ChunkSlice {
    uint32_t first = 0;
    uint32_t count = 0;
};

struct GatherGeometry {
    CopyMode mode = CopyMode::Matched;

    // --- Pages ---
    uint32_t in_page = 0;           // input content page size (Buffer::page_size)
    uint32_t in_aligned_page = 0;   // input page as laid out in memory (Buffer::aligned_page_size)
    uint32_t out_page = 0;          // output content page size
    uint32_t out_aligned_page = 0;  // the spec's page aligned to its memory; checked against the buffer (WalkGeometry)
    // NoC read alignment of each side's memory: hal DRAM alignment (64 B Blackhole, 32 B Wormhole) or hal L1
    // alignment (16 B). A chunk must stay a multiple of both (choose_k); the relay reads the output back.
    uint32_t in_read_alignment = 0;
    uint32_t out_read_alignment = 0;

    // --- Walk pages ---
    uint32_t walk_page_bytes = 0;
    uint32_t in_walk_pages_per_page = 1;   // split ratio
    uint32_t out_walk_pages_per_page = 1;  // concat ratio
    uint32_t num_input_pages = 0;
    uint32_t num_walk_pages = 0;         // this device's contribution
    uint32_t walk_pages_per_stripe = 0;  // per stripe row

    // --- Predicates ---
    // Unicast's relay read-back is legal at k = 1: gather_rules.hpp relay_read_ok() -- the ONE definition. True for
    // every one-walk-page-per-page output (relay_needs_padded_slot widens the CB slot to the output's aligned page
    // where it is padded), and for Concat only when the page is exactly tiled by walk pages AND every walk-page
    // offset is aligned to the output's read alignment. Consumers: route_all_gather (all_gather() and
    // select_program_factory) and the unicast factory's require_relay_ok(); the factory re-checks the chunk it
    // actually walks with WalkGeometry::relay_ok.
    bool relay_ok = false;
    // The relay needs the padded CB slot (slot = out_aligned_page). Unicast's policy turns the padded-slot walk on
    // always (a no-op wherever this is false); multicast's leaves it off.
    bool relay_needs_padded_slot = false;

    // One derivation for everyone: select_program_factory passes the persistent output's spec when there is
    // one, else AllGatherParams::output_spec; the factories pass the output tensor's spec (the same tensor).
    // The aligned output page is the spec's page aligned to its memory's hal alignment, which is what
    // Buffer::aligned_page_size() gives once the buffer exists (WalkGeometry::make TT_FATALs otherwise).
    // TT_FATALs unless the larger content page is a whole number of the smaller (route_all_gather's layout check
    // routes those to composite first).
    static GatherGeometry make(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec, int32_t dim_from_end);

    // Unicast-only relay precondition: TT_FATALs unless relay_ok, or `relays` is false (no device reads its
    // output back: 2 devices, a 3-device ring). route_all_gather never picks unicast otherwise; reaching the
    // FATAL means the op was built through ttnn::prim::all_gather with a hand-made program factory choice.
    void require_relay_ok(bool relays) const;

    // Slice `slice_idx` of `num_slices`. The split is over *input pages*, then scaled by in_walk_pages_per_page,
    // so a slice never cuts an input page.
    WalkPageSlice slice_of(uint32_t slice_idx, uint32_t num_slices) const;

    // Walk pages per whole input page (split) or output page (concat); one of the two is always 1.
    uint32_t walk_pages_per_group() const {
        return in_walk_pages_per_page > out_walk_pages_per_page ? in_walk_pages_per_page : out_walk_pages_per_page;
    }
};

// --- Algorithm routing: the ONE place unicast / multicast / composite is decided ---
// ttnn::all_gather() calls route_all_gather() to choose composite vs native, and select_program_factory calls
// it again (same params, same spec) to pick the factory, so the two cannot disagree.
enum class GatherRoute : uint8_t { Unicast, Multicast, Composite };

struct RouteDecision {
    GatherRoute route = GatherRoute::Multicast;
    std::string reason;  // why; for Composite this is the user-facing log line
};

// A gather-dim-padded input needs composite. Kept separate because it must run BEFORE an output spec is built:
// compute_output_specs_helper can TT_FATAL on a legacy output config that cannot hold the gathered shape, and
// such an input has to route to composite instead. route_all_gather() re-checks it (cheap) for prim callers.
std::optional<std::string> gather_dim_padding_reason(const Tensor& input, int32_t dim_from_end);

// Everything else, in this order:
//   1. layout   -- shapes/pages the native kernels cannot address (formerly all_gather.cpp's
//                  use_composite_all_gather body)                                         -> Composite
//   2. topology -- true 2D                                                                -> Multicast
//                  bent 2D axis / FABRIC_1D_NEIGHBOR_EXCHANGE (unicast only)              -> Unicast if
//                  relay_ok || !relays, else TT_FATAL (see the .cpp: composite's all_broadcast multicasts
//                  N-1 hops, which these topologies forbid, so it is not a fallback there)
//   3. heuristic (prefers_unicast, all_gather_policy.hpp), then unicast only if relay_ok || !relays
//                                                                                         -> Unicast/Multicast
// `output_spec` is the persistent output's own spec when there is one, else AllGatherParams::output_spec; it feeds
// the page checks of stage 1, the relay rule and the heuristic (what the factories will see). Stage 1's shape check
// uses AllGatherParams::output_spec, so a persistent output of the wrong shape still reaches the op's validate and
// its specific error rather than composite.
RouteDecision route_all_gather(
    const AllGatherParams& args, const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);

// gather_rules.hpp unicast_relays() for these params (1D axis only).
bool unicast_relays(const AllGatherParams& args);

// Packed bytes of the gathered output crossing one link. Pass the same spec as GatherGeometry::make.
uint64_t per_link_bytes(const tt::tt_metal::TensorSpec& output_spec, uint32_t num_links);

// Host mirror of TensorAccessor::contiguous_page_stride() (tt_metal/hw/inc/api/tensor/tensor_accessor.h):
//   interleaved -> bank count (the kernel's NUM_DRAM_BANKS / NUM_L1_BANKS JIT define)
//   sharded     -> tensor_strides[contiguous_dim()] over the dspec the kernel receives.
// The kernels trust this value: cross-page runs are linear only if it matches their accessor. They check it
// with a debug ASSERT only (gather_walk.hpp checked_run_source); this function TT_FATALs where it cannot
// mirror the kernel's value (see the .cpp for why the bank counts agree).
uint32_t contiguous_page_stride(const tt::tt_metal::Buffer& buffer);

// Its sharded half, on the dspec's shapes in pages: tensor_strides[contiguous_dim()] (dspec.h precompute_strides_ct
// and TensorAccessor::contiguous_dim). Separate so the host gtest can hold it against a real TensorAccessor.
uint32_t sharded_contiguous_page_stride(
    const tt::tt_metal::Shape& tensor_shape_in_pages, const tt::tt_metal::Shape& shard_shape_in_pages);

// What WalkGeometry::make needs from the buffers and the hal, gathered so the walk itself is a pure function (the
// host gtest builds it from these numbers alone).
struct WalkInputs {
    uint32_t in_page_stride = 1;           // contiguous_page_stride(input buffer)
    uint32_t out_page_stride = 1;          // contiguous_page_stride(output buffer)
    uint32_t out_buffer_aligned_page = 0;  // output Buffer::aligned_page_size(), what the kernels' accessor steps by
    uint32_t packet_size = 0;              // fabric payload bytes
    uint32_t burst = 0;                    // hal::get_noc_max_burst_size_bytes()
    uint32_t run_cap_bytes = 0;            // SharedPolicy::run_cap_bytes
    uint32_t packets_per_cb_entry = 1;     // SharedPolicy::packets_per_cb_entry (packet-load entry only)
    uint32_t cb_depth = 2;                 // SharedPolicy::cb_depth (packet-load entry's L1 clamp)
    uint32_t max_l1_space = 0;             // get_max_l1_space(input) (packet-load entry's L1 clamp)
};

struct WalkGeometry {
    // Everything below is a kernel CT arg (gather_args.hpp), in chunks.
    uint32_t k = 1;           // chunks per walk page
    uint32_t chunk_size = 0;  // walk page / k
    uint32_t slot_bytes = 0;  // CB slot: chunk_size, or the output's aligned page (padded slot)
    uint32_t in_cpp = 1;      // input chunks per input page
    uint32_t out_cpp = 1;     // output chunks per output page
    uint32_t cps = 0;         // chunks per stripe row
    uint32_t stride = 1;      // walk stride (gather_rules.hpp walk_stride_for)
    uint32_t xfer = 1;        // chunks per column
    RunSource in_runs{};      // the input's run source
    RunSource out_runs{};     // the output's run source
    uint32_t chunks_per_entry = 1;
    uint32_t cb_entry_bytes = 0;  // chunks_per_entry * slot_bytes

    // The pure walk: k from the policy (kAutoK = choose_k over min(packet, run cap)), slot, stride, xfer, run
    // sources and the CB entry. TT_FATALs where `in` contradicts `geom` (the output buffer's aligned page), or
    // the policy asks for something the geometry cannot do. A packet-load entry logs the L1-pressure warning
    // when max_l1_space clamps packets_per_cb_entry.
    static WalkGeometry make(const GatherGeometry& geom, const WalkPolicy& policy, const WalkInputs& in);

    // The factories' form: fills WalkInputs from the tensors' buffers and the hal, then also TT_FATALs unless every
    // local device reports the mesh allocator's bank count for each interleaved side whose runs cross pages (those
    // runs are linear only if the host's stride is every device's contiguity).
    static WalkGeometry make(
        const GatherGeometry& geom,
        const SharedPolicy& policy,
        const Tensor& input,
        const Tensor& output,
        uint32_t packet_size);

    // A walk-page slice in chunks.
    ChunkSlice chunks_of(const WalkPageSlice& s) const { return {s.first * k, s.count * k}; }

    // relay_read_ok on the chunk this walk actually moves, and the padded slot where the relay needs it: what the
    // unicast relay reads back. GatherGeometry::relay_ok is the k = 1 form routing uses.
    bool relay_ok(const GatherGeometry& geom) const;
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
