// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "high_bw_all_gather_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "high_bw_all_gather.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather::detail {

void bind_experimental_high_bw_all_gather_operation(nb::module_& mod) {
    ttnn::bind_function<"high_bw_all_gather", "ttnn.experimental.">(
        mod,
        R"doc(
            Gathers a large row-major or tile-layout DRAM tensor over one device-mesh
            axis, or over a direct-neighbor ring linearized across a complete 2D mesh.
            This is a one-dimensional Fabric line or ring collective. The
            operation uses a native one-hop store-and-forward transport and does not
            provide a composite fallback.

            Fabric2D supports a direct physical line or ring. A Torus may wrap
            ``cluster_axis`` when the axis has a distinct wrap neighbor and every
            ring edge is one physical hop. A size-two torus axis is treated as a
            line because its ordinary and wrap neighbors are the same device.
            Plain ``FABRIC_2D`` uses the direct-line schedule; only a Torus
            configuration that wraps ``cluster_axis`` selects the ring schedule.

            Channel trimming compatibility:
                Channel trimming is selected during Fabric initialization and is shared
                by all CCLs in the workload; it is not configurable per operation. A
                trimming capture must cover every CCL and shape that will run. When
                using a trimming profile with this operation, set
                ``TT_METAL_FABRIC_TRIMMING_OVERRIDE`` to an override that force-enables
                all VC0 sender and receiver channels. This avoids a trim-derived VC0
                fast path that can substantially regress this high-rate, multi-worker
                collective while preserving correctness. See this operation's README
                for the YAML and launch example.

            Args:
                input_tensor: Row-major or tile-layout device tensor in DRAM.
                dim: Tensor dimension along which device shards are concatenated.
                output_tensor: Preallocated persistent output tensor.

            Keyword Args:
                cluster_axis: Device-mesh axis (0 or 1) participating in the
                    one-dimensional collective. Other mesh axes run independent
                    all-gathers. Pass ``None`` to gather across every device in a 2D
                    mesh using a snake Hamiltonian ring. The full-mesh mode requires
                    at least one even mesh dimension, direct links for every edge of
                    a row or column snake, and a tensor sharded over ``dim`` across
                    all mesh devices. The host prefers a row snake, then tries a
                    column snake if the row route cannot close directly.
                num_links: Optional number of Fabric links to use. ``None`` uses every
                    link reported usable across the selected axis, or the minimum
                    discovered across both axes in full-mesh mode. An explicit value
                    must be greater than zero and cannot exceed that discovered count;
                    use ``2`` to keep the same link count across QuietBox, LoudBox, and
                    Galaxy.
                subdevice_id: Subdevice containing the worker cores.
                sub_core_grids: Optional worker-core restriction.
                input_batch_index: Optional batch slot selected from a persistent input cache.
                    When set, input has shape [B, 1, ...], output has batch 1, and only that
                    slot is transported.
                gathered_dim_size: Optional active global gathered extent along ``dim``. The
                    output tensor must still be allocated at its worst-case full gathered size.
                    Each rank writes its active local prefix into that rank's fixed worst-case
                    slot; bytes outside those prefixes are left unchanged. ``gathered_dim_size``
                    is the total valid extent, not a contiguous output prefix: consumers must
                    preserve the fixed per-rank stride when locating every rank's valid data.
                input_batch_index_tensor: TRACE-SAFE form of ``input_batch_index``. A 1-element
                    uint32 ROW_MAJOR DRAM tensor holding the USER id; the reader reads it on-device
                    and recomposes the flat cache slot as
                    ``user_id * batch_slot_num_layers + batch_slot_layer_idx`` (the cache batch dim is
                    user-major). Use this instead of ``input_batch_index`` whenever the call is captured
                    into a ttnn trace: a host scalar is patched per dispatch, and a replay never re-runs
                    that patch, so every replay would re-read the slot live at capture time. Mutually
                    exclusive with ``input_batch_index``.
                batch_slot_num_layers: Layers per user in the input cache's batch dim. Runtime (NOT
                    hashed, so one program serves caches of any depth) and only read on the
                    ``input_batch_index_tensor`` path.
                batch_slot_layer_idx: This call's layer index within a user's slots. Runtime (not
                    hashed, so all layers share one program) and only read on the
                    ``input_batch_index_tensor`` path.
                gathered_prefix_tensor: TRACE-SAFE form of ``gathered_dim_size``. A 1-element uint32
                    ROW_MAJOR DRAM tensor holding this chunk's START position in the gathered dim; the
                    reader derives the active extent on-device as
                    ``min(round_up(start + gathered_slab_global, gathered_slab_global), full_extent)``
                    and recomputes its own page partition from it. Use this instead of
                    ``gathered_dim_size`` whenever the call is captured into a ttnn trace: the scalar
                    grows every chunk, and a replay would re-gather only the captured chunk's prefix.
                    Mutually exclusive with ``gathered_dim_size``.
                gathered_slab_global: Block-cyclic slab width in gathered-dim elements
                    (``chunk_local * num_devices``). Required with ``gathered_prefix_tensor`` and hashed,
                    being structural rather than per-chunk.
                ready_semaphore: Optional caller-owned persistent startup semaphore. Must be
                    supplied together with ``data_valid_semaphore`` and initialized to zero on
                    the complete worker-core restriction before the first call.
                data_valid_semaphore: Optional caller-owned persistent relay/completion semaphore.
                    Supplying both semaphore handles selects the allocation-free, no-internal-sync
                    dispatch path intended for sub-device overlap.
                window_start_tensor: Optional resumable-window start, a 1-element uint32 DRAM tensor in
                    gathered-dim elements (slab-aligned; absent = 0). Typically the ``progress_tensor`` of an
                    earlier partial call. Requires ``gathered_prefix_tensor``.
                window_max_slabs: Optional cap on the slabs this call gathers past the window start
                    (runtime, not hashed; None = to the end of the extent).
                prefix_only: Gather only slabs written before this chunk -- the extent becomes
                    ``floor(start / gathered_slab_global)`` slabs instead of including the current chunk.
                    For prefetching a later layer's prefix while the current chunk is still being written.
                progress_tensor: Optional 1-element uint32 DRAM tensor that receives the window end this
                    call reached (gathered-dim elements). Must not alias any input metadata tensor.
                push_runs_tensor: PROTOTYPE. Per-device uint32 [1, 1, workers, words] run lists: word 0 = number
                    of runs, then from byte 64 triples (dest_rank, local_row_start, count). Each owner pushes
                    only the listed rows of the selected cache slot into the requester's output, at the address
                    the dense full-mesh gather would use. Needs caller-owned semaphores and a 2D torus.
                push_stripe_chunk: PROTOTYPE. When > 0, ``push_runs_tensor`` is instead the all-gathered
                    ``sparse_push_plan`` table (cache of ``push_max_rows`` rows, ``push_stripe_chunk`` rows
                    per block) and each owner derives what to push from it on device.
                push_max_rows: PROTOTYPE. Row capacity the plan tables were built for.
                push_part: PROTOTYPE. With plan tables: 0 pushes every needed row, 1 only rows before the
                    plan's chunk start (prefix, e.g. prefetched during the previous layer), 2 only the rest.
        )doc",
        &high_bw_all_gather,
        nb::arg("input_tensor").noconvert(),
        nb::arg("dim"),
        nb::arg("output_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::arg("num_links") = nb::none(),
        nb::arg("input_batch_index") = nb::none(),
        nb::arg("gathered_dim_size") = nb::none(),
        nb::arg("input_batch_index_tensor") = nb::none(),
        nb::arg("batch_slot_num_layers") = 1,
        nb::arg("batch_slot_layer_idx") = 0,
        nb::arg("gathered_prefix_tensor") = nb::none(),
        nb::arg("gathered_slab_global") = 0,
        nb::arg("ready_semaphore") = nb::none(),
        nb::arg("data_valid_semaphore") = nb::none(),
        nb::arg("window_start_tensor") = nb::none(),
        nb::arg("window_max_slabs") = nb::none(),
        nb::arg("prefix_only") = false,
        nb::arg("progress_tensor") = nb::none(),
        nb::arg("push_runs_tensor") = nb::none(),
        nb::arg("push_stripe_chunk") = 0,
        nb::arg("push_max_rows") = 0,
        nb::arg("push_part") = 0);

    ttnn::bind_function<"sparse_push_plan", "ttnn.experimental.">(
        mod,
        R"doc(
            PROTOTYPE. On-device plan for the sparse KV push, from one requester's top-k indices (per device uint32
            row-major [1, 1, rows, topk], 0xFFFFFFFF = masked). Returns ``[table, compact_indices]``: the plan table
            (uint32 [1, 1, pages, 1024]) to all-gather for the owners, and the indices rewritten to slots in the
            requester's compact buffer.
        )doc",
        &sparse_push_plan,
        nb::arg("indices").noconvert(),
        nb::kw_only(),
        nb::arg("num_devices"),
        nb::arg("stripe_chunk"),
        nb::arg("max_rows"),
        nb::arg("pages_per_packet"),
        nb::arg("banks"),
        nb::arg("chunk_start") = nb::none());
}

}  // namespace ttnn::operations::experimental::high_bw_all_gather::detail
