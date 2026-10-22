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
            Gathers a row-major or tile-layout DRAM tensor over one device-mesh axis, or over the
            whole 2D mesh, into a preallocated DRAM output.

            Per chip, per ring direction and per link one fabric link worker core reads (its own shard
            from the input, forwarded shards from the output) and sends each fabric chunk one fabric hop
            into the same pages of the neighbour's output; local copy cores write the chip's own shard
            (and convert an ND-sharded input). Link workers sit as close as the core grid allows to their
            link's Ethernet core. Calls are fenced: no chip writes into a neighbour's output before the
            neighbour has started the same call. Even rings are balanced (the opposite shard travels half
            each way). With ``cluster_axis=None`` on a torus whose sides are both at least 3, the mesh is
            covered by two edge-disjoint Hamiltonian cycles, so every chip uses all four neighbours;
            otherwise by a snake ring. The output is always in row-major chip order.

            Pages move whole, so a ROW_MAJOR tensor cannot be gathered along its innermost dim (every
            output row would mix ranks); gather a TILE_LAYOUT tensor or another dim instead.

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
                cluster_axis: Device-mesh axis (0 or 1) participating in the collective; other
                    mesh axes run independent all-gathers. ``None`` gathers across every device of
                    the 2D mesh (two Hamiltonian cycles on a torus with both sides >= 3, else a snake
                    ring); the tensor must be sharded over ``dim`` across all mesh devices.
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
                push_table: Sparse KV push instead of the dense gather. The all-gathered
                    ``sparse_push_plan`` tables of every chip (uint32 [1, 1, pages * devices, 1024]); each owner
                    pushes only the rows of the selected cache slot that a requester's top-k references, into
                    that requester's compact ``output_tensor``. Full mesh (``cluster_axis=None``) on a 2D torus,
                    with caller-owned semaphores.
                push_stripe_chunk: Rows per block-cyclic block of the cache the plan was built for.
                push_max_rows: Row capacity the plan tables were built for.
                push_part: 0 pushes every needed row; 2 only the rows from the plan's chunk start on (the
                    prefix rows were pushed earlier by a ``sparse_sdpa`` with a fused push).
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
        nb::arg("push_table") = nb::none(),
        nb::arg("push_stripe_chunk") = 0,
        nb::arg("push_max_rows") = 0,
        nb::arg("push_part") = 0);

    ttnn::bind_function<"sparse_push_plan", "ttnn.experimental.">(
        mod,
        R"doc(
            On-device plan for the sparse KV push, from one requester's top-k indices (per device uint32
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
