// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_all_gather_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "fabric_all_gather.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::fabric_all_gather::detail {

void bind_experimental_fabric_all_gather_operation(nb::module_& mod) {
    ttnn::bind_function<"fabric_all_gather", "ttnn.experimental.">(
        mod,
        R"doc(
            All-gather over fabric with the same contract as ``ttnn.experimental.high_bw_all_gather``
            (a drop-in replacement): a row-major or tile-layout DRAM tensor is gathered over one
            device-mesh axis, or over the whole 2D mesh, into a preallocated DRAM output.

            Per chip, per ring direction and per link one fabric link worker core reads (its own shard
            from the input, forwarded shards from the output) and sends each fabric chunk one fabric hop
            into the same pages of the neighbour's output; local copy cores write the chip's own shard
            (and convert an ND-sharded input). Link workers sit as close as the core grid allows to their
            link's Ethernet core. A fabric chunk is pages that are physically contiguous in one DRAM bank,
            one packet each; the last packet of every shard a link worker sends increments the
            neighbour's shards-arrived counter, and the neighbour forwards that shard once it has arrived. Calls are fenced (no chip writes into a
            neighbour's output before the neighbour has started the same call). Even rings are balanced
            (the opposite shard travels half each way). With ``cluster_axis=None`` on a torus whose
            sides are both at least 3, the mesh is covered by two edge-disjoint Hamiltonian cycles, so
            every chip uses all four neighbours; otherwise by a snake ring. The output is always in
            row-major chip order.

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
                gathered_dim_size: Optional valid global gathered length along ``dim``. The
                    output tensor must still be allocated at its worst-case full gathered size.
                    Each rank writes its active local prefix into that rank's fixed worst-case
                    slot; bytes outside those prefixes are left unchanged. ``gathered_dim_size``
                    is the total valid length, not a contiguous output prefix: consumers must
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
                    reader derives the valid length on-device as
                    ``min(round_up(start + gathered_slab_global, gathered_slab_global), full_length)``
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
        )doc",
        &fabric_all_gather,
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
        nb::arg("data_valid_semaphore") = nb::none());
}

}  // namespace ttnn::operations::experimental::fabric_all_gather::detail
