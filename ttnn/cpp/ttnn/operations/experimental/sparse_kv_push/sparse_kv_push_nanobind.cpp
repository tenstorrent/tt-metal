// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_push_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "sparse_kv_push.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::sparse_kv_push::detail {

void bind_experimental_sparse_kv_push_operations(nb::module_& mod) {
    ttnn::bind_function<"sparse_kv_push", "ttnn.experimental.">(
        mod,
        R"doc(
            Each owner chip pushes only the cache rows a requester's top-k references into that requester's
            compact ``output_tensor``, at the slots ``sparse_push_plan`` rewrote its indices to.
            Needs a FABRIC_2D_TORUS_XY fabric on a 2D mesh of at most 32 chips.

            Args:
                input_tensor: The KVPE cache, [slots, 1, rows per chip, width] ROW_MAJOR in DRAM (interleaved or
                    ND-sharded), sharded over its rows (dim 2) across every chip in row-major order. A row page
                    must fit in one fabric packet payload.
                output_tensor: The requester's compact buffer, [1, 1, rows, width] ROW_MAJOR interleaved DRAM with
                    the input's dtype and row page; normally far shorter than the cache. Each owner's rows start
                    on a whole unit of ``pages_per_packet * banks`` rows (``sparse_kv_push_packet_shape``), so it
                    needs ``sum over owners o of round_up(count_o, pages_per_packet * banks)`` rows, where
                    ``count_o`` is the rows the requester needs from owner ``o``; ``min(distinct rows, cache rows)
                    + num_devices * pages_per_packet * banks`` always suffices, and at least
                    ``num_devices * pages_per_packet * banks`` is required. Packets past the end are dropped.
                table: The ``sparse_push_plan`` tables of every chip, all-gathered over dim 2 in row-major chip
                    order (uint32 [1, 1, pages * devices, 1024], ROW_MAJOR interleaved DRAM).
                stripe_chunk: Rows per block-cyclic block of the cache; must equal the plan's.
                max_rows: Cache rows the plan covers; must equal the plan's.
                ready_semaphore, data_valid_semaphore: Two distinct global semaphores on the input's mesh device
                    (L1_SMALL when the device has it, else L1), zero-initialised. The push leaves them at zero, so
                    they can be reused by every later push.
                part: 0 pushes every needed row; 1 only the rows of the blocks before the one holding the plan's
                    chunk start (rows earlier chunks wrote, so it can run before this chunk's KV is written); 2 the
                    rest. Part 2 needs a plan built with ``chunk_start``; without one it pushes nothing.
                num_links: Fabric links per direction (1 or 2).
                batch_index: Cache slot pushed.
                batch_index_tensor: Trace-safe slot: a 1-element uint32 ROW_MAJOR interleaved DRAM tensor holding
                    the user id; the slot is read on device as
                    ``batch_index_tensor[0] * batch_slot_num_layers + batch_slot_layer_idx`` and ``batch_index``
                    is ignored.
                batch_slot_num_layers: Cache slots per user; must divide the cache's slots.
                batch_slot_layer_idx: The pushed layer's slot within its user, below ``batch_slot_num_layers``.
                core_grid: Cores for the push (default: the first sub-device's cores); it needs
                    ``num_links * 4 * 3`` of them: two senders and one fabric mux per link and torus direction.

            Returns:
                ttnn.Tensor: ``output_tensor``, holding the pushed rows, replicated over the mesh.
        )doc",
        &sparse_kv_push,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("output_tensor").noconvert(),
        nb::arg("table").noconvert(),
        nb::arg("stripe_chunk"),
        nb::arg("max_rows"),
        nb::arg("ready_semaphore"),
        nb::arg("data_valid_semaphore"),
        nb::arg("part") = 0,
        nb::arg("num_links") = 2,
        nb::arg("batch_index") = 0,
        nb::arg("batch_index_tensor") = nb::none(),
        nb::arg("batch_slot_num_layers") = 1,
        nb::arg("batch_slot_layer_idx") = 0,
        nb::arg("core_grid") = nb::none());

    ttnn::bind_function<"sparse_push_plan", "ttnn.experimental.">(
        mod,
        R"doc(
            On-device plan for the sparse KV push, from one requester's top-k indices. Needs the fabric the push
            will run on, whose payload size sets the packet shape the compact slots follow.

            Args:
                indices: The requester's top-k cache rows, per device uint32 ROW_MAJOR DRAM [1, 1, rows, topk]
                    with ``topk`` a multiple of 16; 0xFFFFFFFF = masked. Rows are global cache rows below
                    ``max_rows``.
                num_devices: Chips owning the cache (block ``b`` of ``stripe_chunk`` rows lives on chip
                    ``b % num_devices``), at most 32.
                stripe_chunk: Rows per block-cyclic block of the cache, a positive multiple of 32 up to 512. The
                    push must use the same value.
                max_rows: Cache rows covered, a multiple of ``stripe_chunk`` and of 512. The push must use the same
                    value.
                page_bytes: Bytes per cache row page as the push reads it (the DRAM-aligned row width), at most
                    the fabric's max packet payload.
                chunk_start: Optional 1-element uint32 DRAM tensor holding the current chunk's first row, read on
                    device. Blocks before the one holding it are the prefix part (push part 1), the
                    rest the remainder (part 2): that block is not fully written until this chunk's KV is. Without
                    it every row is in the prefix part.

            Returns:
                List[ttnn.Tensor]: ``[table, compact_indices]``: the plan table (uint32 [1, 1, pages, 1024]) to
                all-gather for the owners, and ``indices`` rewritten to slots in the requester's compact buffer.
        )doc",
        &sparse_push_plan,
        nb::arg("indices").noconvert(),
        nb::kw_only(),
        nb::arg("num_devices"),
        nb::arg("stripe_chunk"),
        nb::arg("max_rows"),
        nb::arg("page_bytes"),
        nb::arg("chunk_start") = nb::none());

    mod.def(
        "sparse_kv_push_packet_shape",
        &sparse_kv_push_packet_shape,
        R"doc(
            ``(pages_per_packet, banks)`` of the sparse KV push for cache rows of ``page_bytes`` bytes on
            ``mesh_device``: the push sends packets of ``pages_per_packet`` rows, dealt round robin over ``banks``
            DRAM banks, so each owner's run in the compact buffer is padded to whole ``pages_per_packet * banks``
            row units.
        )doc",
        nb::arg("mesh_device"),
        nb::arg("page_bytes"));
}

}  // namespace ttnn::operations::experimental::sparse_kv_push::detail
