// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_push_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "plan/sparse_push_plan.hpp"
#include "push/sparse_kv_push.hpp"
#include "dispatch/sparse_kv_dispatch.hpp"
#include "common/sparse_kv_push_common.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::sparse_kv_push::detail {

void bind_experimental_sparse_kv_push_operations(nb::module_& mod) {
    nb::enum_<SparseKvPushPart>(mod, "SparseKvPushPart", "Which of a requester's needed rows a sparse KV push sends.")
        .value("ALL", SparseKvPushPart::kAll, "Every needed row.")
        .value(
            "PREFIX",
            SparseKvPushPart::kPrefix,
            "The rows of the blocks before the one holding the plan's chunk start: rows earlier chunks wrote, so they "
            "can be pushed before this chunk's KV is written.")
        .value("REST", SparseKvPushPart::kRest, "Every needed row after the prefix.");

    // Cores the push needs per fabric link on a chip with fabric neighbours in all four directions, the most it takes,
    // for callers laying out its core grid beside other work.
    mod.attr("SPARSE_KV_PUSH_CORES_PER_LINK") = cores_needed(1);
    // Test-only: the compact layout's (pages_per_packet, banks), so tests lay out reference buffers as the ops do.
    mod.def(
        "_sparse_kv_push_packet_shape",
        [](ttnn::MeshDevice& mesh_device, uint32_t page_bytes) {
            const auto shape = packet_shape(&mesh_device, page_bytes);
            return std::make_tuple(shape.pages_per_packet, shape.banks);
        },
        nb::arg("mesh_device"),
        nb::arg("page_bytes"));

    ttnn::bind_function<"is_sparse_kv_push_supported", "ttnn.experimental.">(
        mod,
        R"doc(
            Whether ``sparse_push_plan`` and ``sparse_kv_push`` can serve a cache of ``max_rows`` rows in blocks of
            ``stripe_chunk`` rows on ``mesh_device``: a FABRIC_2D_TORUS_XY fabric, at least 3 chips on each mesh axis
            and at most 32 in all, and a cache geometry the plan table and the plan's reduce fit. When this returns
            False the ops would raise, so callers can gather the whole prefix instead.

            Args:
                mesh_device (ttnn.MeshDevice): The mesh the ops would run on.
                stripe_chunk: Rows per block-cyclic block of the cache.
                max_rows: Cache rows the plan would cover.

            Returns:
                bool
        )doc",
        &is_sparse_kv_push_supported,
        nb::arg("mesh_device"),
        nb::arg("stripe_chunk"),
        nb::arg("max_rows"));

    ttnn::bind_function<"sparse_kv_push_output_spec", "ttnn.experimental.">(
        mod,
        R"doc(
            The spec of the compact buffer ``sparse_kv_push`` writes for up to ``distinct_rows`` needed rows per chip
            (a requester's query rows x top-k bounds them), for a cache of ``max_rows`` rows in blocks of
            ``stripe_chunk`` with rows of ``row_width`` ``dtype`` elements: [1, 1, rows, row_width] ROW_MAJOR
            interleaved DRAM. Use it to size a buffer without allocating one, e.g. to check whether an existing
            buffer can serve.

            Returns:
                ttnn.TensorSpec
        )doc",
        &sparse_kv_push_output_spec,
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("distinct_rows"),
        nb::arg("max_rows"),
        nb::arg("stripe_chunk"),
        nb::arg("row_width"),
        nb::arg("dtype"));

    ttnn::bind_function<"sparse_kv_push_create_output_buffer", "ttnn.experimental.">(
        mod,
        R"doc(
            Allocates the compact buffer ``sparse_kv_push_output_spec`` describes, labelled sharded over its rows
            across every chip: each chip's buffer holds the rows its own top-k needs. Create it once and pass it as
            ``output_tensor`` on every push. Contents are undefined until a push writes them; only pushed rows are
            read.

            Returns:
                ttnn.Tensor
        )doc",
        &sparse_kv_push_create_output_buffer,
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("distinct_rows"),
        nb::arg("max_rows"),
        nb::arg("stripe_chunk"),
        nb::arg("row_width"),
        nb::arg("dtype"));

    ttnn::bind_function<"sparse_push_plan", "ttnn.experimental.">(
        mod,
        R"doc(
            Builds one chip's plan table for ``ttnn.experimental.sparse_kv_push`` from its top-k indices, and
            rewrites the indices to compact-buffer slots. Needs the fabric the push will run on: its payload size
            sets the packet shape the slots follow. Every chip of the mesh, at most 32, owns a share of the cache:
            block ``b`` of ``stripe_chunk`` rows lives on owner ``b % chips``; the push's ``sp_axis`` maps owners to
            chips.

            Args:
                indices: The requester's top-k cache rows, per device uint32 ROW_MAJOR DRAM [1, 1, rows, topk]
                    with ``topk`` a multiple of 16; 0xFFFFFFFF = masked. Rows are global cache rows below
                    ``max_rows``.
                stripe_chunk: Rows per block-cyclic block of the cache, a positive multiple of 32 up to 512. The
                    push must use the same value.
                max_rows: Cache rows covered, a multiple of ``stripe_chunk`` and of 512. The push must use the same
                    value.
                page_bytes: Bytes per cache row page as the push reads it (the DRAM-aligned row width), at most
                    the fabric's max packet payload.
                chunk_start: Optional current chunk's first row, below ``max_rows``. Blocks before the one holding
                    it are the prefix (push ``PREFIX``), the rest ``REST``: that block isn't fully written until this
                    chunk's KV is. Without a chunk start, every row is in the prefix. A runtime arg: one program
                    serves every chunk, but a trace replays the captured value, so use ``chunk_start_tensor`` there.
                chunk_start_tensor: Trace-safe form of ``chunk_start``: a 1-element uint32 DRAM tensor holding the
                    row, read on device. Mutually exclusive with ``chunk_start``.
                dense_below: Dense mode below this chunk start (0: off): ``compact_indices`` become rows of a dense
                    gather into the compact buffer, as high_bw_all_gather with ``gathered_active_below=dense_below``
                    lays it out (chip ``c``'s rows from ``c * slot`` on, in its cache order, with the slot the
                    largest extent below the threshold in whole slabs / chips), and the table lists no rows, so the
                    push sends nothing. Needs ``chunk_start`` or ``chunk_start_tensor``. Hashed.
                dense_below_2: A second, higher dense threshold (0: off) for a reader whose buffer holds the dense
                    gather below it (``gathered_active_below=dense_below_2``): adds a third output, the indices in
                    that dense layout for chunk starts below it and the same as ``compact_indices`` otherwise. Hashed.

            Returns:
                List[ttnn.Tensor]: ``[table, compact_indices]``: the plan table (uint32 [1, 1, pages, 1024]) to
                all-gather for the owners, and ``indices`` rewritten to slots in the requester's compact buffer.
        )doc",
        &sparse_push_plan,
        nb::arg("indices").noconvert(),
        nb::kw_only(),
        nb::arg("stripe_chunk"),
        nb::arg("max_rows"),
        nb::arg("page_bytes"),
        nb::arg("chunk_start") = nb::none(),
        nb::arg("chunk_start_tensor") = nb::none(),
        nb::arg("dense_below") = 0,
        nb::arg("dense_below_2") = 0,
        nb::arg("owner_major") = false,
        nb::arg("rank_table") = false);

    ttnn::bind_function<"sparse_kv_push", "ttnn.experimental.">(
        mod,
        R"doc(
            Each owner chip pushes only the cache rows a requester's top-k references into that requester's
            compact ``output_tensor``, at the slots ``sparse_push_plan`` rewrote its indices to.
            Needs a 2D fabric (FABRIC_2D, or a torus on either or both axes) on a 2D mesh of 2 to 32 chips. Each
            chip sends on the fabric links it has: both ways along an axis the fabric wraps, towards the existing
            neighbours along one it does not.

            Args:
                input_tensor: The KVPE cache, [slots, 1, rows per chip, width] ROW_MAJOR in DRAM (interleaved or
                    ND-sharded), sharded over its rows (dim 2) across every chip, in the stripe order of ``sp_axis``. A row page
                    must fit in one fabric packet payload.
                output_tensor: The requester's compact buffer, [1, 1, rows, width] ROW_MAJOR interleaved DRAM with
                    the input's dtype and row page; normally far shorter than the cache. Each owner's rows start on
                    a whole unit of ``pages_per_packet * banks`` rows, where ``pages_per_packet`` is the fabric's max
                    payload // the row page and ``banks`` the DRAM bank count. So it needs ``sum over owners o of
                    round_up(count_o, pages_per_packet * banks)`` rows, where ``count_o`` is the rows the requester
                    needs from owner ``o``; ``min(distinct rows, cache rows) + num_devices * pages_per_packet *
                    banks`` always suffices, and at least ``num_devices * pages_per_packet * banks`` is required.
                    Packets past the end are dropped.
                table: The ``sparse_push_plan`` tables of every chip, all-gathered over dim 2 in row-major chip
                    order (uint32 [1, 1, pages * devices, 1024], ROW_MAJOR interleaved DRAM).
                stripe_chunk: Rows per block-cyclic block of the cache, a positive multiple of 32 up to 512; must
                    equal the plan's.
                max_rows: Cache rows the plan covers; must equal the plan's.
                ready_semaphore, done_semaphore: Two distinct global semaphores on the input's mesh device
                    (L1_SMALL when the device has it, else L1), zero-initialised. The push leaves them at zero, so
                    they can be reused by every later push.
                part: A ``SparseKvPushPart``: ``ALL`` every needed row, ``PREFIX`` only the rows
                    of the blocks before the one holding the plan's chunk start (rows earlier chunks wrote, so it can
                    run before this chunk's KV is written), ``REST`` the others. ``PREFIX`` then ``REST`` fill
                    exactly what ``ALL`` fills. ``REST`` needs a plan built with a chunk start; without one it
                    pushes nothing.
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
                sp_axis: The mesh axis the cache's blocks stripe along first (0 or 1): the chip at position ``s`` on
                    it and ``t`` on the other axis owns blocks ``s * chips_on_other_axis + t`` (+ ``k * chips``).
                    Axis 0 is row-major chip order. Must match how the plan's owners are ordered (block ``b`` on
                    owner ``b % chips``).

            Returns:
                ttnn.Tensor: ``output_tensor``, holding the pushed rows. Each chip's rows differ, so it is labelled
                sharded over its rows (dim 2) across every chip, like ``input_tensor``.
        )doc",
        &sparse_kv_push,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("output_tensor").noconvert(),
        nb::arg("table").noconvert(),
        nb::arg("stripe_chunk"),
        nb::arg("max_rows"),
        nb::arg("ready_semaphore"),
        nb::arg("done_semaphore"),
        nb::arg("part") = SparseKvPushPart::kAll,
        nb::arg("num_links") = 2,
        nb::arg("batch_index") = 0,
        nb::arg("batch_index_tensor") = nb::none(),
        nb::arg("batch_slot_num_layers") = 1,
        nb::arg("batch_slot_layer_idx") = 0,
        nb::arg("core_grid") = nb::none(),
        nb::arg("sp_axis") = 0);

    ttnn::bind_function<"sparse_kv_dispatch", "ttnn.experimental.">(
        mod,
        R"doc(
            Sends the KV rows every chip's top-k needs over sparse_dispatch_2d-style trees (one copy per tree edge,
            store-and-forward through relay chips) into ``output_tensor``, the full-size gather buffer, at the rows'
            owner-major pages: the layout high_bw_all_gather writes and sparse_sdpa's block-cyclic remap reads. See
            models/demos/deepseek_v3_d_p/tests/sparse_mla/push_experiment/KV_DISPATCH_2D_PLAN.md.

            Args:
                input_tensor: The KVPE cache, sharded over its rows (dim 2) across every chip.
                output_tensor: [1, 1, rows, width], rows >= max_rows and a multiple of the chip count, replicated
                    interleaved DRAM with the input's row page.
                table: This chip's owner-major ``sparse_push_plan`` table (owner_major=True, no dense threshold).
                inbox: One inbox slot per chip, as ``sparse_kv_push``'s inbox.
                scratch: A tensor of ``sparse_kv_dispatch_scratch_spec``.
                stripe_chunk, max_rows: As the plan's; max_rows a whole number of slabs.
                fence_semaphore, arrived_semaphore, credit_semaphore, inbox_semaphore: Distinct global semaphores
                    on the core grid, zero-initialised; the op leaves them at zero.
                num_links: 1 or 2.
                rows_per_packet: Rows per hop packet (0: as many as the fabric payload fits).
                landing_slots: Packets each port can have in flight to its neighbour.
                planners: Cores that turn the slices into destination masks.
                lanes_per_link: Independent sender/unpacker core pairs per direction and link (rows dealt by hash);
                    more than 1 shares each link through a fabric mux core.
                batch_index, batch_index_tensor, batch_slot_num_layers, batch_slot_layer_idx: The cache slot, as
                    ``sparse_kv_push``'s.
                plan_indices, plan_chunk_start, plan_chunk_start_tensor: Experimental (KVD_DESIGN=3): this chip's
                    top-k indices; the op then builds ``table`` itself first (the dispatch-only plan, fused).
                active_from: Experimental (KVD_DESIGN=3): active only for chunks starting at or after this row
                    (a multiple of stripe_chunk; 0: always). Below it the op reads the chunk start from the plan
                    table's header and moves nothing.
                core_grid: At least 4 x lanes port cores (lanes = num_links x lanes_per_link), then 4 x num_links
                    mux cores when lanes_per_link > 1, then the planners: 72 with the defaults.
        )doc",
        &sparse_kv_dispatch,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("output_tensor").noconvert(),
        nb::arg("table").noconvert(),
        nb::arg("inbox").noconvert(),
        nb::arg("scratch").noconvert(),
        nb::arg("stripe_chunk"),
        nb::arg("max_rows"),
        nb::arg("fence_semaphore"),
        nb::arg("arrived_semaphore"),
        nb::arg("credit_semaphore"),
        nb::arg("inbox_semaphore"),
        nb::arg("num_links") = 2,
        nb::arg("rows_per_packet") = 0,
        nb::arg("landing_slots") = 16,
        nb::arg("planners") = 48,
        nb::arg("batch_index") = 0,
        nb::arg("batch_index_tensor") = nb::none(),
        nb::arg("batch_slot_num_layers") = 1,
        nb::arg("batch_slot_layer_idx") = 0,
        nb::arg("core_grid") = nb::none(),
        nb::arg("lanes_per_link") = 2,
        nb::arg("plan_indices") = nb::none(),
        nb::arg("plan_chunk_start") = nb::none(),
        nb::arg("plan_chunk_start_tensor") = nb::none(),
        nb::arg("active_from") = 0,
        nb::arg("part") = 0,
        nb::arg("skip_exchange") = false,
        nb::arg("rank_table") = nb::none(),
        nb::arg("sp_axis") = 0);

    ttnn::bind_function<"sparse_kv_dispatch_scratch_spec", "ttnn.experimental.">(
        mod,
        R"doc(The spec of ``sparse_kv_dispatch``'s scratch tensor for this mesh, cache and planner count.)doc",
        &sparse_kv_dispatch_scratch_spec,
        nb::arg("mesh_device"),
        nb::arg("max_rows"),
        nb::arg("stripe_chunk"),
        nb::arg("planners") = 48,
        nb::arg("num_links") = 2,
        nb::arg("lanes_per_link") = 2);

    mod.def(
        "sparse_kv_push_inbox_slot_bytes",
        [](uint32_t max_rows, uint32_t stripe_chunk, uint32_t num_devices) {
            return inbox_slot_bytes(max_rows, stripe_chunk, num_devices);
        },
        nb::arg("max_rows"),
        nb::arg("stripe_chunk"),
        nb::arg("num_devices"));
}

}  // namespace ttnn::operations::experimental::sparse_kv_push::detail
