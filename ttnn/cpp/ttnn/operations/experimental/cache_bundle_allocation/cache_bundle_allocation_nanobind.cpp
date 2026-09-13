// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cache_bundle_allocation_nanobind.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "cache_bundle_allocation.hpp"

namespace ttnn::operations::experimental::cache_bundle_allocation::detail {

namespace nb = nanobind;

void bind_cache_bundle_allocation_operation(nb::module_& mod) {
    ttnn::bind_function<"update_cache_bundle_allocation", "ttnn.experimental.">(
        mod,
        R"doc(Update cache bundle allocator metadata in place, independently on each device.

Metadata is rank-2, row-major, interleaved DRAM: page_table UINT32 [slots, max_pages],
allocated_pages UINT32 [1, slots], free_list UINT32 [banks * SP, bundles_per_bank],
and free_count UINT32 [1, banks * SP]. banks is the device DRAM channel count;
SP is inferred from free_list.shape[0] / banks. Pool rows are bank-major:
row = bank * SP + sp. Each row is a stack of BANK-LOCAL indices in
[0, bundles_per_bank), initially e.g. [bundles_per_bank - 1, ..., 0].
Only the first free_count[row] stack entries are valid.

chunk_size is required and must be a positive multiple of page_size * SP
(32 * SP with the default page size). Each chunk assigns chunk_size / (page_size * SP)
consecutive logical pages to SP 0, then SP 1, and so on. For SP=8, chunk_size=5120,
and page_size=32: pages 0..19 belong to SP 0, 20..39 to SP 1, ..., 140..159 to SP 7;
the next chunk starts with pages 160..179 on SP 0.

Let C = chunk_size / page_size and L = C / SP. Logical page p belongs to
sp = (p % C) / L and local_page = (p / C) * L + (p % L), using integer division.
Its bank is local_page % banks; bank rotation continues across chunk boundaries.
chunk_size and page_size must remain fixed while any slot in the pool has live pages.
A popped bank-local index i becomes bundle ID i * banks + bank in page_table.
Thus on one eight-bank device, bank 0 owns B0, B8, B16, ... and bank 7 owns
B7, B15, B23, ... . Each request starts its bank rotation at logical page zero.
The bundle ID is shared by TP devices at that SP. Only the first
allocated_pages[slot] table entries are valid. The caller's KV storage and
consumers must map the entire bundle to that bank; this op updates metadata only.

actual_end is an exclusive token position, rounded up to page_size. When
actual_start is zero, release the old slot lifetime before allocating the new
one. (0, 0) releases only. Otherwise grow without moving existing mappings;
actual_start must lie within allocated capacity. Nonzero-start calls are safe
to repeat; zero-start calls always reset. This operation does not write KV data.

Returns the input page_table, updated in place; no output buffer is allocated.
The caller must reserve sufficient capacity in EACH affected (SP, bank) pool before calling. The kernel assumes
valid counts, unique bundle ownership, and a nonzero start within allocated
capacity; it does not detect or report OOM. Host-side shape, dtype, placement,
and scalar validation errors raise. Tensor contents are not copied to the host.

Initialize metadata on the host and replicate all four tensors across the mesh.
Every replica must execute identical requests in identical order; no CCL is used.
Serialize calls sharing a pool, finish prior slot accesses before reset, and
order consumers after this update before accessing new pages.

slot_id, actual_start, and actual_end must be all scalars or all UINT32 [1, 1]
row-major interleaved DRAM tensors on the same device. Mixed scalar/tensor
inputs are rejected. For trace replay, update
these tensors in place on the same command queue before replay; their addresses
remain fixed. Replicate request values across the mesh. Scalars, chunk_size, and page_size
remain fixed within a trace. Tensor request values must satisfy the same range
constraints as scalars; the caller ensures this without host readback.

The kernel stages one table row, the counter rows, and at most 4 KiB of a
free-list row. Total scratch must fit the device's L1 capacity after its reserved
region, and must not overlap existing L1 allocations. Pools may
exceed 65,536 bundles per SP; size them for all simultaneously live slots.
Page-table IDs are in [0, banks * bundles_per_bank); no ID is reserved as a sentinel. Metadata row
byte sizes must fit 32-bit NoC offsets; tensor storage is limited by available DRAM.
)doc",
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                uint32_t,
                uint32_t,
                uint32_t,
                uint32_t,
                uint32_t>(&ttnn::experimental::update_cache_bundle_allocation),
            nb::arg("page_table").noconvert(),
            nb::arg("allocated_pages").noconvert(),
            nb::arg("free_list").noconvert(),
            nb::arg("free_count").noconvert(),
            nb::kw_only(),
            nb::arg("slot_id"),
            nb::arg("actual_start"),
            nb::arg("actual_end"),
            nb::arg("chunk_size"),
            nb::arg("page_size") = 32),
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                uint32_t,
                uint32_t>(&ttnn::experimental::update_cache_bundle_allocation),
            nb::arg("page_table").noconvert(),
            nb::arg("allocated_pages").noconvert(),
            nb::arg("free_list").noconvert(),
            nb::arg("free_count").noconvert(),
            nb::kw_only(),
            nb::arg("slot_id").noconvert(),
            nb::arg("actual_start").noconvert(),
            nb::arg("actual_end").noconvert(),
            nb::arg("chunk_size"),
            nb::arg("page_size") = 32));
}

}  // namespace ttnn::operations::experimental::cache_bundle_allocation::detail
