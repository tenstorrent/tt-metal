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
allocated_pages UINT32 [1, slots], free_list UINT32 [SP, bundles_per_sp], and
free_count UINT32 [1, SP]. Logical page p belongs to SP p % SP; the bundle ID
is shared by the TP devices at that SP. Only the first allocated_pages[slot]
table entries and the first free_count[sp] stack entries are valid.

actual_end is an exclusive token position, rounded up to page_size. When
actual_start is zero, release the old slot lifetime before allocating the new
one. (0, 0) releases only. Otherwise grow without moving existing mappings;
actual_start must lie within allocated capacity. Nonzero-start calls are safe
to repeat; zero-start calls always reset. This operation does not write KV data.

Returns the input page_table, updated in place; no output buffer is allocated.
The caller must reserve sufficient capacity before calling. The kernel assumes
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
remain fixed. Replicate request values across the mesh. Scalars and page_size
remain fixed within a trace. Tensor request values must satisfy the same range
constraints as scalars; the caller ensures this without host readback.

The kernel stages one table row, the counter rows, and at most 4 KiB of a
free-list row. Total scratch must fit the device's L1 capacity after its reserved
region, and must not overlap existing L1 allocations. Pools may
exceed 65,536 bundles per SP; size them for all simultaneously live slots.
IDs are in [0, bundles_per_sp); no ID is reserved as a sentinel. Metadata row
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
                uint32_t>(&ttnn::experimental::update_cache_bundle_allocation),
            nb::arg("page_table").noconvert(),
            nb::arg("allocated_pages").noconvert(),
            nb::arg("free_list").noconvert(),
            nb::arg("free_count").noconvert(),
            nb::kw_only(),
            nb::arg("slot_id"),
            nb::arg("actual_start"),
            nb::arg("actual_end"),
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
                uint32_t>(&ttnn::experimental::update_cache_bundle_allocation),
            nb::arg("page_table").noconvert(),
            nb::arg("allocated_pages").noconvert(),
            nb::arg("free_list").noconvert(),
            nb::arg("free_count").noconvert(),
            nb::kw_only(),
            nb::arg("slot_id").noconvert(),
            nb::arg("actual_start").noconvert(),
            nb::arg("actual_end").noconvert(),
            nb::arg("page_size") = 32));
}

}  // namespace ttnn::operations::experimental::cache_bundle_allocation::detail
