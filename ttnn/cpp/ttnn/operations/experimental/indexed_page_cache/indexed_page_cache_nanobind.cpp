// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "indexed_page_cache_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/indexed_page_cache/indexed_page_cache.hpp"

namespace ttnn::operations::experimental::indexed_page_cache::detail {

void bind_experimental_indexed_page_cache_operations(nb::module_& mod) {
    ttnn::bind_function<"indexed_fused_update_cache", "ttnn.experimental.">(
        mod,
        R"doc(
            Writes packed rows from two input tensors into two cache tensors in parallel.

            ``physical_update_idxs_tensor`` is a row-major INT32 tensor with shape
            ``[1, num_indices]``. Entry ``i`` gives the physical cache row for input
            row ``i``; negative and out-of-range entries are skipped. Physical rows
            flatten cache dimensions 0 and 2: ``page = index // cache.shape[2]`` and
            ``row_in_page = index % cache.shape[2]``. This operation does not perform
            logical page-table translation.

            Both caches and inputs must be interleaved BF16 TILE tensors. Cache shape
            is ``[num_pages, num_heads, rows_per_page, head_dim]`` and packed input
            shape is ``[1, num_heads, packed_rows, head_dim]``, where
            ``1 <= packed_rows <= 256`` and
            ``packed_rows <= num_indices <= 256``. The operation updates the cache
            tensors in place and serializes all rows owned by a head/width worker, so
            multiple rows may safely target the same physical page.

            Replicated tensors are supported on single- or multi-device meshes. The
            four cache/input tensors may also use one identical tensor-parallel mesh
            topology sharded over ``num_heads`` (dimension 1), ``head_dim``
            (dimension 3), or both. Physical update indices must use the same mesh
            shape and coordinates and remain replicated across those mesh axes.
            Cache page/row sharding, sharded physical indices, and per-device index
            remapping are not supported. Initial hardware support is Wormhole and
            Blackhole.

            Args:
                cache_tensor1 (ttnn.Tensor): First paged cache, updated in place.
                input_tensor1 (ttnn.Tensor): Packed rows for ``cache_tensor1``.
                cache_tensor2 (ttnn.Tensor): Second paged cache, updated in place.
                input_tensor2 (ttnn.Tensor): Packed rows for ``cache_tensor2``.
                physical_update_idxs_tensor (ttnn.Tensor): Physical destination rows.

            Returns:
                Tuple[ttnn.Tensor, ttnn.Tensor]: The two in-place cache tensors.
        )doc",
        &ttnn::experimental::indexed_fused_update_cache,
        nb::arg("cache_tensor1").noconvert(),
        nb::arg("input_tensor1").noconvert(),
        nb::arg("cache_tensor2").noconvert(),
        nb::arg("input_tensor2").noconvert(),
        nb::arg("physical_update_idxs_tensor").noconvert());
}

}  // namespace ttnn::operations::experimental::indexed_page_cache::detail
