// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "zero_padded_kv_cache_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "zero_padded_kv_cache.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache::detail {

void bind_zero_padded_kv_cache(nb::module_& mod) {
    using ttnn::Tensor;
    using ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache::zero_padded_kv_cache;
    ttnn::bind_function<"zero_padded_kv_cache", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Zero the migration pad window [valid_global, ceil_pad_align(valid_global)) of a
            block-cyclic KV cache, in place.

            Both the single-shot fill and chunked update_padded_kv_cache write full 32-row
            tiles, so the tokens between the last real token (valid_global) and the next
            pad_align boundary hold stale data; this op clears them so the decode side reads
            clean zeros. The window is up to pad_align-1 tokens (1-4 tiles) and may straddle a
            chip boundary -- each device zeroes its own share, derived from valid_global,
            chunk_size_global and its coordinate along cluster_axis. The boundary (partial)
            tile is read, multiplied by an in-kernel row-mask and written back; the fully-pad
            tiles are written from the L1 zeros buffer.

            Cache slot is linearized users-outer, layers-inner: batch_idx = slot_idx*num_layers
            + layer_idx. ``valid_global`` and ``slot_idx`` stay out of the program hash, so
            successive chunks reuse one cached program.

            Two call forms (identical results):
              - scalar: ``(cache, slot_idx, layer_idx, num_layers, valid_global, chunk_size_global,
                cluster_axis, pad_align)`` -- host scalars patched on cache hits.
              - tensor: ``(cache, slot_idx, valid_global, layer_idx, num_layers, chunk_size_global,
                cluster_axis, pad_align)`` -- ``slot_idx`` and ``valid_global`` are 1-element uint32
                tensors; the reader/writer read element 0 of each on-device, so they never touch the
                host dispatch path. This form is trace-safe.

            Args:
                cache (ttnn.Tensor): 4D KV cache tensor on device, TILE layout, head dim 1.
                slot_idx (int, scalar form / ttnn.Tensor, tensor form): user slot in the batched prefill
                    cache. As a tensor: a 1-element uint32 DRAM tensor (replicated across the mesh, the
                    runner's h2d_socket_sync payload element 0 [slot_id]); read on-device from element 0.
                valid_global (int, scalar form / ttnn.Tensor, tensor form): number of real (non-pad)
                    global tokens; window starts here. As a tensor: a 1-element uint32 DRAM tensor
                    (replicated across the mesh, the runner's payload element 2 [actual_end]); read
                    on-device from element 0.
                layer_idx (int): Transformer layer index for this call (hashed/structural).
                num_layers (int): Total layers folded into the cache batch dim (structural).
                chunk_size_global (int): block-cyclic chunk size (= sp_factor * chunk_local).
                cluster_axis (int): Cluster axis along which the cache is sharded (0 or 1).
                pad_align (int): migration read alignment; window ends at ceil_pad_align (default 128).

            Returns:
                ttnn.Tensor: handle to `cache`, pad window zeroed in place.
        )doc",
        // Scalar form (original signature preserved).
        ttnn::overload_t(
            nb::overload_cast<const Tensor&, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(
                &zero_padded_kv_cache),
            nb::arg("cache").noconvert(),
            nb::arg("slot_idx"),
            nb::arg("layer_idx"),
            nb::arg("num_layers"),
            nb::arg("valid_global"),
            nb::arg("chunk_size_global"),
            nb::arg("cluster_axis"),
            nb::arg("pad_align") = 128),
        // Tensor form (traceable): slot_idx + valid_global as 1-element uint32 tensors.
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                const Tensor&,
                uint32_t,
                uint32_t,
                uint32_t,
                uint32_t,
                uint32_t>(&zero_padded_kv_cache),
            nb::arg("cache").noconvert(),
            nb::arg("slot_idx").noconvert(),
            nb::arg("valid_global").noconvert(),
            nb::arg("layer_idx"),
            nb::arg("num_layers"),
            nb::arg("chunk_size_global"),
            nb::arg("cluster_axis"),
            nb::arg("pad_align") = 128));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_zero_padded_kv_cache(::nanobind::module_& mod) {
    zero_padded_kv_cache::detail::bind_zero_padded_kv_cache(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
