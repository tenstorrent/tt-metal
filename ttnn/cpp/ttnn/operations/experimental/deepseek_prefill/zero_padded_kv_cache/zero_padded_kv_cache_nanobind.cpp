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
            chunk_size_global and its coordinate along cluster_axis. TILE layout uses the
            mask/compute path: the boundary (partial) tile is read, multiplied by an in-kernel
            row-mask and written back; fully-pad tiles are written from an L1 zeros buffer.
            ROW_MAJOR BF16 and FP8_E4M3 caches use a dataflow-only path that zeroes complete
            token rows directly, so the payload never enters the unpack/compute engine.

            Cache slot is linearized users-outer, layers-inner: batch_idx = slot_idx*num_layers
            + layer_idx. ``valid_global`` and ``slot_idx`` are per-call scalars patched on cache
            hits, out of the program hash.

            Args:
                cache (ttnn.Tensor): 4D, DRAM-backed KV cache tensor with head dim 1. Supports
                    TILE layout, or ROW_MAJOR layout with BF16 or FP8_E4M3 dtype.
                slot_idx (int): user slot in the batched prefill cache.
                layer_idx (int): Transformer layer index for this call (hashed/structural).
                num_layers (int): Total layers folded into the cache batch dim (structural).
                valid_global (int): number of real (non-pad) global tokens; window starts here.
                chunk_size_global (int): block-cyclic chunk size (= sp_factor * chunk_local).
                cluster_axis (int): Cluster axis along which the cache is sharded (0 or 1).
                pad_align (int): migration read alignment; window ends at ceil_pad_align (default 128).

            Returns:
                ttnn.Tensor: handle to `cache`, pad window zeroed in place.
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache::zero_padded_kv_cache,
        nb::arg("cache").noconvert(),
        nb::arg("slot_idx"),
        nb::arg("layer_idx"),
        nb::arg("num_layers"),
        nb::arg("valid_global"),
        nb::arg("chunk_size_global"),
        nb::arg("cluster_axis"),
        nb::arg("pad_align") = 128);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_zero_padded_kv_cache(::nanobind::module_& mod) {
    zero_padded_kv_cache::detail::bind_zero_padded_kv_cache(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
