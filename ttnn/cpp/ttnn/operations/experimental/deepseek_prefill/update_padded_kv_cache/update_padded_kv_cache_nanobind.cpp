// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "update_padded_kv_cache_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "update_padded_kv_cache.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache::detail {

void bind_update_padded_kv_cache(nb::module_& mod) {
    ttnn::bind_function<"update_padded_kv_cache", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Write a chunk_local-row input slab into a KV cache at a per-device start offset
            derived from a single global token count.

            Each device participating in the mesh derives its own write offset from
            `kv_actual_global` and its coordinate along `cluster_axis`, so that new tokens
            overwrite the trailing pad cells of the prior cache before spilling into the
            next slab. When `kv_actual_global` aligns to a chunk-global boundary the math
            degenerates to a uniform per-device offset (the chunked-natural case).

            In place: returns a handle to `cache`.

            Cache slot is linearized users-outer, layers-inner — internally the op composes
            ``batch_idx = slot_idx * num_layers + layer_idx``.

            ``slot_idx`` and ``kv_actual_global`` are read on-device from the ``metadata`` tensor,
            so their values stay off the host dispatch path and out of the program hash; successive
            users/chunks reuse one cached program (per layer).

            Args:
                cache (ttnn.Tensor): 4D KV cache tensor on device, TILE layout. Sharded across
                    `cluster_axis` with `sp_factor` slots per chip. Outermost dim equals
                    ``num_slots * num_layers``.
                input (ttnn.Tensor): 4D input slab on device, TILE layout, same dtype and head
                    dim as cache. Per-chip seq length = chunk_local.
                metadata (ttnn.Tensor): small uint32 DRAM tensor, replicated across the mesh (the
                    runner's h2d_socket_sync payload), canonical layout
                    ``[slot_id, actual_start, actual_end]``. The writer kernel reads ``slot_idx``
                    from index 0 and ``kv_actual_global`` (= ``actual_start``, tokens, tile-aligned)
                    from index 1 on-device, so neither touches the host dispatch path. The caller
                    packs valid values (host-side validation).
                layer_idx (int): Transformer layer index for this call. Structural (hashed): one
                    cached program per layer is reused across users and chunks.
                num_layers (int): Total layers folded into the cache batch dim. Structural —
                    fixed for the lifetime of the workload.
                cluster_axis (int): Cluster axis along which the cache is sharded (0 or 1).

            Returns:
                ttnn.Tensor: handle to `cache` with the new slab written in place.
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache::update_padded_kv_cache,
        nb::arg("cache").noconvert(),
        nb::arg("input").noconvert(),
        nb::arg("metadata").noconvert(),
        nb::arg("layer_idx"),
        nb::arg("num_layers"),
        nb::arg("cluster_axis"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_update_padded_kv_cache(::nanobind::module_& mod) {
    update_padded_kv_cache::detail::bind_update_padded_kv_cache(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
