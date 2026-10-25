// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "update_padded_kv_cache_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "update_padded_kv_cache.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache::detail {

void bind_update_padded_kv_cache(nb::module_& mod) {
    using ttnn::Tensor;
    using ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache::update_padded_kv_cache;
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

            Without ``paged_kv_table`` the cache slot is linearized users-outer, layers-inner — the op composes
            ``batch_idx = slot_idx * num_layers + layer_idx``. ``slot_idx`` and ``kv_actual_global``
            stay out of the program hash, so successive users/chunks reuse one cached program (per
            layer).

            Two call forms (identical results):
              - scalar: ``(cache, input, slot_idx, layer_idx, num_layers, kv_actual_global,
                cluster_axis)`` — the per-call values are host scalars patched on cache hits.
              - per-element-tensor: ``(cache, input, slot_idx, kv_actual_global, layer_idx,
                num_layers, cluster_axis)`` — ``slot_idx`` and ``kv_actual_global`` are 1-element
                uint32 tensors; the writer kernel reads element [0] of each on-device, so they never
                touch the host dispatch path. This form is trace-safe (one captured program replays
                across chunks/users).

            Args:
                cache (ttnn.Tensor): 4D KV cache tensor on device, TILE or ROW_MAJOR layout. Without
                    ``paged_kv_table``: sharded across `cluster_axis` with `sp_factor` slots per chip, outermost dim
                    ``num_slots * num_layers``. With it, a per-chip page pool: see ``paged_kv_table``.
                    Layout/dtype must match ``input``: block-float
                    (bfloat8_b/bfloat4_b) requires TILE; FP8_E4M3 requires ROW_MAJOR (Blackhole).
                    Seq dims stay 32-aligned in both layouts, and TILE layout requires the standard
                    32x32 tile.
                input (ttnn.Tensor): 4D input slab on device, same layout, dtype and head dim as
                    cache. Per-chip seq length = chunk_local.
                slot_idx (int | ttnn.Tensor): scalar form: user slot in the batched prefill cache, or with
                    ``paged_kv_table`` the request's page-table row.
                    per-element-tensor form: a 1-element uint32 DRAM tensor ([1,1,1,1], ROW_MAJOR,
                    replicated across the mesh) whose element [0] is the user slot (with ``paged_kv_table``, the
                    row); read on-device.
                kv_actual_global (int | ttnn.Tensor): scalar form: prior valid global KV length in
                    tokens (tile-aligned). per-element-tensor form: a 1-element uint32 DRAM tensor
                    (same layout as ``slot_idx``) whose element [0] is that length; read on-device.
                    The caller packs valid values for the tensor form (host-side validation).
                layer_idx (int): Transformer layer index for this call. Structural (hashed): one
                    cached program per layer is reused across users and chunks.
                num_layers (int): Total layers folded into the cache batch dim. Structural —
                    fixed for the lifetime of the workload.
                cluster_axis (Optional[int]): Cluster axis along which the cache is sharded (0 or 1).
                    Explicit None means that sequence shards span the complete 2D mesh in canonical
                    row-major coordinate order.
                valid_global (int | ttnn.Tensor, optional): end of this chunk's REAL tokens — an int
                    (scalar form) or a 1-element uint32 tensor read on-device (tensor form). Given it,
                    only the rows holding real tokens are written, so the cache must fit
                    ``ceil32(valid_global)`` rather than ``kv_actual_global + chunk_global``. The
                    32-token block holding the last real token is still written; clear it with
                    ``zero_padded_kv_cache``, which takes no ``paged_kv_table``: a page pool keeps the input's pad
                    rows there. Omitted: the whole padded slab is written.
                tp_axis (int, optional): Second axis to also shard the cache across (KV dedup).
                    ``input`` must then be TP-replicated and single-head, and each chip persists only
                    its own ``1/tp`` seq window. Must differ from ``cluster_axis``.

                rope (ttnn.Tensor, optional): Separate TILE BF16 RoPE values. When supplied,
                    input is TILE BF16 latent and cache is ROW_MAJOR BF16, with width equal to
                    the sum of the two input widths. Packing and untilization are fused into
                    the cache write; both inputs must be interleaved, single-batch, single-head.
                scales (ttnn.Tensor, optional): Select scaled-FP8 packing. Input is ROW_MAJOR
                    FP8_E4M3 latent (512 values), scales is ROW_MAJOR FP32 (4 values), and rope
                    is BF16 (64 values), ROW_MAJOR or TILE. Tiled RoPE is untilized inside this
                    operation using LLK. Cache is ROW_MAJOR FP8_E4M3 with 656-byte
                    rows containing the three fields. All fields share leading shapes.
                input_row_offset (int, optional, tensor form): first input row (token, multiple of 32)
                    to write; with ``input_rows``, write rows ``[input_row_offset, input_row_offset +
                    input_rows)`` of every head as if that slice were the input. Plain page copy only.
                input_rows (int, optional, tensor form): rows in the window (multiple of 32).
                paged_kv_table (ttnn.Tensor, optional): Selects a paged cache (not the ``paged_update_cache`` /
                    SDPA ``page_table`` format). ``cache`` is then, on every chip, its own pool of single-head
                    pages ``[num_pages * num_layers, 1, page, head_dim]`` (replicate it, or shard a
                    ``[.., sp * page, ..]`` tensor along ``cluster_axis``) whose per-chip seq dim is the page size (a
                    multiple of 32 dividing ``chunk_local``); page ``p`` of layer ``l`` is entry
                    ``p * num_layers + l``.
                    ``paged_kv_table`` is an INT32 ROW_MAJOR DRAM tensor ``[rows, width]`` replicated across the
                    mesh, and ``slot_idx`` selects the request's row, which holds at most ``width * page`` tokens:
                    the scalar form rejects a longer write, the tensor form writes only the row's pages.
                    Entry ``q`` names the physical page holding global positions ``[q * page, (q + 1) * page)``
                    in the pool of the chip at SP coordinate ``(q * page // chunk_local) % sp`` with
                    ``chunk_local`` the rows the call writes per chip (the input row window when one is given, else the
                    per-chip input seq) and ``sp`` the mesh extent along ``cluster_axis``: every write of a request must
                    use the same ``chunk_local``. -1 is unmapped and its rows are not
                    written; neither is an entry outside the pool, which is invalid. The tensor form does not check
                    ``slot_idx`` against the table's rows: a row past them is read from beyond the table, and its
                    writes land on arbitrary pool pages. The table's width is part of the program key, so a
                    workload keeps one fixed-width table. Plain copy path only (no ``rope``, ``scales`` or
                    ``tp_axis``); requires ``cluster_axis``. Format:
                    ``ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/paged_kv_table.hpp``.

            Returns:
                ttnn.Tensor: handle to `cache` with the new slab written in place.
        )doc",
        // Scalar form (original signature preserved).
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                uint32_t,
                uint32_t,
                uint32_t,
                uint32_t,
                std::optional<uint32_t>,
                std::optional<uint32_t>,
                std::optional<uint32_t>,
                const std::optional<Tensor>&,
                const std::optional<Tensor>&,
                const std::optional<Tensor>&>(&update_padded_kv_cache),
            nb::arg("cache").noconvert(),
            nb::arg("input").noconvert(),
            nb::arg("slot_idx"),
            nb::arg("layer_idx"),
            nb::arg("num_layers"),
            nb::arg("kv_actual_global"),
            nb::arg("cluster_axis"),
            nb::arg("valid_global") = nb::none(),
            nb::arg("tp_axis") = nb::none(),
            nb::arg("rope").noconvert() = nb::none(),
            nb::arg("scales").noconvert() = nb::none(),
            nb::arg("paged_kv_table").noconvert() = nb::none()),
        // Per-element-tensor form (traceable).
        ttnn::overload_t(
            nb::overload_cast<
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                uint32_t,
                uint32_t,
                std::optional<uint32_t>,
                const std::optional<Tensor>&,
                std::optional<uint32_t>,
                const std::optional<Tensor>&,
                const std::optional<Tensor>&,
                uint32_t,
                std::optional<uint32_t>,
                const std::optional<Tensor>&>(&update_padded_kv_cache),
            nb::arg("cache").noconvert(),
            nb::arg("input").noconvert(),
            nb::arg("slot_idx").noconvert(),
            nb::arg("kv_actual_global").noconvert(),
            nb::arg("layer_idx"),
            nb::arg("num_layers"),
            nb::arg("cluster_axis"),
            nb::arg("valid_global").noconvert() = nb::none(),
            nb::arg("tp_axis") = nb::none(),
            nb::arg("rope").noconvert() = nb::none(),
            nb::arg("scales").noconvert() = nb::none(),
            nb::arg("input_row_offset") = 0,
            nb::arg("input_rows") = nb::none(),
            nb::arg("paged_kv_table").noconvert() = nb::none()));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_update_padded_kv_cache(::nanobind::module_& mod) {
    update_padded_kv_cache::detail::bind_update_padded_kv_cache(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
