// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "turbo_quant_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/turbo_quant/turbo_quant.hpp"

namespace ttnn::operations::experimental::turbo_quant {

void bind_turbo_quant_operations(nb::module_& mod) {
    ttnn::bind_function<"turbo_quant_bucketize", "ttnn.experimental.">(
        mod,
        R"doc(
Fused TurboQuant bucketize.  Maps normalised rotated values to integer
bucket indices using a single device kernel (replaces 13 cascaded TTNN ops).

Args:
    input_tensor: BF16 TILE tensor of normalised values.
    boundaries:   List of inner boundary floats (len = 2^bits - 1).

Returns:
    BF16 TILE tensor with integer indices 0 … 2^bits-1.
)doc",
        &ttnn::turbo_quant_bucketize,
        nb::arg("input_tensor").noconvert(),
        nb::arg("boundaries"));

    ttnn::bind_function<"turbo_quant_gather_centroids", "ttnn.experimental.">(
        mod,
        R"doc(
Fused TurboQuant gather centroids.  Maps integer indices to their
corresponding centroid values using a single device kernel (replaces 21
cascaded TTNN ops).

Args:
    input_tensor: BF16 TILE tensor with integer indices.
    centroids:    List of centroid floats (len = 2^bits).

Returns:
    BF16 TILE tensor with centroid values.
)doc",
        &ttnn::turbo_quant_gather_centroids,
        nb::arg("input_tensor").noconvert(),
        nb::arg("centroids"));
    ttnn::bind_function<"turbo_quant_sdpa_decode", "ttnn.experimental.">(
        mod,
        R"doc(
Fused TurboQuant SDPA decode.  Reads BFP4 quantized KV indices + BF16 norms
from paged cache and dequantizes on-the-fly during SDPA computation.
Eliminates the full-cache BF16 dequantize temporary.

Args:
    q:           BF16 query [B, NQH, 1, DH].
    k_indices:   BFP4 paged K indices [B, NKH, Sk, DH].
    k_norms:     BF16 K norms [B, NKH, Sk, 1].
    v_indices:   BFP4 paged V indices [B, NKH, Sk, vDH].
    v_norms:     BF16 V norms [B, NKH, Sk, 1].
    page_table:  Int32 page table [B, max_pages].
    cur_pos:     Int32 current position [B].
    centroids:   Centroid float values (len = 2^bits).
    scale:       Attention scale factor.
    pre_rescaled: If true, BFP4 values are already centroid*norm — skip gather+norm.
    num_cores_per_head: Tier 2A — number of worker cores per (batch, q_head) tuple
        that cooperate on the chunk loop. Default 1 = legacy behavior. Larger
        values split the chunk range across cores and merge partial state via a
        cross-core reduce; cap is the number of compute cores available divided
        by total (B * NQH) tuples.
    return_lse: If true, returns [out, lse] where lse holds LSE = max + log(sum)
        per (B, NQH) for the sliding-window hybrid host-side online-softmax
        combine. Default false returns [out].
    recent_window: Sliding-window fused hybrid (Phase 1, plumbing only). When > 0,
        the kernel will run a single fused SDPA over both the TQ cache (older
        positions) and a BFP8 ring buffer (most recent `recent_window` positions),
        replacing the dual-call + host-combine flow. Requires k_ring, v_ring,
        ring_page_table to be provided. Default 0 = legacy behavior.
    k_ring, v_ring: BFP8 paged ring caches over the most recent `recent_window`
        positions. Required when recent_window > 0. Layout matches the TQ paged
        cache ([B, NKH, ring_W_padded, head_dim]).
    ring_page_table: Int32 page table for the ring (identity mapping for now).
)doc",
        &ttnn::turbo_quant_sdpa_decode,
        nb::arg("q").noconvert(),
        nb::arg("k_indices").noconvert(),
        nb::arg("k_norms").noconvert(),
        nb::arg("v_indices").noconvert(),
        nb::arg("v_norms").noconvert(),
        nb::arg("page_table").noconvert(),
        nb::arg("cur_pos").noconvert(),
        nb::arg("centroids"),
        nb::arg("scale"),
        nb::arg("pre_rescaled") = false,
        nb::arg("num_cores_per_head") = 1,
        nb::arg("return_lse") = false,
        nb::arg("recent_window") = 0,
        nb::arg("k_ring") = std::nullopt,
        nb::arg("v_ring") = std::nullopt,
        nb::arg("ring_page_table") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::turbo_quant
