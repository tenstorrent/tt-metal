// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_decode_nanobind.hpp"
#include "kda_decode.hpp"

#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>

namespace ttnn::operations::transformer {

void bind_kda_decode(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Fused KDA (Kimi Delta Attention) single-token decode block: everything between the packed
        projection matmuls and o_proj in one program. For every user b and head h it computes the
        causal conv (kernel 4) over the device-owned history + SiLU, the per-head L2 norms, the
        per-key-channel forget gate g = lb * sigmoid(exp(A_log) * (f + dt_bias)), beta = sigmoid(b),
        the gated delta-rule step S = diag(exp g) S; u = beta (v - k S); S += k^T u; o = q S (q scaled by
        K^-0.5) and the gated RMSNorm rmsnorm(o) * w * sigmoid(og). Work items are (head, 32-user tile
        row) dealt over the cores; the state and the conv history are updated IN PLACE.

        Args:
            qkv (ttnn.Tensor):         [1, 1, B, 3*H*K] bf16 TILE, raw projections [q | k | v], head-major per branch.
            gates (ttnn.Tensor):       [1, 1, B, H*K + H*V + Hb] fp32 TILE: [f pre-activation | og pre-activation | beta logits],
                                       Hb = H rounded up to 32 (the padding columns are ignored).
            conv_state (ttnn.Tensor):  [B, 3, 3*H*K] bf16 ROW_MAJOR, the three previous raw rows (oldest first); updated in place.
            recurrent_state (ttnn.Tensor): [B, H, K, V] fp32 TILE; updated in place.
            taps (ttnn.Tensor):        [1, 1, 4, 3*H*K] fp32 TILE, row j = conv tap j (row 3 multiplies the current token).
            chan_consts (ttnn.Tensor): [1, 1, 2, H*K] fp32 TILE, row 0 = dt_bias, row 1 = exp(A_log) per key channel.
            o_norm_weight (ttnn.Tensor): [1, 1, 1, V] fp32 TILE.
            selectors (ttnn.Tensor):   [33*32, 32] bf16 TILE constant: tile r < 32 has a 1 at (0, r), tile 32 is all ones.
            num_heads (int):           H (the local heads).

        Keyword Args:
            gate_lower_bound (float): lb, default -5.0.
            l2_eps (float): l2norm eps inside the root, default 1e-6 (FLA).
            norm_eps (float): gated RMSNorm eps, default 1e-5.
            memory_config, compute_kernel_config: as usual (HiFi4 / fp32 acc by default; fp32 acc is required).

        Returns:
            ttnn.Tensor: o [1, 1, B, H*V] bf16 ROW_MAJOR (one row per user; tilize it for o_proj). Padding rows of
            qkv/gates must be finite (they are zero when produced by ttnn matmuls on zero-padded inputs). K = V = 128.
            Work items are (head, 32-user tile row, part): the users of a tile row are split over as many cores as
            the grid allows beyond heads x tile rows, so a TP shard with few local heads still fills the chip.
        )doc";

    ttnn::bind_function<"kda_decode", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::kda_decode,
        nb::arg("qkv").noconvert(),
        nb::arg("gates").noconvert(),
        nb::arg("conv_state").noconvert(),
        nb::arg("recurrent_state").noconvert(),
        nb::arg("taps").noconvert(),
        nb::arg("chan_consts").noconvert(),
        nb::arg("o_norm_weight").noconvert(),
        nb::arg("selectors").noconvert(),
        nb::arg("num_heads"),
        nb::kw_only(),
        nb::arg("gate_lower_bound") = -5.0f,
        nb::arg("l2_eps") = 1e-6f,
        nb::arg("norm_eps") = 1e-5f,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
