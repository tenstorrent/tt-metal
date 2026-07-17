// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/gated_delta_attn/gated_delta_attn_nanobind.hpp"
#include "ttnn/operations/transformer/gated_delta_attn/gated_delta_attn.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

namespace ttnn::operations::transformer {

void bind_gated_delta_attn_seq(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Gated DeltaNet attention — sequential inter-chunk scan (Path A).

        All inputs must be float32. Python pre-normalises L_unit to unit-diagonal
        form and precomputes L_inv (diagonal block inverses). The C++ kernel performs
        blocked forward substitution and the sequential inter-chunk state update.

        Args:
            L_unit (ttnn.Tensor):      [BH, NC, C, C]    unit-diagonal lower-tri
            v_beta_sc (ttnn.Tensor):   [BH, NC, C, Dv]   D^{-1} @ v_beta
            k_bd_sc (ttnn.Tensor):     [BH, NC, C, Dk]   D^{-1} @ k_beta_decay
            intra_attn (ttnn.Tensor):  [BH, NC, C, C]    intra-chunk attention
            q_decay (ttnn.Tensor):     [BH, NC, C, Dk]   queries with decay
            k_decay_t (ttnn.Tensor):   [BH, NC, Dk, C]   transposed keys with decay
            dl_exp (ttnn.Tensor):      [BH, NC, 1, 1]    state decay scalar (fp32)
            L_inv (ttnn.Tensor):       [BH, NC, C, 32]   4 diagonal block inverses per chunk

        Keyword Args:
            initial_state (ttnn.Tensor, optional): [BH, Dk, Dv] initial state (zeros if absent).
            memory_config (ttnn.MemoryConfig, optional): output memory config.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional).
            token_major_output (bool, optional): when True, the writer scatters heads directly into
                a token-major output [B, seq_len, num_v_heads*Dv] (folds the head->token permute).
                Requires num_v_heads and seq_len. Defaults to False (legacy [BH, NC, C, Dv]).
            num_v_heads (int, optional): H (value heads per batch); B = BH / H. Used iff token_major_output.
            seq_len (int, optional): T (logical sequence length, pre chunk-pad). Used iff token_major_output.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]:
                output [BH, NC, C, Dv] (or [B, seq_len, num_v_heads*Dv] if token_major_output),
                final_state [BH, Dk, Dv]
        )doc";

    ttnn::bind_function<"gated_delta_attn_seq", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::gated_delta_attn_seq,
        nb::arg("L_unit").noconvert(),
        nb::arg("v_beta_sc").noconvert(),
        nb::arg("k_bd_sc").noconvert(),
        nb::arg("intra_attn").noconvert(),
        nb::arg("q_decay").noconvert(),
        nb::arg("k_decay_t").noconvert(),
        nb::arg("dl_exp").noconvert(),
        nb::arg("L_inv").noconvert(),
        nb::kw_only(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("token_major_output") = false,
        nb::arg("num_v_heads") = 0,
        nb::arg("seq_len") = 0);
}

}  // namespace ttnn::operations::transformer
