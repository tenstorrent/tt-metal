// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_heads_matmul_decode_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "concat_heads_matmul_decode.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_concat_heads_matmul_decode(nb::module_& mod) {
    ttnn::bind_function<"concat_heads_matmul_decode", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused (free-view) concat-heads + output-projection via matmul_decode. The matmul_decode-based
        sibling of ttnn.experimental.concat_heads_matmul.

        attn [1, num_heads, seq, head_dim] (TILE, interleaved, seq <= 1 tile) is reinterpreted as
        [1, 1, seq, K = num_heads*head_dim] via a free build-time view (the concat-heads is the
        contiguous tile order for seq <= 1 tile), then resharded to a WIDTH_SHARDED input-A
        ([seq, K/reshard_cores] over reshard_cores cores) and fed to matmul_decode
        (partial_width_sharded=True, interleaved_output=True). Returns an INTERLEAVED L1 [1, 1, seq, N].

        Args:
            attn (ttnn.Tensor): [1, num_heads, seq, head_dim], TILE, interleaved.
            weight (ttnn.Tensor): partial-width-sharded resident-L1 weight (the _pws_B layout of a
                [K=num_heads*head_dim, N] projection).

        Keyword args:
            output_dtype (ttnn.DataType, optional): Defaults to bfloat16.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.
            reshard_cores (int, optional): cores to width-shard the input activation over. Defaults to 2.
            residual (ttnn.Tensor, optional): interleaved [seq, N] residual; when given with `gate`,
                folds in the gated residual epilogue out = residual + gate * (attn @ weight).
            gate (ttnn.Tensor, optional): per-channel gate, resident width-sharded across the N_blocks
                base cores (row-replicated), consumed by the fused-residual epilogue.

        Returns:
            ttnn.Tensor: the projected output [1, 1, seq, N], interleaved L1 (or the gated residual).
        )doc",
        &ttnn::experimental::concat_heads_matmul_decode,
        nb::arg("attn"),
        nb::arg("weight"),
        nb::kw_only(),
        nb::arg("output_dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("reshard_cores") = 2,
        nb::arg("residual") = nb::none(),
        nb::arg("gate") = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer
