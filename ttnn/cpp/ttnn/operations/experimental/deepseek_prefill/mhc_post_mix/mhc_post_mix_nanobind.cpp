// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_post_mix_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "mhc_post_mix.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_post_mix::detail {

void bind_experimental_mhc_post_mix_operation(nb::module_& mod) {
    ttnn::bind_function<"mhc_post_mix", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Fused mHC residual update (exact fp32): for the packed streams ``x`` [1,1,T,n*C], the
            sublayer output ``y`` [1,1,T,C] and the ``mhc_split_sinkhorn`` coefficients ``post`` [T,n],
            ``comb`` [T,n*n]:

                out[t, j*C + c] = sum_i comb[t, i*n + j] * x[t, i*C + c] + post[t, j] * y[t, c]

            One program: every output tile is computed with SFPU fp32 multiply-adds from the 4 stream
            tiles, the y tile and the per-token coefficient columns broadcast across the tile by the
            unpacker's 32-bit column-broadcast datacopy. Rows are independent (pad rows stay pad rows).

            Args:
                * :attr:`x`: [1, 1, T, n*C] FLOAT32 TILE, T % 32 == 0.
                * :attr:`y`: [1, 1, T, C] FLOAT32 TILE.
                * :attr:`post`: [T, n] FLOAT32 TILE.
                * :attr:`comb`: [T, n*n] FLOAT32 TILE.
                * :attr:`n`: number of streams (n*n + n <= 32).
                * :attr:`collapse`: return the mean over the n new streams [1, 1, T, C] instead.

            Returns:
                [1, 1, T, n*C] FLOAT32 TILE (DRAM interleaved), or [1, 1, T, C] with ``collapse``.
        )doc",
        &mhc_post_mix,
        nb::arg("x").noconvert(),
        nb::arg("y").noconvert(),
        nb::arg("post").noconvert(),
        nb::arg("comb").noconvert(),
        nb::arg("n"),
        nb::arg("collapse") = false);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_post_mix::detail
