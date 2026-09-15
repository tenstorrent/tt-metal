// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_post_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "mhc_post.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_post::detail {

void bind_experimental_mhc_post_operation(nb::module_& mod) {
    ttnn::bind_function<"mhc_post", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Fused mHC post-mix (Manifold-Constrained Hyper-Connections, DeepSeek-V4).

            Applies H_post and H_res to the sublayer output and the n residual streams in one
            pass over the packed [1,1,T,n*C] layout:

                out[:, j*C:(j+1)*C] = post[:,j] * y + sum_i comb[:, i*n+j] * residual[:, i*C:(i+1)*C]

            Replaces the n column slices, n*n addcmuls and concat of the composite form, each of
            which round-trips a full [1,1,T,C] accumulator through DRAM.

            Args:
                * :attr:`y`: [1,1,T,C] FLOAT32 TILE, the sublayer output F(H_pre.X).
                * :attr:`residual`: [1,1,T,n*C] FLOAT32 TILE, stream i at columns [i*C,(i+1)*C).
                * :attr:`post`: [1,1,T,n] FLOAT32 TILE, from mhc_split_sinkhorn.
                * :attr:`comb`: [1,1,T,n*n] FLOAT32 TILE, from mhc_split_sinkhorn; entry (i,j) at
                  column i*n+j.
                * :attr:`consts`: [n*n,32,32] FLOAT32 TILE column-broadcast tiles (tile k has row k
                  all ones); see the Python wrapper.
                * :attr:`n`: expansion rate (streams).

            Returns:
                out [1,1,T,n*C] FLOAT32 TILE.
        )doc",
        &mhc_post,
        nb::arg("y").noconvert(),
        nb::arg("residual").noconvert(),
        nb::arg("post").noconvert(),
        nb::arg("comb").noconvert(),
        nb::arg("consts").noconvert(),
        nb::arg("n"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_post::detail
