// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_split_sinkhorn_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "mhc_split_sinkhorn.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_split_sinkhorn::detail {

void bind_experimental_mhc_split_sinkhorn_operation(nb::module_& mod) {
    ttnn::bind_function<"mhc_split_sinkhorn", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Fused mHC parametrization (Manifold-Constrained Hyper-Connections, DeepSeek-V4).

            Splits the fused-projection output ``mixes`` into H_pre, H_post and H_res and
            Sinkhorn-normalizes H_res on the doubly-stochastic manifold. Token-parallel across
            the Tensix grid (also single-core, L1-sharded, and multi-device paths).
            Matches models/demos/deepseek_v3_d_p/reference/mhc/mhc_reference.py::parametrize.

            The scalars a and biases b, plus the row/col-sum selection matrices, are baked
            into ``consts`` host-side (see the Python wrapper), so the kernel is pure tile ops.

            Args:
                * :attr:`mixes`: [T, (2+n)*n] FLOAT32 TILE.
                * :attr:`consts`: [8, 32, 32] FLOAT32 TILE (SEL_pre, SEL_post, SEL_comb,
                  base_pre, base_post, base_comb, RB, CB).
                * :attr:`n`: expansion rate (streams).
                * :attr:`sinkhorn_iters`: Sinkhorn-Knopp iterations.
                * :attr:`eps`: stochasticity epsilon.

            Returns:
                (pre [T,n], post [T,n], comb [T,n*n]) FLOAT32 TILE. For a sharded ``mixes`` the
                outputs come back 32-wide sharded (real values in the first n / n*n columns, the
                rest padding); the caller slices [:, :w].
        )doc",
        &mhc_split_sinkhorn,
        nb::arg("mixes").noconvert(),
        nb::arg("consts").noconvert(),
        nb::arg("n"),
        nb::arg("sinkhorn_iters"),
        nb::arg("eps"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_split_sinkhorn::detail
