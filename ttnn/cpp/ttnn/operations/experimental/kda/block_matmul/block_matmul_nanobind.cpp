// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "block_matmul_nanobind.hpp"

#include "block_matmul.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>

namespace ttnn::operations::experimental::kda::block_matmul::detail {

void bind_block_matmul(nb::module_& mod) {
    ttnn::bind_function<"block_matmul", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        BFLOAT16 ``a @ w`` on a rectangle of worker cores. Each core owns a tile block of the output and streams
        K in ``k_block``-tile blocks; A is multicast along rows (NoC 1) and W down columns (NoC 0), K blocks
        accumulate in FP32 and the last packs BFLOAT16 subblocks that are written as soon as they are packed.
        ``ring_relay_prefix`` and ``select_final_carry_with_projection`` run the same program in their dispatch.
        )doc",
        &ttnn::experimental::kda::block_matmul,
        nb::arg("a").noconvert(),
        nb::arg("w").noconvert(),
        nb::kw_only(),
        nb::arg("cores"),
        nb::arg("k_block"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::block_matmul::detail
