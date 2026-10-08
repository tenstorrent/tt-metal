// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "simple_add_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/quasar/simple_add/simple_add.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_simple_add(nb::module_& mod) {
    ttnn::bind_function<"simple_add", "ttnn.experimental.quasar.">(
        mod,
        R"doc(
            Element-wise C = A + B on a single node. On a Quasar Neo cluster the reader runs on 4 DM cores, the
            writer on 2 DM cores and the compute kernel on all 4 Tensix engines; on Wormhole/Blackhole each is a
            single thread.

            Both inputs must be bfloat16, TILE layout and DRAM interleaved, with the same shape (no broadcast).

            Args:
                input_a (ttnn.Tensor): the first addend.
                input_b (ttnn.Tensor): the second addend.
                memory_config (ttnn.MemoryConfig, optional): output memory configuration; must be DRAM
                    interleaved. Defaults to `input_a`'s.

            Returns:
                ttnn.Tensor: A + B, bfloat16, TILE layout.
        )doc",
        ttnn::overload_t(
            &ttnn::operations::experimental::quasar::simple_add,
            nb::arg("input_a"),
            nb::arg("input_b"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace ttnn::operations::experimental::quasar::detail
