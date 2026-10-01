// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reallocate_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/quasar/reallocate/reallocate.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_reallocate(nb::module_& mod) {
    // Bind as a proper ttnn operation object (callable wrapper with the __ttnn_operation__ marker),
    // matching the matmul/linear form in this op group, rather than a plain mod.def() free function.
    ttnn::bind_function<"reallocate", "ttnn.experimental.quasar.">(
        mod,
        R"doc(
            Deallocates device tensor and returns a reallocated tensor.

            Args:
                tensor (ttnn.Tensor): the input tensor.
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the reallocated tensor. Defaults to `None`.

            Returns:
                ttnn.Tensor: the reallocated tensor.
        )doc",
        ttnn::overload_t(
            &ttnn::operations::experimental::quasar::reallocate,
            nb::arg("tensor"),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace ttnn::operations::experimental::quasar::detail
