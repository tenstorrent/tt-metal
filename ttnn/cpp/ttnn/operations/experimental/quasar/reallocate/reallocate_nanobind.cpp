// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reallocate_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn/operations/experimental/quasar/reallocate/reallocate.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_reallocate(nb::module_& mod) {
    mod.def(
        "reallocate",
        [](ttnn::Tensor& input_tensor, const std::optional<ttnn::MemoryConfig>& memory_config) -> ttnn::Tensor {
            return ttnn::operations::experimental::quasar::reallocate(input_tensor, memory_config);
        },
        nb::arg("tensor"),
        nb::arg("memory_config") = nb::none(),
        R"doc(
            Deallocates device tensor and returns a reallocated tensor.

            Args:
                tensor (ttnn.Tensor): the input tensor.
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the reallocated tensor. Defaults to `None`.

            Returns:
                ttnn.Tensor: the reallocated tensor.
        )doc");
}

}  // namespace ttnn::operations::experimental::quasar::detail
