// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotate_half_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "rotate_half.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_rotate_half(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::rotate_half,
        R"doc(rotate_half(input: ttnn.Tensor, *, memory_config: Optional[MemoryConfig] = None) -> ttnn.Tensor

            Performs a rotate half operation used by RotaryEmbedding

            Args:
                * :attr:`input`: Input Tensor

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor, if None then it gets set to input.memory_config()
        )doc",
        ttnn::nanobind_arguments_t{nb::arg("input"), nb::kw_only(), nb::arg("memory_config") = nb::none()});
}

}  // namespace ttnn::operations::experimental::transformer
