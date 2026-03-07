// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_nanobind.hpp"

#include <optional>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "repeat.hpp"

namespace ttnn::operations::data_movement {
namespace nb = nanobind;

void bind_repeat(nb::module_& mod) {
    const auto* doc = R"doc(
        Returns a new tensor filled with repetition of input :attr:`input_tensor` according to number of times specified in :attr:`shape`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            repetition_vector (SmallVector): The number of repetitions for each dimension.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
    )doc";

    ttnn::bind_function<"repeat">(
        mod,
        doc,
        ttnn::overload_t(
            static_cast<ttnn::Tensor (*)(
                const ttnn::Tensor&, const ttnn::SmallVector<uint32_t>&, const std::optional<MemoryConfig>&)>(
                &ttnn::repeat),
            nb::arg("input_tensor"),
            nb::arg("repeat_dims"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace ttnn::operations::data_movement
