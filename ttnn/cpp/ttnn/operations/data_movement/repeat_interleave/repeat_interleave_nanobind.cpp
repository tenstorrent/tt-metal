// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_interleave_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "repeat_interleave.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_repeat_interleave(nb::module_& mod) {
    auto doc =
        R"doc(
        Repeats elements of a :attr:`tensor` in the given :attr:`dim`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            repeats (number): he number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.
            dim (number): the dimension to expand with the repetitions.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:

        torch_input_tensor =
            torch_result = torch.repeat_interleave(torch_input_tensor, repeats, dim=dim)

            input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

            output = ttnn.repeat_interleave(input_tensor, repeats, dim=dim)
            >>> a = ttnn.from_torch(torch.rand(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> b = ttnn.repeat_interleave(a, 2, dim=0)
            >>> print(a.shape, b.shape)
            ttnn.Shape([1, 1, 32, 32]) ttnn.Shape([2, 1, 32, 32])
        )doc";

    ttnn::bind_registered_operation(
        mod,
        ttnn::repeat_interleave,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("repeats"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()});
}

}  // namespace ttnn::operations::data_movement::detail
