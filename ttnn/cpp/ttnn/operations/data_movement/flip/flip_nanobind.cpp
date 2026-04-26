// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flip_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include "flip.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_flip(nb::module_& mod) {
    const auto* doc = R"doc(
        mod,
        ttnn::flip,
        R"doc(
        Reverse the order of an n-D tensor along given axis in dims.

        Input Specs:
            - **Supported:**
                - Interleaved row-major layout tensors of following dtypes: `bfloat16`, `float32`, `int32`
                - Interleaved tiled layout tensors of following dtypes: `bfloat16`, `float32`, `int32`

            - **Not Supported:**
                - Sharded tensors

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            dims (List[int]): the dimensions to reverse.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> # Create a simple tensor
            >>> torch_tensor = torch.arange(12).reshape(3, 4).float()
            >>> tensor = ttnn.from_torch(torch_tensor, device=device)
            >>>
            >>> # Flip along dimension 0 (rows)
            >>> flipped = ttnn.flip(tensor, dims=[0])
            >>>
            >>> # Flip along dimension 1 (columns)
            >>> flipped_cols = ttnn.flip(tensor, dims=[1])
            >>>
            >>> # Flip along both dimensions
            >>> flipped_both = ttnn.flip(tensor, dims=[0, 1])

            >>> x = ttnn.to_device(ttnn.from_torch(torch.arrange(8).view(2, 2, 2), dtype=torch.bfloat16)), device)
            >>> x
            tensor([[[ 0,  1],
                     [ 2,  3]],

                    [[ 4,  5],
                     [ 6,  7]]])
            >>> flipped_x = ttnn.flip(x, (0, 1))
            >>> flipped_x
            tensor([[[ 6,  7],
                     [ 4,  5]],

                    [[ 2,  3],
                     [ 0,  1]]])
        )doc";

    ttnn::bind_function<"flip">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttnn::SmallVector<int64_t>&,
                const std::optional<ttnn::MemoryConfig>&>(&ttnn::flip),
            nb::arg("input_tensor").noconvert(),
            nb::arg("dims"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()),
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, const ttnn::SmallVector<int64_t>&>(&ttnn::flip),
            nb::arg("input_tensor").noconvert(),
            nb::arg("dims")));
}

}  // namespace ttnn::operations::data_movement::detail
