// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"
#include "permute.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_permute(nb::module_& mod) {
    const auto* doc = R"doc(
        Permutes the dimensions of :attr:`input_tensor` (Quasar-native). Same contract as :func:`ttnn.permute`
        for interleaved tensors: output dimension ``k`` takes input dimension ``dims[k]``.

        Args:
            input_tensor (ttnn.Tensor): the input tensor (TILE or ROW_MAJOR layout, interleaved, rank 1 to 8).
            dims (List[int]): the desired ordering of dimensions; negative values count from the end.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): interleaved output memory config. Defaults to the
                input's.
            pad_value (float): value of the tile padding of a TILE output. Defaults to `0.0`.

        Returns:
            ttnn.Tensor: the permuted tensor, in the layout and data type of the input.
        )doc";

    ttnn::bind_function<"permute", "ttnn.experimental.quasar.">(
        mod,
        doc,
        &ttnn::operations::experimental::quasar::permute,
        nb::arg("input_tensor").noconvert(),
        nb::arg("dims"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("pad_value") = 0.0f);
}

}  // namespace ttnn::operations::experimental::quasar::detail
