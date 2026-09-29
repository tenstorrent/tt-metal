// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concatenate_heads_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "concatenate_heads.hpp"

namespace ttnn::operations::transformer {

void bind_concatenate_heads(nb::module_& mod) {
    const auto* const doc =
        R"doc(
            Takes in a tensor of shape ``[batch_size, num_heads, sequence_size, head_size]``, concatenates heads back along the width dimension and returns the tensor of shape ``[batch_size, sequence_size, num_heads * head_size]``

            Args:
                input_tensor (ttnn.Tensor): the input tensor.

            Keyword Args:
                memory_config: Memory Config of the output tensor, if `None` then it gets set to input_tensor.memory_config(). Defaults to `None`.
                head_split (bool): interleaved input and output with fewer tile rows than cores: split the work per (tile row, head) instead of per tile row, so short sequences use the whole grid. Same output; ignored otherwise. Defaults to `False`.

            Returns:
                ttnn.Tensor: the output tensor.

        )doc";

    ttnn::bind_function<"concatenate_heads", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::concatenate_heads,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("head_split") = false);
}

}  // namespace ttnn::operations::transformer
