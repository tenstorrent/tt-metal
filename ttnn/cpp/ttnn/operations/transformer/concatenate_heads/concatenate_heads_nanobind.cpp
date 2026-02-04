// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concatenate_heads_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "concatenate_heads.hpp"

namespace ttnn::operations::transformer {

void bind_concatenate_heads(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::transformer::concatenate_heads,
        R"doc(
            Takes in a tensor of shape ``[batch_size, num_heads, sequence_size, head_size]``, concatenates heads back along the width dimension and returns the tensor of shape ``[batch_size, sequence_size, num_heads * head_size]``

            Args:
                input_tensor (ttnn.Tensor): the input tensor.

            Keyword Args:
                memory_config: Memory Config of the output tensor, if `None` then it gets set to input_tensor.memory_config(). Defaults to `None`.

            Returns:
                ttnn.Tensor: the output tensor.

        )doc",
        ttnn::nanobind_arguments_t{nb::arg("input_tensor"), nb::kw_only(), nb::arg("memory_config") = nb::none()});
}

}  // namespace ttnn::operations::transformer
