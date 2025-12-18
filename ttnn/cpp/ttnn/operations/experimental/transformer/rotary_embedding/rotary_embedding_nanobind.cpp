// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "rotary_embedding.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_rotary_embedding(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::rotary_embedding,
        R"doc(
        Applies the rotary embedding to the input_tensor tensor using the cos_cache and sin_cache tensors.

        When token_idx is passed, this assumes input is transposed to [seq_len, 1, B, head_dim], and seq_len is 1.


        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            cod_cache (ttnn.Tensor): the Cosine Cache tensor.
            sin_cache (ttnn.Tensor): the Sine Cache tensor.
            token_index (int, optional): Defaults to `None`.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.

        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("cos_cache"),
            nb::arg("sin_cache"),
            nb::arg("token_index") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}

}  // namespace ttnn::operations::experimental::transformer
