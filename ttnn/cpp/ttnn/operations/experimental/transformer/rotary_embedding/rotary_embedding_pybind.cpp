// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "rotary_embedding.hpp"

namespace ttnn::operations::experimental::transformer {

void py_bind_rotary_embedding(pybind11::module& module) {
    namespace py = pybind11;
    ttnn::bind_registered_operation(module,
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
                                    ttnn::pybind_arguments_t{py::arg("input_tensor"),
                                                             py::arg("cos_cache"),
                                                             py::arg("sin_cache"),
                                                             py::arg("token_index") = std::nullopt,
                                                             py::kw_only(),
                                                             py::arg("memory_config") = std::nullopt,
                                                             py::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::transformer
