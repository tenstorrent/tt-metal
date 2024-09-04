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
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::rotary_embedding,
        R"doc(rotary_embedding(input_tensor: ttnn.Tensor, cos_cache: ttnn.Tensor, sin_cache: ttnn.Tensor, token_index: int, memory_config: MemoryConfig, compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor

            Applies the rotary embedding to the input_tensor tensor using the cos_cache and sin_cache tensors.

            When token_idx is passed, this assumes input is transposed to [seq_len, 1, B, head_dim], and seq_len is 1.

            Args:
                * :attr:`input_tensor`: Input Tensor
                * :attr:`cos_cache`: Cosine Cache Tensor
                * :attr:`sin_cache`: Sine Cache Tensor
                * :attr:`token_index`: Token Index = None
                * :attr:`memory_config`: Memory Config of the output tensor = None
                * :attr:`compute_kernel_config`: Optional[DeviceComputeKernelConfig] = None
        )doc",
        ttnn::pybind_arguments_t {
            py::arg("input_tensor"),
            py::arg("cos_cache"),
            py::arg("sin_cache"),
            py::arg("token_index") = std::nullopt,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::transformer
