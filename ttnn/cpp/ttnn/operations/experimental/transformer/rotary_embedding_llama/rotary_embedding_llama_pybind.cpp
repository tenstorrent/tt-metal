// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/ttnn-pybind/decorators.hpp"

#include "rotary_embedding_llama.hpp"

namespace ttnn::operations::experimental::transformer {

void py_bind_rotary_embedding_llama(pybind11::module& module) {
    namespace py = pybind11;

    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::rotary_embedding_llama,
        R"doc(rotary_embedding_llama(input_tensor: ttnn.Tensor, cos_cache: ttnn.Tensor, sin_cache: ttnn.Tensor, trans_mat: ttnn.Tensor, is_decode_mode: bool, memory_config: MemoryConfig, compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor

            Applies the rotary embedding to the input_tensor tensor using the cos_cache and sin_cache tensors.

            When token_idx is passed, this assumes input is transposed to [seq_len, 1, B, head_dim], and seq_len is 1.

            `cos_cache` and `sin_cache` must be of shape [1, n_heads, seq_len, head_dim] or [1, 1, seq_len, head_dim].
            If shape[1] is 1 then the sin/cos will be broadcasted across heads.

            Args:
                * :attr:`input_tensor`: Input Tensor
                * :attr:`cos_cache`: Cosine Cache Tensor
                * :attr:`sin_cache`: Sine Cache Tensor
                * :attr:`trans_mat`: Transformation Matrix Tensor
                * :attr:`is_decode_mode`: Specify mode of operation
                * :attr:`memory_config`: Memory Config of the output tensor = DEFAULT_OUTPUT_MEMORY_CONFIG
                * :attr:`compute_kernel_config`: Optional[DeviceComputeKernelConfig] = None
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("cos_cache"),
            py::arg("sin_cache"),
            py::arg("trans_mat"),
            py::kw_only(),
            py::arg("is_decode_mode") = false,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::transformer
