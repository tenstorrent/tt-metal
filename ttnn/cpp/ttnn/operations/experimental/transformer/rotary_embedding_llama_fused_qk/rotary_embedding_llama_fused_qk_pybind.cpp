// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_fused_qk_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "rotary_embedding_llama_fused_qk.hpp"

namespace ttnn::operations::experimental::transformer {

void py_bind_rotary_embedding_llama_fused_qk(pybind11::module& module) {
    namespace py = pybind11;

    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::rotary_embedding_llama_fused_qk,
        R"doc(rotary_embedding_llama_fused_qk(q_input_tensor: ttnn.Tensor, k_input_tensor: ttnn.Tensor, cos_cache: ttnn.Tensor, sin_cache: ttnn.Tensor, trans_mat: ttnn.Tensor, compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor

            Applies the rotary embedding to the q_input_tensor and k_input_tensor parallely using the cos_cache and sin_cache tensors.

            Args:
                * :attr:`q_input_tensor`: Q Input Tensor [1, q_batch, num_heads, head_dim]
                * :attr:`k_input_tensor`: K Input Tensor [1, k_batch, num_kv_heads, head_dim]
                * :attr:`cos_cache`: Cosine Cache Tensor [1, (q_batch + k_batch), 1(32), head_dim]
                * :attr:`sin_cache`: Sine Cache Tensor [1, (q_batch + k_batch), 1(32), head_dim]
                * :attr:`trans_mat`: Transformation Matrix Tensor  [1, (q_batch + k_batch), 32, 32]
                * :attr:`compute_kernel_config`: Optional[DeviceComputeKernelConfig] = None
        )doc",
        ttnn::pybind_arguments_t {
            py::arg("q_input_tensor"),
            py::arg("k_input_tensor"),
            py::arg("cos_cache"),
            py::arg("sin_cache"),
            py::arg("trans_mat"),
            py::kw_only(),
            py::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::transformer
