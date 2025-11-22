// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "rotary_embedding_llama.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_rotary_embedding_llama(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
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
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("cos_cache"),
            nb::arg("sin_cache"),
            nb::arg("trans_mat"),
            nb::kw_only(),
            nb::arg("is_decode_mode") = false,
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}

}  // namespace ttnn::operations::experimental::transformer
