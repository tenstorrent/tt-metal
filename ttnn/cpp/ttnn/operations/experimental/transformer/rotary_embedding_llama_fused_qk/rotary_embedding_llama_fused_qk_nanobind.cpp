// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_fused_qk_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "rotary_embedding_llama_fused_qk.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_rotary_embedding_llama_fused_qk(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::rotary_embedding_llama_fused_qk,
        R"doc(
            Applies rotary embeddings to both `q_input_tensor` and `k_input_tensor` in parallel using precomputed sine and cosine values. This function is optimized for parallel execution, and both input tensors should share the same batch size and head dimensions.

            Args:
                q_input_tensor (ttnn.Tensor): The Q input tensor, with shape [1, batch, num_heads, head_dim].
                k_input_tensor (ttnn.Tensor): The K input tensor, with shape [1, batch, num_kv_heads, head_dim].
                cos_cache (ttnn.Tensor): Precomputed cosine values, with shape [1, 2 * batch, 32, head_dim].
                sin_cache (ttnn.Tensor): Precomputed sine values, with shape [1, 2 * batch, 32, head_dim].
                trans_mat (ttnn.Tensor): Transformation matrix tensor, with shape [1, 2 * batch, 32, 32].

            Keyword args:
                compute_kernel_config (DeviceComputeKernelConfig, optional): Optional configuration for the device compute kernel. Defaults to None.

            Returns:
                ttnn.Tensor, ttnn.Tensor: q and k output tensors with rotary embeddings applied.

        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("q_input_tensor"),
            nb::arg("k_input_tensor"),
            nb::arg("cos_cache"),
            nb::arg("sin_cache"),
            nb::arg("trans_mat"),
            nb::kw_only(),
            nb::arg("compute_kernel_config") = nb::none()});
}

}  // namespace ttnn::operations::experimental::transformer
