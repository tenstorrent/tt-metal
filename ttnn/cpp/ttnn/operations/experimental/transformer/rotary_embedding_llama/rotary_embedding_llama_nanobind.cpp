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

void bind_rotary_embedding_llama_transpose_enum(nb::module_& mod) {
    nb::enum_<ttnn::experimental::prim::RotaryEmbeddingTranspose>(mod, "RotaryEmbeddingTranspose")
        .value("NONE", ttnn::experimental::prim::RotaryEmbeddingTranspose::NONE)
        .value("HC", ttnn::experimental::prim::RotaryEmbeddingTranspose::HC);
}

void bind_rotary_embedding_llama(nb::module_& mod) {
    bind_rotary_embedding_llama_transpose_enum(mod);

    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::rotary_embedding_llama,
        R"doc(
            Applies the rotary embedding to the input_tensor tensor using the cos_cache and sin_cache tensors.

            When token_idx is passed, this assumes input is transposed to [seq_len, 1, B, head_dim], and seq_len is 1.

            `cos_cache` and `sin_cache` must be of shape [1, n_heads, seq_len, head_dim] or [1, 1, seq_len, head_dim].
            If shape[1] is 1 then the sin/cos will be broadcasted across heads.

            When ``input_transpose`` is set to ``RotaryEmbeddingTranspose.HC``, the operation will internally
            transpose dims 1 and 2 (H and C in NCHW notation) of the input tensor before applying rotary
            embeddings, and transpose the output back. This is useful for fusing explicit transpose calls
            around the rotary embedding operation (e.g. in the Deepseek MLA decode path).

            Args:
                * :attr:`input_tensor`: Input Tensor
                * :attr:`cos_cache`: Cosine Cache Tensor
                * :attr:`sin_cache`: Sine Cache Tensor
                * :attr:`trans_mat`: Transformation Matrix Tensor
                * :attr:`is_decode_mode`: Specify mode of operation
                * :attr:`memory_config`: Memory Config of the output tensor = DEFAULT_OUTPUT_MEMORY_CONFIG
                * :attr:`compute_kernel_config`: Optional[DeviceComputeKernelConfig] = None
                * :attr:`input_transpose`: Optional transpose to apply to input before (and after) embedding.
                  Use ``RotaryEmbeddingTranspose.HC`` to transpose dims 1 and 2. Default: ``RotaryEmbeddingTranspose.NONE``
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("cos_cache"),
            nb::arg("sin_cache"),
            nb::arg("trans_mat"),
            nb::kw_only(),
            nb::arg("is_decode_mode") = false,
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("input_transpose") = ttnn::experimental::prim::RotaryEmbeddingTranspose::NONE});
}

}  // namespace ttnn::operations::experimental::transformer
