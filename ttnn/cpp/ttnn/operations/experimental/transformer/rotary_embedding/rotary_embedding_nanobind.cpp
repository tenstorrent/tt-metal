// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "rotary_embedding.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_rotary_embedding(nb::module_& mod) {
    ttnn::bind_function<"rotary_embedding", "ttnn.experimental.">(
        mod,
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
        &ttnn::experimental::rotary_embedding,
        nb::arg("input_tensor"),
        nb::arg("cos_cache"),
        nb::arg("sin_cache"),
        nb::arg("token_index") = nb::none(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

void bind_rotary_embedding_to_cache(nb::module_& mod) {
    ttnn::bind_function<"rotary_embedding_to_cache", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused rotary_embedding + write-to-cache. Applies RoPE to input_tensor and writes
        the rotated tiles directly into output_cache at the tile offset corresponding to
        update_idx. Eliminates the intermediate L1 allocation and separate fill_cache dispatch.

        Requires:
            - input_tensor shape [1, H, seq, D]
            - output_cache shape [1, H, cache_seq, D] (must match input head_dim)
            - update_idx % TILE_HEIGHT (32) == 0
            - input.dtype == output_cache.dtype

        Args:
            input_tensor (ttnn.Tensor): the tensor to rotate.
            cos_cache (ttnn.Tensor): the Cosine Cache tensor.
            sin_cache (ttnn.Tensor): the Sine Cache tensor.
            output_cache (ttnn.Tensor): pre-allocated destination cache tensor (written in-place).
            update_idx (int): sequence index in the cache where rotated tiles will be written.

        Keyword args:
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.

        Returns:
            ttnn.Tensor: the output_cache tensor (same as input `output_cache`, modified in place).
        )doc",
        &ttnn::experimental::rotary_embedding_to_cache,
        nb::arg("input_tensor"),
        nb::arg("cos_cache"),
        nb::arg("sin_cache"),
        nb::arg("output_cache"),
        nb::arg("update_idx"),
        nb::kw_only(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer
