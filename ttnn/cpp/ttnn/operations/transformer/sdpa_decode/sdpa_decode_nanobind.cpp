// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "sdpa_decode.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::transformer {

void bind_sdpa_decode(nb::module_& mod) {
    auto doc =
        R"doc(
        A version of scaled dot product attention specifically for decode.
        The implementation is Flash-Decode and it currently only supports MQA on decoding single token.


        Accepts a `SDPAMultiCoreProgramConfig` which specifies the grid size and chunk tiles in the K/V/Mask sequence lengths (Q chunk tiles is not used). The op parallelizes over `b` and K/V/Mask's `s` dimension.


        Args:
            input_tensor_q (ttnn.Tensor): the input tensor [1 x b x nh x dh]
            input_tensor_k (ttnn.Tensor): the input tensor [b x nkv x   s x dh]
            input_tensor_v (ttnn.Tensor): the input tensor [b x nkv x   s x dh]


        Keyword args:
            is_causal (bool): whether the attention is is_causal. Defaults to `True`.
            attn_mask (ttnn.Tensor, optional): the input tensor [b x 1 x s x s]. Defaults to `None`.
            cur_pos (List of int, optional): list of integers of length b. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            cur_pos_tensor (ttnn.Tensor, optional): [b] tensor of integers of length b. Defaults to `None`.
            scale (float, optional): Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.
            sliding_window_size (int, optional): The size of sliding window for sliding window attention. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor [1 x b x pnh x dh].


        "Accepts a `SDPAMultiCoreProgramConfig` which specifies the grid size and chunk tiles in the K/V/Mask sequence lengths (Q chunk tiles is not used). The op parallelizes over `b` and K/V/Mask's `s` dimension."
        "If a position is given as (-1), compute for the corresponding index in the batch is skipped."
        )doc";

    using OperationType = decltype(ttnn::transformer::scaled_dot_product_attention_decode);
    ttnn::bind_registered_operation(
        mod,
        ttnn::transformer::scaled_dot_product_attention_decode,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const ttnn::Tensor& input_tensor_v,
               const bool is_causal,
               const std::optional<const Tensor>& attn_mask,
               const std::vector<uint32_t>& cur_pos,
               const std::optional<const Tensor>& cur_pos_tensor,
               const std::optional<const Tensor>& attention_sink,
               std::optional<float> scale,
               std::optional<uint32_t> sliding_window_size,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<SDPAProgramConfig> program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
                return self(
                    input_tensor_q,
                    input_tensor_k,
                    input_tensor_v,
                    is_causal,
                    attn_mask,
                    cur_pos,
                    cur_pos_tensor,
                    attention_sink,
                    scale,
                    sliding_window_size,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
            nb::arg("input_tensor_q").noconvert(),
            nb::arg("input_tensor_k").noconvert(),
            nb::arg("input_tensor_v").noconvert(),
            nb::kw_only(),
            nb::arg("is_causal").noconvert() = true,
            nb::arg("attn_mask").noconvert() = nb::none(),
            nb::arg("cur_pos").noconvert() = std::vector<uint32_t>(),
            nb::arg("cur_pos_tensor").noconvert() = nb::none(),
            nb::arg("attention_sink").noconvert() = nb::none(),
            nb::arg("scale").noconvert() = nb::none(),
            nb::arg("sliding_window_size").noconvert() = nb::none(),
            nb::arg("memory_config").noconvert() = nb::none(),
            nb::arg("program_config").noconvert() = nb::none(),
            nb::arg("compute_kernel_config").noconvert() = nb::none()});

    using PagedOperationType = decltype(ttnn::transformer::paged_scaled_dot_product_attention_decode);
    ttnn::bind_registered_operation(
        mod,
        ttnn::transformer::paged_scaled_dot_product_attention_decode,
        doc,
        ttnn::nanobind_overload_t{
            [](const PagedOperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const ttnn::Tensor& input_tensor_v,
               const ttnn::Tensor& page_table_tensor,
               const bool is_causal,
               const std::optional<const Tensor>& attn_mask,
               const std::optional<const Tensor>& cur_pos_tensor,
               const std::optional<const Tensor>& attention_sink,
               std::optional<float> scale,
               std::optional<uint32_t> sliding_window_size,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<SDPAProgramConfig> program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
                return self(
                    input_tensor_q,
                    input_tensor_k,
                    input_tensor_v,
                    page_table_tensor,
                    is_causal,
                    attn_mask,
                    cur_pos_tensor,
                    attention_sink,
                    scale,
                    sliding_window_size,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
            nb::arg("input_tensor_q").noconvert(),
            nb::arg("input_tensor_k").noconvert(),
            nb::arg("input_tensor_v").noconvert(),
            nb::arg("page_table_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("is_causal").noconvert() = true,
            nb::arg("attn_mask").noconvert() = nb::none(),
            nb::arg("cur_pos_tensor").noconvert() = nb::none(),
            nb::arg("attention_sink").noconvert() = nb::none(),
            nb::arg("scale").noconvert() = nb::none(),
            nb::arg("sliding_window_size").noconvert() = nb::none(),
            nb::arg("memory_config").noconvert() = nb::none(),
            nb::arg("program_config").noconvert() = nb::none(),
            nb::arg("compute_kernel_config").noconvert() = nb::none()});

    using MLAOperationType = decltype(ttnn::transformer::flash_multi_latent_attention_decode);
    ttnn::bind_registered_operation(
        mod,
        ttnn::transformer::flash_multi_latent_attention_decode,
        doc,
        ttnn::nanobind_overload_t{
            [](const MLAOperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const uint32_t head_dim_v,
               const bool is_causal,
               const std::optional<const Tensor>& attn_mask,
               const std::vector<uint32_t>& cur_pos,
               const std::optional<const Tensor>& cur_pos_tensor,
               const std::optional<const Tensor>& attention_sink,
               std::optional<float> scale,
               std::optional<uint32_t> sliding_window_size,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<SDPAProgramConfig> program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
                return self(
                    input_tensor_q,
                    input_tensor_k,
                    head_dim_v,
                    is_causal,
                    attn_mask,
                    cur_pos,
                    cur_pos_tensor,
                    attention_sink,
                    scale,
                    sliding_window_size,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
            nb::arg("input_tensor_q").noconvert(),
            nb::arg("input_tensor_k").noconvert(),
            nb::arg("head_dim_v").noconvert(),
            nb::kw_only(),
            nb::arg("is_causal").noconvert() = true,
            nb::arg("attn_mask").noconvert() = nb::none(),
            nb::arg("cur_pos").noconvert() = std::vector<uint32_t>(),
            nb::arg("cur_pos_tensor").noconvert() = nb::none(),
            nb::arg("attention_sink").noconvert() = nb::none(),
            nb::arg("scale").noconvert() = nb::none(),
            nb::arg("sliding_window_size").noconvert() = nb::none(),
            nb::arg("memory_config").noconvert() = nb::none(),
            nb::arg("program_config").noconvert() = nb::none(),
            nb::arg("compute_kernel_config").noconvert() = nb::none()});

    using PagedMLAOperationType = decltype(ttnn::transformer::paged_flash_multi_latent_attention_decode);
    ttnn::bind_registered_operation(
        mod,
        ttnn::transformer::paged_flash_multi_latent_attention_decode,
        doc,
        ttnn::nanobind_overload_t{
            [](const PagedMLAOperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const uint32_t head_dim_v,
               const ttnn::Tensor& page_table_tensor,
               const bool is_causal,
               const std::optional<const Tensor>& attn_mask,
               const std::optional<const Tensor>& cur_pos_tensor,
               const std::optional<const Tensor>& attention_sink,
               std::optional<float> scale,
               std::optional<uint32_t> sliding_window_size,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<SDPAProgramConfig> program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
                return self(
                    input_tensor_q,
                    input_tensor_k,
                    head_dim_v,
                    page_table_tensor,
                    is_causal,
                    attn_mask,
                    cur_pos_tensor,
                    attention_sink,
                    scale,
                    sliding_window_size,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
            nb::arg("input_tensor_q").noconvert(),
            nb::arg("input_tensor_k").noconvert(),
            nb::arg("head_dim_v").noconvert(),
            nb::arg("page_table_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("is_causal").noconvert() = true,
            nb::arg("attn_mask").noconvert() = nb::none(),
            nb::arg("cur_pos_tensor").noconvert() = nb::none(),
            nb::arg("attention_sink").noconvert() = nb::none(),
            nb::arg("scale").noconvert() = nb::none(),
            nb::arg("sliding_window_size").noconvert() = nb::none(),
            nb::arg("memory_config").noconvert() = nb::none(),
            nb::arg("program_config").noconvert() = nb::none(),
            nb::arg("compute_kernel_config").noconvert() = nb::none()});
}
}  // namespace ttnn::operations::transformer
