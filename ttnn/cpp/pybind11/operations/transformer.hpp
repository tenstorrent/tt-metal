// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/transformer.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace transformer {

void py_module(py::module& module) {

    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::rotary_embedding,
        R"doc(rotary_embedding(input_tensor: ttnn.Tensor, cos_cache: ttnn.Tensor, sin_cache: ttnn.Tensor, token_index: int, memory_config: MemoryConfig, compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor

            Applies the rotary embedding to the input_tensor tensor using the cos_cache and sin_cache tensors.

            When token_idx is passed, this assumes input is transposed to [seq_len, 1, B, head_dim], and seq_len is 1.

            Args:
                * :attr:`input_tensor`: Input Tensor
                * :attr:`cos_cache`: Cosine Cache Tensor
                * :attr:`sin_cache`: Sine Cache Tensor
                * :attr:`token_index`: Token Index
                * :attr:`memory_config`: Memory Config of the output tensor
                * :attr:`compute_kernel_config`: Optional[DeviceComputeKernelConfig] = None
        )doc",
        ttnn::pybind_arguments_t{py::arg("input_tensor"),py::arg("cos_cache"),py::arg("sin_cache"),py::arg("token_index"), py::arg("memory_config") = std::nullopt, py::arg("compute_kernel_config") = std::nullopt});

    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::concatenate_heads,
        R"doc(concatenate_heads(input_tensor: ttnn.Tensor, *, memory_config: Optional[MemoryConfig] = None) -> ttnn.Tensor

            Takes in a tensor of shape ``[batch_size, num_heads, sequence_size, head_size]``, concatenates heads back along the width dimension and returns the tensor of shape ``[batch_size, sequence_size, num_heads * head_size]``

            Args:
                * :attr:`input_tensor`: Input Tensor

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor, if None then it gets set to input_tensor.memory_config()
        )doc",
        ttnn::pybind_arguments_t{py::arg("input_tensor"), py::kw_only(), py::arg("memory_config") = std::nullopt});

    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::attention_softmax,
        R"doc(attention_softmax(tensor: ttnn.Tensor, *, head_size: Optional[int] = None, attention_mask: Optional[ttnn.Tensor] = None, program_config: Optional[SoftmaxProgramConfig] = SoftmaxDefaultProgramConfig(), causal_mask: bool = False,  memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Divides :attr:`tensor` by the square root of :attr:`head_size`, adds :attr:`attention_mask` (optionally) and computes softmax.

            Args:
                * :attr:`tensor`: Input Tensor

            Keyword Args:
                * :attr:`head_size`: Number of heads
                * :attr:`attention_mask`: Attention Mask
                * :attr:`program_config`: Program Config of the output tensor
                * :attr:`causal_mask`: the attention mask is causal
                * :attr:`memory_config`: Memory Config of the output tensor, defaults to input_tensor.memory_config()
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("tensor"),
            py::kw_only(),
            py::arg("head_size") = std::nullopt,
            py::arg("attention_mask") = std::nullopt,
            py::arg("program_config").noconvert() =
                ttnn::operations::normalization::SoftmaxDefaultProgramConfig{},
            py::arg("causal_mask") = false,
            py::arg("memory_config") = std::nullopt});

    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::attention_softmax_,
        R"doc(attention_softmax_(tensor: ttnn.Tensor, *, head_size: Optional[int] = None, attention_mask: Optional[ttnn.Tensor] = None, program_config: Optional[SoftmaxProgramConfig] = SoftmaxDefaultProgramConfig(), causal_mask: bool = False,  memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            In-Place divides :attr:`tensor` by the square root of :attr:`head_size`, adds :attr:`attention_mask` (optionally) and computes softmax.

            Args:
                * :attr:`tensor`: Input Tensor

            Keyword Args:
                * :attr:`head_size`: Number of heads
                * :attr:`attention_mask`: Attention Mask
                * :attr:`program_config`: Program Config of the output tensor
                * :attr:`causal_mask`: the attention mask is causal
                * :attr:`memory_config`: Memory Config of the output tensor, defaults to input_tensor.memory_config()
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("tensor"),
            py::kw_only(),
            py::arg("head_size") = std::nullopt,
            py::arg("attention_mask") = std::nullopt,
            py::arg("program_config").noconvert() =
                ttnn::operations::normalization::SoftmaxDefaultProgramConfig{},
            py::arg("causal_mask") = false,
            py::arg("memory_config") = std::nullopt});

    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::split_query_key_value_and_split_heads,
        R"doc(split_query_key_value_and_split_heads(input_tensor: ttnn.Tensor, kv_input_tensor: Optional[ttnn.Tensor] = None, *, num_heads: int, num_kv_heads: Optional[int] = None, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]

            Splits :attr:`input_tensor` of shape ``[batch_size, sequence_size, 3 * hidden_size]`` into 3 tensors (Query, Key, Value) of shape ``[batch_size, sequence_size, hidden_size]``.
            Then, reshapes and permutes the output tensors, to make them ready for computing attention scores.

            If :attr:`kv_input_tensor` is passed in, then :attr:`input_tensor` of shape ``[batch_size, sequence_size, hidden_size]`` is only used for Query,
            and :attr:`kv_input_tensor` of shape ``[batch_size, sequence_size, 2 * hidden_size]`` is used for Key and Value.

            For the sharded implementation, the input query, key and value are expected to be concatenated such that the heads are interleaved (q1 k1 v1...qn kn vn).

            Equivalent pytorch code:

            .. code-block:: python

                if kv_input_tensor is not None:
                    input_tensor = torch.cat([input_tensor, kv_input_tensor], dim=-1)

                if num_kv_heads is None:
                    num_kv_heads = num_heads

                batch_size, sequence_size, hidden_size = input_tensor.shape
                # Subtract head sizes for key and value
                head_size = (hidden_size) // (num_heads + num_kv_heads * 2)
                tensor = torch.reshape(input_tensor, (batch_size, sequence_size, num_heads + num_kv_heads * 2, head_size))
                query, key, value = (
                    tensor[..., :num_heads, :],
                    tensor[..., num_heads:num_heads + num_kv_heads, :],
                    tensor[..., num_heads + num_kv_heads:, :],
                )

                query = torch.reshape(query, (batch_size, sequence_size, num_heads, head_size))
                key = torch.reshape(key, (batch_size, sequence_size, num_kv_heads, head_size))
                value = torch.reshape(value, (batch_size, sequence_size, num_kv_heads, head_size))

                query = torch.permute(query, (0, 2, 1, 3)).contiguous().clone()
                key = torch.permute(key, (0, 2, 1, 3)).contiguous().clone()
                value = torch.permute(value, (0, 2, 1, 3)).contiguous().clone()
                if transpose_key:
                    key = torch.permute(key, (0, 1, 3, 2)).contiguous().clone()

                return query, key, value

            Args:
                * :attr:`input_tensor`: Input Tensor for Query, Key and Value. If :attr:`kv_input_tensor` is not None, then :attr:`input_tensor` is only used for Query.
                * :attr:`kv_input_tensor`: Input Tensor for Key and Value. If passed in, :attr:`input_tensor` has to be used only for Query.
                * :attr:`num_heads`: num heads to split into
                * :attr:`num_kv_heads`: num heads of Key and num heads of Value. If not passed in, then :attr:`num_kv_heads` is set to :attr:`num_heads`
                * :attr:`transpose_key`: Whether to transpose the Key tensor on the last two dimensions
                * :attr:`memory_config`: Memory Config of the output tensor
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::transformer::split_query_key_value_and_split_heads) &self,
               const Tensor &input_tensor,
               const std::optional<Tensor> &kv_input_tensor,
               const uint32_t num_heads,
               const std::optional<uint32_t> num_kv_heads,
               const bool transpose_key,
               const std::optional<MemoryConfig> &memory_config)
                -> std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> {
                return self(input_tensor, kv_input_tensor, num_heads, num_kv_heads, transpose_key, memory_config);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("kv_input_tensor") = std::nullopt,
            py::kw_only(),
            py::arg("num_heads"),
            py::arg("num_kv_heads") = std::nullopt,
            py::arg("transpose_key") = true,
            py::arg("memory_config") = std::nullopt});
}
}  // namespace transformer
}  // namespace operations
}  // namespace ttnn
