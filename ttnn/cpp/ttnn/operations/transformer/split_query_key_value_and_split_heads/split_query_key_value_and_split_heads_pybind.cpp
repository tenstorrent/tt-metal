// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_query_key_value_and_split_heads_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "split_query_key_value_and_split_heads.hpp"

namespace ttnn::operations::transformer {

void py_bind_split_query_key_value_and_split_heads(pybind11::module& module) {
    namespace py = pybind11;
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

}  // namespace ttnn::operations::transformer
