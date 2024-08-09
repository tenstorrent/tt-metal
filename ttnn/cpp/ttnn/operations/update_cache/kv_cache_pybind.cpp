// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kv_cache.hpp"
#include "kv_cache_pybind.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace kv_cache {

namespace detail {

template <typename kv_cache_operation_t>
void bind_fill_cache_for_user_(py::module& module, const kv_cache_operation_t& operation) {
    auto doc = fmt::format(
        R"doc({0}(cache: ttnn.Tensor, input: ttnn.Tensor, batch_index: int) -> ttnn.Tensor

        Populates the :attr:`cache` tensor in-place with values sourced from :attr:`input` at :attr:`batch_index`.

        Args:
            * :attr:`cache` (ttnn.Tensor): The cache tensor to be written to.
            * :attr:`input` (ttnn.Tensor): The input tensor to be written to the cache.
            * :attr:`batch_index` (int): The index into the cache tensor.

    )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const kv_cache_operation_t& self,
               const ttnn::Tensor& cache,
               const ttnn::Tensor& input,
               const uint32_t batch_index) -> ttnn::Tensor {
                return self(cache, input, batch_index);
            },
            py::arg("cache"), py::arg("input"), py::arg("batch_index")});
}


template <typename kv_cache_operation_t>
void bind_update_cache_for_token_(py::module& module, const kv_cache_operation_t& operation) {
    auto doc = fmt::format(
        R"doc({0}(cache: ttnn.Tensor, input: ttnn.Tensor, update_index: int, batch_offset: int) -> ttnn.Tensor

        Updates the :attr:`cache` tensor in-place with values from :attr:`input` at :attr:`update_index` and :attr:`batch_offset`.

        Args:
            * :attr:`cache` (ttnn.Tensor): The cache tensor to be written to.
            * :attr:`token` (ttnn.Tensor): The token tensor to be written to the cache.
            * :attr:`update_index` (int): The index into the cache tensor.
            * :attr:`batch_offset` (int): The batch_offset into the cache tensor.

    )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const kv_cache_operation_t& self,
               const ttnn::Tensor& cache,
               const ttnn::Tensor& input,
               const uint32_t update_index,
               const uint32_t batch_offset) -> ttnn::Tensor {
                return self(cache, input, update_index, batch_offset);
            },
            py::arg("cache"), py::arg("input"), py::arg("update_index"), py::arg("batch_offset") = 0});
}

template <typename update_cache_operation_t>
void bind_update_cache(pybind11::module& module, const update_cache_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(update_cache(cache: ttnn.Tensor, input: ttnn.Tensor, update_idx: int, batch_offset: int, *, compute_kernel_config : Optional[DeviceComputeKernelConfig] = None, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

        Updates the cache tensor in place with the values from input at the specified update_idx. When cache has batch less than 32, input is assumed to have batch padded to 32 and [batch_offset:batch_offset+batch] from dim[-2] of input is used to update the cache.

        Args:
            * :attr:`cache` (ttnn.Tensor): The cache tensor to be written to.
            * :attr:`input` (ttnn.Tensor): The token tensor to be written to the cache.
            * :attr:`update_index` (int): The index into the cache tensor.
            * :attr:`batch_offset` (int): The batch_offset into the cache tensor. Default = 0 .

        Keyword Args:
            * :attr:`compute_kernel_config` Optional[DeviceComputeKernelConfig]

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.update_cache(tensor1, tensor2, update_index)

    )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const update_cache_operation_t& self,
               const ttnn::Tensor& cache,
               const ttnn::Tensor& input,
               const uint32_t update_idx,
               const uint32_t batch_offset,
               std::optional<const DeviceComputeKernelConfig> compute_kernel_config) -> ttnn::Tensor {
                return self(cache, input, update_idx, batch_offset, compute_kernel_config);
            },
            py::arg("cache"),
            py::arg("input"),
            py::arg("update_idx"),
            py::kw_only(),
            py::arg("batch_offset") = 0,
            py::arg("compute_kernel_config") = std::nullopt});
}

template <typename update_cache_operation_t>
void bind_fill_cache(pybind11::module& module, const update_cache_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(fill_cache(cache_tensor: ttnn.Tensor, input_tensor: ttnn.Tensor, batch_idx: int, *, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

        Fills the cache tensor in place with the values from input at the specified batch_idx.

        Args:
            * :attr:`cache_tensor` (ttnn.Tensor): The cache tensor to be written to.
            * :attr:`input_tensor` (ttnn.Tensor): The token tensor to be written to the cache.
            * :attr:`batch_idx` (int): The index into the cache tensor.

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.update_cache(tensor1, tensor2, batch_idx)

    )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const update_cache_operation_t& self,
               const ttnn::Tensor& cache_tensor,
               const ttnn::Tensor& input_tensor,
               const uint32_t batch_idx) -> ttnn::Tensor {
                return self(cache_tensor, input_tensor, batch_idx);
            },
            py::arg("cache_tensor"),
            py::arg("input_tensor"),
            py::arg("batch_idx")});
}

}  // namespace detail

void py_bind_kv_cache(py::module& module) {
    detail::bind_fill_cache_for_user_(module, ttnn::fill_cache_for_user_);
    detail::bind_update_cache_for_token_(module, ttnn::update_cache_for_token_);
    detail::bind_update_cache(module, ttnn::update_cache);
    detail::bind_fill_cache(module, ttnn::fill_cache);
}

}  // namespace kv_cache
}  // namespace operations
}  // namespace ttnn
