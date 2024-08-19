// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/kv_cache.hpp"
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

}  // namespace detail


void py_module(py::module& module) {
    detail::bind_fill_cache_for_user_(module, ttnn::kv_cache::fill_cache_for_user_);
    detail::bind_update_cache_for_token_(module, ttnn::kv_cache::update_cache_for_token_);
}

}  // namespace kv_cache
}  // namespace operations
}  // namespace ttnn
