// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_cache_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <fmt/format.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "kv_cache.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::kv_cache {

namespace {

template <typename kv_cache_operation_t>
void bind_fill_cache_for_user_(nb::module_& mod, const kv_cache_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Populates the :attr:`cache` tensor in-place with values sourced from :attr:`input` at :attr:`batch_index`.


        Args:
            cache (ttnn.Tensor): the cache tensor to be written to.
            input_tensor (ttnn.Tensor): the input tensor to be written to the cache.
            batch_index (int): the index into the cache tensor.


        Returns:
            ttnn.Tensor: the output tensor.


        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const kv_cache_operation_t& self,
               const ttnn::Tensor& cache,
               const ttnn::Tensor& input,
               const uint32_t batch_index) -> ttnn::Tensor { return self(cache, input, batch_index); },
            nb::arg("cache"),
            nb::arg("input"),
            nb::arg("batch_index")});
}

template <typename kv_cache_operation_t>
void bind_update_cache_for_token_(nb::module_& mod, const kv_cache_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Updates the :attr:`cache` tensor in-place with values from :attr:`input` at :attr:`update_index` and :attr:`batch_offset`.


        Args:
            cache (ttnn.Tensor): the cache tensor to be written to.
            token (ttnn.Tensor): the token tensor to be written to the cache.
            update_index (int): the index into the cache tensor.
            batch_offset (int): the batch_offset into the cache tensor.


        Returns:
            ttnn.Tensor: the output tensor.


        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const kv_cache_operation_t& self,
               const ttnn::Tensor& cache,
               const ttnn::Tensor& input,
               const uint32_t update_index,
               const uint32_t batch_offset,
               std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) -> ttnn::Tensor {
                return self(cache, input, update_index, batch_offset, compute_kernel_config);
            },
            nb::arg("cache"),
            nb::arg("input"),
            nb::arg("update_index"),
            nb::arg("batch_offset") = 0,
            nb::arg("compute_kernel_config") = nb::none()});
}

template <typename update_cache_operation_t>
void bind_update_cache(nb::module_& mod, const update_cache_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
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
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const update_cache_operation_t& self,
               const ttnn::Tensor& cache,
               const ttnn::Tensor& input,
               const uint32_t update_idx,
               const uint32_t batch_offset,
               std::optional<const DeviceComputeKernelConfig> compute_kernel_config) -> ttnn::Tensor {
                return self(cache, input, update_idx, batch_offset, compute_kernel_config);
            },
            nb::arg("cache"),
            nb::arg("input"),
            nb::arg("update_idx"),
            nb::kw_only(),
            nb::arg("batch_offset") = 0,
            nb::arg("compute_kernel_config") = nb::none()});
}

template <typename update_cache_operation_t>
void bind_fill_cache(nb::module_& mod, const update_cache_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
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
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const update_cache_operation_t& self,
               const ttnn::Tensor& cache_tensor,
               const ttnn::Tensor& input_tensor,
               const uint32_t batch_idx) -> ttnn::Tensor { return self(cache_tensor, input_tensor, batch_idx); },
            nb::arg("cache_tensor"),
            nb::arg("input_tensor"),
            nb::arg("batch_idx")});
}

}  // namespace

void bind_kv_cache(nb::module_& mod) {
    bind_fill_cache_for_user_(mod, ttnn::fill_cache_for_user_);
    bind_update_cache_for_token_(mod, ttnn::update_cache_for_token_);
    bind_update_cache(mod, ttnn::update_cache);
    bind_fill_cache(mod, ttnn::fill_cache);
}

}  // namespace ttnn::operations::kv_cache
