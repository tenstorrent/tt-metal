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
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::kv_cache {

namespace {

void bind_fill_cache_for_user_(nb::module_& mod) {
    auto doc = R"doc(
        Populates the :attr:`cache` tensor in-place with values sourced from :attr:`input` at :attr:`batch_index`.


        Args:
            cache (ttnn.Tensor): the cache tensor to be written to.
            input_tensor (ttnn.Tensor): the input tensor to be written to the cache.
            batch_index (int): the index into the cache tensor.


        Returns:
            ttnn.Tensor: the output tensor.


        )doc";

    ttnn::bind_function<"fill_cache_for_user_">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::fill_cache_for_user_,
            nb::arg("cache"),
            nb::arg("input"),
            nb::arg("batch_index")));
}

void bind_update_cache_for_token_(nb::module_& mod) {
    auto doc = R"doc(
        Updates the :attr:`cache` tensor in-place with values from :attr:`input` at :attr:`update_index` and :attr:`batch_offset`.


        Args:
            cache (ttnn.Tensor): the cache tensor to be written to.
            token (ttnn.Tensor): the token tensor to be written to the cache.
            update_index (int): the index into the cache tensor.
            batch_offset (int): the batch_offset into the cache tensor.


        Returns:
            ttnn.Tensor: the output tensor.


        )doc";

    ttnn::bind_function<"update_cache_for_token_">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::update_cache_for_token_,
            nb::arg("cache"),
            nb::arg("input"),
            nb::arg("update_index"),
            nb::arg("batch_offset") = 0,
            nb::arg("compute_kernel_config") = nb::none()));
}

void bind_update_cache(nb::module_& mod) {
    auto doc = R"doc(
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

    )doc";

    ttnn::bind_function<"update_cache">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::update_cache,
            nb::arg("cache"),
            nb::arg("input"),
            nb::arg("update_idx"),
            nb::kw_only(),
            nb::arg("batch_offset") = 0,
            nb::arg("compute_kernel_config") = nb::none()));
}

void bind_fill_cache(nb::module_& mod) {
    auto doc = R"doc(
        Fills the cache tensor in place with the values from input at the specified batch_idx.

        Args:
            * :attr:`cache_tensor` (ttnn.Tensor): The cache tensor to be written to.
            * :attr:`input_tensor` (ttnn.Tensor): The token tensor to be written to the cache.
            * :attr:`batch_idx` (int): The index into the cache tensor.

        Example:
            >>> tensor1 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.update_cache(tensor1, tensor2, batch_idx)

    )doc";

    ttnn::bind_function<"fill_cache">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::fill_cache,
            nb::arg("cache_tensor"),
            nb::arg("input_tensor"),
            nb::arg("batch_idx")));
}

}  // namespace

void bind_kv_cache(nb::module_& mod) {
    bind_fill_cache_for_user_(mod);
    bind_update_cache_for_token_(mod);
    bind_update_cache(mod);
    bind_fill_cache(mod);
}

}  // namespace ttnn::operations::kv_cache
