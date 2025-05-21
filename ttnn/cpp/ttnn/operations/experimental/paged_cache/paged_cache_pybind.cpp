// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/paged_cache/paged_cache.hpp"
#include "ttnn/operations/experimental/paged_cache/paged_cache_pybind.hpp"

namespace ttnn::operations::experimental::paged_cache::detail {

namespace py = pybind11;

void bind_experimental_paged_cache_operations(py::module& module) {
    auto paged_update_cache_doc =
        R"doc(
         Paged update cache operation. This operation expects the following inputs: cache_tensor of shape [B, 1, kv_len, head_dim] and input_tensor of shape [1, B, 1[32], head_dim] where input_tensor is height sharded on B cores. update_idxs will specify for each batch element which token to update in the cache.
        )doc";

    using PagedUpdateCacheType = decltype(ttnn::experimental::paged_update_cache);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::paged_update_cache,
        paged_update_cache_doc,
        ttnn::pybind_overload_t{
            [](const PagedUpdateCacheType& self,
               const ttnn::Tensor& cache_tensor,
               const ttnn::Tensor& input_tensor,
               const std::vector<uint32_t>& update_idxs,
               const std::optional<const ttnn::Tensor>& update_idxs_tensor,
               const std::optional<bool> share_cache,
               const std::optional<const ttnn::Tensor>& page_table,
               const uint32_t batch_offset,
               std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
                return self(
                    cache_tensor,
                    input_tensor,
                    update_idxs,
                    update_idxs_tensor,
                    share_cache,
                    page_table,
                    batch_offset,
                    compute_kernel_config);
            },
            py::arg("cache_tensor").noconvert(),
            py::arg("input_tensor").noconvert(),
            py::kw_only(),
            py::arg("update_idxs").noconvert() = std::vector<uint32_t>(),
            py::arg("update_idxs_tensor").noconvert() = std::nullopt,
            py::arg("share_cache").noconvert() = std::nullopt,
            py::arg("page_table").noconvert() = std::nullopt,
            py::arg("batch_offset") = 0,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
        });

    auto paged_fused_update_cache_doc =
        R"doc(

            Updates the cache tensors `cache_tensor1` and `cache_tensor2` in parallel with values derived from the corresponding input tensors. This function supports fine-grained updates using specified index lists or tensors.

            Positional Arguments:
                cache_tensor1 (ttnn.Tensor): The first cache tensor to update.
                input_tensor1 (ttnn.Tensor): The input tensor corresponding to `cache_tensor1`.
                cache_tensor2 (ttnn.Tensor): The second cache tensor to update.
                input_tensor2 (ttnn.Tensor): The input tensor corresponding to `cache_tensor2`.

            Keyword Args:
                update_idxs (List[int]): A list of indices specifying the cache update positions. Defaults to an empty list.
                update_idxs_tensor (ttnn.Tensor, optional): A tensor specifying update indices. Defaults to None.
                share_cache (bool, optional): Whether the cache tensors share memory regions. Defaults to None.
                page_table (ttnn.Tensor, optional): The page table for managing memory regions during updates. Defaults to None.
                batch_offset (int): Offset for batching updates. Defaults to 0.
                compute_kernel_config (DeviceComputeKernelConfig, Optional): Optional configuration for the device compute kernel. Defaults to None.

            Returns:
                ttnn.Tensor, ttnn.Tensor: Tensors representing the updated cache states.
        )doc";

    using PagedFusedUpdateCacheType = decltype(ttnn::experimental::paged_fused_update_cache);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::paged_fused_update_cache,
        paged_fused_update_cache_doc,
        ttnn::pybind_overload_t{
            [](const PagedFusedUpdateCacheType& self,
               const ttnn::Tensor& cache_tensor1,
               const ttnn::Tensor& input_tensor1,
               const ttnn::Tensor& cache_tensor2,
               const ttnn::Tensor& input_tensor2,
               const std::vector<uint32_t>& update_idxs,
               const std::optional<const ttnn::Tensor>& update_idxs_tensor,
               const std::optional<bool> share_cache,
               const std::optional<const ttnn::Tensor>& page_table,
               const uint32_t batch_offset,
               std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
                return self(
                    cache_tensor1,
                    input_tensor1,
                    cache_tensor2,
                    input_tensor2,
                    update_idxs,
                    update_idxs_tensor,
                    share_cache,
                    page_table,
                    batch_offset,
                    compute_kernel_config);
            },
            py::arg("cache_tensor1").noconvert(),
            py::arg("input_tensor1").noconvert(),
            py::arg("cache_tensor2").noconvert(),
            py::arg("input_tensor2").noconvert(),
            py::kw_only(),
            py::arg("update_idxs").noconvert() = std::vector<uint32_t>(),
            py::arg("update_idxs_tensor").noconvert() = std::nullopt,
            py::arg("share_cache").noconvert() = std::nullopt,
            py::arg("page_table").noconvert() = std::nullopt,
            py::arg("batch_offset") = 0,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
        });

    auto paged_fill_cache_doc =
        R"doc(
        Paged fill cache operation. This operation expects the following inputs: cache_tensor, input_tensor, and page_table.
        It uses either batch_idx_tensor (if provided, kwarg batch_idx_tensor) or batch_idx (kwarg batch_idx) as a fallback to determine the batch index for updating the cache.
        cache_tensor shape: [max_num_blocks, 1, block_size, head_dim]
        input_tensor shape: [1, num_heads, input_seq_len, head_dim]
        page_table shape: [batch_size, max_num_blocks_per_seq]
        batch_idx_tensor (optional) shape: [1] (scalar uint32 tensor)
        batch_idx (scalar, defaults to 0) is used if batch_idx_tensor is not provided.
        )doc";

    using PagedFillCacheType = decltype(ttnn::experimental::paged_fill_cache);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::paged_fill_cache,
        paged_fill_cache_doc,
        ttnn::pybind_overload_t{
            [](const PagedFillCacheType& self,
               const ttnn::Tensor& cache_tensor,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& page_table,
               std::optional<const ttnn::Tensor> batch_idx_tensor,
               const uint32_t batch_idx,
               std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
                return self(cache_tensor, input_tensor, page_table, batch_idx_tensor, batch_idx, compute_kernel_config);
            },
            py::arg("cache_tensor").noconvert(),
            py::arg("input_tensor").noconvert(),
            py::arg("page_table").noconvert(),
            py::kw_only(),
            py::arg("batch_idx_tensor").noconvert() = std::nullopt,
            py::arg("batch_idx") = 0,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
        });
}

}  // namespace ttnn::operations::experimental::paged_cache::detail
