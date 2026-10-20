// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather {

Tensor high_bw_all_gather(
    const Tensor& input_tensor,
    int32_t dim,
    const Tensor& output_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt,
    std::optional<uint32_t> num_links = std::nullopt,
    std::optional<uint32_t> input_batch_index = std::nullopt,
    std::optional<uint32_t> gathered_dim_size = std::nullopt,
    // Trace-safe slot select: 1-element uint32 tensor holding the USER id, recomposed on-device as
    // user_id * batch_slot_num_layers + batch_slot_layer_idx (with page_table: the request's page-table row, the
    // layer being batch_slot_layer_idx). Mutually exclusive with input_batch_index.
    const std::optional<Tensor>& input_batch_index_tensor = std::nullopt,
    uint32_t batch_slot_num_layers = 1,
    uint32_t batch_slot_layer_idx = 0,
    // Trace-safe active extent: 1-element uint32 tensor holding this chunk's start position in the
    // gathered dim; the reader derives the extent from it. Mutually exclusive with gathered_dim_size.
    const std::optional<Tensor>& gathered_prefix_tensor = std::nullopt,
    uint32_t gathered_slab_global = 0,
    const std::optional<GlobalSemaphore>& ready_semaphore = std::nullopt,
    const std::optional<GlobalSemaphore>& data_valid_semaphore = std::nullopt,
    // Paged source (update_padded_kv_cache/device/kernels/paged_kv_table.hpp): `input_tensor` is a pool of single-head
    // pages [num_pages * batch_slot_num_layers, 1, page, D] and `page_table` an INT32 table whose request row
    // locates each page. A request is required: the scalar input_batch_index is the flat slot
    // row * batch_slot_num_layers + layer, as for a slot cache, or the tensor form names the row.
    // gathered_slab_global (chunk_local * ranks) is required and holds whole pages per rank. Unmapped pages gather as
    // zeros.
    const std::optional<Tensor>& page_table = std::nullopt);

}  // namespace ttnn::operations::experimental::high_bw_all_gather

namespace ttnn::experimental {
using operations::experimental::high_bw_all_gather::high_bw_all_gather;
}  // namespace ttnn::experimental
