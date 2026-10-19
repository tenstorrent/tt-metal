// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

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
    // user_id * batch_slot_num_layers + batch_slot_layer_idx. Mutually exclusive with input_batch_index.
    const std::optional<Tensor>& input_batch_index_tensor = std::nullopt,
    uint32_t batch_slot_num_layers = 1,
    uint32_t batch_slot_layer_idx = 0,
    // Trace-safe active extent: 1-element uint32 tensor holding this chunk's start position in the
    // gathered dim; the reader derives the extent from it. Mutually exclusive with gathered_dim_size.
    const std::optional<Tensor>& gathered_prefix_tensor = std::nullopt,
    uint32_t gathered_slab_global = 0,
    const std::optional<GlobalSemaphore>& ready_semaphore = std::nullopt,
    const std::optional<GlobalSemaphore>& data_valid_semaphore = std::nullopt,
    // Resumable window (metadata extent only): gather slabs [window_start, window_start + window_max_slabs)
    // of the active extent and record the reached end in progress_tensor. prefix_only limits the extent to
    // slabs written before this chunk.
    const std::optional<Tensor>& window_start_tensor = std::nullopt,
    std::optional<uint32_t> window_max_slabs = std::nullopt,
    bool prefix_only = false,
    const std::optional<Tensor>& progress_tensor = std::nullopt,
    // PROTOTYPE: index-driven sparse push instead of the dense gather (see HighBwSparsePushFactory).
    const std::optional<Tensor>& push_runs_tensor = std::nullopt);

// PROTOTYPE on-device plan for the sparse push: returns {plan table, indices rewritten to compact slots}
// (see device/sparse_push_plan_device_operation.hpp).
std::vector<Tensor> sparse_push_plan(
    const Tensor& indices,
    uint32_t num_devices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t pages_per_packet,
    uint32_t banks);

}  // namespace ttnn::operations::experimental::high_bw_all_gather

namespace ttnn::experimental {
using operations::experimental::high_bw_all_gather::high_bw_all_gather;
using operations::experimental::high_bw_all_gather::sparse_push_plan;
}  // namespace ttnn::experimental
