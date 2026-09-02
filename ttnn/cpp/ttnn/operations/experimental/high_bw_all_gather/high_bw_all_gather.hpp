// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather {

Tensor high_bw_all_gather(
    const Tensor& input_tensor,
    int32_t dim,
    const Tensor& output_tensor,
    uint32_t cluster_axis,
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
    uint32_t gathered_slab_global = 0);

}  // namespace ttnn::operations::experimental::high_bw_all_gather

namespace ttnn::experimental {
using operations::experimental::high_bw_all_gather::high_bw_all_gather;
}  // namespace ttnn::experimental
