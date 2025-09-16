// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#pragma once

#include <tt-metalium/core_coord.hpp>

#include "ttnn/types.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/experimental/ccl/all_broadcast_async/device/all_broadcast_async_op.hpp"
#include "ttnn/distributed/types.hpp"

namespace composite_common {

bool use_composite_reduce_scatter(const ttnn::Tensor& input_tensor, int32_t dim, std::optional<uint32_t> cluster_axis);
bool use_all_gather_async_llama_sharded(const ttnn::Tensor& input_tensor, const ttnn::MemoryConfig& output_mem_config);
bool use_composite_all_gather(
    const ttnn::Tensor& input_tensor, int32_t dim, const std::optional<ttnn::MemoryConfig>& memory_config);
ttnn::Tensor composite_all_gather(
    ttnn::Tensor input_tensor,
    int32_t dim,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis);

}  // namespace composite_common
