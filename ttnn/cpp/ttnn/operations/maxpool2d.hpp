// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"
#include "ttnn/core.hpp"
#include "ttnn/operations/core/core.hpp"
#include "tt_metal/common/math.hpp"
#include "ttnn/operations/conv2d/conv2d.hpp"
#include "ttnn/operations/pool/max_pool.hpp"
#include "ttnn/experimental/tt_dnn/op_library/sliding_window_op_infra/halo_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/sliding_window_op_infra/sliding_window.hpp"


namespace ttnn::operations {
namespace maxpool2d {

using array2_t = std::array<uint32_t, 2>;

// maxpool macro-op
inline Tensor maxpool2d(const Tensor& input_tensor, uint32_t batch_size, uint32_t input_h, uint32_t input_w, uint32_t channels, array2_t kernel_size, array2_t stride, array2_t padding, array2_t dilation, Device& device) {
    MemoryConfig memory_config = input_tensor.memory_config();
    const auto shard_grid = memory_config.shard_spec.value().grid;
    const auto shard_scheme = memory_config.memory_layout;
    const auto shard_orientation = memory_config.shard_spec.value().orientation;

    TT_FATAL(shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED, "Only height sharded tensors are supported.");
    TT_FATAL(shard_orientation == ShardOrientation::ROW_MAJOR, "Only row major orientation is supported.");

    ParallelConfig parallel_config = conv2d::determine_parallel_config(
                                        shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED,
                                        batch_size,
                                        0,          // in_channels -- not used
                                        input_h,
                                        input_w,
                                        0,          // out_channels -- not used
                                        device,
                                        shard_orientation);
    uint32_t num_cores_nhw = conv2d::get_num_cores_nhw_from_parallel_config(parallel_config);

    SlidingWindowConfig sliding_window_config = SlidingWindowConfig(batch_size,
                                                                    input_h, input_w,
                                                                    kernel_size.at(0), kernel_size.at(1),
                                                                    stride.at(0), stride.at(1),
                                                                    padding.at(0), padding.at(1),
                                                                    dilation.at(0), dilation.at(1),
                                                                    num_cores_nhw,
                                                                    parallel_config.grid);
    uint32_t neg_inf_pad_val = 0xf7ff;  // TODO: double check

    auto haloed_tensor = ttnn::operations::halo::halo_op(input_tensor, sliding_window_config, neg_inf_pad_val, false, parallel_config.shard_orientation == ShardOrientation::COL_MAJOR, 0, memory_config);
    return tt::tt_metal::maxpool2d_new(haloed_tensor, sliding_window_config, channels, memory_config);
}

}  // namespace maxpool
}  // namespace ttnn::operations
