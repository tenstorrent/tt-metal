// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "untilize_op.hpp"

namespace ttnn::operations::data_movement {
namespace detail {

tt::tt_metal::operation::ProgramWithCallbacks untilize_multi_core_sub_core_grids(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en, const CoreRangeSet& sub_core_grids);

tt::tt_metal::operation::ProgramWithCallbacks untilize_multi_core_block(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en);

tt::tt_metal::operation::ProgramWithCallbacks untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en);

tt::tt_metal::operation::ProgramWithCallbacks untilize_multi_core(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en);

tt::tt_metal::operation::ProgramWithCallbacks untilize_single_core(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en);

tt::tt_metal::operation::ProgramWithCallbacks untilize_row_wise_fuseable(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    bool use_pack_untilize,
    bool fp32_dest_acc_en,
    const CoreRangeSet& sub_core_grids,
    const std::optional<tt::tt_metal::GlobalSemaphore>& _internal_semaphore,
    const std::optional<CoreRangeSet>& sync_core_grids,
    uint32_t max_tiles_per_block = 1);
}  // namespace detail

namespace untilize {

using FuseableUntilizeCallback = std::function<void(
    const std::optional<tt::tt_metal::GlobalSemaphore>&, tt::tt_metal::Program&, const Tensor&, const Tensor&)>;

FuseableUntilizeCallback untilize_row_wise_fuseable(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    bool use_pack_untilize,
    bool fp32_dest_acc_en,
    const CoreRangeSet& sub_core_grids,
    const std::optional<tt::tt_metal::GlobalSemaphore>& _internal_semaphore,
    const std::optional<CoreRangeSet>& sync_core_grids,
    uint32_t max_tiles_per_block = 1);
}  // namespace untilize
}  // namespace ttnn::operations::data_movement
