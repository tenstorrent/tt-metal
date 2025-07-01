// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement::detail {

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

}  // namespace ttnn::operations::data_movement::detail
