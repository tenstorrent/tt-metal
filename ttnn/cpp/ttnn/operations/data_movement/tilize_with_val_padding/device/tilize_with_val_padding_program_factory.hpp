// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::data_movement::detail {
tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_single_core(
    const Tensor& a,
    Tensor& output,
    tt::tt_metal::PadValue pad_value,
    const std::optional<CoreRangeSet>& sub_core_grids);

tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_interleaved(
    const Tensor& a,
    Tensor& output,
    tt::tt_metal::PadValue pad_value,
    const std::optional<CoreRangeSet>& sub_core_grids);

tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_sharded(
    const Tensor& a, Tensor& output, tt::tt_metal::PadValue pad_value);

tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_col_interleaved(
    const Tensor& a,
    Tensor& output,
    tt::tt_metal::PadValue pad_value,
    const std::optional<CoreRangeSet>& sub_core_grids);

tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_block_interleaved(
    const Tensor& a,
    Tensor& output,
    tt::tt_metal::PadValue pad_value,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::detail
