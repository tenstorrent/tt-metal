// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding_common.hpp"

using namespace tt::constants;

namespace ttnn::operations::data_movement::detail {
tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_single_core(
    const Tensor& a, Tensor& output, ttnn::PadValue pad_value);

tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_interleaved(
    const Tensor& a, Tensor& output, ttnn::PadValue pad_value);

tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_sharded(
    const Tensor& a, Tensor& output, ttnn::PadValue pad_value);

tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_col_interleaved(
    const Tensor& a, Tensor& output, ttnn::PadValue pad_value);

tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_block_interleaved(
    const Tensor& a, Tensor& output, ttnn::PadValue pad_value);

}  // namespace ttnn::operations::data_movement::detail
