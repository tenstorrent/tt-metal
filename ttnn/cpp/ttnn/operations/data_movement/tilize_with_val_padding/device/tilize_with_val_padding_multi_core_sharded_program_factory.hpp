// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding_common.hpp"

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_sharded(
    const Tensor& a, Tensor& output, const ttnn::PadValue pad_value);

}  // namespace ttnn::operations::data_movement::detail
