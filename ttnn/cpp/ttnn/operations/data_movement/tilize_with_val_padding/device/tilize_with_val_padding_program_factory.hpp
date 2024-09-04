// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks tilize_with_val_padding_single_core(
    const Tensor& a, Tensor& output, const float pad_value);



operation::ProgramWithCallbacks tilize_with_val_padding_multi_core(
    const Tensor& a, Tensor& output, const float pad_value); 

}  // namespace ttnn::operations::data_movement::detail
