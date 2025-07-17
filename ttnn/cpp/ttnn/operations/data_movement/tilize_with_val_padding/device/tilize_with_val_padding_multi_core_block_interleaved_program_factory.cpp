// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_block_interleaved_program_factory.hpp"
#include "tilize_with_val_padding_single_core_program_factory.hpp"

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding_common.hpp"
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

// get_packed_value function is defined in single_core_program_factory.cpp

tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_block_interleaved(
    const Tensor& a, Tensor& output, const ttnn::PadValue pad_value) {
    // Temporary fallback to single core implementation
    return tilize_with_val_padding_single_core(a, output, pad_value);
}

}  // namespace ttnn::operations::data_movement::detail
