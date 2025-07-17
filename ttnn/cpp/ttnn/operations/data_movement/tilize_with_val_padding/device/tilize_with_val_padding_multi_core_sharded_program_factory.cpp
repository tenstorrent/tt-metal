// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_sharded_program_factory.hpp"
#include "tilize_with_val_padding_single_core_program_factory.hpp"

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_sharded(
    const Tensor& a, Tensor& output, const ttnn::PadValue pad_value) {
    // Temporary fallback to single core implementation
    return tilize_with_val_padding_single_core(a, output, pad_value);
}

}  // namespace ttnn::operations::data_movement::detail
