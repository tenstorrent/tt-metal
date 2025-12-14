// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::data_movement::tilize_with_val_padding::program {
struct shared_variables_interleaved {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<CoreCoord> cores;
    uint32_t ncores;
};
}  // namespace ttnn::operations::data_movement::tilize_with_val_padding::program
