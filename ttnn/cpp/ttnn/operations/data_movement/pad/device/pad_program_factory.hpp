// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::data_movement::pad::program {

struct PadSharedVariables {
    tt::tt_metal::KernelHandle unary_reader_kernel_id{};
    tt::tt_metal::KernelHandle unary_writer_kernel_id{};
};

struct PadProgramFactory {
    using shared_variables_t = PadSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
};

}  // namespace ttnn::operations::data_movement::pad::program
