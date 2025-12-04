// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <vector>

namespace ttnn::operations::data_movement::detail {

struct UntilizeWithUnpaddingMultiCoreSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id;
    tt::tt_metal::KernelHandle writer_kernel_id;
    std::vector<tt::tt_metal::CoreCoord> cores;
};

}  // namespace ttnn::operations::data_movement::detail
