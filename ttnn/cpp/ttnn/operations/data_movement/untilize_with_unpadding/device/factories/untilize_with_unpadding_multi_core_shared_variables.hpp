// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <vector>

namespace ttnn::prim {

struct UntilizeWithUnpaddingMultiCoreSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id {};
    tt::tt_metal::KernelHandle writer_kernel_id {};
    std::vector<tt::tt_metal::CoreCoord> cores;
    uint32_t ncores = 0;
};

}  // namespace ttnn::prim
