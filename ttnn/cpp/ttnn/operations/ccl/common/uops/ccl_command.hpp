// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/common/types/ccl_types.hpp"

#include <cstdint>

namespace ttnn {
namespace ccl {
namespace cmd {

struct CclCommand {
    Shape4D<uint32_t> tensor_slice_shape;
    Shape4D<uint32_t> tensor_slice_offset;
    Shape4D<uint32_t> worker_start_offset_in_slice;
    uint32_t worker_pages_per_slice;
};

} // namespace cmd
} // namespace ccl
} // namespace ttnn
