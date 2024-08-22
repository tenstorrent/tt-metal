// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command_host.hpp"

#include <ranges>

namespace ttnn {
namespace ccl {
namespace cmd {


std::vector<uint32_t> add_ccl_command_to_args(CclCommand const& cmd ) {
    return {
        cmd.tensor_slice_shape.w,
        cmd.tensor_slice_shape.z,
        cmd.tensor_slice_shape.y,
        cmd.tensor_slice_shape.x,
        cmd.worker_start_offset_in_slice.w,
        cmd.worker_start_offset_in_slice.z,
        cmd.worker_start_offset_in_slice.y,
        cmd.worker_start_offset_in_slice.x,
        cmd.worker_pages_per_slice
    };
}


} // namespace cmd
} // namespace ccl
} // namespace ttnn
