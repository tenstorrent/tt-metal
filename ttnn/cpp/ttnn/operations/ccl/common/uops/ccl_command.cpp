// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types.hpp"

namespace ttnn {
namespace ccl {
namespace cmd {

void pack_field_without_header(ttnn::ccl::cmd::args_elem_t* args, ttnn::ccl::Shape4D<uint32_t> const& out) {
    std::size_t i = 0;
    args[i++] = out.w;
    args[i++] = out.z;
    args[i++] = out.y;
    args[i++] = out.x;
}


}  // namespace cmd
}  // namespace ccl
}  // namespace ttnn
