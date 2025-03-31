// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d_op_program_factory_common.hpp"

namespace ttnn::operations::conv {
namespace conv2d {

uint32_t CBIndices::get_next_cb_index() { return next_cb_index++; }

}  // namespace conv2d
}  // namespace ttnn::operations::conv
