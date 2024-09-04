// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn {

enum class BcastOpMath { ADD, SUB, MUL };
enum class BcastOpDim { H, W, HW };


namespace bcast_op_utils {
  std::map<std::string, std::string> get_defines(ttnn::BcastOpDim bcast_dim, ttnn::BcastOpMath bcast_math);
}

} // namespace ttnn
