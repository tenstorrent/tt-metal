// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/reduce_op_types.hpp"

#include <tt_stl/assert.hpp>

namespace reduce_op_utils {

std::map<std::string, std::string> get_defines(
    tt::tt_metal::ReduceOpMath reduce_op, tt::tt_metal::ReduceOpDim reduce_dim) {
    std::map<std::string, std::string> defines;
    std::string reduce_dim_str;
    switch (reduce_dim) {
        case tt::tt_metal::ReduceOpDim::W: reduce_dim_str = "ckernel::ReduceDim::REDUCE_ROW"; break;
        case tt::tt_metal::ReduceOpDim::H: reduce_dim_str = "ckernel::ReduceDim::REDUCE_COL"; break;
        case tt::tt_metal::ReduceOpDim::HW: reduce_dim_str = "ckernel::ReduceDim::REDUCE_SCALAR"; break;
        default: TT_THROW("Invalid reduce_dim!");
    }
    switch (reduce_op) {
        case tt::tt_metal::ReduceOpMath::MAX: defines["REDUCE_OP"] = "ckernel::PoolType::MAX"; break;
        case tt::tt_metal::ReduceOpMath::MIN: defines["REDUCE_OP"] = "ckernel::PoolType::MIN"; break;
        case tt::tt_metal::ReduceOpMath::AVG: defines["REDUCE_OP"] = "ckernel::PoolType::AVG"; break;
        default: defines["REDUCE_OP"] = "ckernel::PoolType::SUM"; break;
    }
    defines["REDUCE_DIM"] = reduce_dim_str;
    return defines;
}

}  // namespace reduce_op_utils
