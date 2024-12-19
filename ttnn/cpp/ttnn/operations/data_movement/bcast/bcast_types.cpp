// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_types.hpp"

namespace ttnn {

namespace bcast_op_utils {

std::map<std::string, std::string> get_defines(BcastOpDim bcast_dim, BcastOpMath bcast_math) {
    std::map<std::string, std::string> defines;
    const char* math_to_op_define[] = {"add_tiles_bcast", "sub_tiles_bcast", "mul_tiles_bcast"};
    const char* math_to_llkop_define[] = {"ELWADD", "ELWSUB", "ELWMUL"};
    const char* bdim_to_llkdim_define[] = {"BroadcastType::ROW", "BroadcastType::COL", "BroadcastType::SCALAR"};
    defines["BCAST_OP"] = math_to_op_define[int(bcast_math)];
    defines["BCAST_LLKOP"] = math_to_llkop_define[int(bcast_math)];
    defines["BCAST_DIM"] = bdim_to_llkdim_define[int(bcast_dim)];
    return defines;
}

}  // namespace bcast_op_utils

}  // namespace ttnn
