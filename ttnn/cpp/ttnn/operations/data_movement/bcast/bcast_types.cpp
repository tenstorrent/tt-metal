// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_types.hpp"

namespace ttnn::bcast_op_utils {

std::map<std::string, std::string> get_defines(BcastOpDim bcast_dim, BcastOpMath bcast_math) {
    std::map<std::string, std::string> defines;
    const char* math_to_op_define[] = {"add_tiles_bcast", "sub_tiles_bcast", "mul_tiles_bcast"};
    const char* math_to_llkop_define[] = {
        "EltwiseBinaryType::ELWADD", "EltwiseBinaryType::ELWSUB", "EltwiseBinaryType::ELWMUL"};
    const char* bdim_to_llkdim_define[] = {"BroadcastType::ROW", "BroadcastType::COL", "BroadcastType::SCALAR"};
    const char* math_to_chain_op[] = {
        "compute_kernel_lib::BinaryFpuOp::Add",
        "compute_kernel_lib::BinaryFpuOp::Sub",
        "compute_kernel_lib::BinaryFpuOp::Mul"};
    const char* bdim_to_chain_dim[] = {
        "compute_kernel_lib::BroadcastDim::Row",
        "compute_kernel_lib::BroadcastDim::Col",
        "compute_kernel_lib::BroadcastDim::Scalar"};
    defines["BCAST_OP"] = math_to_op_define[int(bcast_math)];
    defines["BCAST_LLKOP"] = math_to_llkop_define[int(bcast_math)];
    defines["BCAST_DIM"] = bdim_to_llkdim_define[int(bcast_dim)];
    defines["CHAIN_BCAST_OP"] = math_to_chain_op[int(bcast_math)];
    defines["CHAIN_BCAST_DIM"] = bdim_to_chain_dim[int(bcast_dim)];
    return defines;
}

}  // namespace ttnn::bcast_op_utils
