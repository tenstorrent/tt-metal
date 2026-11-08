// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A combine pool cell's NCRISC in one program for the mesh (common/chip_program.hpp): a cell is a port on some chips
// and an accumulator on others, so its binary carries both roles - the port reader (combine_port_reader.cpp) and the
// accumulator reader (accumulator_reader.cpp) - and runs this chip's. Runtime args: [role (0 none, 1 port, 2
// accumulator), the accumulator block (padded: the port block at CMB_P0 on every chip), the port block]; each role
// reads its own block from its arg 0 (the args' base moved). Compile-time args: the port reader's, then the accumulator
// reader's (CMB_AR_CT0 on); the roles' plan accessors CMB_PR_PLAN_CT / CMB_AR_PLAN_CT.

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_loop.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "combine_zero_gate.hpp"
#include "combine_pool.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/combine_plan_words.hpp"

namespace cmb_port {
#define CMB_ROLE_FN body
#define CMB_CT0 0
#define CMB_PLAN_CT CMB_PR_PLAN_CT
#include "combine_port_reader.cpp"
#undef CMB_PLAN_CT
#undef CMB_CT0
#undef CMB_ROLE_FN
}  // namespace cmb_port

namespace cmb_acc {
#define CMB_ROLE_FN body
#define CMB_CT0 CMB_AR_CT0
#define CMB_PLAN_CT CMB_AR_PLAN_CT
#include "accumulator_reader.cpp"
#undef CMB_PLAN_CT
#undef CMB_CT0
#undef CMB_ROLE_FN
}  // namespace cmb_acc

MB_LOOP_BODY {
    const uint32_t role = get_arg_val<uint32_t>(0);
    if (role == 1) {
        rta_l1_base += CMB_P0;  // (the port block: past the accumulator block - the same length on every chip)
        cmb_port::body();
    } else if (role == 2) {
        rta_l1_base += 1;
        cmb_acc::body();
    }
}

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_loop_main.hpp"
