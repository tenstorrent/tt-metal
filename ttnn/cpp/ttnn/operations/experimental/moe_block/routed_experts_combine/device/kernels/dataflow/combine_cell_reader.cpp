// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// combine_cell_reader: a combine pool cell's NCRISC binary - the port reader and accumulator reader roles in one
// Runs on: combine pool cells (plan.cells, ports-apart mode only), RISCV_1 (NCRISC), NoC 0
//          (combine dm_kernel: RISCV_1 is pinned to NoC 0)
// Responsibilities:
//  - one program for the mesh: a cell is a port on some chips and an accumulator on others; runtime arg 0 picks this
//    chip's role (0 none, 1 port -> combine_port_reader body, 2 accumulator -> accumulator_reader body)
//  - role != 2: pushes a 0 meta page so the cell's accumulator_compute returns at once
//  - reads / writes / syncs exactly as the selected role's kernel (both #included in their own namespaces)

// A combine pool cell's NCRISC in one program for the mesh (common/chip_program.hpp): a cell is a port on some chips
// and an accumulator on others, so its binary carries both roles - the port reader (combine_port_reader.cpp) and the
// accumulator reader (accumulator_reader.cpp) - and runs this chip's. Runtime args: [role (0 none, 1 port, 2
// accumulator), the accumulator block (padded: the port block at CMB_P0 on every chip), the port block]; each role
// reads its own block from its arg 0 (the args' base moved). Compile-time args: the port reader's, then the accumulator
// reader's (CMB_AR_CT0 on); the roles' plan accessors CMB_PR_PLAN_CT / CMB_AR_PLAN_CT.

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "combine_zero_gate.hpp"
#include "combine_pool.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/combine_plan_words.hpp"

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

void kernel_main() {
    moe_block_args::load();  // (this core's args - the role, the accumulator block, the port block - from the DRAM
                             // table)
    const uint32_t role = moe_block_args::arg(0);
    if (role != 2) {  // (the cell's compute: no accumulator here on this chip - its first meta page 0, it returns)
        cb_reserve_back(cmb_acc::cb_meta, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cmb_acc::cb_meta))[0] = 0;
        cb_push_back(cmb_acc::cb_meta, 1);
    }
    if (role == 1) {
        moe_block_args::set_base(
            CMB_P0);  // (the port block: past the accumulator block - the same length on every chip)
        cmb_port::body();
    } else if (role == 2) {
        moe_block_args::set_base(1);
        cmb_acc::body();
    }
}

