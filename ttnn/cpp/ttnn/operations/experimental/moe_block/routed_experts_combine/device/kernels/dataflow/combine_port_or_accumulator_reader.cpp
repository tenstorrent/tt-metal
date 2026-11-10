// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// combine_port_or_accumulator_reader: the NCRISC kernel of a core that is a combine port on some chips and an
// accumulator on others - the port reader and the accumulator reader in one binary, running this chip's role
// Runs on: plan.port_or_accumulator_cores (separate accumulator cores only), RISCV_1 (NCRISC), NoC 0
//          (combine dm_kernel: RISCV_1 is pinned to NoC 0)
//
// The op builds one program for the whole mesh (common/chip_program.hpp), and a core can be a port on one chip and an
// accumulator on another, so its kernel carries both roles and picks this chip's at run time:
//
//   runtime arg 0 (role) | runs
//   ---------------------+----------------------------------------------------------------------------------------
//   0 none               | nothing (pushes the 0 meta page that ends the core's accumulator_compute at once)
//   1 port               | combine_port_reader.cpp (and the 0 meta page, as above)
//   2 accumulator        | accumulator_reader.cpp
//
// Each role is compiled from its own kernel file, #included into a namespace of its own with:
//   COMBINE_ROLE_FUNCTION_NAME           the role's kernel_main is compiled as this function (role_main), called below
//   COMBINE_ROLE_FIRST_COMPILE_TIME_ARG  where the role's compile-time args start in this kernel's
//   COMBINE_PLAN_ACCESSOR_FIRST_ARG      where the role's device-plan accessor args start, past its first arg
//
// Compile-time args: the port reader's, then the accumulator reader's from ACCUMULATOR_READER_FIRST_COMPILE_TIME_ARG;
// the roles' plan accessors at PORT_READER_PLAN_ACCESSOR_FIRST_ARG / ACCUMULATOR_READER_PLAN_ACCESSOR_FIRST_ARG (host
// defines, combine_program.cpp). Runtime args: [role, the accumulator role's args (padded to the same length on every
// chip), the port role's args from PORT_ROLE_FIRST_RUNTIME_ARG]; each role reads its own args from its arg 0 (the
// args' base moved). Named compile-time args: the combine's options (combine_options.hpp).

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "combine_zero_gate.hpp"
#include "combine_pool.hpp"
#include "combine_formats.hpp"
#include "combine_options.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/combine_plan_words.hpp"

namespace port_role {
#define COMBINE_ROLE_FUNCTION_NAME role_main
#define COMBINE_ROLE_FIRST_COMPILE_TIME_ARG 0
#define COMBINE_PLAN_ACCESSOR_FIRST_ARG PORT_READER_PLAN_ACCESSOR_FIRST_ARG
#include "combine_port_reader.cpp"
#undef COMBINE_PLAN_ACCESSOR_FIRST_ARG
#undef COMBINE_ROLE_FIRST_COMPILE_TIME_ARG
#undef COMBINE_ROLE_FUNCTION_NAME
}  // namespace port_role

namespace accumulator_role {
#define COMBINE_ROLE_FUNCTION_NAME role_main
#define COMBINE_ROLE_FIRST_COMPILE_TIME_ARG ACCUMULATOR_READER_FIRST_COMPILE_TIME_ARG
#define COMBINE_PLAN_ACCESSOR_FIRST_ARG ACCUMULATOR_READER_PLAN_ACCESSOR_FIRST_ARG
#include "accumulator_reader.cpp"
#undef COMBINE_PLAN_ACCESSOR_FIRST_ARG
#undef COMBINE_ROLE_FIRST_COMPILE_TIME_ARG
#undef COMBINE_ROLE_FUNCTION_NAME
}  // namespace accumulator_role

namespace {
enum Role : uint32_t { kNoRole = 0, kPortRole = 1, kAccumulatorRole = 2 };
}  // namespace

void kernel_main() {
    moe_block_args::load();  // (this core's args - the role, the accumulator role's, the port role's - from the DRAM
                             // table)
    const uint32_t role = moe_block_args::arg(0);
    if (role != kAccumulatorRole) {  // (no accumulator here on this chip: the core's compute gets a 0 meta page and
                                     // returns)
        cb_reserve_back(accumulator_role::cb_meta, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(accumulator_role::cb_meta))[0] = 0;
        cb_push_back(accumulator_role::cb_meta, 1);
    }
    if (role == kPortRole) {
        moe_block_args::set_base(PORT_ROLE_FIRST_RUNTIME_ARG);
        port_role::role_main();
    } else if (role == kAccumulatorRole) {
        moe_block_args::set_base(1);
        accumulator_role::role_main();
    }
}
