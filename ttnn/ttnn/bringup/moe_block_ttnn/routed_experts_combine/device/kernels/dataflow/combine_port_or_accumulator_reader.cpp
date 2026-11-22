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
// Each role is compiled from its own kernel file, #included into a namespace of its own with
//   COMBINE_ROLE_FUNCTION_NAME           the role's kernel_main is compiled as this function (role_main), called below
// (the one macro left: it names a function, which a compile-time arg cannot).
//
// Compile-time args: all named (combine_names.hpp): the combine's options (combine_options.hpp) and each role's own
// values, one name a meaning, so the two roles' names merge (equal names have equal values). The positional ones
// are the roles' device-plan accessor args - the port reader's, then the accumulator reader's - at the absolute indices
// port_reader_plan_accessor_first_arg / accumulator_reader_plan_accessor_first_arg. Runtime args: [role, the
// accumulator role's args (padded to the same length on every chip), the port role's args from the named compile-time
// arg port_role_first_runtime_arg]; each role reads its own args from its arg 0 (the args' base moved).

#include "../../../../common/kernels/moe_block_args.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "combine_zero_gate.hpp"
#include "combine_pool.hpp"
#include "combine_formats.hpp"
#include "combine_options.hpp"
#include "../combine_names.hpp"
#include "../../../../common/kernels/combine_plan_words.hpp"

namespace port_role {
#define COMBINE_ROLE_FUNCTION_NAME role_main
#include "combine_port_reader.cpp"
#undef COMBINE_ROLE_FUNCTION_NAME
}  // namespace port_role

namespace accumulator_role {
#define COMBINE_ROLE_FUNCTION_NAME role_main
#include "accumulator_reader.cpp"
#undef COMBINE_ROLE_FUNCTION_NAME
}  // namespace accumulator_role

namespace {
enum Role : uint32_t { kNoRole = 0, kPortRole = 1, kAccumulatorRole = 2 };
}  // namespace

// where the port role's runtime args start (the args' base moves there for it)
constexpr uint32_t port_role_first_runtime_arg =
    get_named_compile_time_arg_val(combine_names::kPortRoleFirstRuntimeArg);

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
        moe_block_args::set_base(port_role_first_runtime_arg);
        port_role::role_main();
    } else if (role == kAccumulatorRole) {
        moe_block_args::set_base(1);
        accumulator_role::role_main();
    }
}
