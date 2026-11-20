// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// combine_port_or_accumulator_writer: the BRISC kernel of a core that is a combine port on some chips and an
// accumulator on others - the port writer (the fabric sender) and the accumulator writer in one binary, running this
// chip's role
// Runs on: plan.port_or_accumulator_cores (separate accumulator cores only), RISCV_0 (BRISC), NoC 1
//          (combine dm_kernel: RISCV_0 is pinned to NoC 1)
//
// The same arrangement as combine_port_or_accumulator_reader.cpp, which runs the same role on the core:
//
//   runtime arg 0 (role) | runs
//   ---------------------+-------------------------------
//   0 none               | nothing
//   1 port               | combine_port_writer.cpp
//   2 accumulator        | accumulator_writer.cpp
//
// Each role is compiled from its own kernel file, #included into a namespace of its own with
//   COMBINE_ROLE_FUNCTION_NAME           the role's kernel_main is compiled as this function (role_main)
// (the one macro left: it names a function, which a compile-time arg cannot).
//
// Compile-time args: all named (combine_names.hpp), as the reader's: the options (combine_options.hpp) and the roles'
// values, merged. The positional ones are the roles' accessor args - the port writer's output, device plan,
// routed-header cache and per-chip accumulator table, then the accumulator writer's output - at the absolute indices
// the *_accessor_first_arg names give. Runtime args as the reader's, the port role's from the named compile-time arg
// port_role_first_runtime_arg.

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "combine_zero_gate.hpp"
#include "combine_pool.hpp"
#include "combine_formats.hpp"
#include "combine_options.hpp"
#include "../combine_names.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/combine_plan_words.hpp"

namespace port_role {
#define COMBINE_ROLE_FUNCTION_NAME role_main
#include "combine_port_writer.cpp"
#undef COMBINE_ROLE_FUNCTION_NAME
}  // namespace port_role

namespace accumulator_role {
#define COMBINE_ROLE_FUNCTION_NAME role_main
#include "accumulator_writer.cpp"
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
    if (role == kPortRole) {
        moe_block_args::set_base(port_role_first_runtime_arg);
        port_role::role_main();
    } else if (role == kAccumulatorRole) {
        moe_block_args::set_base(1);
        accumulator_role::role_main();
    }
}
