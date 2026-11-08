// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A combine pool cell's BRISC in one program for the mesh (combine_cell_reader.cpp): the port sender
// (combine_port_writer.cpp) or the accumulator writer (accumulator_writer.cpp), this chip's role. Runtime args as the
// reader's; compile-time args: the sender's, then the accumulator writer's (CMB_AW_CT0 on); the sender's accessors
// CMB_PW_PLAN_CT / CMB_PW_HC_CT / CMB_PW_LUT_CT.

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_loop.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "combine_zero_gate.hpp"
#include "combine_pool.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/combine_plan_words.hpp"

namespace cmb_port {
#define CMB_ROLE_FN body
#define CMB_CT0 0
#define CMB_PLAN_CT CMB_PW_PLAN_CT
#define CMB_HC_CT CMB_PW_HC_CT
#define CMB_LUT_CT CMB_PW_LUT_CT
#include "combine_port_writer.cpp"
#undef CMB_LUT_CT
#undef CMB_HC_CT
#undef CMB_PLAN_CT
#undef CMB_CT0
#undef CMB_ROLE_FN
}  // namespace cmb_port

namespace cmb_acc {
#define CMB_ROLE_FN body
#define CMB_CT0 CMB_AW_CT0
#include "accumulator_writer.cpp"
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
