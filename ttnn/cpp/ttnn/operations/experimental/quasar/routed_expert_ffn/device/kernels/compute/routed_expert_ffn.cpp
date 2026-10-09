// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// routed_expert_ffn compute: y = ((x @ w_gate) * (x @ w_up)) @ w_down on one Tensix engine, one tile at a time.
// For each tile row m of x:
//   phase 1, for each h: gate = x[m, :] @ w_gate[:, h], up = x[m, :] @ w_up[:, h], act[h] = gate * up
//   phase 2, for each n: y[m, n] = act[0..Ht) @ w_down[:, n]
// gate, up and act are compute-only DFBs: the pack thread produces them and the unpack thread consumes them.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace {
// On Quasar the pack destination is baked in at pack_init, so every switch of the output DFB re-inits the packer.
ALWI void pack_to(uint32_t dfb_id) {
#ifdef ARCH_QUASAR
    pack_reconfig_data_format(dfb_id);
    pack_init(dfb_id);
#endif
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Mt = get_arg(args::Mt);
    constexpr uint32_t Kt = get_arg(args::Kt);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t gate_dst = 0;
    constexpr uint32_t up_dst = 1;

    DataflowBuffer dfb_x(dfb::x);
    DataflowBuffer dfb_w(dfb::w);
    DataflowBuffer dfb_gate(dfb::gate);
    DataflowBuffer dfb_up(dfb::up);
    DataflowBuffer dfb_act(dfb::act);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::x, dfb::w, dfb::gate);

    for (uint32_t m = 0; m < Mt; ++m) {
        dfb_x.wait_front(Kt);
        for (uint32_t h = 0; h < Ht; ++h) {
            matmul_init(dfb::x, dfb::w);
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; ++k) {
                dfb_w.wait_front(1);
                matmul_tiles(dfb::x, dfb::w, k, 0, gate_dst);
                dfb_w.pop_front(1);
            }
            for (uint32_t k = 0; k < Kt; ++k) {
                dfb_w.wait_front(1);
                matmul_tiles(dfb::x, dfb::w, k, 0, up_dst);
                dfb_w.pop_front(1);
            }
            tile_regs_commit();

            dfb_gate.reserve_back(1);
            dfb_up.reserve_back(1);
            tile_regs_wait();
            pack_to(dfb::gate);
            pack_tile(gate_dst, dfb::gate);
            pack_to(dfb::up);
            pack_tile(up_dst, dfb::up);
            tile_regs_release();
            dfb_gate.push_back(1);
            dfb_up.push_back(1);

            // acc_to_dest defaults to true, which on Quasar adds the product to what DST already holds.
            mul_init(dfb::gate, dfb::up, /*acc_to_dest=*/false);
            dfb_gate.wait_front(1);
            dfb_up.wait_front(1);
            tile_regs_acquire();
            mul_tiles(dfb::gate, dfb::up, 0, 0, 0);
            tile_regs_commit();

            dfb_act.reserve_back(1);
            tile_regs_wait();
            pack_to(dfb::act);
            pack_tile(0, dfb::act);
            tile_regs_release();
            dfb_act.push_back(1);
            dfb_gate.pop_front(1);
            dfb_up.pop_front(1);
        }
        dfb_x.pop_front(Kt);

        matmul_init(dfb::act, dfb::w);
        dfb_act.wait_front(Ht);
        for (uint32_t n = 0; n < Kt; ++n) {
            tile_regs_acquire();
            for (uint32_t h = 0; h < Ht; ++h) {
                dfb_w.wait_front(1);
                matmul_tiles(dfb::act, dfb::w, h, 0, 0);
                dfb_w.pop_front(1);
            }
            tile_regs_commit();

            dfb_out.reserve_back(1);
            tile_regs_wait();
            pack_to(dfb::out);
            pack_tile(0, dfb::out);
            tile_regs_release();
            dfb_out.push_back(1);
        }
        dfb_act.pop_front(Ht);
    }
}
