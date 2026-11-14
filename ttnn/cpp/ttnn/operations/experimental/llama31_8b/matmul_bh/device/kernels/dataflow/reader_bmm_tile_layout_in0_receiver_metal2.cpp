// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#ifndef ARCH_QUASAR
#include "ckernel.h"
#include "ckernel_defs.h"
#endif
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#ifdef OUT_WRITER_ON_IN0
#include "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/dataflow/out_writer_on_in0.hpp"
#endif
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args_metal2.hpp"

void kernel_main() {
    constexpr auto in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr auto num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr auto num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr auto num_blocks_h_dim = get_arg(args::num_blocks_h_dim);
    constexpr auto batch = get_arg(args::batch);
    constexpr bool get_batch_from_reader = static_cast<bool>(get_arg(args::get_batch_from_reader));

    const Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    constexpr auto in0_mcast = MCAST_ARGS(in0_mcast);
    auto receiver = in0_mcast.receiver(noc);

    for (uint32_t b = 0; b < batch; ++b) {
        if constexpr (get_batch_from_reader) {
            const auto is_batch_valid = receiver.receive_signal() == VALID;

#ifndef ARCH_QUASAR
            ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, static_cast<uint32_t>(is_batch_valid));
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, static_cast<uint32_t>(is_batch_valid));
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, static_cast<uint32_t>(is_batch_valid));
#endif

            if (!is_batch_valid) {
                continue;
            }
        }

        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
#ifdef IN0_RESIDENT
                for (uint32_t block = 0; block < (bw == 0 ? num_blocks_inner_dim : 0u); ++block) {
#else
                for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
#endif
                    dfb_in0.reserve_back(in0_block_num_tiles);

                    receiver.receive();

                    dfb_in0.push_back(in0_block_num_tiles);
                }
            }
        }
    }

    noc.async_atomic_barrier();
#ifdef OUT_WRITER_ON_IN0
    write_output_blocks(noc);
#endif
}
