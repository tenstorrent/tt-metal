// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Kernel args
    // This kernel accepts a RM row-interleaved tensor laid out as NC,H,(Wt*32)-RM
    // H should be < 32 at the moment
    // It will write out a tensor NC,32,Wt*32

    // Note: this kernel is written with maximum simplicity in mind and (deliberately) doesn't pursue performance

    std::uint32_t NC = get_arg(args::NC);
    std::uint32_t H = get_arg(args::H);
    std::uint32_t W = get_arg(args::W);
    std::uint32_t fillH = get_arg(args::fillH);
    std::uint32_t fillW = get_arg(args::fillW);
    std::uint32_t val_hi = get_arg(args::val_hi);
    std::uint32_t val_lo = get_arg(args::val_lo);

    const auto s0 = TensorAccessor(tensor::out);

    // DPRINT("fill_rm_8bank: NC={} H={} W={} fillH={} fillW={}\n", NC, H, W, fillH, fillW);
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    dfb_in0.reserve_back(16);
    dfb_in1.reserve_back(16);
    std::uint32_t l1_w_addr = dfb_in0.get_write_ptr();
    std::uint32_t l1_zeros_addr = dfb_in1.get_write_ptr();
    std::uint32_t w;
    for (w = 0; w < fillW; w++) {
        reinterpret_cast<std::uint16_t*>(l1_w_addr)[w] = val_hi;
    }
    for (w = fillW; w < W; w++) {
        reinterpret_cast<std::uint16_t*>(l1_w_addr)[w] = val_lo;
    }
    for (w = 0; w < W; w++) {
        reinterpret_cast<std::uint16_t*>(l1_zeros_addr)[w] = val_lo;
    }
    dfb_in0.push_back(16);
    dfb_in1.push_back(16);

    Noc noc;
    std::uint32_t nch_dst = 0;
    // input is NCH(Wt*32) unpadded RM
    for (std::uint32_t nc = 0; nc < NC; nc++) {
        for (std::uint32_t h = 0; h < H; h++) {
            if (h < fillH) {
                noc.async_write(
                    dfb_in0, s0, (W << 1), {.offset_bytes = 0}, {.page_id = nch_dst});  // TODO(AP): segment this write
            } else {
                noc.async_write(
                    dfb_in1, s0, (W << 1), {.offset_bytes = 0}, {.page_id = nch_dst});  // TODO(AP): segment this write
            }
            noc.async_write_barrier();
            nch_dst++;
        }  // h<paddedH
    }  // nc
}
