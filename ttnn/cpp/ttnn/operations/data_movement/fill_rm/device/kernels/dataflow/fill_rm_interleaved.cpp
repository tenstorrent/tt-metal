// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
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
    Scratchpad<uint16_t> in0(scratch::in0);
    Scratchpad<uint16_t> in1(scratch::in1);

    std::uint32_t w;
    for (w = 0; w < fillW; w++) {
        in0[w] = val_hi;
    }
    for (w = fillW; w < W; w++) {
        in0[w] = val_lo;
    }
    for (w = 0; w < W; w++) {
        in1[w] = val_lo;
    }

    Noc noc;
    std::uint32_t nch_dst = 0;
    // input is NCH(Wt*32) unpadded RM
    for (std::uint32_t nc = 0; nc < NC; nc++) {
        for (std::uint32_t h = 0; h < H; h++) {
            if (h < fillH) {
                noc.async_write(
                    in0, s0, (W << 1), {.offset_bytes = 0}, {.page_id = nch_dst});  // TODO(AP): segment this write
            } else {
                noc.async_write(
                    in1, s0, (W << 1), {.offset_bytes = 0}, {.page_id = nch_dst});  // TODO(AP): segment this write
            }
            noc.async_write_barrier();
            nch_dst++;
        }  // h<paddedH
    }  // nc
}
