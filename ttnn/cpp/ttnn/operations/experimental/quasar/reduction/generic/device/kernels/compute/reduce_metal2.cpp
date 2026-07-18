// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reduce.cpp. Identical compute; CB indices → dfb:: bindings (in / scaler / out, no
// self-loop), CTAs → named args. The legacy reduce.cpp is retained for not-yet-ported reduce factories.

#include <cstdint>
#include "api/compute/cb_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "experimental/kernel_args.h"
#include "api/debug/dprint.h"        // [DIAG avgpool x1.15] remove after
#include "api/debug/dprint_pages.h"  // [DIAG avgpool x1.15] print_full_tile — remove after

void kernel_main() {
    uint32_t Ht = get_arg(args::Ht);
    uint32_t Wt = get_arg(args::Wt);
    uint32_t NC = get_arg(args::NC);

    // [DIAG avgpool x1.15 -- remove after] Does the COMPILED kernel actually have REDUCE_POST_MUL +
    // the right post_mul bits? JIT kernel, so this shows on rerun with no ninja. If RPM_OFF prints
    // (or bits != 0x3ca72f05) while the host factory said use_post_mul=true, the compiled kernel is
    // stale/wrong (hash/cache). If RPM_ON with correct bits but output is still x1.15, the bug is in
    // the reduce/GAPOOL sum, not the post-mul.
#ifdef REDUCE_POST_MUL
    UNPACK(DPRINT("RPM_ON post_mul_bits={}\n", (uint32_t)get_arg(args::post_mul_scaler_bits)));
#else
    UNPACK(DPRINT("RPM_OFF\n"));
#endif

    compute_kernel_hw_startup(dfb::in, dfb::scaler, dfb::out);

    compute_kernel_lib::reduce<
        REDUCE_OP,
        REDUCE_DIM,
        dfb::in,
        dfb::scaler,
        dfb::out,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
#ifdef REDUCE_POST_MUL
        [](uint32_t dst_idx) {
            constexpr uint32_t post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
            constexpr DataFormat reduce_format = static_cast<DataFormat>(unpack_src_format[dfb::in]);
            compute_kernel_lib::detail::reduce_post_mul_tile<reduce_format>(dst_idx, post_mul_scaler_bits);
        }
#else
        compute_kernel_lib::NoOp{}
#endif
    );

    // [DIAG avgpool x1.15 -- remove after] Dump the ACTUAL GAPOOL SrcB scaler tile the reduce just
    // consumed (proper untilized readout; no wait_front, the reduce already has it front-valid -> no hang).
    // The scaler lives in row 0 of each face. If those read ~1.0 => GAPOOL saw scaler=1.0 and the x1.1504
    // is a GAPOOL-HW gain (mechanism B, LLK/HW). If ~0.0204 (1/49) => GAPOOL saw the fractional scaler and
    // quantized it to 3/128 (mechanism A). Also dumps the reduced output tile for context.
#if defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)
    PACK(DPRINT("SCALER_TILE (GAPOOL SrcB):\n"));
    PACK(tt::compute::common::print_full_tile(dfb::scaler, 0, true));
    PACK(DPRINT("OUT_TILE (reduced result):\n"));
    PACK(tt::compute::common::print_full_tile(dfb::out, 0, true));
#endif

    // The reduce helper waits on the scaler DFB but never pops it (the single scaler tile is reused for
    // the whole reduction). Pop it here so the DFB is left balanced.
    DataflowBuffer(dfb::scaler).pop_front(1);
}
