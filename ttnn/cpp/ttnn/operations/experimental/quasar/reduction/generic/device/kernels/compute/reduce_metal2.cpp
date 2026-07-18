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
#include "api/debug/dprint.h"  // [DIAG avgpool x1.15] remove after

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

    // [DIAG avgpool x1.15 -- remove after] The DECISIVE read: what scaler value did the reader actually
    // pack into the GAPOOL SrcB CB? bf16 1.0 = 16256 (0x3f80) => the scaler=1.0 workaround reached GAPOOL
    // (so a x1.1504 gain would be a GAPOOL-HW issue even at SrcB=1.0 -> LLK/HW ticket). bf16 1/49 = 15527
    // (0x3ca7) => the workaround did NOT propagate; GAPOOL sees the fractional scaler and quantizes it to
    // 3/128 (=x1.1504) -> the fix is app-side (make the reduce truly use 1.0 + external post-mul).
    {
        DataflowBuffer scaler_cb(dfb::scaler);
        scaler_cb.wait_front(1);
        volatile tt_l1_ptr uint16_t* sp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scaler_cb.get_read_ptr());
        UNPACK(DPRINT("SCALER_CB s0={} s1={} s16={}\n", (uint32_t)sp[0], (uint32_t)sp[1], (uint32_t)sp[16]));
    }

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

    // The reduce helper waits on the scaler DFB but never pops it (the single scaler tile is reused for
    // the whole reduction). Pop it here so the DFB is left balanced.
    DataflowBuffer(dfb::scaler).pop_front(1);
}
