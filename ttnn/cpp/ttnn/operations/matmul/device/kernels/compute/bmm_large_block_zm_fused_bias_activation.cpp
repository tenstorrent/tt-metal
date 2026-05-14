// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "experimental/circular_buffer.h"
#include "internal/mod_div_lib.h"
#include "api/debug/dprint.h"
#ifdef CHLKC_UNPACK
#include "llk_operands.h"
#endif

#ifdef FUSE_BIAS
#include "api/compute/bcast.h"
#endif

#include "api/compute/eltwise_unary/sfpu_split_includes.h"

#include "sanitizer/impl.h"

void kernel_main() {
    llk::san::State<uint32_t> math_fmt_A = 3;
    llk::san::State<uint32_t> math_fmt_AB = 4;

    DEVICE_PRINT("1\n");


    operand_error_assert(
        math_fmt_A,
        math_fmt_AB,
        CTSTR("crkni"),
        llk::san::sanitizer->context.pack.configure_pack,
        llk::san::sanitizer->context.pack.current);


    DEVICE_PRINT("2\n");


}
