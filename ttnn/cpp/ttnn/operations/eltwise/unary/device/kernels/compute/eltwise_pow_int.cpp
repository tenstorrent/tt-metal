// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/bitwise.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t exponent = get_arg(args::exponent);
    constexpr auto data_format = static_cast<DataFormat>(get_arg(args::data_format));
    const uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr bool is_uint16 = data_format == DataFormat::UInt16;
    // UInt16 is widened to UInt32 in the 32-bit Dest, multiplied there, and truncated back on the way out.
    constexpr DataFormat math_format = is_uint16 ? DataFormat::UInt32 : data_format;
    constexpr auto uint16_id = static_cast<uint32_t>(DataFormat::UInt16);
    constexpr auto uint32_id = static_cast<uint32_t>(DataFormat::UInt32);
    constexpr uint32_t uint16_mask = 0xFFFF;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst_base = 0;
    // x^1 is x itself, so it is packed straight from dst_base.
    constexpr uint32_t dst_result = exponent == 1 ? dst_base : 1;
    // Bit below the leading one; the leading one itself is consumed by seeding the result with x.
    // TODO: use std::countl_zero when C++20 becomes available
    constexpr int first_bit = exponent == 0 ? -2 : 30 - __builtin_clz(exponent);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    copy_init(dfb::in);
    // UInt16 interleaves several SFPU ops whose inits clobber each other, so it re-inits per tile instead.
    if constexpr (!is_uint16) {
        if constexpr (exponent == 0) {
            fill_tile_init();
        } else {
            mul_int_tile_init<math_format>();
        }
    }

    for (uint32_t t = 0; t < num_tiles; ++t) {
        dfb_in.wait_front(onetile);
        tile_regs_acquire();

        if constexpr (exponent == 0) {
            if constexpr (is_uint16) {
                fill_tile_init();
            }
            fill_tile_int<math_format>(dst_result, 1u);
        } else {
            copy_tile(dfb::in, 0, dst_base);
            if constexpr (is_uint16) {
                typecast_tile_init<uint16_id, uint32_id>();
                typecast_tile<uint16_id, uint32_id>(dst_base);
                mul_int_tile_init<math_format>();
            }
            // Left-to-right square-and-multiply; products wrap modulo 2^32 like torch integer pow.
            for (int bit = first_bit; bit >= 0; --bit) {
                const uint32_t to_square = bit == first_bit ? dst_base : dst_result;
                mul_int_tile<math_format>(to_square, to_square, dst_result);
                if ((exponent >> bit) & 1u) {
                    mul_int_tile<math_format>(dst_result, dst_base, dst_result);
                }
            }
        }

        if constexpr (is_uint16) {
            // Mask first: the UInt32 -> UInt16 typecast saturates rather than truncates.
            bitwise_and_tile_init();
            bitwise_and_tile<math_format>(dst_result, uint16_mask);
            typecast_tile_init<uint32_id, uint16_id>();
            typecast_tile<uint32_id, uint16_id>(dst_result);
        }

        tile_regs_commit();
        dfb_in.pop_front(onetile);

        dfb_out.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst_result, dfb::out);
        tile_regs_release();
        dfb_out.push_back(onetile);
    }
}
