// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"
#include "api/dataflow/dataflow_buffer.h"

#if defined(ARCH_BLACKHOLE)
// The copy init of c_0 serves an operand that shares its formats and tile geometry.
template <uint32_t cb>
constexpr bool copy_init_shared_with_c0() {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    constexpr uint32_t c0 = tt::CBIndex::c_0;
    return unpack_src_format[cb] == unpack_src_format[c0] && unpack_dst_format[cb] == unpack_dst_format[c0] &&
           unpack_tile_num_faces[cb] == unpack_tile_num_faces[c0] &&
           unpack_tile_face_r_dim[cb] == unpack_tile_face_r_dim[c0] &&
           unpack_partial_face[cb] == unpack_partial_face[c0] && unpack_narrow_tile[cb] == unpack_narrow_tile[c0] &&
           unpack_tile_r_dim[cb] == unpack_tile_r_dim[c0] && unpack_tile_c_dim[cb] == unpack_tile_c_dim[c0];
#else
    return true;
#endif
}
#endif

#if defined(ARCH_BLACKHOLE)
// With 32-bit operands the inits stay per tile: addcdiv on float32 measured slower without them.
constexpr bool init_once = !DST_ACCUM_MODE;
#else
constexpr bool init_once = false;
#endif

ALWI void process_tile(
    uint32_t cb_in0_id,
    uint32_t cb_in1_id,
    uint32_t cb_in2_id,
    uint32_t cb_out_id,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle,
    uint32_t scalar_arg) {
    using namespace ckernel;
#if defined(ARCH_BLACKHOLE)
    constexpr bool shared_copy_init =
        init_once && copy_init_shared_with_c0<tt::CBIndex::c_1>() && copy_init_shared_with_c0<tt::CBIndex::c_2>();
#else
    constexpr bool shared_copy_init = false;
#endif

    DataflowBuffer dfb_in0(cb_in0_id);
    DataflowBuffer dfb_in1(cb_in1_id);
    DataflowBuffer dfb_in2(cb_in2_id);
    DataflowBuffer dfb_out(cb_out_id);

    // 3-tensor broadcast-aware synchronization - wait for broadcast CBs outside loop
#if BCAST_A
    dfb_in0.wait_front(num_tiles_per_cycle);  // input_a is broadcast
#endif
#if BCAST_B
    dfb_in1.wait_front(num_tiles_per_cycle);  // input_b is broadcast
#endif
#if BCAST_C
    dfb_in2.wait_front(num_tiles_per_cycle);  // input_c is broadcast
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        // Wait for non-broadcast CBs inside loop
#if !BCAST_A
        dfb_in0.wait_front(num_tiles_per_cycle);
#endif
#if !BCAST_B
        dfb_in1.wait_front(num_tiles_per_cycle);
#endif
#if !BCAST_C
        dfb_in2.wait_front(num_tiles_per_cycle);
#endif

        dfb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();

        // Load all three inputs into DST registers
        if constexpr (!shared_copy_init) {
            copy_init(dfb_in0.get_id());
        }
        copy_tile(dfb_in0.get_id(), 0 /*in_tile_index*/, 0 /*dst_tile_index*/);

        if constexpr (!shared_copy_init) {
            copy_init(dfb_in1.get_id());
        }
        copy_tile(dfb_in1.get_id(), 0 /*in_tile_index*/, 1 /*dst_tile_index*/);

        if constexpr (!shared_copy_init) {
            copy_init(dfb_in2.get_id());
        }
        copy_tile(dfb_in2.get_id(), 0 /*in_tile_index*/, 2 /*dst_tile_index*/);

        // Use direct addcmul kernel: computes input_a + scalar_arg * input_b * input_c -> DST[0]
        if constexpr (!init_once) {
            TERNARY_SFPU_OP_INIT();
        }
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg);

        tile_regs_commit();
        tile_regs_wait();

        // Pack the result from DST[0] to output
        pack_tile(0, dfb_out.get_id());

        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);

        // Pop non-broadcast CBs inside loop
#if !BCAST_A
        dfb_in0.pop_front(num_tiles_per_cycle);
#endif
#if !BCAST_B
        dfb_in1.pop_front(num_tiles_per_cycle);
#endif
#if !BCAST_C
        dfb_in2.pop_front(num_tiles_per_cycle);
#endif
    }

    // Pop broadcast CBs outside loop
#if BCAST_A
    dfb_in0.pop_front(num_tiles_per_cycle);
#endif
#if BCAST_B
    dfb_in1.pop_front(num_tiles_per_cycle);
#endif
#if BCAST_C
    dfb_in2.pop_front(num_tiles_per_cycle);
#endif
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_in0_id = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1_id = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2_id = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out_id = tt::CBIndex::c_3;  // output

    compute_kernel_hw_startup(cb_in0_id, cb_out_id);
    copy_init(cb_in0_id);
    if constexpr (init_once) {
        TERNARY_SFPU_OP_INIT();
    }

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(
            cb_in0_id, cb_in1_id, cb_in2_id, cb_out_id, tile_freq, tile_start, num_tiles_per_cycle, scalar_arg);
    }

    if (remaining_iterations > 0) {
        process_tile(
            cb_in0_id,
            cb_in1_id,
            cb_in2_id,
            cb_out_id,
            remaining_iterations,
            tile_start,
            num_tiles_per_cycle,
            scalar_arg);
    }
}
