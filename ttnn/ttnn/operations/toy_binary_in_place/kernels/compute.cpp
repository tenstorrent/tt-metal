// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for toy_binary_in_place.
//
// Single full init (compute_kernel_hw_startup) at kernel start.
// Phase transitions use reconfig (inside the helpers) instead of
// a second full init.
//
// Supports: add(0), sub(1), mul(2), square(3)
// Supports: in_place(1) and normal(0) modes

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"

// Dispatch macro: calls the correct in-place helper based on op_code and bcast_code.
// SQUARE ignores broadcast and B policy.
#define DISPATCH_IN_PLACE(op_code, bcast_code, cb_work, cb_b, shape)                                                \
    do {                                                                                                            \
        if constexpr (op_code == 3) {                                                                               \
            square_in_place(cb_work, shape);                                                                        \
        } else if constexpr (bcast_code == 0) {                                                                     \
            OP_IN_PLACE<BroadcastDim::NONE, BinaryInputPolicy::WaitUpfrontPopAtEnd>(op_code, cb_work, cb_b, shape); \
        } else if constexpr (bcast_code == 1) {                                                                     \
            OP_IN_PLACE<BroadcastDim::ROW, BinaryInputPolicy::WaitUpfrontNoPop>(op_code, cb_work, cb_b, shape);     \
        } else if constexpr (bcast_code == 2) {                                                                     \
            OP_IN_PLACE<BroadcastDim::COL, BinaryInputPolicy::WaitUpfrontPopAtEnd>(op_code, cb_work, cb_b, shape);  \
        } else {                                                                                                    \
            OP_IN_PLACE<BroadcastDim::SCALAR, BinaryInputPolicy::WaitUpfrontNoPop>(op_code, cb_work, cb_b, shape);  \
        }                                                                                                           \
    } while (0)

// Dispatch macro: calls the correct normal (non-in-place) helper.
#define DISPATCH_NORMAL(op_code, bcast_code, cb_input, cb_b, cb_out, shape)                                            \
    do {                                                                                                               \
        if constexpr (op_code == 3) {                                                                                  \
            square(cb_input, cb_out, shape);                                                                           \
        } else if constexpr (bcast_code == 0) {                                                                        \
            OP_NORMAL<BroadcastDim::NONE, BinaryInputPolicy::WaitAndPopPerTile>(                                       \
                op_code, cb_input, cb_b, cb_out, shape);                                                               \
        } else if constexpr (bcast_code == 1) {                                                                        \
            OP_NORMAL<BroadcastDim::ROW, BinaryInputPolicy::WaitUpfrontNoPop>(op_code, cb_input, cb_b, cb_out, shape); \
        } else if constexpr (bcast_code == 2) {                                                                        \
            OP_NORMAL<BroadcastDim::COL, BinaryInputPolicy::WaitAndPopPerTile>(                                        \
                op_code, cb_input, cb_b, cb_out, shape);                                                               \
        } else {                                                                                                       \
            OP_NORMAL<BroadcastDim::SCALAR, BinaryInputPolicy::WaitUpfrontNoPop>(                                      \
                op_code, cb_input, cb_b, cb_out, shape);                                                               \
        }                                                                                                              \
    } while (0)

using namespace compute_kernel_lib;

// op_code: 0=add, 1=sub, 2=mul
template <BroadcastDim bcast_dim, BinaryInputPolicy b_policy, uint32_t op_code>
ALWI void op_in_place_impl(uint32_t cb_work, uint32_t cb_b, BinaryInputBlockShape shape) {
    if constexpr (op_code == 0) {
        add_in_place<bcast_dim, b_policy>(cb_work, cb_b, shape);
    } else if constexpr (op_code == 1) {
        sub_in_place<bcast_dim, b_policy>(cb_work, cb_b, shape);
    } else {
        mul_in_place<bcast_dim, b_policy>(cb_work, cb_b, shape);
    }
}

template <BroadcastDim bcast_dim, BinaryInputPolicy b_policy, uint32_t op_code>
ALWI void op_normal_impl(uint32_t cb_input, uint32_t cb_b, uint32_t cb_out, BinaryInputBlockShape shape) {
    if constexpr (op_code == 0) {
        add<bcast_dim, BinaryInputPolicy::WaitAndPopPerTile, b_policy>(cb_input, cb_b, cb_out, shape);
    } else if constexpr (op_code == 1) {
        sub<bcast_dim, BinaryInputPolicy::WaitAndPopPerTile, b_policy>(cb_input, cb_b, cb_out, shape);
    } else {
        mul<bcast_dim, BinaryInputPolicy::WaitAndPopPerTile, b_policy>(cb_input, cb_b, cb_out, shape);
    }
}

// Workaround: macros that forward constexpr op_code to template parameter
#define OP_IN_PLACE(bcast, bpol, opc, w, b, s)     \
    if constexpr (opc == 0) {                      \
        op_in_place_impl<bcast, bpol, 0>(w, b, s); \
    } else if constexpr (opc == 1) {               \
        op_in_place_impl<bcast, bpol, 1>(w, b, s); \
    } else {                                       \
        op_in_place_impl<bcast, bpol, 2>(w, b, s); \
    }

#define OP_NORMAL(bcast, bpol, opc, i, b, o, s)     \
    if constexpr (opc == 0) {                       \
        op_normal_impl<bcast, bpol, 0>(i, b, o, s); \
    } else if constexpr (opc == 1) {                \
        op_normal_impl<bcast, bpol, 1>(i, b, o, s); \
    } else {                                        \
        op_normal_impl<bcast, bpol, 2>(i, b, o, s); \
    }

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t bcast_code = get_compile_time_arg_val(2);
    constexpr uint32_t in_place_flag = get_compile_time_arg_val(3);
    constexpr uint32_t op_code = get_compile_time_arg_val(4);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_work = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t total_a_tiles = Ht * Wt;
    constexpr auto shape = BinaryInputBlockShape::of(Ht, Wt);

    if constexpr (in_place_flag == 1) {
        // === IN-PLACE MODE ===
        compute_kernel_hw_startup(cb_input, cb_work);

        // Phase 1: Copy A tiles from cb_input → cb_work
        copy_tiles<CopyInputPolicy::WaitAndPop, CopyDataFormatReconfig::NONE>(cb_input, cb_work, total_a_tiles);

        // Phase 2: In-place op on cb_work (reconfig handles format transition)
        if constexpr (op_code == 4) {
            // SFPU SQUARE: unary square via SFPU (copy to DEST, square_tile, pack back)
            // In-place pop-before-pack cycle, same as binary but using SFPU math.
            square_tile_init();
            for (uint32_t i = 0; i < total_a_tiles; ++i) {
                cb_wait_front(cb_work, 1);
                tile_regs_acquire();
                copy_tile(cb_work, 0, 0);
                square_tile(0);
                cb_pop_front(cb_work, 1);
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_work, 1);
                pack_tile(0, cb_work);
                cb_push_back(cb_work, 1);
                tile_regs_release();
            }
        } else if constexpr (op_code == 3) {
            // FPU SQUARE: cb_work = cb_work * cb_work (binary MUL with same operand)
            square_in_place(cb_work, shape);
        } else if constexpr (bcast_code == 0) {
            OP_IN_PLACE(BroadcastDim::NONE, BinaryInputPolicy::WaitUpfrontPopAtEnd, op_code, cb_work, cb_b, shape)
        } else if constexpr (bcast_code == 1) {
            OP_IN_PLACE(BroadcastDim::ROW, BinaryInputPolicy::WaitUpfrontNoPop, op_code, cb_work, cb_b, shape)
        } else if constexpr (bcast_code == 2) {
            OP_IN_PLACE(BroadcastDim::COL, BinaryInputPolicy::WaitUpfrontPopAtEnd, op_code, cb_work, cb_b, shape)
        } else {
            OP_IN_PLACE(BroadcastDim::SCALAR, BinaryInputPolicy::WaitUpfrontNoPop, op_code, cb_work, cb_b, shape)
        }

        // Phase 3: Copy modified tiles from cb_work → cb_out
        copy_tile_to_dst_init_short(cb_work);
        copy_tiles<CopyInputPolicy::WaitAndPop, CopyDataFormatReconfig::OUTPUT>(cb_work, cb_out, total_a_tiles);

    } else {
        // === NORMAL (NON-IN-PLACE) MODE ===
        compute_kernel_hw_startup(cb_input, cb_b, cb_out);

        if constexpr (op_code == 4) {
            // SFPU SQUARE (non-in-place): copy to DEST, square_tile, pack to cb_out
            square_tile_init();
            copy_tiles<CopyInputPolicy::WaitAndPop, CopyDataFormatReconfig::NONE>(
                cb_input, cb_out, total_a_tiles, [](uint32_t dst_idx) { square_tile(dst_idx); });
        } else if constexpr (op_code == 3) {
            // FPU SQUARE: cb_out = cb_input * cb_input
            square(cb_input, cb_out, shape);
        } else if constexpr (bcast_code == 0) {
            OP_NORMAL(BroadcastDim::NONE, BinaryInputPolicy::WaitAndPopPerTile, op_code, cb_input, cb_b, cb_out, shape)
        } else if constexpr (bcast_code == 1) {
            OP_NORMAL(BroadcastDim::ROW, BinaryInputPolicy::WaitUpfrontNoPop, op_code, cb_input, cb_b, cb_out, shape)
        } else if constexpr (bcast_code == 2) {
            OP_NORMAL(BroadcastDim::COL, BinaryInputPolicy::WaitAndPopPerTile, op_code, cb_input, cb_b, cb_out, shape)
        } else {
            OP_NORMAL(BroadcastDim::SCALAR, BinaryInputPolicy::WaitUpfrontNoPop, op_code, cb_input, cb_b, cb_out, shape)
        }
    }
}
