// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"

constexpr uint32_t cb_e = get_compile_time_arg_val(0);
constexpr uint32_t cb_b = get_compile_time_arg_val(1);
constexpr uint32_t cb_ein = get_compile_time_arg_val(2);
constexpr uint32_t cb_eout = get_compile_time_arg_val(3);
constexpr uint32_t kt = get_compile_time_arg_val(4);
// The same landed entry state as cb_ein, unpacked straight to DST (FP32, exact).
constexpr uint32_t cb_ein_exact = get_compile_time_arg_val(5);
// One BF16 identity tile: B enters the sum as I @ B.
constexpr uint32_t cb_identity = get_compile_time_arg_val(6);
// The final carry (see the reader): FP32 E and B of the final transition, and the relayed final state P in its matmul
// and exact views.
constexpr uint32_t cb_fe = get_compile_time_arg_val(7);
constexpr uint32_t cb_fb = get_compile_time_arg_val(8);
constexpr uint32_t cb_p = get_compile_time_arg_val(9);
constexpr uint32_t cb_p_exact = get_compile_time_arg_val(10);
// The reader's {holder, split, chunks, group chunks}.
constexpr uint32_t cb_ctl = get_compile_time_arg_val(11);
// The entry correction: the seeds, the chunk terms, their differences and the chunk outputs.
constexpr uint32_t cb_seed = get_compile_time_arg_val(12);
constexpr uint32_t cb_tail = get_compile_time_arg_val(13);
constexpr uint32_t cb_terms = get_compile_time_arg_val(14);
constexpr uint32_t cb_diff = get_compile_time_arg_val(15);
constexpr uint32_t cb_out = get_compile_time_arg_val(16);
constexpr uint32_t core_chunks = get_compile_time_arg_val(17);
constexpr bool streamed = get_compile_time_arg_val(18);
// The cores are heads x cores_per_head consecutive cores of the compute grid in row-major order, from first_core,
// head by head.
constexpr uint32_t cores_per_head = get_compile_time_arg_val(19);
constexpr uint32_t grid_x = get_compile_time_arg_val(20);
constexpr uint32_t first_core = get_compile_time_arg_val(21);
constexpr uint32_t state_tiles = kt * kt;
constexpr uint32_t dst_tiles =
    ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
// A tile column of the next state is formed in one DST pass (the op checks K against the DST).
static_assert(kt <= dst_tiles);
// The carry pass forms `carry_rows` tiles of a column per DST pass, with P's tiles beside them.
constexpr uint32_t carry_rows = kda::largest_divisor_at_most(kt, dst_tiles / 2);
// The terms buffer holds a chunk row of F(0) at tile row, and of F(I) at fi_offset further: in place, the core's whole
// shard (core_chunks rows of F(0), then of F(I)); streamed, one chunk at a time.
constexpr uint32_t fi_offset = (streamed ? 1 : core_chunks) * kt;
constexpr uint32_t terms_tiles = 2 * fi_offset;

// The relay's hop on the head core: the next entry state S_out = A S_in + B for one head, from the complement form
// E = A - I: S_out = S_in + E S_in + B, summed in the FP32 DST. The exact copy of S_in unpacks to DST losslessly; E
// S_in accumulates on top at the affine prefix's fidelity, so the matmul's source-register copy of S_in reaches only
// the small E S_in term; B accumulates as I @ B, exact at HiFi2 and above (a matmul's in1 goes to srcA, whose HiFi2
// phases cover a BF16 mantissa, and 1.0 is exact in srcB). Per output tile column (column-major state): column j needs
// only column j of S_in, so it is computed as soon as that column lands.
void relay_hop() {
    pack_reconfig_data_format(cb_eout);
    cb_wait_front(cb_e, state_tiles);
    cb_wait_front(cb_b, state_tiles);
    cb_wait_front(cb_identity, 1);
    for (uint32_t j = 0; j < kt; ++j) {
        cb_reserve_back(cb_eout, kt);
        cb_wait_front(cb_ein, (j + 1) * kt);
        cb_wait_front(cb_ein_exact, (j + 1) * kt);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_ein_exact);
        copy_init(cb_ein_exact);
        for (uint32_t r = 0; r < kt; ++r) {
            copy_tile(cb_ein_exact, j * kt + r, r);
        }
        reconfig_data_format<SrcOrder::Reverse>(cb_e, cb_ein);
        matmul_block_init(cb_e, cb_ein, false, 1, kt, kt);
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_block(cb_e, cb_ein, k, j * kt + k, 0, false, 1, kt, kt);
        }
        reconfig_data_format<SrcOrder::Reverse>(cb_identity, cb_b);
        matmul_block_init(cb_identity, cb_b, false, 1, 1, 1);
        for (uint32_t r = 0; r < kt; ++r) {
            matmul_block(cb_identity, cb_b, 0, r * kt + j, r, false, 1, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t r = 0; r < kt; ++r) {
            pack_tile(r, cb_eout);
        }
        tile_regs_release();
        cb_push_back(cb_eout, kt);
    }
    cb_pop_front(cb_e, state_tiles);
    cb_pop_front(cb_ein, state_tiles);
    cb_pop_front(cb_ein_exact, state_tiles);
    cb_pop_front(cb_b, state_tiles);
    cb_pop_front(cb_identity, 1);
}

// The final carry on the first rank: P + (E P + B) per column with the final transition when the interval ends in a
// separated tail, else P itself. B and P unpack to DST exactly and are added on the SFPU, E P accumulates on top.
void carry_pass(bool split) {
    pack_reconfig_data_format(cb_eout);
    if (split) {
        cb_wait_front(cb_fe, state_tiles);
        cb_wait_front(cb_fb, state_tiles);
    }
    for (uint32_t j = 0; j < kt; ++j) {
        cb_reserve_back(cb_eout, kt);
        cb_wait_front(cb_p, (j + 1) * kt);
        cb_wait_front(cb_p_exact, (j + 1) * kt);
        for (uint32_t r0 = 0; r0 < kt; r0 += carry_rows) {
            tile_regs_acquire();
            if (split) {
                reconfig_data_format_srca(cb_fb);
                copy_init(cb_fb);
                for (uint32_t i = 0; i < carry_rows; ++i) {
                    copy_tile(cb_fb, (r0 + i) * kt + j, i);
                }
                reconfig_data_format_srca(cb_p_exact);
                copy_init(cb_p_exact);
                for (uint32_t i = 0; i < carry_rows; ++i) {
                    copy_tile(cb_p_exact, j * kt + r0 + i, carry_rows + i);
                }
                add_binary_tile_init();
                for (uint32_t i = 0; i < carry_rows; ++i) {
                    add_binary_tile(i, carry_rows + i, i);
                }
                reconfig_data_format<SrcOrder::Reverse>(cb_fe, cb_p);
                matmul_block_init(cb_fe, cb_p, false, 1, carry_rows, kt);
                for (uint32_t k = 0; k < kt; ++k) {
                    matmul_block(cb_fe, cb_p, r0 * kt + k, j * kt + k, 0, false, 1, carry_rows, kt);
                }
            } else {
                reconfig_data_format_srca(cb_p_exact);
                copy_init(cb_p_exact);
                for (uint32_t i = 0; i < carry_rows; ++i) {
                    copy_tile(cb_p_exact, j * kt + r0 + i, i);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < carry_rows; ++i) {
                pack_tile(i, cb_eout);
            }
            tile_regs_release();
        }
        cb_push_back(cb_eout, kt);
    }
    cb_pop_front(cb_p, state_tiles);
    cb_pop_front(cb_p_exact, state_tiles);
    if (split) {
        cb_pop_front(cb_fe, state_tiles);
        cb_pop_front(cb_fb, state_tiles);
    }
}

// FI - F0 for tile rows [first, first + tiles) of the terms buffer, packed to the same rows of the differences.
void subtract(uint32_t first, uint32_t tiles) {
    pack_reconfig_data_format(cb_diff);
    reconfig_data_format(cb_terms, cb_terms);
    sub_init(cb_terms, cb_terms);
    for (uint32_t t = first; t < first + tiles; t += kt) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < kt; ++i) {
            sub_tiles(cb_terms, cb_terms, fi_offset + t + i, t + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < kt; ++i) {
            pack_tile(i, cb_diff);
        }
        tile_regs_release();
    }
}

// Chunk outputs F0_c + (FI_c - F0_c) @ seed for this core's chunks [begin, end), accumulated in the FP32 DST on top of
// the copied F0. The seed is column-major by tile and lands a tile column at a time: output column j needs only seed
// column j, so the columns are formed as they land, up to dst_tiles chunks per pass, and packed into their rows. In
// place the differences are already formed; streamed, each chunk's is formed just before its products.
void correct(uint32_t begin, uint32_t end, uint32_t seed) {
    const uint32_t block = streamed ? 1 : end - begin;
    for (uint32_t first = begin; first < end; first += block) {
        if constexpr (streamed) {
            cb_wait_front(cb_terms, terms_tiles);
            cb_reserve_back(cb_diff, kt);
            subtract(0, kt);
            cb_push_back(cb_diff, kt);
            cb_wait_front(cb_diff, kt);
        }
        pack_reconfig_data_format(cb_out);
        cb_reserve_back(cb_out, block * kt);
        for (uint32_t j = 0; j < kt; ++j) {
            cb_wait_front(seed, (j + 1) * kt);
            for (uint32_t c = 0; c < block; c += dst_tiles) {
                const uint32_t rows = block - c < dst_tiles ? block - c : dst_tiles;
                const uint32_t row = streamed ? 0 : (first + c) * kt;
                tile_regs_acquire();
                reconfig_data_format_srca(cb_terms);
                copy_init(cb_terms);
                for (uint32_t i = 0; i < rows; ++i) {
                    copy_tile(cb_terms, row + i * kt + j, i);
                }
                reconfig_data_format<SrcOrder::Reverse>(cb_diff, seed);
                matmul_block_init(cb_diff, seed, false, 1, rows, kt);
                for (uint32_t k = 0; k < kt; ++k) {
                    matmul_block(cb_diff, seed, row + k, j * kt + k, 0, false, 1, rows, kt);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < rows; ++i) {
                    pack_tile<true>(i, cb_out, (c + i) * kt + j);
                }
                tile_regs_release();
            }
        }
        cb_push_back(cb_out, block * kt);
        if constexpr (streamed) {
            cb_pop_front(cb_terms, terms_tiles);
            cb_pop_front(cb_diff, kt);
        }
    }
}

// Every core of a head corrects its chunks; the head's first core also runs the relay's hop first and, on the first
// rank, the final carry between the chunks that start from the entry state and those of the separated tail, which
// start from P.
void kernel_main() {
    const bool relay_head =
        (get_absolute_logical_y() * grid_x + get_absolute_logical_x() - first_core) % cores_per_head == 0;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_terms, cb_seed, cb_out);
    cb_wait_front(cb_ctl, 1);
    const uint32_t holder = read_tile_value(cb_ctl, 0, 0);
    const uint32_t split = read_tile_value(cb_ctl, 0, 1);
    const uint32_t chunks = read_tile_value(cb_ctl, 0, 2);
    const uint32_t group_chunks = read_tile_value(cb_ctl, 0, 3);
    cb_pop_front(cb_ctl, 1);

    if (relay_head) {
        relay_hop();
    }
    // In place, the differences for all of this core's chunks first: they need no seed.
    if (!streamed && chunks != 0) {
        cb_wait_front(cb_terms, terms_tiles);
        cb_reserve_back(cb_diff, chunks * kt);
        subtract(0, chunks * kt);
        cb_push_back(cb_diff, chunks * kt);
        cb_wait_front(cb_diff, chunks * kt);
    }
    correct(0, group_chunks, cb_seed);
    if (relay_head && holder) {
        carry_pass(split);
    }
    correct(group_chunks, chunks, cb_tail);
    if (group_chunks != 0) {
        cb_pop_front(cb_seed, state_tiles);
    }
    if (chunks != group_chunks) {
        cb_pop_front(cb_tail, state_tiles);
    }
    if (!streamed && chunks != 0) {
        cb_pop_front(cb_terms, terms_tiles);
        cb_pop_front(cb_diff, chunks * kt);
    }
}
