// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows, compute of one core of one expert group. Per job (one expert, at most 32 routed rows per row tile):
//  gate/up projection: for each of this core's W0/W1 column groups, [g0 u0 g1 u1] = x . [W0 j0, W1 j0, W0 j1, W1 j1]
//      over all hidden tiles, then the activation on the packer's SFPU, act(g0) u0 and act(g1) u1 -> cb_activated; a
//      half block-column (the last group of a ring position with an odd column count in the compact layout) is
//      [g0 u0] = x . [W0 j0, W1 j0] -> act(g0) u0.
//  down projection: the job's intermediate (every column, from the exchange in rows_writer.cpp) . W2 for each of this
//      core's 4-tile output groups (2 tiles for the half-width last a2a iteration), the W2 rows in moe_compute's
//      per-core rotated order -> cb_rows (4 bf16 tiles; the writer sends the real rows).
// Each output tile sums the same K rows in the same order as the ring program (compute.cpp), at ct_dim 2 where it
// uses 2. Job order: gate/up(0), gate/up(1), down(0), gate/up(2), down(1), ..., down(jobs - 1).
// A job of several row tiles (row_tiles > 1) streams its weights once for all its row tiles: the gate/up projection
// walks x in chunks of chunk_tiles hidden tiles (chunk-major weights) and the down projection walks each output group's
// W2 rows in chunks of w2_chunk_rows; between chunks each (group, row tile) leaves its DEST tiles in cb_partials (FP32
// with FP32 DEST) and the next chunk loads them back before it accumulates, so every output tile is summed over K in
// the same order as with one row tile per job. With a single chunk (chunk_tiles covers x, w2_chunk_rows the W2 rows;
// always with row_tiles = 1) nothing is spilled.
#include <cstdint>
#include "../moe_ring_common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/fill.h"
#include "../moe_activation.h"

namespace {

using ttnn::experimental::prim::detail::MoEActivationFunction;

constexpr uint32_t block_tiles = get_named_compile_time_arg_val("block_tiles");
constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
constexpr uint32_t intermediate_tiles = get_named_compile_time_arg_val("intermediate_tiles");
constexpr uint32_t ring_cores = get_named_compile_time_arg_val("ring_cores");
constexpr bool has_bias = get_named_compile_time_arg_val("has_bias") == 1;
constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
constexpr uint32_t cb_activated = get_named_compile_time_arg_val("cb_activated");  // this core's activated columns
constexpr uint32_t cb_intermediate = get_named_compile_time_arg_val("cb_intermediate");  // the job's whole intermediate
constexpr uint32_t cb_rows = get_named_compile_time_arg_val("cb_rows");
constexpr uint32_t cb_ctl = get_named_compile_time_arg_val("cb_ctl");
constexpr uint32_t cb_ones = get_named_compile_time_arg_val("cb_ones");
constexpr uint32_t cb_partials = get_named_compile_time_arg_val("cb_partials");
constexpr uint32_t activated_tiles = get_named_compile_time_arg_val("activated_tiles");  // per row tile
constexpr uint32_t row_tiles = get_named_compile_time_arg_val("row_tiles");
constexpr uint32_t chunk_tiles = get_named_compile_time_arg_val("chunk_tiles");
constexpr uint32_t chunk_blocks = get_named_compile_time_arg_val("chunk_blocks");
constexpr uint32_t w2_chunk_rows = get_named_compile_time_arg_val("w2_chunk_rows");
constexpr auto activation_function = MoEActivationFunction(get_named_compile_time_arg_val("activation_function"));
static_assert(block_tiles % 4 == 0, "a block holds whole 4-tile K rows");

constexpr uint32_t gate_up_chunks = (hidden_tiles + chunk_tiles - 1) / chunk_tiles;
constexpr uint32_t down_chunks = (intermediate_tiles + w2_chunk_rows - 1) / w2_chunk_rows;
constexpr uint32_t unit_tiles = chunk_blocks * block_tiles;  // weight pages of one chunk unit (rows_reader.cpp)
// a one-row-tile job holds nothing for other row tiles: x chunks as large as its slot, its weights streamed in whole
// runs, its pass padded to whole units when held units exist (row_tiles > 1, rows_reader.cpp)
constexpr uint32_t single_chunk_tiles = chunk_tiles * row_tiles < hidden_tiles ? chunk_tiles * row_tiles : hidden_tiles;
constexpr uint32_t single_chunks = (hidden_tiles + single_chunk_tiles - 1) / single_chunk_tiles;
constexpr uint32_t unit_blocks = row_tiles > 1 ? chunk_blocks : 1;
// cb_partials holds one 4-tile slot per (group, row tile) whatever the group's width (a half group fills 2 of them):
// every reserve then starts at a multiple of 4 tiles of the CB (4 row_tiles groups tiles) and none runs past its end
constexpr uint32_t partial_slot_tiles = 4;

// intermediate columns of each ring position, and the first of them
constexpr auto shard_columns = moe_ring::make_shard_lut<intermediate_tiles, ring_cores>();
constexpr moe_ring::ShardLUT<ring_cores> make_first_column_lut() {
    moe_ring::ShardLUT<ring_cores> first{};
    uint32_t column = 0;
    for (uint32_t s = 0; s < ring_cores; ++s) {
        first.data[s] = column;
        column += moe_ring::shard_tiles(intermediate_tiles, s, ring_cores);
    }
    return first;
}
constexpr auto shard_first_column = make_first_column_lut();

// This core's W0/W1 column groups and W2 output groups (runtime args); the last of each may be half wide.
struct CoreGroups {
    uint32_t column_groups;
    uint32_t output_groups;
    uint32_t ring_position;  // of this core's bank: the W2 row rotation
    bool half_column;        // the last column group is a half block-column
    bool half_output;        // the last output group is the half-width last a2a iteration
    // tiles per K row (matmul ct_dim) of column group u and of output group q: 4, or 2 for a half group
    FORCE_INLINE uint32_t column_group_width(uint32_t u) const { return half_column && u + 1 == column_groups ? 2 : 4; }
    FORCE_INLINE uint32_t output_group_width(uint32_t q) const { return half_output && q + 1 == output_groups ? 2 : 4; }
};

// Weight tiles of one pass arrive in blocks of block_tiles; a run (one group) ends with a short block.
struct WeightStream {
    uint32_t left = 0, cur = 0;
    bool held = false;
    FORCE_INLINE uint32_t take(uint32_t n) {
        if (left == 0) {
            if (held) {
                cb_pop_front(cb_w, block_tiles);
            }
            cb_wait_front(cb_w, block_tiles);
            held = true;
            left = block_tiles;
            cur = 0;
        }
        const uint32_t i = cur;
        cur += n;
        left -= n;
        return i;
    }
    FORCE_INLINE void end_run() {
        if (held) {
            cb_pop_front(cb_w, block_tiles);
        }
        held = false;
        left = 0;
    }
};

// The packer's output format: the partials, or the bf16 outputs (cb_activated and cb_rows share a format).
struct PackTarget {
    bool partials = false;
    FORCE_INLINE void select(bool to_partials) {
        if (to_partials != partials) {
            if (to_partials) {
                pack_reconfig_data_format(cb_activated, cb_partials);
            } else {
                pack_reconfig_data_format(cb_partials, cb_activated);
            }
            partials = to_partials;
        }
    }
};

// The W2 rows of an output group in moe_compute's order (prepare_w2): this ring position's intermediate columns first,
// then the previous ring positions'. next() is the intermediate column of the next W2 row.
struct IntermediateColumnWalk {
    uint32_t position, column, left;
    explicit IntermediateColumnWalk(uint32_t ring_position) :
        position(ring_position), column(shard_first_column[ring_position]), left(shard_columns[ring_position]) {}
    FORCE_INLINE uint32_t next() {
        while (left == 0) {
            position = position == 0 ? ring_cores - 1 : position - 1;
            left = shard_columns[position];
            column = shard_first_column[position];
        }
        --left;
        return column++;
    }
};

// ones tile: matmul(ones, bias row) adds the bias to every row
FORCE_INLINE void fill_ones_tile() {
    pack_reconfig_data_format(cb_ones);
    copy_init(cb_ones);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 1.f);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_ones, 1);
    pack_tile(0, cb_ones);
    tile_regs_release();
    cb_push_back(cb_ones, 1);
    cb_wait_front(cb_ones, 1);
}

// DEST 0..width-1 <- this (group, row tile)'s partials of the previous chunk, in the order they were packed; then the
// matmul of a group `width` tiles wide
FORCE_INLINE void reload_partials(uint32_t cb_in0, uint32_t width) {
    reconfig_data_format_srca(cb_w, cb_partials);
    copy_init(cb_partials);
    cb_wait_front(cb_partials, partial_slot_tiles);
    for (uint32_t t = 0; t < width; ++t) {
        copy_tile(cb_partials, t, t);
    }
    cb_pop_front(cb_partials, partial_slot_tiles);
    reconfig_data_format_srca(cb_partials, cb_w);
    matmul_block_init(cb_in0, cb_w, /*transpose=*/false, /*ct_dim=*/width, /*rt_dim=*/1, /*kt_dim=*/1);
}

FORCE_INLINE void pack_partials(uint32_t width) {
    tile_regs_wait();
    cb_reserve_back(cb_partials, partial_slot_tiles);
    for (uint32_t t = 0; t < width; ++t) {
        pack_tile<true>(t, cb_partials, t);
    }
    cb_push_back(cb_partials, partial_slot_tiles);
}

// The activation of a group `width` tiles wide on the packer's SFPU (DEST 0..width-1), its activated tiles ->
// cb_activated [first_tile, first_tile + width / 2) (as compute.cpp's compute_w0_w1_block_column: wait for math plus a
// CFG stall before the SFPU's DEST offset)
FORCE_INLINE void pack_activation(uint32_t width, uint32_t first_tile) {
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    if (width == 4) {
        ::moe_activation::PackActivation<activation_function, /*kPairs=*/2>::compute();
    } else {
        ::moe_activation::PackActivation<activation_function, /*kPairs=*/1>::compute();
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    pack_tile<true>(0, cb_activated, first_tile);
    if (width == 4) {
        pack_tile<true>(2, cb_activated, first_tile + 1);
    }
}

// An output group's 4 bf16 tiles of one row tile -> cb_rows.
FORCE_INLINE void pack_output_rows() {
    tile_regs_wait();
    cb_reserve_back(cb_rows, 4);
    for (uint32_t t = 0; t < 4; ++t) {
        pack_tile<true>(t, cb_rows, t);
    }
    cb_push_back(cb_rows, 4);
}

// The weight blocks that pad a one-row-tile pass to whole units (row_tiles > 1, rows_reader.cpp).
FORCE_INLINE void pop_unit_padding(uint32_t blocks) {
    const uint32_t pad = (unit_blocks - blocks % unit_blocks) % unit_blocks;
    if (pad) {
        cb_wait_front(cb_w, pad * block_tiles);
        cb_pop_front(cb_w, pad * block_tiles);
    }
}

FORCE_INLINE void init_gate_up_projection() {
    // GELU initializes in PackActivation, per group
    if constexpr (activation_function != MoEActivationFunction::GELU) {
        ::moe_activation::pack_init_activation<activation_function>();
    }
    matmul_block_init(cb_x, cb_w, /*transpose=*/false, /*ct_dim=*/4, /*rt_dim=*/1, /*kt_dim=*/1);
}

// Gate/up projection and activation of a job of one row tile: x in chunks of single_chunk_tiles, each column group's
// weights in whole runs from the stream.
FORCE_INLINE void gate_up_one_row_tile(const CoreGroups& groups, WeightStream& weights, PackTarget& pack_target) {
    init_gate_up_projection();
    uint32_t blocks = 0;
    for (uint32_t c = 0; c < single_chunks; ++c) {
        const bool last = c + 1 == single_chunks;
        const uint32_t chunk = last ? hidden_tiles - c * single_chunk_tiles : single_chunk_tiles;
        pack_target.select(!last);
        cb_wait_front(cb_x, chunk_tiles * row_tiles);
        if (last) {
            cb_reserve_back(cb_activated, row_tiles * activated_tiles);
        }
        for (uint32_t u = 0; u < groups.column_groups; ++u) {
            const uint32_t width = groups.column_group_width(u);
            tile_regs_acquire();
            if (c > 0) {
                reload_partials(cb_x, width);
            } else if (width != 4) {
                matmul_block_init(cb_x, cb_w, /*transpose=*/false, /*ct_dim=*/width, /*rt_dim=*/1, /*kt_dim=*/1);
            }
            for (uint32_t kt = 0; kt < chunk; ++kt) {
                matmul_block(cb_x, cb_w, kt, weights.take(width), 0, false, width, 1, 1);
            }
            if constexpr (has_bias) {
                if (last) {
                    matmul_block(cb_ones, cb_w, 0, weights.take(width), 0, false, width, 1, 1);
                }
            }
            weights.end_run();
            blocks += (width * (chunk + (has_bias && last ? 1 : 0)) + block_tiles - 1) / block_tiles;
            tile_regs_commit();
            if (last) {
                pack_activation(width, 2 * u);
            } else {
                pack_partials(width);
            }
            tile_regs_release();
        }
        cb_pop_front(cb_x, chunk_tiles * row_tiles);
    }
    cb_push_back(cb_activated, row_tiles * activated_tiles);
    pop_unit_padding(blocks);
}

// Gate/up projection and activation of a job of m row tiles: x chunk slot = [hidden tile of the chunk][row tile],
// activated tiles -> cb_activated [row tile][activated_tiles]; each weight unit is held for all m row tiles.
FORCE_INLINE void gate_up_row_tiles(uint32_t m, const CoreGroups& groups, PackTarget& pack_target) {
    init_gate_up_projection();
    for (uint32_t c = 0; c < gate_up_chunks; ++c) {
        const bool last = c + 1 == gate_up_chunks;
        const uint32_t chunk = last ? hidden_tiles - c * chunk_tiles : chunk_tiles;
        pack_target.select(!last);
        cb_wait_front(cb_x, chunk_tiles * row_tiles);
        if (last) {
            cb_reserve_back(cb_activated, row_tiles * activated_tiles);
        }
        for (uint32_t u = 0; u < groups.column_groups; ++u) {
            const uint32_t width = groups.column_group_width(u);
            if (c == 0 && width != 4) {
                matmul_block_init(cb_x, cb_w, /*transpose=*/false, /*ct_dim=*/width, /*rt_dim=*/1, /*kt_dim=*/1);
            }
            cb_wait_front(cb_w, unit_tiles);
            for (uint32_t row_tile = 0; row_tile < m; ++row_tile) {
                tile_regs_acquire();
                if (c > 0) {
                    reload_partials(cb_x, width);
                }
                for (uint32_t kt = 0; kt < chunk; ++kt) {
                    matmul_block(cb_x, cb_w, kt * m + row_tile, width * kt, 0, false, width, 1, 1);
                }
                if constexpr (has_bias) {
                    if (last) {
                        matmul_block(cb_ones, cb_w, 0, width * chunk, 0, false, width, 1, 1);
                    }
                }
                tile_regs_commit();
                if (last) {
                    pack_activation(width, row_tile * activated_tiles + 2 * u);
                } else {
                    pack_partials(width);
                }
                tile_regs_release();
            }
            cb_pop_front(cb_w, unit_tiles);
        }
        cb_pop_front(cb_x, chunk_tiles * row_tiles);
    }
    cb_push_back(cb_activated, row_tiles * activated_tiles);
}

// Down projection of a job of one row tile: each output group's W2 rows in one run from the stream.
FORCE_INLINE void down_one_row_tile(const CoreGroups& groups, WeightStream& weights, PackTarget& pack_target) {
    pack_target.select(false);
    cb_wait_front(cb_intermediate, row_tiles * intermediate_tiles);
    matmul_block_init(cb_intermediate, cb_w, /*transpose=*/false, /*ct_dim=*/4, /*rt_dim=*/1, /*kt_dim=*/1);
    uint32_t blocks = 0;
    for (uint32_t q = 0; q < groups.output_groups; ++q) {
        const uint32_t width = groups.output_group_width(q);
        if (width != 4) {
            matmul_block_init(cb_intermediate, cb_w, /*transpose=*/false, /*ct_dim=*/width, /*rt_dim=*/1, /*kt_dim=*/1);
        }
        tile_regs_acquire();
        IntermediateColumnWalk walk(groups.ring_position);
        for (uint32_t i = 0; i < intermediate_tiles; ++i) {
            const uint32_t column = walk.next();
            matmul_block(cb_intermediate, cb_w, column, weights.take(width), 0, false, width, 1, 1);
        }
        if constexpr (has_bias) {
            matmul_block(cb_ones, cb_w, 0, weights.take(width), 0, false, width, 1, 1);
        }
        weights.end_run();
        blocks += (width * (intermediate_tiles + (has_bias ? 1 : 0)) + block_tiles - 1) / block_tiles;
        tile_regs_commit();
        pack_output_rows();
        tile_regs_release();
    }
    cb_pop_front(cb_intermediate, row_tiles * intermediate_tiles);
    pop_unit_padding(blocks);
}

// Down projection of a job of m row tiles: intermediate slot = [row tile][intermediate column], rows -> cb_rows per
// (output group, row tile); each W2 chunk is held for all m row tiles.
FORCE_INLINE void down_row_tiles(uint32_t m, const CoreGroups& groups, PackTarget& pack_target) {
    cb_wait_front(cb_intermediate, row_tiles * intermediate_tiles);
    matmul_block_init(cb_intermediate, cb_w, /*transpose=*/false, /*ct_dim=*/4, /*rt_dim=*/1, /*kt_dim=*/1);
    for (uint32_t q = 0; q < groups.output_groups; ++q) {
        const uint32_t width = groups.output_group_width(q);
        if (width != 4) {
            matmul_block_init(cb_intermediate, cb_w, /*transpose=*/false, /*ct_dim=*/width, /*rt_dim=*/1, /*kt_dim=*/1);
        }
        IntermediateColumnWalk chunk_start(groups.ring_position);
        for (uint32_t p = 0; p < down_chunks; ++p) {
            const bool last = p + 1 == down_chunks;
            const uint32_t chunk = last ? intermediate_tiles - p * w2_chunk_rows : w2_chunk_rows;
            pack_target.select(!last);
            cb_wait_front(cb_w, unit_tiles);
            IntermediateColumnWalk walk = chunk_start;
            for (uint32_t row_tile = 0; row_tile < m; ++row_tile) {
                tile_regs_acquire();
                if (p > 0) {
                    reload_partials(cb_intermediate, width);
                }
                walk = chunk_start;
                for (uint32_t i = 0; i < chunk; ++i) {
                    const uint32_t in0_tile = row_tile * intermediate_tiles + walk.next();
                    matmul_block(cb_intermediate, cb_w, in0_tile, width * i, 0, false, width, 1, 1);
                }
                if constexpr (has_bias) {
                    if (last) {
                        matmul_block(cb_ones, cb_w, 0, width * chunk, 0, false, width, 1, 1);
                    }
                }
                tile_regs_commit();
                if (last) {
                    pack_output_rows();
                } else {
                    pack_partials(width);
                }
                tile_regs_release();
            }
            chunk_start = walk;
            cb_pop_front(cb_w, unit_tiles);
        }
    }
    cb_pop_front(cb_intermediate, row_tiles * intermediate_tiles);
}

// row tiles of this group's job `job` (cb_ctl, rows_reader.cpp); always 1 with row_tiles = 1
FORCE_INLINE uint32_t job_row_tiles(uint32_t job) { return row_tiles > 1 ? read_tile_value(cb_ctl, 0, 1 + job) : 1; }

FORCE_INLINE void gate_up_projection(
    uint32_t job, const CoreGroups& groups, WeightStream& weights, PackTarget& pack_target) {
    const uint32_t m = job_row_tiles(job);
    if (m == 1) {
        gate_up_one_row_tile(groups, weights, pack_target);
    } else {
        gate_up_row_tiles(m, groups, pack_target);
    }
}

FORCE_INLINE void down_projection(
    uint32_t job, const CoreGroups& groups, WeightStream& weights, PackTarget& pack_target) {
    const uint32_t m = job_row_tiles(job);
    if (m == 1) {
        down_one_row_tile(groups, weights, pack_target);
    } else {
        down_row_tiles(m, groups, pack_target);
    }
}

}  // namespace

void kernel_main() {
    const CoreGroups groups{
        .column_groups = get_arg_val<uint32_t>(0),
        .output_groups = get_arg_val<uint32_t>(1),
        .ring_position = get_arg_val<uint32_t>(2),
        .half_column = get_arg_val<uint32_t>(3) == 1,
        .half_output = get_arg_val<uint32_t>(4) == 1};

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_activated);
    if constexpr (has_bias) {
        fill_ones_tile();
    }
    cb_wait_front(cb_ctl, 1);
    const uint32_t jobs = read_tile_value(cb_ctl, 0, 0);  // this group's jobs
    if (jobs == 0) {
        return;
    }
    pack_reconfig_data_format(cb_activated);
    reconfig_data_format_srcb(cb_x);
    reconfig_data_format_srca(cb_w);
    MATH((ckernel::zeroacc()));
    WeightStream weights;
    PackTarget pack_target;
    // step s: the gate/up projection of job s, then the down projection of job s - 1, whose intermediate the writers
    // exchanged meanwhile
    for (uint32_t step = 0; step <= jobs; ++step) {
        if (step < jobs) {
            gate_up_projection(step, groups, weights, pack_target);
        }
        if (step > 0) {
            down_projection(step - 1, groups, weights, pack_target);
        }
    }
}
