// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows, compute of one core of one expert group. Per job (one expert, at most 32 routed rows per row tile):
//  gate/up projection: for each of this core's W0/W1 column groups, [g0 u0 g1 u1] = x . [W0 j0, W1 j0, W0 j1, W1 j1]
//      over all hidden tiles, then the activation on the packer's SFPU, act(g0) u0 and act(g1) u1 -> cb_activated; a
//      half block-column (the last group of a ring position with an odd column count in the compact layout) is
//      [g0 u0] = x . [W0 j0, W1 j0] -> act(g0) u0.
//  down projection: the job's intermediate (every column, from the exchange in rows_writer.cpp) . W2 for each of this
//      core's 4-tile output groups (the last output group of a ring position can be two tiles wide), the W2 rows in
//      moe_compute's per-core rotated order -> cb_rows (4 bf16 tiles; the writer sends the real rows).
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
#include "api/dataflow/circular_buffer.h"
#include "../moe_activation.h"
#include "expert_rows_common.h"

namespace {

using expert_rows::GroupRun;
using ttnn::experimental::prim::detail::MoEActivationFunction;

constexpr uint32_t block_tiles = get_named_compile_time_arg_val("block_tiles");
constexpr uint32_t cb_w_id = get_named_compile_time_arg_val("cb_w");
constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
constexpr uint32_t intermediate_tiles = get_named_compile_time_arg_val("intermediate_tiles");
constexpr uint32_t ring_cores = get_named_compile_time_arg_val("ring_cores");
constexpr bool has_bias = get_named_compile_time_arg_val("has_bias") == 1;
constexpr uint32_t cb_x_id = get_named_compile_time_arg_val("cb_x");
constexpr uint32_t cb_activated_id = get_named_compile_time_arg_val("cb_activated");        // this core's columns
constexpr uint32_t cb_intermediate_id = get_named_compile_time_arg_val("cb_intermediate");  // the job's every column
constexpr uint32_t cb_rows_id = get_named_compile_time_arg_val("cb_rows");
constexpr uint32_t cb_ctl_id = get_named_compile_time_arg_val("cb_ctl");
constexpr uint32_t cb_ones_id = get_named_compile_time_arg_val("cb_ones");
constexpr uint32_t cb_partials_id = get_named_compile_time_arg_val("cb_partials");
constexpr uint32_t activated_tiles = get_named_compile_time_arg_val("activated_tiles");  // per row tile
constexpr uint32_t row_tiles = get_named_compile_time_arg_val("row_tiles");
constexpr uint32_t chunk_tiles = get_named_compile_time_arg_val("chunk_tiles");
constexpr uint32_t chunk_blocks = get_named_compile_time_arg_val("chunk_blocks");
constexpr uint32_t w2_chunk_rows = get_named_compile_time_arg_val("w2_chunk_rows");
constexpr auto activation_function = MoEActivationFunction(get_named_compile_time_arg_val("activation_function"));

constexpr uint32_t group_tiles = expert_rows::group_tiles;
static_assert(block_tiles % group_tiles == 0, "a block holds whole K rows of a column group");

constexpr uint32_t gate_up_chunks = expert_rows::div_up(hidden_tiles, chunk_tiles);
constexpr uint32_t down_chunks = expert_rows::div_up(intermediate_tiles, w2_chunk_rows);
constexpr uint32_t unit_tiles = chunk_blocks * block_tiles;  // weight pages of one chunk unit (rows_reader.cpp)
// A job of one row tile holds nothing for other row tiles: x chunks as large as its slot, its weights streamed in whole
// runs. When units are held (row_tiles > 1, rows_reader.cpp), its pass is padded to whole units of unit_blocks blocks.
constexpr uint32_t single_chunk_tiles = expert_rows::single_chunk_tiles(chunk_tiles, row_tiles, hidden_tiles);
constexpr uint32_t single_chunks = expert_rows::div_up(hidden_tiles, single_chunk_tiles);
constexpr uint32_t unit_blocks = row_tiles > 1 ? chunk_blocks : 1;
// cb_partials holds one slot of partial_slot_tiles per (group, row tile), whatever the group's width (a half group
// fills 2 of them): partial CB = row_tiles * groups * partial_slot_tiles tiles, so every reserve starts at a multiple
// of partial_slot_tiles and none runs past the CB's end.
constexpr uint32_t partial_slot_tiles = group_tiles;

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
    bool half_output;        // the last output group is two tiles wide
    FORCE_INLINE GroupRun columns() const { return {column_groups, half_column}; }
    FORCE_INLINE GroupRun outputs() const { return {output_groups, half_output}; }
};

// Weight tiles of one pass arrive in blocks of block_tiles; a run (one group) ends with a short block.
struct WeightStream {
    uint32_t left = 0, cur = 0;
    bool held = false;
    FORCE_INLINE uint32_t take(uint32_t n) {
        if (left == 0) {
            CircularBuffer cb_w(cb_w_id);
            if (held) {
                cb_w.pop_front(block_tiles);
            }
            cb_w.wait_front(block_tiles);
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
            CircularBuffer(cb_w_id).pop_front(block_tiles);
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
                pack_reconfig_data_format(cb_activated_id, cb_partials_id);
            } else {
                pack_reconfig_data_format(cb_partials_id, cb_activated_id);
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
    CircularBuffer cb_ones(cb_ones_id);
    pack_reconfig_data_format(cb_ones_id);
    copy_init(cb_ones_id);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 1.f);
    tile_regs_commit();
    tile_regs_wait();
    cb_ones.reserve_back(1);
    pack_tile(0, cb_ones_id);
    tile_regs_release();
    cb_ones.push_back(1);
    cb_ones.wait_front(1);
}

// DEST 0..width-1 <- this (group, row tile)'s partials of the previous chunk, in the order they were packed; then the
// matmul of a group `width` tiles wide
FORCE_INLINE void reload_partials(uint32_t cb_in0_id, uint32_t width) {
    CircularBuffer cb_partials(cb_partials_id);
    reconfig_data_format_srca(cb_w_id, cb_partials_id);
    copy_init(cb_partials_id);
    cb_partials.wait_front(partial_slot_tiles);
    for (uint32_t t = 0; t < width; ++t) {
        copy_tile(cb_partials_id, t, t);
    }
    cb_partials.pop_front(partial_slot_tiles);
    reconfig_data_format_srca(cb_partials_id, cb_w_id);
    matmul_block_init(cb_in0_id, cb_w_id, /*transpose=*/false, /*ct_dim=*/width, /*rt_dim=*/1, /*kt_dim=*/1);
}

// DEST 0..tiles-1 -> one slot of group_tiles tiles of `cb_id`: a (group, row tile)'s partials (cb_partials), or an
// output group's 4 bf16 tiles of one row tile (cb_rows).
FORCE_INLINE void pack_slot(uint32_t cb_id, uint32_t tiles) {
    CircularBuffer cb(cb_id);
    tile_regs_wait();
    cb.reserve_back(group_tiles);
    for (uint32_t t = 0; t < tiles; ++t) {
        pack_tile<true>(t, cb_id, t);
    }
    cb.push_back(group_tiles);
}

// The activation of a group `width` tiles wide on the packer's SFPU (DEST 0..width-1), its activated tiles (DEST 0 and,
// for a full group, 2) -> cb_activated [first_tile, first_tile + width / 2) (as compute.cpp's
// compute_w0_w1_block_column: wait for math plus a CFG stall before the SFPU's DEST offset)
FORCE_INLINE void pack_activation(uint32_t width, uint32_t first_tile) {
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    if (width == group_tiles) {
        ::moe_activation::PackActivation<activation_function, /*kPairs=*/2>::compute();
    } else {
        ::moe_activation::PackActivation<activation_function, /*kPairs=*/1>::compute();
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    pack_tile<true>(0, cb_activated_id, first_tile);
    if (width == group_tiles) {
        pack_tile<true>(2, cb_activated_id, first_tile + 1);
    }
}

// The weight blocks that pad a one-row-tile pass to whole units (row_tiles > 1, rows_reader.cpp).
FORCE_INLINE void pop_unit_padding(uint32_t blocks) {
    const uint32_t pad = (unit_blocks - blocks % unit_blocks) % unit_blocks;
    if (pad) {
        CircularBuffer cb_w(cb_w_id);
        cb_w.wait_front(pad * block_tiles);
        cb_w.pop_front(pad * block_tiles);
    }
}

FORCE_INLINE void init_gate_up_projection() {
    // GELU initializes in PackActivation, per group
    if constexpr (activation_function != MoEActivationFunction::GELU) {
        ::moe_activation::pack_init_activation<activation_function>();
    }
    matmul_block_init(cb_x_id, cb_w_id, /*transpose=*/false, /*ct_dim=*/group_tiles, /*rt_dim=*/1, /*kt_dim=*/1);
}

// A job of one row tile has its own walk: its weights arrive block by block from the stream (WeightStream) in whole
// runs, while a job of several row tiles holds each chunk unit in cb_w for all of its row tiles.

// Gate/up projection and activation of a job of one row tile: x in chunks of single_chunk_tiles, each column group's
// weights in whole runs from the stream.
FORCE_INLINE void gate_up_one_row_tile(const CoreGroups& groups, WeightStream& weights, PackTarget& pack_target) {
    CircularBuffer cb_x(cb_x_id);
    CircularBuffer cb_activated(cb_activated_id);
    init_gate_up_projection();
    uint32_t blocks = 0;
    for (uint32_t chunk = 0; chunk < single_chunks; ++chunk) {
        const bool last = chunk + 1 == single_chunks;
        const uint32_t k_tiles = last ? hidden_tiles - chunk * single_chunk_tiles : single_chunk_tiles;
        pack_target.select(!last);
        cb_x.wait_front(chunk_tiles * row_tiles);
        if (last) {
            cb_activated.reserve_back(row_tiles * activated_tiles);
        }
        for (uint32_t column_group = 0; column_group < groups.column_groups; ++column_group) {
            const uint32_t width = groups.columns().width(column_group);
            tile_regs_acquire();
            if (chunk > 0) {
                reload_partials(cb_x_id, width);
            } else if (width != group_tiles) {
                matmul_block_init(cb_x_id, cb_w_id, /*transpose=*/false, /*ct_dim=*/width, /*rt_dim=*/1, /*kt_dim=*/1);
            }
            for (uint32_t kt = 0; kt < k_tiles; ++kt) {
                matmul_block(cb_x_id, cb_w_id, kt, weights.take(width), 0, false, width, 1, 1);
            }
            if constexpr (has_bias) {
                if (last) {
                    matmul_block(cb_ones_id, cb_w_id, 0, weights.take(width), 0, false, width, 1, 1);
                }
            }
            weights.end_run();
            blocks += expert_rows::run_blocks(width, k_tiles + (has_bias && last ? 1 : 0), block_tiles);
            tile_regs_commit();
            if (last) {
                pack_activation(width, expert_rows::group_columns * column_group);
            } else {
                pack_slot(cb_partials_id, width);
            }
            tile_regs_release();
        }
        cb_x.pop_front(chunk_tiles * row_tiles);
    }
    cb_activated.push_back(row_tiles * activated_tiles);
    pop_unit_padding(blocks);
}

// Gate/up projection and activation of a job of job_row_tiles row tiles: x chunk slot = [hidden tile of the chunk][row
// tile], activated tiles -> cb_activated [row tile][activated_tiles]; each weight unit is held for all row tiles.
FORCE_INLINE void gate_up_row_tiles(uint32_t job_row_tiles, const CoreGroups& groups, PackTarget& pack_target) {
    CircularBuffer cb_x(cb_x_id);
    CircularBuffer cb_w(cb_w_id);
    CircularBuffer cb_activated(cb_activated_id);
    init_gate_up_projection();
    for (uint32_t chunk = 0; chunk < gate_up_chunks; ++chunk) {
        const bool last = chunk + 1 == gate_up_chunks;
        const uint32_t k_tiles = last ? hidden_tiles - chunk * chunk_tiles : chunk_tiles;
        pack_target.select(!last);
        cb_x.wait_front(chunk_tiles * row_tiles);
        if (last) {
            cb_activated.reserve_back(row_tiles * activated_tiles);
        }
        for (uint32_t column_group = 0; column_group < groups.column_groups; ++column_group) {
            const uint32_t width = groups.columns().width(column_group);
            if (chunk == 0 && width != group_tiles) {
                matmul_block_init(cb_x_id, cb_w_id, /*transpose=*/false, /*ct_dim=*/width, /*rt_dim=*/1, /*kt_dim=*/1);
            }
            cb_w.wait_front(unit_tiles);
            for (uint32_t row_tile = 0; row_tile < job_row_tiles; ++row_tile) {
                tile_regs_acquire();
                if (chunk > 0) {
                    reload_partials(cb_x_id, width);
                }
                for (uint32_t kt = 0; kt < k_tiles; ++kt) {
                    matmul_block(cb_x_id, cb_w_id, kt * job_row_tiles + row_tile, width * kt, 0, false, width, 1, 1);
                }
                if constexpr (has_bias) {
                    if (last) {
                        matmul_block(cb_ones_id, cb_w_id, 0, width * k_tiles, 0, false, width, 1, 1);
                    }
                }
                tile_regs_commit();
                if (last) {
                    pack_activation(width, row_tile * activated_tiles + expert_rows::group_columns * column_group);
                } else {
                    pack_slot(cb_partials_id, width);
                }
                tile_regs_release();
            }
            cb_w.pop_front(unit_tiles);
        }
        cb_x.pop_front(chunk_tiles * row_tiles);
    }
    cb_activated.push_back(row_tiles * activated_tiles);
}

// Down projection of a job of one row tile: each output group's W2 rows in one run from the stream.
FORCE_INLINE void down_one_row_tile(const CoreGroups& groups, WeightStream& weights, PackTarget& pack_target) {
    CircularBuffer cb_intermediate(cb_intermediate_id);
    pack_target.select(false);
    cb_intermediate.wait_front(row_tiles * intermediate_tiles);
    matmul_block_init(
        cb_intermediate_id, cb_w_id, /*transpose=*/false, /*ct_dim=*/group_tiles, /*rt_dim=*/1, /*kt_dim=*/1);
    uint32_t blocks = 0;
    for (uint32_t output_group = 0; output_group < groups.output_groups; ++output_group) {
        const uint32_t width = groups.outputs().width(output_group);
        if (width != group_tiles) {
            matmul_block_init(
                cb_intermediate_id, cb_w_id, /*transpose=*/false, /*ct_dim=*/width, /*rt_dim=*/1, /*kt_dim=*/1);
        }
        tile_regs_acquire();
        IntermediateColumnWalk walk(groups.ring_position);
        for (uint32_t i = 0; i < intermediate_tiles; ++i) {
            const uint32_t column = walk.next();
            matmul_block(cb_intermediate_id, cb_w_id, column, weights.take(width), 0, false, width, 1, 1);
        }
        if constexpr (has_bias) {
            matmul_block(cb_ones_id, cb_w_id, 0, weights.take(width), 0, false, width, 1, 1);
        }
        weights.end_run();
        blocks += expert_rows::run_blocks(width, intermediate_tiles + (has_bias ? 1 : 0), block_tiles);
        tile_regs_commit();
        pack_slot(cb_rows_id, group_tiles);
        tile_regs_release();
    }
    cb_intermediate.pop_front(row_tiles * intermediate_tiles);
    pop_unit_padding(blocks);
}

// Down projection of a job of job_row_tiles row tiles: intermediate slot = [row tile][intermediate column], rows ->
// cb_rows per (output group, row tile); each W2 chunk is held for all row tiles.
FORCE_INLINE void down_row_tiles(uint32_t job_row_tiles, const CoreGroups& groups, PackTarget& pack_target) {
    CircularBuffer cb_w(cb_w_id);
    CircularBuffer cb_intermediate(cb_intermediate_id);
    cb_intermediate.wait_front(row_tiles * intermediate_tiles);
    matmul_block_init(
        cb_intermediate_id, cb_w_id, /*transpose=*/false, /*ct_dim=*/group_tiles, /*rt_dim=*/1, /*kt_dim=*/1);
    for (uint32_t output_group = 0; output_group < groups.output_groups; ++output_group) {
        const uint32_t width = groups.outputs().width(output_group);
        if (width != group_tiles) {
            matmul_block_init(
                cb_intermediate_id, cb_w_id, /*transpose=*/false, /*ct_dim=*/width, /*rt_dim=*/1, /*kt_dim=*/1);
        }
        IntermediateColumnWalk chunk_start(groups.ring_position);
        for (uint32_t chunk = 0; chunk < down_chunks; ++chunk) {
            const bool last = chunk + 1 == down_chunks;
            const uint32_t k_rows = last ? intermediate_tiles - chunk * w2_chunk_rows : w2_chunk_rows;
            pack_target.select(!last);
            cb_w.wait_front(unit_tiles);
            IntermediateColumnWalk walk = chunk_start;
            for (uint32_t row_tile = 0; row_tile < job_row_tiles; ++row_tile) {
                tile_regs_acquire();
                if (chunk > 0) {
                    reload_partials(cb_intermediate_id, width);
                }
                walk = chunk_start;
                for (uint32_t i = 0; i < k_rows; ++i) {
                    const uint32_t in0_tile = row_tile * intermediate_tiles + walk.next();
                    matmul_block(cb_intermediate_id, cb_w_id, in0_tile, width * i, 0, false, width, 1, 1);
                }
                if constexpr (has_bias) {
                    if (last) {
                        matmul_block(cb_ones_id, cb_w_id, 0, width * k_rows, 0, false, width, 1, 1);
                    }
                }
                tile_regs_commit();
                if (last) {
                    pack_slot(cb_rows_id, group_tiles);
                } else {
                    pack_slot(cb_partials_id, width);
                }
                tile_regs_release();
            }
            chunk_start = walk;
            cb_w.pop_front(unit_tiles);
        }
    }
    cb_intermediate.pop_front(row_tiles * intermediate_tiles);
}

// row tiles of this group's job `job` (cb_ctl, rows_reader.cpp); always 1 with row_tiles = 1
FORCE_INLINE uint32_t row_tiles_of_job(uint32_t job) {
    return row_tiles > 1 ? CircularBuffer(cb_ctl_id).read_tile_value(0, 1 + job) : 1;
}

FORCE_INLINE void gate_up_projection(
    uint32_t job, const CoreGroups& groups, WeightStream& weights, PackTarget& pack_target) {
    const uint32_t job_row_tiles = row_tiles_of_job(job);
    if (job_row_tiles == 1) {
        gate_up_one_row_tile(groups, weights, pack_target);
    } else {
        gate_up_row_tiles(job_row_tiles, groups, pack_target);
    }
}

FORCE_INLINE void down_projection(
    uint32_t job, const CoreGroups& groups, WeightStream& weights, PackTarget& pack_target) {
    const uint32_t job_row_tiles = row_tiles_of_job(job);
    if (job_row_tiles == 1) {
        down_one_row_tile(groups, weights, pack_target);
    } else {
        down_row_tiles(job_row_tiles, groups, pack_target);
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

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_id, cb_w_id, cb_activated_id);
    if constexpr (has_bias) {
        fill_ones_tile();
    }
    CircularBuffer cb_ctl(cb_ctl_id);
    cb_ctl.wait_front(1);
    const uint32_t jobs = cb_ctl.read_tile_value(0, 0);  // this group's jobs
    if (jobs == 0) {
        return;
    }
    pack_reconfig_data_format(cb_activated_id);
    reconfig_data_format_srcb(cb_x_id);
    reconfig_data_format_srca(cb_w_id);
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
