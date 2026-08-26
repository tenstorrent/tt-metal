// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/groupnorm_constants.hpp"
#include "api/dataflow/dataflow_buffer.h"

// clang-format off
// Definitions
//   block_h: This the length of the row we wish to processes in terms of tiles
//
//   out_block_...: This is the length of our Circular Buffer, sometimes the length of out tensors(block_h) are larger than SRAM space, so we
//   have to process chunks of this data at a time
//   this chunk is called an out_block
//
//   num_out_blocks: This is the number of chunks specified by the use, such that a CBs (length defined by out_block) fit in SRAM
//   (Users should minimize the number of num_out_blocks for better perf)
//
//   ...normal:  If num_out_blocks evenly divides block_h, then all chunks are the size normal
//
//   ...last: If num_out_blocks does not divides block_h, the leftovers are put into a chunk of length last
//
//   sender: This refers to a core that does aggregation calculations
//   for the group of cores
//
//   receiver: This the cores that receive the aggregated results from sender, they only do
//   local computations that they send to the sender for final aggregation
//
// This is a high level description of the stages of this kernel. Each stage below is one function
// named after it; kernel_main() at the bottom is the batch/group loop that calls them in order.
//
// Batch Loop:
//   Group Loop:
//     This is the process which repeats for every group
//     Average Calc: E[x]                                  -> accumulate_local_sum, global_reduce
//       Local Reduce:
//           First we apply an input mask
//           This is where we sum up our core's subtensor
//           After summing up, we pass our scalar tile to cb_ex_partial_id
//           The reader kernels then aggregate all of the local scalars into a single tile
//       Global Reduce:
//           This single tile (cb_ex_external_id) is a tile that contains each partial reduce from all the other cores
//           Only the core designated as the sender reduces this tile to produce the global scalar reduce value.
//           The reader core then sends this data out to all other cores as cb_ex_global_id
//
//     Variance Calc: ∑(x-E[x])^2                          -> accumulate_local_sq_dev, global_reduce
//     This follows the same pattern as the average calculation
//       Local Reduce:
//           First we subtract each value from our core's subtensor by the average value
//           We next apply our input mask to zero our the values we wish to ignore
//           Next we square our residuals to obtain the squared residuals
//           After summing up, we pass our scalar tile to cb_ex2_partial_id
//           The reader kernels then aggregate all of the local scalars into a single tile
//       Global Reduce:
//           This single tile (cb_ex_external_id) is a tile that contains each partial reduce from all the other cores
//           Only the core designated as the sender reduces this tile to produce the global scalar reduce value.
//           The reader core then sends this data out to all other cores as cb_ex2_global_id
//
//     cb_ex2pe_id Calculation:                             -> compute_rstd
//       First we add cb_ex2_global_id with cb_eps_id
//       Then we take the sqrt
//       Lastly we take the reciprocal and he have the denominator of our calculation
//     Final Val Calc:                                      -> write_output_block
//       First we subtract each value from our core's subtensor by the average value
//       We next apply our input mask to zero our the values we wish to ignore
//       Next we multiply our residual with our denominator
//       Optional Gamma:
//           We multiply this value to gamma
//       Optional Beta:
//           We add beta to this value
//
// We are now done! Nice
// clang-format on

// The kernel's own vocabulary (block_w, group, batch, ...) is scoped so it cannot collide with
// anything the compute API headers declare.
namespace groupnorm_compute {

// ============================================================================
// Compile-time configuration
// ============================================================================

constexpr uint32_t is_mcast_sender = get_named_compile_time_arg_val("is_mcast_sender");
constexpr uint32_t do_gamma = get_named_compile_time_arg_val("do_gamma");
constexpr uint32_t do_beta = get_named_compile_time_arg_val("do_beta");
constexpr uint32_t num_cores_per_mcast_group = get_named_compile_time_arg_val("num_cores_per_mcast_group");
// True when a reconfig-relevant operand is fp32: the per-group reconfig_data_format calls below
// are then required. All-bf16 compiles them out (no-ops). See program factory.
constexpr bool enable_fp32_reconfig = get_named_compile_time_arg_val("enable_fp32_reconfig") != 0;

constexpr uint32_t batch = get_named_compile_time_arg_val("batch");
constexpr uint32_t group = get_named_compile_time_arg_val("group");

constexpr uint32_t block_h = get_named_compile_time_arg_val("block_h");
constexpr uint32_t block_w = get_named_compile_time_arg_val("block_w");
constexpr uint32_t block_hw [[maybe_unused]] = get_named_compile_time_arg_val("block_hw");

constexpr uint32_t subblock_w = get_named_compile_time_arg_val("subblock_w");
constexpr uint32_t num_subblocks_w = get_named_compile_time_arg_val("num_subblocks_w");

constexpr uint32_t per_core_M [[maybe_unused]] = get_named_compile_time_arg_val("per_core_M");
constexpr uint32_t per_core_N = get_named_compile_time_arg_val("per_core_N");
constexpr uint32_t per_core_MN [[maybe_unused]] = get_named_compile_time_arg_val("per_core_MN");

constexpr uint32_t per_core_N_tile_bytes [[maybe_unused]] = get_named_compile_time_arg_val("per_core_N_tile_bytes");
constexpr uint32_t num_groups_per_reset = get_named_compile_time_arg_val("num_groups_per_reset");

constexpr uint32_t single_tile_size_bytes = get_named_compile_time_arg_val("single_tile_size_bytes");
constexpr uint32_t num_tiles_per_batch [[maybe_unused]] = get_named_compile_time_arg_val("num_tiles_per_batch");

constexpr uint32_t num_tiles_input_mask [[maybe_unused]] = get_named_compile_time_arg_val("num_tiles_input_mask");
constexpr uint32_t num_cols_per_group = get_named_compile_time_arg_val("num_cols_per_group");

constexpr uint32_t block_w_last = get_named_compile_time_arg_val("block_w_last");
constexpr uint32_t GROUP_SIZE_IS_POWER_OF_2 = get_named_compile_time_arg_val("GROUP_SIZE_IS_POWER_OF_2");
constexpr uint32_t GROUP_SIZE_SMALLER_THAN_TILE_W = get_named_compile_time_arg_val("GROUP_SIZE_SMALLER_THAN_TILE_W");
constexpr uint32_t group_row_offset = get_named_compile_time_arg_val("group_row_offset");
constexpr uint32_t num_out_blocks = get_named_compile_time_arg_val("num_out_blocks");
constexpr uint32_t tile_width = get_named_compile_time_arg_val("TILE_WIDTH");

// Non-tile-aligned H*W: the tile-padding rows are excluded from both accumulation passes by
// switching to a second, row-masked set of mask tiles on the batch's final row-tile. The writer
// gives cores that do not hold that row-tile a copy of the normal set, so the switch here is
// unconditional. The divisor is corrected separately, in the reduce scaler.
// logical_hw / padded_hw are carried only so two shapes padding to the same size cannot share a
// cached program; has_row_mask is what this kernel branches on.
constexpr uint32_t logical_hw [[maybe_unused]] = get_named_compile_time_arg_val("logical_hw");
constexpr uint32_t padded_hw [[maybe_unused]] = get_named_compile_time_arg_val("padded_hw");
constexpr bool has_row_mask = get_named_compile_time_arg_val("has_row_mask") == 1;
constexpr uint32_t mask_tiles_per_group = has_row_mask ? 2 * block_w : block_w;
constexpr uint32_t last_row_tile = block_h - 1;

constexpr uint32_t block_w_minus_one = block_w - 1;
constexpr uint32_t block_w_minus_two = block_w - 2;
constexpr uint32_t tile_w_minux_group_size = tile_width - num_cols_per_group;

constexpr uint32_t data_per_core_N_per_group = (per_core_N * tile_width / group);

// The three layout variants the program factory selects with defines, hoisted to constants so the
// stages below can branch on them with if constexpr instead of repeating the preprocessor condition
// at every use. The preprocessor is still needed at the few points noted below.
#ifdef TILIZE_IN
constexpr bool tilize_in = true;
#else
constexpr bool tilize_in = false;
#endif
#ifdef UNTILIZE_OUT
constexpr bool untilize_out = true;
#else
constexpr bool untilize_out = false;
#endif
#ifdef READER_REPACK
constexpr bool reader_repack = true;
#else
constexpr bool reader_repack = false;
#endif

// dst regs
constexpr uint32_t dst0 = 0;
constexpr uint32_t scaler0 [[maybe_unused]] = 0;

// input cbs
constexpr uint32_t dfb_in0_id = tt::CBIndex::c_0;
constexpr uint32_t dfb_in_id = tt::CBIndex::c_29;
// Holds the whole per-core group, tilized once and kept resident for all three passes.
constexpr uint32_t dfb_in_resident_id = tt::CBIndex::c_17;
constexpr uint32_t dfb_scaler_id = tt::CBIndex::c_2;
constexpr uint32_t dfb_scaler_global_id = tt::CBIndex::c_4;
constexpr uint32_t dfb_eps_id = tt::CBIndex::c_3;
constexpr uint32_t dfb_gamma_id = tt::CBIndex::c_5;
constexpr uint32_t dfb_beta_id = tt::CBIndex::c_6;
constexpr uint32_t dfb_input_mask_id = tt::CBIndex::c_28;

// interm cbs
constexpr uint32_t dfb_repack_id = tt::CBIndex::c_26;
constexpr uint32_t dfb_repack_out_id = tt::CBIndex::c_31;
constexpr uint32_t dfb_x_id = tt::CBIndex::c_24;
constexpr uint32_t dfb_xmm_id = tt::CBIndex::c_25;
constexpr uint32_t dfb_ex_partial_id = tt::CBIndex::c_8;
constexpr uint32_t dfb_ex2_partial_id = tt::CBIndex::c_21;
constexpr uint32_t dfb_ex_id = tt::CBIndex::c_9;
constexpr uint32_t dfb_ex2_id = tt::CBIndex::c_13;
constexpr uint32_t dfb_ex_external_id = tt::CBIndex::c_10;
constexpr uint32_t dfb_ex_global_id = tt::CBIndex::c_15;
constexpr uint32_t dfb_ex2_global_id = tt::CBIndex::c_14;
constexpr uint32_t dfb_ex2pe_id = tt::CBIndex::c_27;

// interm cbs reuse
constexpr uint32_t dfb_fusion_id [[maybe_unused]] = dfb_xmm_id;
constexpr uint32_t dfb_reread_out_id = tt::CBIndex::c_23;
constexpr uint32_t dfb_reread_write_out_id = tt::CBIndex::c_22;
// Scratch for the row-major output reread; tilized into c_23 below.
constexpr uint32_t dfb_reread_rm_id = tt::CBIndex::c_20;

// output cb
constexpr uint32_t dfb_out0_id = tt::CBIndex::c_16;
constexpr uint32_t dfb_out_rm_id = tt::CBIndex::c_30;
constexpr uint32_t dfb_out_id =
    untilize_out ? dfb_out_rm_id : ((do_gamma or do_beta) ? dfb_out0_id : dfb_reread_write_out_id);

constexpr uint32_t dfb_outgamma_id = do_beta ? dfb_in_id : dfb_out0_id;
constexpr uint32_t dfb_inbeta_id = do_gamma ? dfb_outgamma_id : dfb_reread_write_out_id;
constexpr uint32_t dfb_outbeta_id = dfb_out0_id;

// Row-major source the resident group is tilized from.
constexpr uint32_t dfb_in_rm_id = reader_repack ? dfb_repack_id : dfb_in0_id;
// Untilize the tiled result into the row-major output c_30.
constexpr uint32_t dfb_untilize_in_id = (do_gamma or do_beta) ? dfb_out0_id : dfb_reread_write_out_id;
constexpr uint32_t dfb_untilize_out_id = reader_repack ? dfb_repack_out_id : dfb_out_id;

// Where the three passes read their input tiles from: the resident copy when this kernel tilizes
// the input itself, otherwise the tiles the reader delivered.
constexpr uint32_t dfb_input_id = tilize_in ? dfb_in_resident_id : dfb_in0_id;

// ============================================================================
// Dataflow buffers
// ============================================================================

// A DataflowBuffer's constructor waits for its CB to be published, so all of them are built once at
// kernel entry, in this declaration order, and handed to the stages as one bundle.
struct Buffers {
    DataflowBuffer beta{dfb_beta_id};
    DataflowBuffer eps{dfb_eps_id};
    DataflowBuffer ex{dfb_ex_id};
    DataflowBuffer ex2{dfb_ex2_id};
    DataflowBuffer ex2_global{dfb_ex2_global_id};
    DataflowBuffer ex2_partial{dfb_ex2_partial_id};
    DataflowBuffer ex2pe{dfb_ex2pe_id};
    DataflowBuffer ex_external{dfb_ex_external_id};
    DataflowBuffer ex_global{dfb_ex_global_id};
    DataflowBuffer ex_partial{dfb_ex_partial_id};
    DataflowBuffer gamma{dfb_gamma_id};
    DataflowBuffer in{dfb_in_id};
#ifdef TILIZE_IN
    DataflowBuffer in_resident{dfb_in_resident_id};
#endif
    DataflowBuffer in0{dfb_in0_id};
    DataflowBuffer inbeta{dfb_inbeta_id};
    DataflowBuffer input_mask{dfb_input_mask_id};
    DataflowBuffer outbeta{dfb_outbeta_id};
    DataflowBuffer outgamma{dfb_outgamma_id};
    DataflowBuffer reread_out{dfb_reread_out_id};
    DataflowBuffer reread_write_out{dfb_reread_write_out_id};
    DataflowBuffer scaler{dfb_scaler_id};
    DataflowBuffer scaler_global{dfb_scaler_global_id};
    DataflowBuffer x{dfb_x_id};
    DataflowBuffer xmm{dfb_xmm_id};
#ifndef TILIZE_IN
    // The resident CB is not allocated on this path, and constructing a DataflowBuffer for it would
    // wait on a producer that does not exist. in_resident stands in as in0 and is never touched,
    // since every use of it is behind if constexpr (tilize_in).
    DataflowBuffer& in_resident = in0;
#endif
};

// ============================================================================
// Out-block chunking
// ============================================================================

constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
constexpr uint32_t out_block_hw_normal = out_block_h_normal * block_w;

// The block_h row-tiles of a group are processed in out-blocks so the intermediate CBs fit in SRAM.
// When num_out_blocks does not divide block_h the leftover rows form one shorter final out-block,
// and the CB slots that out-block does not fill still have to be released.
struct OutBlocks {
    uint32_t num_out_blocks_padded;
    uint32_t out_block_h_last;
    uint32_t out_block_hw_last;
    bool extra_out_block;

    // True for the shorter final out-block.
    bool is_short_last(uint32_t out_block_index) const {
        return extra_out_block && (out_block_index == (num_out_blocks_padded - 1));
    }

    // Row-tiles actually held by the given out-block.
    uint32_t h_actual(uint32_t out_block_index) const {
        return is_short_last(out_block_index) ? out_block_h_last : out_block_h_normal;
    }

    // Tiles the shorter final out-block leaves unused. pop_front(0) advances the tile-counter
    // round-robin, so callers must guard this with is_short_last rather than popping zero.
    uint32_t unused_tiles() const { return out_block_hw_normal - out_block_hw_last; }
};

ALWI OutBlocks make_out_blocks() {
    OutBlocks ob;
    ob.num_out_blocks_padded = num_out_blocks;
    ob.extra_out_block = false;
    ob.out_block_h_last = out_block_h_normal;
    ob.out_block_hw_last = out_block_hw_normal;
    if constexpr (block_h % num_out_blocks != 0) {
        ob.extra_out_block = true;
        uint32_t residual = block_h - (num_out_blocks * out_block_h_normal);
        ob.num_out_blocks_padded += (residual / out_block_h_normal + 1);
        ob.out_block_h_last = residual % out_block_h_normal;
        ob.out_block_hw_last = ob.out_block_h_last * block_w;
    }
    return ob;
}

// Tiles the sender has to reduce to turn the per-core partials into one global value. The reader
// packs one partial per core per out-block into dfb_ex_external at a fixed byte pitch, which need
// not come out to a whole number of tiles.
ALWI uint32_t ex_external_tiles_required(const OutBlocks& ob) {
    const uint32_t total_bytes =
        ob.num_out_blocks_padded * num_cores_per_mcast_group * dfb_ex_external_slot_pitch_bytes;
    uint32_t tiles = total_bytes / single_tile_size_bytes;
    if (total_bytes % single_tile_size_bytes) {
        tiles++;
    }
    return tiles;
}

// ============================================================================
// Shared tile loop
// ============================================================================

// Every elementwise stage below walks an out-block the same way: for each row-tile, split the row
// into subblocks of subblock_w tiles, compute one subblock into DEST, and pack it out. per_tile
// computes the tile that belongs at DEST index w, given the row and the offset of the subblock
// within that row; after_row runs once a row's tiles have been packed, and is where a stage
// releases the input rows it has finished with.
template <uint32_t out_dfb_id, typename PerTile, typename AfterRow>
ALWI void for_each_subblock(uint32_t out_block_h_actual, PerTile per_tile, AfterRow after_row) {
    for (uint32_t i = 0; i < out_block_h_actual; ++i) {
        uint32_t index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; ++j) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; ++w) {
                per_tile(i, index_subblock_w_offset, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t w = 0; w < subblock_w; ++w) {
                pack_tile(w, out_dfb_id);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        after_row(i);
    }
}

// Stages that keep their input available for a later pass have nothing to do per row. A closure
// rather than a function so the call inlines away instead of going through a pointer.
constexpr auto no_after_row = [](uint32_t) {};

// ============================================================================
// Stages shared by the variance pass and the output pass
// ============================================================================

// x - E[x] over one out-block, from the kernel's input tiles into dfb_xmm. On the tiled input path
// the rows are popped as they are consumed, so the caller cannot read them again; on the resident
// path the whole group stays available until the group is done.
ALWI void center_out_block(Buffers& dfb, const OutBlocks& ob, uint32_t out_block_index) {
    // The resident group is already there; only the tiled path waits on new rows.
    if constexpr (!tilize_in) {
        dfb.in0.wait_front(out_block_hw_normal);
    }
    // x - E[x]
    sub_bcast_scalar_init(dfb_input_id, dfb_ex_global_id);
    // fp32: reset both srcs so fp32 input/mean aren't read through the format the preceding stage
    // left behind.
    if constexpr (enable_fp32_reconfig) {
        reconfig_data_format_srca(dfb_input_id);
        reconfig_data_format_srcb(dfb_ex_global_id);
    }

    dfb.xmm.reserve_back(out_block_hw_normal);
    dfb.ex_global.wait_front(1);
    for_each_subblock<dfb_xmm_id>(
        ob.h_actual(out_block_index),
        [&](uint32_t i, uint32_t index_subblock_w_offset, uint32_t w) {
            // The resident copy holds the whole group, so its tiles are addressed from the start of
            // the group; the tiled path only ever has the current row in front of it.
            uint32_t row_base = 0;
            if constexpr (tilize_in) {
                row_base = out_block_index * out_block_hw_normal + i * block_w;
            }
            uint32_t index = w + index_subblock_w_offset + row_base;
            sub_tiles_bcast_scalar(dfb_input_id, dfb_ex_global_id, index, 0, w);
        },
        [&](uint32_t) {
            if constexpr (!tilize_in) {
                dfb.in0.pop_front(block_w);
            }
        });
    if (ob.is_short_last(out_block_index)) {
        if constexpr (!tilize_in) {
            dfb.in0.pop_front(ob.unused_tiles());
        }
    }
    dfb.xmm.push_back(out_block_hw_normal);
}

// Zero the garbage columns of a centered out-block by multiplying dfb_xmm by the input mask, into
// dfb_x. Leaves srcb configured for the mask.
//
// use_row_mask selects the second, row-masked set of mask tiles on the batch's final row-tile. That
// set varies down the rows of a tile, so it can only be consumed by a full-tile multiply, which
// rules out the cheaper row broadcast.
template <bool use_row_mask>
ALWI void mask_out_block(Buffers& dfb, const OutBlocks& ob, uint32_t out_block_index) {
    const uint32_t row_tile_base = out_block_index * out_block_h_normal;

    // zero out the garbage values by mult mask again
    reconfig_data_format_srcb(dfb_ex_global_id, dfb_input_mask_id);
    if constexpr (use_row_mask) {
        mul_init(dfb_xmm_id, dfb_input_mask_id);
    } else {
        mul_bcast_rows_init(dfb_xmm_id, dfb_input_mask_id);
    }

    dfb.x.reserve_back(out_block_hw_normal);
    dfb.xmm.wait_front(out_block_hw_normal);
    for_each_subblock<dfb_x_id>(
        ob.h_actual(out_block_index),
        [&](uint32_t i, uint32_t index_subblock_w_offset, uint32_t w) {
            // Rows are popped as they are consumed, so tile indices restart at every row.
            uint32_t index = w + index_subblock_w_offset;
            if constexpr (use_row_mask) {
                // Without this switch each padding row is centered to (garbage - E[x]) and squared
                // into the variance.
                uint32_t mask_set_offset = ((row_tile_base + i) == last_row_tile) ? block_w : 0;
                mul_tiles(dfb_xmm_id, dfb_input_mask_id, index, index + mask_set_offset, w);
            } else {
                mul_tiles_bcast_rows(dfb_xmm_id, dfb_input_mask_id, index, index, w);
            }
        },
        [&](uint32_t) { dfb.xmm.pop_front(block_w); });
    if (ob.is_short_last(out_block_index)) {
        dfb.xmm.pop_front(ob.unused_tiles());
    }
    dfb.x.push_back(out_block_hw_normal);
}

// ============================================================================
// Average Calc: E[x]
// ============================================================================

// Local Reduce for one out-block: mask the raw input into dfb_x and sum it into one partial tile for
// the reader kernels to aggregate.
ALWI void accumulate_local_sum(Buffers& dfb, const OutBlocks& ob, uint32_t out_block_index) {
    const uint32_t out_block_h_actual = ob.h_actual(out_block_index);
    const uint32_t row_tile_base = out_block_index * out_block_h_normal;

    uint32_t out_block_base = 0;
    if constexpr (tilize_in) {
#ifdef TILIZE_IN
        // Append this out-block; no pop, so the whole group stays available. Guarded by the
        // preprocessor rather than if constexpr because a discarded if constexpr branch in a
        // non-template function still instantiates the templates it names.
        compute_kernel_lib::tilize<
            block_w,
            dfb_in_rm_id,
            dfb_in_resident_id,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(out_block_h_normal);
#endif
        dfb.in_resident.wait_front((out_block_index + 1) * out_block_hw_normal);
        out_block_base = out_block_index * out_block_hw_normal;
    } else {
        dfb.in0.wait_front(out_block_hw_normal);
    }

    reconfig_data_format_srcb(dfb_in0_id, dfb_input_mask_id);
    // mask input
    // The row-masked set varies down the rows of a tile, so it can only be consumed by a full-tile
    // multiply. Row-0-only synthesis and the row broadcast are therefore available exactly when
    // there is no row mask, i.e. on tile-aligned H*W.
    if constexpr (has_row_mask) {
        mul_init(dfb_input_id, dfb_input_mask_id);
    } else {
        mul_bcast_rows_init(dfb_input_id, dfb_input_mask_id);
    }

    dfb.x.reserve_back(out_block_hw_normal);
    for_each_subblock<dfb_x_id>(
        out_block_h_actual,
        [&](uint32_t i, uint32_t index_subblock_w_offset, uint32_t w) {
            uint32_t index = w + index_subblock_w_offset + i * block_w + out_block_base;
            uint32_t index_mask = w + index_subblock_w_offset;
            if constexpr (has_row_mask) {
                // Row-masked set on the batch's final row-tile, so the padding contributes nothing
                // to E[x].
                uint32_t mask_set_offset = ((row_tile_base + i) == last_row_tile) ? block_w : 0;
                mul_tiles(dfb_input_id, dfb_input_mask_id, index, index_mask + mask_set_offset, w);
            } else {
                mul_tiles_bcast_rows(dfb_input_id, dfb_input_mask_id, index, index_mask, w);
            }
        },
        no_after_row);
    // Only the tiled path pops here; the row-major group stays resident.
    if constexpr (!tilize_in) {
        dfb.in0.pop_front(out_block_hw_normal);
    }
    dfb.x.push_back(out_block_hw_normal);
    reconfig_data_format_srcb(dfb_input_mask_id, dfb_scaler_id);

    // Partial/E[x]
    dfb.x.wait_front(out_block_hw_normal);
    compute_kernel_lib::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_SCALAR,
        dfb_x_id,
        dfb_scaler_id,
        dfb_ex_partial_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
        compute_kernel_lib::ReduceInputBlockShape::of(out_block_h_actual, block_w));
    dfb.x.pop_front(out_block_hw_normal);

    dfb.ex_partial.wait_front(1);
}

// ============================================================================
// Variance Calc: ∑(x-E[x])^2
// ============================================================================

// Local Reduce for one out-block: center, re-mask, square, and sum into one partial tile.
ALWI void accumulate_local_sq_dev(Buffers& dfb, const OutBlocks& ob, uint32_t out_block_index) {
    const uint32_t out_block_h_actual = ob.h_actual(out_block_index);

    center_out_block(dfb, ob, out_block_index);
    mask_out_block<has_row_mask>(dfb, ob, out_block_index);

    reconfig_data_format_srcb(dfb_input_mask_id, dfb_x_id);
    // (x - E[x])^2
    mul_init(dfb_x_id, dfb_x_id);
    dfb.xmm.reserve_back(out_block_hw_normal);
    dfb.x.wait_front(out_block_hw_normal);
    for_each_subblock<dfb_xmm_id>(
        out_block_h_actual,
        [&](uint32_t i, uint32_t index_subblock_w_offset, uint32_t w) {
            uint32_t index = w + index_subblock_w_offset + i * block_w;
            mul_tiles(dfb_x_id, dfb_x_id, index, index, w);
        },
        no_after_row);
    dfb.x.pop_front(out_block_hw_normal);
    dfb.xmm.push_back(out_block_hw_normal);

    // Partial-Var(x)
    dfb.xmm.wait_front(out_block_hw_normal);
    compute_kernel_lib::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_SCALAR,
        dfb_xmm_id,
        dfb_scaler_id,
        dfb_ex2_partial_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
        compute_kernel_lib::ReduceInputBlockShape::of(out_block_h_actual, block_w));
    dfb.xmm.pop_front(out_block_hw_normal);
}

// ============================================================================
// Global Reduce
// ============================================================================

// Sender-only: reduce the per-core partials the reader gathered in dfb_ex_external into the one
// global value the reader broadcasts back out, then signal the reader that it is ready.
template <uint32_t global_dfb_id>
ALWI void global_reduce(uint32_t ex_external_tiles, DataflowBuffer& dfb_signal) {
    compute_kernel_lib::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_SCALAR,
        dfb_ex_external_id,
        dfb_scaler_global_id,
        global_dfb_id,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
        compute_kernel_lib::ReduceInputBlockShape::col(ex_external_tiles));
    if (num_cores_per_mcast_group > 1) {
        dfb_signal.reserve_back(1);
        dfb_signal.push_back(1);
    }
}

// ============================================================================
// 1/sqrt(Var + eps)
// ============================================================================

ALWI void compute_rstd(Buffers& dfb) {
    //  global reduce results
    dfb.eps.wait_front(1);
    dfb.ex2_global.wait_front(1);
    dfb.ex2pe.reserve_back(1);

    // The row mask keeps the padding out of both sums, so this is already the variance over
    // the real rows; no back-correction needed.
    // (Var + eps)
    tile_regs_acquire();
    add_init(dfb_ex2_global_id, dfb_eps_id);
    // fp32: reset both srcs so fp32 variance / bf16 eps aren't read through the stale square/reduce format.
    if constexpr (enable_fp32_reconfig) {
        reconfig_data_format_srca(dfb_ex2_global_id);
        reconfig_data_format_srcb(dfb_eps_id);
    }
    add_tiles(dfb_ex2_global_id, dfb_eps_id, 0, 0, dst0);
    tile_regs_wait();
    // 1/[sqrt(Var + eps)]
    rsqrt_tile_init<true>();
    rsqrt_tile<true>(dst0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, dfb_ex2pe_id);
    tile_regs_release();
    dfb.ex2pe.push_back(1);
    dfb.ex2_global.pop_front(1);
}

// ============================================================================
// Final Val Calc
// ============================================================================

// Position of the output accumulation within a tile. A group covers only part of a tile's columns,
// so each output tile is built up over several groups: the first group to reach a tile copies into
// it, later ones add. These three values carry that position across out-blocks and groups.
struct AccumulateState {
    bool copy_or_add;
    uint32_t group_reset_index;
    uint32_t index_block_w;
};

// Fold the normalized out-block into the output tiles, and record per tile of this out-block whether
// the tile still lies inside the current group and so takes gamma and beta.
ALWI void accumulate_into_output(
    Buffers& dfb,
    const OutBlocks& ob,
    uint32_t out_block_index,
    uint32_t block_w_curr,
    uint32_t index_g_offset,
    uint32_t g,
    AccumulateState& state,
    bool (&apply_gamma_beta)[block_w]) {
    const uint32_t out_block_h_actual = ob.h_actual(out_block_index);

    dfb.reread_out.wait_front(out_block_hw_normal);
    dfb.reread_write_out.reserve_back(out_block_hw_normal);
    for (uint32_t w = 0; w < block_w_curr; ++w) {
        uint32_t index_h_offset = 0;
        uint32_t index_h1_offset = 0;

        if (state.copy_or_add == true) {
            copy_tile_init(dfb_xmm_id);
        } else {
            add_init(dfb_reread_out_id, dfb_xmm_id);
        }

        for (uint32_t i = 0; i < out_block_h_actual; ++i) {
            tile_regs_acquire();
            uint32_t index_reread_out = w + index_h_offset;
            uint32_t index_xmm = w + index_h1_offset;

            if (state.copy_or_add == true) {
                copy_tile(dfb_xmm_id, index_xmm, dst0);
            } else {
                add_tiles(dfb_reread_out_id, dfb_xmm_id, index_reread_out, index_xmm, dst0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(dst0, dfb_reread_write_out_id, index_reread_out);
            tile_regs_release();

            index_h_offset += block_w_curr;
            index_h1_offset += block_w;
        }

        // update group tile offset
        if (state.index_block_w >= block_w_curr - 1) {
            state.index_block_w = 0;

            if (state.group_reset_index == num_groups_per_reset - 1) {
                state.copy_or_add = true;

                state.group_reset_index = 0;
            } else {
                state.copy_or_add = false;

                state.group_reset_index += 1;
            }
        } else {
            state.copy_or_add = true;
            state.index_block_w += 1;
        }

        bool is_past_end_of_group = (((w + index_g_offset) + 1) * tile_width) > ((g + 1) * data_per_core_N_per_group);
        apply_gamma_beta[w] = !is_past_end_of_group;
    }
    dfb.xmm.pop_front(out_block_hw_normal);
    dfb.reread_out.pop_front(out_block_hw_normal);
    dfb.reread_write_out.push_back(out_block_hw_normal);
}

// gamma multiplies and beta adds; both walk the out-block the same way and fall back to a plain copy
// on the tiles whose columns lie past the end of the current group.
enum class GammaBetaOp { Multiply, Add };

// wait_input is false when a preceding stage has already waited on the tiles being read: gamma is
// the first consumer of accumulate_into_output's tiles and waits for them, whereas beta reads what
// gamma left waited (or, with no gamma, what accumulate_into_output produced).
template <GammaBetaOp op, uint32_t in_dfb_id, uint32_t out_dfb_id, bool wait_input>
ALWI void apply_gamma_or_beta(
    uint32_t out_block_h_actual,
    uint32_t block_w_curr,
    uint32_t index_g_offset,
    const bool (&apply_gamma_beta)[block_w],
    DataflowBuffer& dfb_param,
    DataflowBuffer& dfb_in,
    DataflowBuffer& dfb_out) {
    constexpr uint32_t param_dfb_id = (op == GammaBetaOp::Multiply) ? dfb_gamma_id : dfb_beta_id;

    uint32_t index_h_offset = 0;
    dfb_out.reserve_back(out_block_hw_normal);
    dfb_param.wait_front(per_core_N);
    if constexpr (wait_input) {
        dfb_in.wait_front(out_block_hw_normal);
    }
    for (uint32_t i = 0; i < out_block_h_actual; ++i) {
        for (uint32_t j = 0; j < block_w_curr; ++j) {
            if (apply_gamma_beta[j]) {
                if constexpr (op == GammaBetaOp::Multiply) {
                    mul_bcast_rows_init(in_dfb_id, param_dfb_id);
                } else {
                    add_bcast_rows_init(in_dfb_id, param_dfb_id);
                }
                // fp32: reset both srcs so the bf16 gamma/beta isn't read through the fp32 format of
                // the stage that produced this input.
                if constexpr (enable_fp32_reconfig) {
                    reconfig_data_format_srca(in_dfb_id);
                    reconfig_data_format_srcb(param_dfb_id);
                }
            } else {
                copy_tile_init(in_dfb_id);
            }
            tile_regs_acquire();
            uint32_t index = j + index_h_offset;
            uint32_t index_param = j + index_g_offset;
            if (apply_gamma_beta[j]) {
                if constexpr (op == GammaBetaOp::Multiply) {
                    mul_tiles_bcast_rows(in_dfb_id, param_dfb_id, index, index_param, dst0);
                } else {
                    add_tiles_bcast_rows(in_dfb_id, param_dfb_id, index, index_param, dst0);
                }
            } else {
                copy_tile(in_dfb_id, index, dst0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, out_dfb_id);
            tile_regs_release();
        }
        index_h_offset += block_w_curr;
    }
    dfb_out.push_back(out_block_hw_normal);
    dfb_in.pop_front(out_block_hw_normal);
    dfb_out.wait_front(out_block_hw_normal);
}

// One out-block of the output: center, re-mask, scale by 1/sqrt(Var + eps), fold into the output
// tiles, then apply gamma and beta.
ALWI void write_output_block(
    Buffers& dfb,
    const OutBlocks& ob,
    uint32_t out_block_index,
    uint32_t block_w_curr,
    uint32_t index_g_offset,
    uint32_t g,
    AccumulateState& state,
    bool (&apply_gamma_beta)[block_w]) {
    const uint32_t out_block_h_actual = ob.h_actual(out_block_index);

    center_out_block(dfb, ob, out_block_index);
    // By this pass the padding rows only feed padding outputs, so the cheaper row broadcast is
    // always enough and the row-masked set is not needed.
    mask_out_block<false>(dfb, ob, out_block_index);
    reconfig_data_format_srcb(dfb_input_mask_id, dfb_x_id);

    // (x - Ex) * 1/[sqrt(Var + eps)]
    mul_bcast_scalar_init(dfb_x_id, dfb_ex2pe_id);
    // fp32: reset both srcs so fp32 x/rstd aren't read through the stale mask/eps format.
    if constexpr (enable_fp32_reconfig) {
        reconfig_data_format_srca(dfb_x_id);
        reconfig_data_format_srcb(dfb_ex2pe_id);
    }
    dfb.xmm.reserve_back(out_block_hw_normal);
    dfb.ex2pe.wait_front(1);
    dfb.x.wait_front(out_block_hw_normal);
    for_each_subblock<dfb_xmm_id>(
        out_block_h_actual,
        [&](uint32_t i, uint32_t index_subblock_w_offset, uint32_t w) {
            uint32_t index = w + index_subblock_w_offset + i * block_w;
            mul_tiles_bcast_scalar(dfb_x_id, dfb_ex2pe_id, index, 0, w);
        },
        no_after_row);
    dfb.x.pop_front(out_block_hw_normal);
    dfb.xmm.push_back(out_block_hw_normal);
    dfb.xmm.wait_front(out_block_hw_normal);

#ifdef UNTILIZE_OUT
    // Tilize the reread rows so the accumulation below sees tiles. Guarded by the preprocessor
    // rather than if constexpr because a discarded if constexpr branch in a non-template function
    // still instantiates the templates it names.
    compute_kernel_lib::tilize<
        block_w,
        dfb_reread_rm_id,
        dfb_reread_out_id,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(out_block_h_normal);
#endif

    accumulate_into_output(dfb, ob, out_block_index, block_w_curr, index_g_offset, g, state, apply_gamma_beta);

    // Start Optional Gamma
    if constexpr (do_gamma) {
        apply_gamma_or_beta<GammaBetaOp::Multiply, dfb_reread_write_out_id, dfb_outgamma_id, true>(
            out_block_h_actual,
            block_w_curr,
            index_g_offset,
            apply_gamma_beta,
            dfb.gamma,
            dfb.reread_write_out,
            dfb.outgamma);
    }
    // End Optional Gamma

    // Start Optional Beta
    if constexpr (do_beta) {
        apply_gamma_or_beta<GammaBetaOp::Add, dfb_inbeta_id, dfb_outbeta_id, false>(
            out_block_h_actual, block_w_curr, index_g_offset, apply_gamma_beta, dfb.beta, dfb.inbeta, dfb.outbeta);
    }
    // End Optional Beta

#ifdef UNTILIZE_OUT
    // untilize - DEST capacity auto-detected. Guarded by the preprocessor for the same reason as the
    // tilize above.
    compute_kernel_lib::untilize<
        block_w,
        dfb_untilize_in_id,
        dfb_untilize_out_id,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(
        out_block_h_normal);
#endif
}

// ============================================================================
// Group bookkeeping
// ============================================================================

// Step index_g_offset to the first tile of the next group. Groups are not tile-aligned in general,
// so consecutive groups can share a tile and the step alternates between block_w and one or two
// tiles less; row_offset tracks how far into the current tile the next group starts.
ALWI void advance_to_next_group(uint32_t& index_g_offset, uint32_t& row_offset) {
    if constexpr (GROUP_SIZE_IS_POWER_OF_2) {
        if (row_offset == tile_width) {
            index_g_offset += block_w;
            row_offset = num_cols_per_group;

        } else {
            index_g_offset += block_w_minus_one;
            row_offset += num_cols_per_group;
        }
    } else if constexpr (GROUP_SIZE_SMALLER_THAN_TILE_W) {
        if (row_offset == tile_width) {
            index_g_offset += block_w_minus_one;
            row_offset = num_cols_per_group;

        } else if (row_offset > tile_width) {
            index_g_offset += block_w_minus_one;
            row_offset = row_offset + group_row_offset;

        } else {
            row_offset += num_cols_per_group;
        }
    } else {
        if (row_offset > tile_width) {
            index_g_offset += block_w_minus_one;
            row_offset = row_offset - tile_w_minux_group_size;
        } else {
            row_offset += num_cols_per_group;
            index_g_offset += block_w_minus_two;
        }
    }
}

}  // namespace groupnorm_compute

void kernel_main() {
    using namespace groupnorm_compute;

    Buffers dfb;

    if constexpr (tilize_in) {
        // Tilize the whole group once and reuse it for all three passes.
        compute_kernel_hw_startup(dfb_in0_id, dfb_in0_id, dfb_in_resident_id);
    } else {
        // Already tiled, so feed compute directly.
        compute_kernel_hw_startup(dfb_in0_id, dfb_input_mask_id, dfb_x_id);
    }

    const OutBlocks ob = make_out_blocks();
    const uint32_t ex_external_tiles = ex_external_tiles_required(ob);

    uint32_t index_g_offset = 0;
    uint32_t row_offset = num_cols_per_group;
    AccumulateState state{true, 0, 0};
    bool apply_gamma_beta[block_w];

    // Start Batch Loop
    for (uint32_t b = 0; b < batch; ++b) {
        index_g_offset = 0;
        row_offset = num_cols_per_group;
        state = AccumulateState{true, 0, 0};

        // Start Group Loop
        for (uint32_t g = 0; g < group; ++g) {
            dfb.input_mask.wait_front(mask_tiles_per_group);

            // Start Average Calc
            for (uint32_t out_block_index = 0; out_block_index < ob.num_out_blocks_padded; out_block_index++) {
                accumulate_local_sum(dfb, ob, out_block_index);
            }
            if constexpr (is_mcast_sender) {
                global_reduce<dfb_ex_global_id>(ex_external_tiles, dfb.ex);
            }
            // End Average Calc

            // Start Variance Calc
            for (uint32_t out_block_index = 0; out_block_index < ob.num_out_blocks_padded; out_block_index++) {
                accumulate_local_sq_dev(dfb, ob, out_block_index);
            }
            if constexpr (is_mcast_sender) {
                global_reduce<dfb_ex2_global_id>(ex_external_tiles, dfb.ex2);
            }
            compute_rstd(dfb);
            // End Variance Calc

            // Every out-block folds into the same output tiles, so each one restarts from the
            // accumulation position this group began at.
            const AccumulateState group_start_state = state;
            const uint32_t block_w_curr = (index_g_offset == (per_core_N - block_w_last)) ? block_w_last : block_w;

            // Start Final Val Calc
            for (uint32_t out_block_index = 0; out_block_index < ob.num_out_blocks_padded; out_block_index++) {
                state = group_start_state;
                write_output_block(dfb, ob, out_block_index, block_w_curr, index_g_offset, g, state, apply_gamma_beta);
            }
            // End Final Val Calc

            if constexpr (tilize_in) {
                // All passes done with the group, resident group popped.
                dfb.in_resident.pop_front(ob.num_out_blocks_padded * out_block_hw_normal);
            }
            advance_to_next_group(index_g_offset, row_offset);
            dfb.ex_global.pop_front(1);
            dfb.ex2pe.pop_front(1);
            dfb.input_mask.pop_front(mask_tiles_per_group);
        }
        // End Group Loop
    }
    // End Batch Loop
}
