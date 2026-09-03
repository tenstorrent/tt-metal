// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of conv_bmm_tilize.cpp (conv2d block-matmul + tilize compute kernel).
//
// The algorithm body is identical to the legacy kernel; only the host-binding surface is migrated:
//   - CB-index CTAs -> dfb:: tokens (act / weights / act_row_major / act_tilized / matmul_partials /
//     out / bias; act_second_reader gated behind SPLIT_READER)
//   - remaining positional CTAs -> get_arg(args::name)
//   - the check_skip_compute RTA -> get_arg(args::skip_compute)
//   - experimental::CB -> DataflowBuffer (kernel_main + helper signatures)
//   - in-place matmul-partials accumulate: on WH/BH it rewinds the partials CB's fifo_rd_ptr/fifo_wr_ptr
//     to re-accumulate in the same L1; on Quasar (no cb_interface) it snapshots/restores the equivalent
//     g_dfb_interface ring position (see the PARTIALS_* macros below).
//
// This fork is bound by the Metal 2.0 width-sharded factory and the non-overlap paths of the
// sharded factory (height-sharded; block-sharded without split_reader_cb_shared).  The split-reader
// activation-reuse / shared-overlap paths are gated by SPLIT_READER / ACTIVATION_REUSE defines.
// The legacy conv_bmm_tilize.cpp this was forked from has since been removed (it had no remaining
// consumer once both quasar conv2d factories moved to this fork); see git history for the original.

#include <cstdint>

#include "internal/mod_div_lib.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
// [block-sharded 0x19 A/B discriminator] MATH-thread markers around the matmul subblock loop and the
// partials reload, to tell whether the ERROR_TRISC1 0x0119 is the pre-existing MATH<->PACK DEST bank-recycle
// deadlock (stalls in the matmul at the first bank0 reuse, subblock 2, on the FIRST K-block with rl=0) or the
// option-2 dedicated-partials reload read (stalls at RLOAD_RD on a later K-block with rl=1). Remove once
// localized.
#include "api/debug/dprint.h"

// [#55076 / 0x19] DIAGNOSTIC STATE -- all switches below default to STOCK, so a build of this file
// reproduces the unmodified failure and is safe to hand to a waveform capture. Full write-up and the
// waypoint trigger-value table are in qsr_x19_rtl_findings.md (§4p covers wave capture).
//
//   conv_bmm_tilize_metal2.cpp
//     kStallwaitWaitsJustForPackDestRead  false   §4c  tested, no effect (write confirmed landed)
//     kDisableReplayBankedMode            false   §4n  tested, no effect (write confirmed landed)
//     kFenceMathRiscToPipePerSubblock     false   §4m  CSR fence; ON proves the pipe never drains
//     kPrintKernelTextAndDfbAddrs         false   §4g  DPRINT destabilises Quasar; leave off
//   tt-llk/tt_llk_quasar/llk_lib/llk_math_matmul.h
//     kHoistDestBaseToInit                false   §4k  ON moves the wedge SDW1 -> MB2
//     kDrainMathBeforeSrcBankClear        false   §4l  ON (with the above) moves it MB2 -> MP0
//
// Stock wedge signature: TRISC0 UPMW / TRISC1 MACQ / TRISC2 PPAK, MATH stalled in matmul_block at
// subblock 1 of ~98. WAYPOINT markers stay compiled in: each is a single L1 store of a compile-time
// constant to the watcher mailbox, so they are near-zero perturbation and double as wave triggers.
// The TRISC named in the error code is NOT the culprit -- it is whichever had a read outstanding.

// In-place matmul-partials accumulate: re-accumulate each inner-K block into the SAME L1 region by
// "rewinding" the partials buffer's producer/consumer position back to the start of the output block.
//
// On WH/BH that is a save/restore of the partials CB's fifo_rd_ptr/fifo_wr_ptr. Quasar compute has no
// cb_interface (it tracks DFB state in g_dfb_interface), so the equivalent position is the DFB ring
// state: tc_slots[].wr_entry_idx/wr_offset + wr_entry_ptr for the packer, tc_slots[].rd_entry_idx/
// rd_offset for the unpacker. Those advance only via dfb_advance_slot() on push_back/pop_front, so
// snapshotting and restoring them reproduces the rewind exactly. The PARTIALS_* macros below abstract
// the two arches; the *_WR/_RD variants are only ever expanded inside PACK()/UNPACK(), so the wr_*/rd_*
// fields they touch only compile on the matching TRISC.
#ifdef ARCH_QUASAR
struct QsrDfbRingPos {
    uint16_t entry_idx[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
    uint16_t offset[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
    uint16_t entry_ptr;  // packer in-order tile offset (wr_entry_ptr); unused on the read side
    uint8_t tc_idx;
};
using PartialsRingPos = QsrDfbRingPos;
// Quasar has no evil_set_*; snapshot/restore the DFB ring via get_local_dfb_interface.
// Macros take a DataflowBuffer so call sites match the WH/BH evil_* path.
// [#48552] f6b15a widened DFBTCSlot.ring_size to uint32 and REMOVED wr_offset/rd_offset -- the cursor
// byte-offset is now DERIVED from *_entry_idx (dfb_slot_cursor_offset_units). The partials-rewind
// snapshot/restore therefore only needs *_entry_idx; restoring it restores the derived offset.
// (PartialsRingPos.offset[] is now unused.)
#define QSR_SNAPSHOT_WR(pos, dfb)                                         \
    do {                                                                  \
        LocalDFBInterface& _qd = get_local_dfb_interface((dfb).get_id()); \
        for (uint8_t _qi = 0; _qi < _qd.num_tcs_to_rr; ++_qi) {           \
            (pos).entry_idx[_qi] = _qd.tc_slots[_qi].wr_entry_idx;        \
        }                                                                 \
        (pos).entry_ptr = _qd.wr_entry_ptr;                               \
        (pos).tc_idx = _qd.tc_idx;                                        \
    } while (0)
#define QSR_RESTORE_WR(pos, dfb)                                          \
    do {                                                                  \
        LocalDFBInterface& _qd = get_local_dfb_interface((dfb).get_id()); \
        for (uint8_t _qi = 0; _qi < _qd.num_tcs_to_rr; ++_qi) {           \
            _qd.tc_slots[_qi].wr_entry_idx = (pos).entry_idx[_qi];        \
        }                                                                 \
        _qd.wr_entry_ptr = (pos).entry_ptr;                               \
        _qd.tc_idx = (pos).tc_idx;                                        \
    } while (0)
#define QSR_SNAPSHOT_RD(pos, dfb)                                         \
    do {                                                                  \
        LocalDFBInterface& _qd = get_local_dfb_interface((dfb).get_id()); \
        for (uint8_t _qi = 0; _qi < _qd.num_tcs_to_rr; ++_qi) {           \
            (pos).entry_idx[_qi] = _qd.tc_slots[_qi].rd_entry_idx;        \
        }                                                                 \
        (pos).tc_idx = _qd.tc_idx;                                        \
    } while (0)
#define QSR_RESTORE_RD(pos, dfb)                                          \
    do {                                                                  \
        LocalDFBInterface& _qd = get_local_dfb_interface((dfb).get_id()); \
        for (uint8_t _qi = 0; _qi < _qd.num_tcs_to_rr; ++_qi) {           \
            _qd.tc_slots[_qi].rd_entry_idx = (pos).entry_idx[_qi];        \
        }                                                                 \
        _qd.tc_idx = (pos).tc_idx;                                        \
    } while (0)
#define SAVE_PARTIALS_WR(var, dfb) \
    PartialsRingPos var;           \
    QSR_SNAPSHOT_WR(var, dfb)
#define SAVE_PARTIALS_RD(var, dfb) \
    PartialsRingPos var;           \
    QSR_SNAPSHOT_RD(var, dfb)
#define RESAVE_PARTIALS_WR(var, dfb) QSR_SNAPSHOT_WR(var, dfb)
#define RESAVE_PARTIALS_RD(var, dfb) QSR_SNAPSHOT_RD(var, dfb)
#define RESTORE_PARTIALS_WR(var, dfb) QSR_RESTORE_WR(var, dfb)
#define RESTORE_PARTIALS_RD(var, dfb) QSR_RESTORE_RD(var, dfb)
#else
// WH/BH: rewind via DataflowBuffer get_*/evil_set_* (do not poke local_cb_interface directly).
using PartialsRingPos = uint32_t;
#define SAVE_PARTIALS_WR(var, dfb) uint32_t var = (dfb).get_write_ptr()
#define SAVE_PARTIALS_RD(var, dfb) uint32_t var = (dfb).get_read_ptr()
#define RESAVE_PARTIALS_WR(var, dfb) var = (dfb).get_write_ptr()
#define RESAVE_PARTIALS_RD(var, dfb) var = (dfb).get_read_ptr()
#define RESTORE_PARTIALS_WR(var, dfb) (dfb).evil_set_write_ptr(var)
#define RESTORE_PARTIALS_RD(var, dfb) (dfb).evil_set_read_ptr(var)
#endif

#ifdef SPLIT_READER
template <
    uint32_t in_block_w,
    uint32_t in_cb_id,
    uint32_t out_cb_id,
    bool init_tilize = true,
    bool uninit_tilize = true,
    compute_kernel_lib::tilize_config::RemapMode remap_mode = compute_kernel_lib::tilize_config::RemapMode::Configure>
__attribute__((noinline)) void tilize_in(
#else
template <
    uint32_t in_block_w,
    uint32_t in_cb_id,
    uint32_t out_cb_id,
    bool init_tilize = true,
    bool uninit_tilize = true,
    compute_kernel_lib::tilize_config::RemapMode remap_mode = compute_kernel_lib::tilize_config::RemapMode::Configure>
void tilize_in(
#endif
    uint32_t in_num_subblocks) {
    constexpr compute_kernel_lib::tilize_config::InitUninitMode init_uninit_mode =
        init_tilize ? (uninit_tilize ? compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit
                                     : compute_kernel_lib::tilize_config::InitUninitMode::InitOnly)
                    : (uninit_tilize ? compute_kernel_lib::tilize_config::InitUninitMode::UninitOnly
                                     : compute_kernel_lib::tilize_config::InitUninitMode::Neither);
    constexpr auto reconfig_mode =
        init_tilize ? compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure
                    : compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure;
    compute_kernel_lib::tilize<
        in_block_w,
        in_cb_id,
        out_cb_id,
        init_uninit_mode,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        reconfig_mode,
        compute_kernel_lib::tilize_config::Fp32Mode::Fast,
        remap_mode>(in_num_subblocks);
}  // tilize_in()

template <uint32_t in_cb_id, uint32_t in_block_w, uint32_t out_cb_id>
inline void tilize_single_block(DataflowBuffer& in_cb) {
    in_cb.wait_front(in_block_w);
    // [from amokan/fused_conv] Un-guarded on Quasar: fast_tilize_block now forwards to plain tilize_block on
    // Quasar (tilize.h ARCH_QUASAR branch), so the name resolves and the guard is no longer needed.
    fast_tilize_block(in_cb_id, in_block_w, out_cb_id);
    in_cb.pop_front(in_block_w);
}

template <uint32_t window_reuse_offset>
inline uint32_t update_in_cb(DataflowBuffer& in_cb, uint32_t in_cb_addr) {
#ifndef ARCH_QUASAR  // activation_reuse/split_reader path is off for resnet; dead on Quasar (no cb_interface)
    UNPACK((in_cb.evil_set_read_ptr(in_cb_addr)));
#endif
    return in_cb_addr + window_reuse_offset;
}

template <uint32_t in_cb_id, uint32_t in_block_w, uint32_t out_cb_id, uint32_t tilized_cb_row_offset>
inline void tilize_single_block_with_out_cb_update(
    DataflowBuffer& in_cb, DataflowBuffer& out_cb, uint32_t& out_cb_addr) {
#ifndef ARCH_QUASAR  // activation_reuse/split_reader path is off for resnet; dead on Quasar (no cb_interface)
    PACK((out_cb.evil_set_write_ptr(out_cb_addr)));
#endif
    PACK((out_cb_addr += tilized_cb_row_offset));
    tilize_single_block<in_cb_id, in_block_w, out_cb_id>(in_cb);
}

template <
    uint32_t in1_cb_id,
    uint32_t in2_cb_id,
    uint32_t in_block_w,
    uint32_t in1_num_subblocks,
    uint32_t in2_num_subblocks,
    uint32_t out_cb_id,
    uint32_t out_cb_tiles,
    uint32_t window_reuse_offset,
    uint32_t tilized_cb_row_offset,
    uint32_t tilized_cb_second_reader_offset,
    uint32_t image_width_in_tiles>
inline void tilize_in_reuse_split_reader(
    DataflowBuffer& in1_cb,
    DataflowBuffer& in2_cb,
    DataflowBuffer& out_cb,
    uint32_t act_cb_start_address,
    uint32_t act_cb_second_reader_start_address) {
    out_cb.reserve_back(out_cb_tiles);
    // [from amokan/fused_conv] Un-guarded on Quasar: fast_tilize_init_with_dt forwards to the plain path.
    fast_tilize_init_with_dt(in1_cb_id, in_block_w, out_cb_id);

    uint32_t in1_cb_addr = act_cb_start_address;
    uint32_t in2_cb_addr = act_cb_second_reader_start_address;

    uint32_t out_cb_addr, out_cb_addr_second_reader, out_cb_addr_init = 0;
#ifndef ARCH_QUASAR  // activation_reuse/split_reader path is off for resnet; dead on Quasar (no cb_interface)
    PACK((out_cb_addr_init = out_cb.get_write_ptr()));
#endif
    PACK((out_cb_addr = out_cb_addr_init));
    PACK((out_cb_addr_second_reader = out_cb_addr_init + tilized_cb_second_reader_offset));

    constexpr uint32_t min_num_subblocks =
        in1_num_subblocks > in2_num_subblocks ? in2_num_subblocks : in1_num_subblocks;
    constexpr uint32_t min_num_image_rows = min_num_subblocks / image_width_in_tiles;
    constexpr uint32_t leftover_in1 = in1_num_subblocks - min_num_image_rows * image_width_in_tiles;
    constexpr uint32_t leftover_in2 = in2_num_subblocks - min_num_image_rows * image_width_in_tiles;
    constexpr uint32_t max_leftover = leftover_in1 > leftover_in2 ? leftover_in1 : leftover_in2;

    for (uint32_t image_row = 0; image_row < min_num_image_rows; ++image_row) {
        in1_cb_addr = update_in_cb<window_reuse_offset>(in1_cb, in1_cb_addr);
        in2_cb_addr = update_in_cb<window_reuse_offset>(in2_cb, in2_cb_addr);
        for (uint32_t image_col = 0; image_col < image_width_in_tiles; ++image_col) {
            tilize_single_block_with_out_cb_update<in1_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                in1_cb, out_cb, out_cb_addr);
            tilize_single_block_with_out_cb_update<in2_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                in2_cb, out_cb, out_cb_addr_second_reader);
        }
    }

    in1_cb_addr = update_in_cb<window_reuse_offset>(in1_cb, in1_cb_addr);
    in2_cb_addr = update_in_cb<window_reuse_offset>(in2_cb, in2_cb_addr);
    for (uint32_t image_col = 0; image_col < max_leftover; ++image_col) {
        if (image_col < leftover_in1) {
            tilize_single_block_with_out_cb_update<in1_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                in1_cb, out_cb, out_cb_addr);

            if (image_col == image_width_in_tiles - 1) {
                in1_cb_addr = update_in_cb<window_reuse_offset>(in1_cb, in1_cb_addr);
            }
        }

        if (image_col < leftover_in2) {
            tilize_single_block_with_out_cb_update<in2_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                in2_cb, out_cb, out_cb_addr_second_reader);

            if (image_col == image_width_in_tiles - 1) {
                in2_cb_addr = update_in_cb<window_reuse_offset>(in2_cb, in2_cb_addr);
            }
        }
    }

#ifndef ARCH_QUASAR  // activation_reuse/split_reader path is off for resnet; dead on Quasar (no cb_interface)
    PACK((out_cb.evil_set_write_ptr(out_cb_addr_init)));
#endif
    out_cb.push_back(out_cb_tiles);
    // [from amokan/fused_conv] Un-guarded on Quasar: fast_tilize_uninit forwards to the plain path.
    fast_tilize_uninit(in2_cb_id, out_cb_id, in_block_w);
}

template <uint32_t out_subblock_w, uint32_t out_block_w>
inline void reblock_and_untilize(
    DataflowBuffer& interm_cb,
    DataflowBuffer& out_cb,
    uint32_t num_out_subblocks_in_col,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h) {
    const uint32_t interm_cb_id = interm_cb.get_id();
    const uint32_t out_cb_id = out_cb.get_id();
    uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles, num_out_subblocks_in_col);
    interm_cb.wait_front(num_tiles_in_row_of_subblocks);
    uint32_t within_block_index = 0;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        uint32_t block_offset = 0;
        out_cb.reserve_back(out_block_w);
        for (uint32_t n = 0; n < num_out_subblocks_in_col; n++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < out_subblock_w; w++) {
                uint32_t tile_index = block_offset + within_block_index + w;
                copy_tile(interm_cb_id, tile_index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<out_subblock_w, out_block_w>(out_cb_id, 1, n);
            tile_regs_release();
            block_offset += out_subblock_num_tiles;
        }
        out_cb.push_back(out_block_w);
        within_block_index += out_subblock_w;
    }
    interm_cb.pop_front(num_tiles_in_row_of_subblocks);
}

void kernel_main() {
    constexpr uint32_t in0_block_w = get_arg(args::in0_block_w);
    constexpr uint32_t in0_num_subblocks = get_arg(args::in0_num_subblocks);
    constexpr uint32_t in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr uint32_t in0_subblock_num_tiles = get_arg(args::in0_subblock_num_tiles);
    constexpr uint32_t reader_num_h_subblocks = get_arg(args::reader_num_h_subblocks);
    constexpr uint32_t in1_num_subblocks = get_arg(args::in1_num_subblocks);
    constexpr uint32_t in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    constexpr uint32_t in1_block_w = get_arg(args::in1_block_w);
    constexpr uint32_t in0_num_blocks_h = get_arg(args::in0_num_blocks_h);
    constexpr uint32_t in0_num_blocks_w = get_arg(args::in0_num_blocks_w);
    constexpr uint32_t in1_num_blocks_w = get_arg(args::in1_num_blocks_w);
    constexpr uint32_t out_subblock_h = get_arg(args::out_subblock_h);
    constexpr uint32_t out_subblock_w = get_arg(args::out_subblock_w);
    constexpr uint32_t out_subblock_num_tiles = get_arg(args::out_subblock_num_tiles);
    constexpr bool height_sharded = get_arg(args::height_sharded);
    constexpr bool untilize_out = get_arg(args::untilize_out);
    constexpr uint32_t in0_cb_id = dfb::act;
    constexpr uint32_t in1_cb_id = dfb::weights;
    // in0_pretilize_cb_id is the row-major tilize input, used only on the !height_sharded (mcast) path.
    // The factory binds dfb::act_row_major and defines HAS_ACT_ROW_MAJOR only on that path; on the
    // height-sharded path there is no act_row_major DFB (compute tilizes dfb::act directly), so alias it
    // to dfb::act — the value is unused there because the `if constexpr (!height_sharded)` branch that
    // references it is discarded.
#ifdef HAS_ACT_ROW_MAJOR
    constexpr uint32_t in0_pretilize_cb_id = dfb::act_row_major;
#else
    constexpr uint32_t in0_pretilize_cb_id = dfb::act;
#endif
#ifdef SPLIT_READER
    constexpr uint32_t in0_cb_second_reader_id = dfb::act_second_reader;
#endif
    constexpr uint32_t matmul_partials_cb = dfb::matmul_partials;
    constexpr uint32_t tilized_in0_cb_id = dfb::act_tilized;
    constexpr uint32_t out_cb_id = dfb::out;
    constexpr bool partials_cb_uses_output = get_arg(args::partials_cb_uses_output);
    constexpr uint32_t in0_nblocks_w_tilize = get_arg(args::in0_nblocks_w_tilize);
    constexpr bool pack_relu = get_arg(args::pack_relu);
    constexpr bool packer_untilize = get_arg(args::packer_untilize);
    constexpr bool packer_l1_acc = get_arg(args::packer_l1_acc);
    constexpr bool fuse_bias = get_arg(args::fuse_bias);
    constexpr bool split_reader = get_arg(args::split_reader);
    constexpr bool activation_reuse = get_arg(args::activation_reuse);

    constexpr uint32_t image_width_in_tiles = get_arg(args::image_width_in_tiles);
    constexpr uint32_t window_reuse_offset = get_arg(args::window_reuse_offset);
    constexpr uint32_t tilized_cb_row_offset = get_arg(args::tilized_cb_row_offset);
    constexpr uint32_t tilized_cb_second_reader_offset = get_arg(args::tilized_cb_second_reader_offset);
    constexpr bool split_reader_cb_shared = get_arg(args::split_reader_cb_shared) == 1;

    constexpr uint32_t out_block_num_tiles = in0_num_subblocks * in1_num_subblocks * out_subblock_num_tiles;
    constexpr uint32_t out_block_w = in1_block_w;
    constexpr bool spill = in0_num_blocks_w > 1;

    // QSR matmul-partials accumulation uses the SAME per-K-block idiom as WH/BH (reserve/pack/push per
    // subblock, then wait_front/pop_front(out_block) + RESTORE the ring each non-last K-block). The tested
    // Quasar L1-acc spill/reload kernel (#43990 multi_block_compute.cpp) proves this idiom works on Quasar:
    // it is credit-BALANCED per block (pop frees the slot before the next reserve, so credit never
    // over-posts), pop_front does not clear L1 (so pack_reconfig_l1_acc keeps summing into the same tiles),
    // and the RESTORE only rewinds the L1 ring position (not credit) so accumulation re-targets the aliased
    // output block. The earlier posted=392/acked=0 deadlock was the craq-sim sub-tile credit bug (since
    // fixed), not a fundamental "no credit-rewind" limit, so no Quasar-specific partials path is needed.

    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? matmul_partials_cb : out_cb_id;

    [[maybe_unused]] uint32_t bias_block_offset = 0;
    [[maybe_unused]] constexpr uint32_t bias_ntiles_w = get_arg(args::bias_ntiles_w);
#ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = dfb::bias;
#endif
    constexpr uint32_t mm_out_cb_id = fuse_bias ? matmul_partials_cb : untilize_mode_out_cb_id;

    constexpr uint32_t mm_in0_cb_id = height_sharded ? tilized_in0_cb_id : in0_cb_id;

    constexpr uint32_t in0_num_subblocks_read_last =
        (split_reader && !split_reader_cb_shared) ? reader_num_h_subblocks / 2 : 0;
    constexpr uint32_t in0_num_subblocks_read = reader_num_h_subblocks - in0_num_subblocks_read_last;

    DataflowBuffer cb_in0(in0_cb_id);
#ifdef SPLIT_READER
    DataflowBuffer cb_in0_second_reader(in0_cb_second_reader_id);
#endif
    DataflowBuffer cb_tilized_in0(tilized_in0_cb_id);
    DataflowBuffer cb_mm_in0(mm_in0_cb_id);
    DataflowBuffer cb_in1(in1_cb_id);
    DataflowBuffer cb_matmul_partials(matmul_partials_cb);
    DataflowBuffer cb_mm_out(mm_out_cb_id);
    DataflowBuffer cb_out(out_cb_id);
#ifdef FUSE_BIAS
    DataflowBuffer cb_bias(bias_cb_id);
#endif
    DataflowBuffer cb_untilize_mode_out(untilize_mode_out_cb_id);

    [[maybe_unused]] const uint32_t out_cb_tiles =
        activation_reuse ? in0_block_w * (in0_num_subblocks_read + in0_num_subblocks_read_last) : 0;
    // activation_reuse base addresses (used only on the split_reader activation_reuse path, off for
    // resnet). Quasar has no evil_set_*; the path is dead here, so source them as 0.
#ifdef ARCH_QUASAR
    [[maybe_unused]] uint32_t act_cb_start_address = 0;
    [[maybe_unused]] const uint32_t tilized_cb_start_address = 0;
#ifdef SPLIT_READER
    [[maybe_unused]] const uint32_t act_cb_second_reader_start_address = 0;
#endif
#else
    [[maybe_unused]] uint32_t act_cb_start_address = activation_reuse ? cb_in0.get_read_ptr() : 0;
    [[maybe_unused]] const uint32_t tilized_cb_start_address = activation_reuse ? cb_tilized_in0.get_write_ptr() : 0;
#ifdef SPLIT_READER
    [[maybe_unused]] const uint32_t act_cb_second_reader_start_address =
        activation_reuse ? cb_in0_second_reader.get_read_ptr() : 0;
#endif
#endif

#ifdef CHECK_SKIP_COMPUTE
    bool skip_compute = (bool)get_arg(args::skip_compute);
#endif

    // WH fast_tilize RACE-GUARD (DPRINT-independent; do NOT remove without the LLK fix). The WH fast_tilize
    // dest/semaphore handshake has a timing-sensitive race that deadlocks the tilize on the PACK thread
    // here) and HANGS once they were removed — NO other WH-path change — and that it only passes when DPRINT
    // compute_kernel_hw_startup. Replicate that latency WITHOUT DPRINT: on the PACK thread, read the same CB
    // interface registers into a volatile sink and spin briefly. `kRaceGuardSpin` is a TUNABLE delay — if WH
    // still hangs, raise it. Real fix = the fast_tilize LLK handshake race (TEN-4746 class).
#ifndef ARCH_QUASAR
    PACK({
        constexpr uint32_t kRaceGuardSpin = 512;  // TUNABLE — raise if WH still hangs
        for (uint32_t g = 0; g < kRaceGuardSpin; ++g) {
            asm volatile("nop");  // volatile asm: cannot be optimized away
        }
    });
#endif

#ifdef ARCH_QUASAR
    // [#55076 / 0x19] STALLWAIT-on-packer semantics -- candidate root cause of the MATH<->PACK wedge.
    //
    // _llk_pack_dest_semaphore_section_done_ (llk_pack_common.h) gates the SEMGET that releases MATH's
    // tile_regs_acquire() behind TTI_STALLWAIT(STALL_MATH, -, -, p_stall::PACK) -- i.e. behind
    // p_stall::PACK0 == i_packer_busy. In the Quasar RTL:
    //     tt_tdma.sv:4798  pack_busy = pack_busy_pre | pack_instrn_thrd_first_vld | packer_pipe_stage_busy
    // and the comment at tt_tdma.sv:1742 says that middle term comes from i_pack_instrn_vld "which is
    // blocked" -- the clock-gating logic deliberately substitutes pack_busy_pre to dodge it, but
    // tt_sync_exu gets the full version (tt_instruction_thread.sv:1763-1767, :1787). So a STALLWAIT
    // waiting for packer-completely-idle can be held up by a pack instruction that is itself blocked,
    // and nothing behind that STALLWAIT can unblock it -> the SEMGET never issues -> MATH sits in
    // SEMWAIT(STALL_ON_MAX) forever. Matches the observed state: PACK's last waypoint is RBD (inside the
    // pack section) in every repro, MATH is wedged, and it needs >= 3 out_subblocks (the first subblock
    // where MATH actually has to wait on PACK at all).
    //
    // The RTL has a chicken bit for exactly this case, reset value 0
    // (ws-tensix-quasar_rtl/src/hardware/tensix/registers/rdl/t6_debug_regs.rdl:1995-2006):
    //     "Normally, when issuing a STALLWAIT that waits for the packer, the STALLWAIT will wait until
    //      the packer is completely idle. Sometimes this is what you want, but the most common case when
    //      we're waiting for the packer is when we're waiting for the dest bank to be ready for new math
    //      outputs. So, as a performance improvement, you can turn on this bit and STALLWAIT will only
    //      wait for packer dest reads to be finished."
    // "waiting for the dest bank to be ready for new math outputs" is precisely what the LLK is doing
    // here, so the LLK wants this bit set and is currently getting the completely-idle behaviour.
    //
    // CHICKEN_BITS is a Neo-local register shared by all four TRISCs on the Neo, so one write covers the
    // whole Neo. Do it from PACK so it is ordered ahead of every pack STALLWAIT in PACK's own program
    // order. Flip kStallwaitWaitsJustForPackDestRead to false to get the old behaviour back.
    PACK({
        // RESULT 2026-09-02: tested true, hang was byte-identical, and the ASSERT below did NOT fire --
        // so the write did land and this is a genuine negative, not an unmodelled register. Turned back
        // off to remove a variable. Flip to true to retest.
        constexpr bool kStallwaitWaitsJustForPackDestRead = false;
        if constexpr (kStallwaitWaitsJustForPackDestRead) {
            RISCV_DEBUG_REGS->CHICKEN_BITS |= T6_DEBUG_REGS__CHICKEN_BITS__STALLWAIT_WAITS_JUST_FOR_PACK_DEST_READ_bm;
            // Read back so the posted MMIO write has retired before the first pack STALLWAIT issues.
            volatile uint32_t chicken_bits_readback = RISCV_DEBUG_REGS->CHICKEN_BITS;
            (void)chicken_bits_readback;
        }

        // [#55076 / 0x19] REPLAY banked-mode chicken bit -- directly indicated by the MFNR result.
        //
        // The per-unit fence (§4m/§4n) proved via CSR polling that `replay` busy never clears: the
        // REPLAY unit is the stuck one. CHICKEN_BITS.replay_disable_banked_mode is a 4-bit
        // per-TRISC field at bit 5 with RESET VALUE 0x1 (t6_debug_regs.rdl:2016), i.e. only TRISC0
        // ships with banked mode disabled -- MATH (TRISC1) runs the replay unit in Quasar-native
        // DOUBLE-BANKED mode. The RDL describes the bit as "Disables the double-banking mode in the
        // replay unit. When double-banking is disabled, the replay unit will appear to have 64
        // entries as it did in BH."
        //
        // The matmul is by far the heaviest replay user here (_llk_math_matmul_load_replay_ loads 15
        // MVMULs into slot 0, and every matmul_block replays them FIDELITY_PHASES times), so forcing
        // the BH-compatible non-banked behaviour on all TRISCs is a one-bit test of whether the
        // banked replay path is what wedges. Setting the full field mask disables banking everywhere.
        constexpr bool kDisableReplayBankedMode = false;  // WAVE: off = stock HW config
        if constexpr (kDisableReplayBankedMode) {
            RISCV_DEBUG_REGS->CHICKEN_BITS |= T6_DEBUG_REGS__CHICKEN_BITS__REPLAY_DISABLE_BANKED_MODE_bm;
            volatile uint32_t replay_cb_readback = RISCV_DEBUG_REGS->CHICKEN_BITS;
            ASSERT(
                (replay_cb_readback & T6_DEBUG_REGS__CHICKEN_BITS__REPLAY_DISABLE_BANKED_MODE_bm) ==
                T6_DEBUG_REGS__CHICKEN_BITS__REPLAY_DISABLE_BANKED_MODE_bm);
        }
    });
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(mm_in0_cb_id, in1_cb_id, out_cb_id);
    matmul_block_init(mm_in0_cb_id, in1_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);

#ifdef ARCH_QUASAR
    // [#55076 / 0x19] WHAT IS AT L1 0x36cc4?
    //
    // Watcher reports "PC 0x00036cc4" for this fault, but 0x36cc4 is NOT in any code region on
    // Quasar (dev_mem_map.h: TRISC firmware 0x11e00-0x16e00, TRISC kernels at MEM_KERNEL_BASE
    // 0x400000 -- and the kernel ELF confirms .text = 0x400000-0x401dfc). 0x36cc4 = 224452 sits
    // above MEM_PACKET_HEADER_POOL_BASE (0x303f0), i.e. in the L1 kernel-config / heap region where
    // the DFB configs and interfaces live. So watcher's "PC" label is wrong for this sub-error --
    // quasar_error_data_is_pc() returns true for ERROR_TRISC0..3 (debug_helpers.hpp:126-135) but
    // error_handling.h:82-86 describes MEM_READ_NO_RESPONSE as "a read went out and nothing
    // answered ... Look at the TARGET rather than the core ... the load still completes, with zero
    // data, so the kernel carries on with a bogus value and whatever breaks next is a knock-on."
    //
    // That reframes the whole failure: the compute-side wedge may be the knock-on, and the root
    // cause an unanswered L1 read at 0x36cc4 -- on TRISC0/UNPACK, whose last waypoint (RBD) sits
    // immediately after reserve_back_impl's read of
    //   local_dfb_interface_.tc_slots[tc_idx].packed_tile_counter
    // (dataflow_buffer.inl:143-145), which is exactly a read into this region.
    //
    // Print the DFB interface addresses once at init so we can see which structure -- if any --
    // 0x36cc4 falls inside, and whether it is a live allocation or a gap. Init-only, so run this
    // once with TT_METAL_DPRINT_CORES set; it does not perturb the loop.
    // ERR_DATA is confirmed (HW team) to be the last committed RISC PC, and TRISC code executes from
    // L1 -- so 0x36cc4 / 0x3713c ARE real PCs. They are absent from the kernel ELF only because the
    // ELF is linked at MEM_KERNEL_BASE 0x400000 while the text actually runs from a lower L1 base.
    // (Confirmation that it is code: the PC moved 0x36cc4 -> 0x3713c, +0x478, when the only change
    // was adding the DPRINT below -- i.e. it shifted with the code.)
    //
    // Print &kernel_main to recover the link->runtime delta so reported PCs can be looked up in the
    // objdump directly:
    //   * prints ~0x35xxx  => text is relocated at load; delta = 0x400000 - printed, and
    //                         PC_link = PC_reported + delta.
    //   * prints ~0x400xxx => text is NOT relocated (an address window maps fetches into L1), so the
    //                         delta must come from the host's kernel-text L1 allocation instead.
    //
    // Provisional mapping, pending that number: if the faulting load is the DFB-interface read right
    // after WAYPOINT("RBD") -- objdump 0x4019f0 sw (RBD), 0x4019f8 lw armed_mask, 0x4019fc lbu
    // local_dfb.tc_idx -- then the runtime base is ~0x352c8 and PC 0x36cc4 lands on 0x4019f8/0x4019fc.
    // That is consistent but circular until &kernel_main pins the base independently.
    // ANSWERED 2026-09-02: KTEXT kernel_main = 0x36C20 and the ELF links it at 0x400950, so
    //   PC_link = PC_runtime + 0x3C9D30   (TRISC0 text base 0x362D0)
    // DFBADDR came back act=0x8020AC wts=0x802104 tilized=0x8021B4 partials=0x80220C out=0x802264 --
    // all in TRISC local data RAM (MEM_LOCAL_BASE 0x802000), evenly spaced 88B apart and sane, which
    // rules out a bad DFB-interface address. Note the delta above is TRISC0-only; each TRISC has its
    // own kernel slot, so re-measure per thread if a TRISC1/2 PC needs mapping.
    // Left off by default: DPRINT itself destabilises Quasar (it tripped a watcher assert on 2026-09-02
    // 19:50 and changes the failure). Flip to true only to re-measure.
    UNPACK({
        constexpr bool kPrintKernelTextAndDfbAddrs = false;
        if constexpr (!kPrintKernelTextAndDfbAddrs) {
        } else {
            DPRINT_UNPACK("KTEXT kernel_main={}\n", (uint32_t)reinterpret_cast<uintptr_t>(&kernel_main));
            DPRINT_UNPACK(
                "DFBADDR act={} wts={} mmin0={} tilized={} partials={} out={}\n",
                (uint32_t)&get_local_dfb_interface(in0_cb_id),
                (uint32_t)&get_local_dfb_interface(in1_cb_id),
                (uint32_t)&get_local_dfb_interface(mm_in0_cb_id),
                (uint32_t)&get_local_dfb_interface(tilized_in0_cb_id),
                (uint32_t)&get_local_dfb_interface(matmul_partials_cb),
                (uint32_t)&get_local_dfb_interface(out_cb_id));
        }
    });
#endif
#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif
    UNPACK(SAVE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials);)
    PACK(SAVE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials);)
    for (uint32_t in1_block_w_i = 0; in1_block_w_i < in1_num_blocks_w; ++in1_block_w_i) {
        for (uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
            bool enable_reload = false;

            if constexpr (pack_relu) {
                PACK((llk_pack_relu_config(ReluConfig::none())));
            }
            if constexpr (partials_cb_uses_output) {
                UNPACK(RESAVE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials);)
                PACK(RESAVE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials);)
            }
            uint32_t curr_matmul_out_cb = matmul_partials_cb;
            for (uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
                bool last_inner_dim_block = (in0_block_w_i == in0_num_blocks_w - 1);
                // TRISC1) so the LAST BS* line in that file before the 0x19 pins the stall point:
                //   BSLOOP present but no "mmin0-ok" -> stuck in cb_mm_in0.wait_front (tilized act never
                //     delivered by the mcast reader / tilize never completed for this K-block).
                //   "mmin0-ok" but no "in1-ok"       -> stuck in cb_in1.wait_front (weights never delivered
                //     by the DM3 weights mcast).
                //   "in1-ok" (+ existing MMBLK/MMMV) then fault -> the matmul MVMULs read missing SrcA/SrcB.
                if constexpr (!height_sharded) {
                    if (in0_block_w_i % in0_nblocks_w_tilize == 0) {
                        if constexpr (pack_relu && !fuse_bias) {
                            if (last_inner_dim_block) {
                                PACK((llk_pack_relu_config(ReluConfig::none())));
                            }
                        }
                        if constexpr (packer_l1_acc) {
                            pack_reconfig_data_format(curr_matmul_out_cb, tilized_in0_cb_id);
                            pack_reconfig_l1_acc(0);
                        }
#ifdef ARCH_QUASAR
                        // [#48552] Re-seed MATH/PACK sync + repoint the pack BD before the block-sharded tilize
                        // (matmul->tilize). This fixed the tilize side (207 blocks OK). NOTE: the dvalid scrub
                        // (llk_math_set_dvalid) was tried here AND before the matmul and did NOT help the 0x19 --
                        // the fault is the SyncFull single-DEST-section datacopy<->pack MOP collision (see
                        // conv2d_op_sharded_program_factory.cpp #47797 notes), not a stale dvalid.
                        MATH((llk_math_pack_sync_init()));
                        PACK((llk_pack_init(tilized_in0_cb_id)));
                        PACK((llk_pack_dest_init()));
#endif
                        tilize_in<
                            in0_block_w,
                            in0_pretilize_cb_id,
                            tilized_in0_cb_id,
                            true,
                            !split_reader || split_reader_cb_shared,
                            compute_kernel_lib::tilize_config::RemapMode::Configure>(in0_num_subblocks_read);

#ifdef SPLIT_READER
                        if constexpr (split_reader && !split_reader_cb_shared) {
                            tilize_in<in0_block_w, in0_cb_second_reader_id, tilized_in0_cb_id, false, true>(
                                in0_num_subblocks_read_last);
                        }
#endif
                        reconfig_data_format(in0_pretilize_cb_id, in1_cb_id, in0_pretilize_cb_id, in0_cb_id);
                        matmul_block_init(mm_in0_cb_id, in1_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
                    }
                } else {
                    if constexpr (pack_relu && !fuse_bias) {
                        if (last_inner_dim_block) {
                            PACK((llk_pack_relu_config(ReluConfig::none())));
                        }
                    }
                    if constexpr (packer_l1_acc) {
                        pack_reconfig_data_format(curr_matmul_out_cb, tilized_in0_cb_id);
                        pack_reconfig_l1_acc(0);
                    }
#ifdef ARCH_QUASAR
                    // ROOT CAUSE (proven via MMBLK-absent + tile-index>obnt localization): on Quasar the plain
                    // `tilize_init` (tilize.h:63-69) does ONLY unpack+math init and omits ALL pack config --
                    // unlike the non-Quasar branch (:106-108), BH (:60-62), which all call llk_pack_hw_configure(ocb) +
                    // llk_pack_init(ocb). So the tilize's pack BUFFER DESCRIPTOR is never pointed at
                    // tilized_in0 -- it keeps the stale dfb::out base from compute_kernel_hw_startup, and the
                    // tilize packs tilized_in0's tiles into the OUT L1 region -> PACR0_TILE_INC / ERROR_TRISC1
                    // OOB (fires BEFORE the matmul pack). A prior workaround called only llk_pack_init here
                    // (sets the MOP buf_desc_id) but NOT llk_pack_hw_configure (which programs the BD BASE),
                    // so the BD base stayed stale. UPDATE (pool cross-check 2026-07-14): re-running
                    // llk_pack_hw_configure PER K-block is the "hw_configure is one-time" corruption that caused
                    // an UNPACKER fault in the Quasar pool -- and is the likely cause of the residual t=4
                    // DEST-bank fault here (state corruption surfacing after a bank rotation, NOT an LLK bug).
                    // The one-time pack hw_configure already ran pre-loop at compute_kernel_hw_startup. Drop the
                    // per-block hw_configure; llk_pack_init repoints the pack BD (the per-use-safe call the pool
                    // relies on per c-block). If this regresses to the earlier t=1 tilize OOB (stale BD base),
                    // the proper fix is to set the pack BD base ONCE in tilize.h's Quasar tilize_init.
                    // Quasar-only: re-seed the MATH<->PACK DEST semaphore + dest-bank phase for the tilize.
                    // Quasar's DEST handshake is a MATH_PACK-semaphore workaround (dest-dvalid gap); the plain
                    // Quasar tilize_init omits llk_math_pack_sync_init, so the tilize inherits the matmul's
                    // stale semaphore count / bank phase -> MATH issues its datacopy MOP into a DEST bank out
                    // of phase with PACK -> Risc IB interrupt (0x19) whose faulting tile/core move with DPRINT
                    // latency (a timing race, not OOB: TZCUR showed wr_entry_idx=0/nent=448). This is the
                    // MATH-side partner of the llk_pack_init/llk_pack_dest_init re-issued just below; runs once
                    // per tilize group, NO per-block hw_configure.
                    // NB: this seeds the MATH side; a RESIDUAL Quasar DEST-handshake race in the per-tile
                    // tilize_block datacopy<->pack loop (ERROR_TRISC1 0x19, fault tile/core move with DPRINT
                    // latency) survives even with correct init -- it is an LLK-team issue, see
                    // ~/llk_conv_tilize_issue.md.
                    //
                    // ROOT CAUSE (LLK hazard audit 2026-07-15): the Quasar MATH<->PACK *semaphore* dest-sync
                    // scheme never issues a CLEARDVALID, so the HW DEST data-valid bit SET by the preceding
                    // matmul's terminal MVMUL is never scrubbed. The tilize's MOVA2D datacopy MOP is then
                    // rejected at issue for targeting a bank whose dvalid is still set -> ERROR_TRISC1 (MATH)
                    // 0x19. (Standalone tilize passes because no prior op set the dvalid.) All prior fixes
                    // stalled on PACK, but the dvalid + ZEROACC retire on MATH/FPU, so a PACK stall can't drain
                    // them -- the fix is a state SCRUB, not a stall. Issue the CLEARDVALID (both banks in
                    // SyncFull, the active bank in SyncHalf) to clear the matmul's stale math dvalid before the
                    // tilize. Cheaper than compute_kernel_hw_startup (no hw_configure).
                    // [#48552] CLEARDVALID scrub REMOVED: llk_math_set_dvalid is static_assert-blocked on Quasar
                    // (it belongs to the dest-dvalid sync scheme and must not mix with the semaphore scheme
                    // tt-metal uses) AND it did not resolve the 0x19 in testing. Leaving it here was a hard
                    // trisc1 compile error once the static_assert was restored. The MATH pack-sync re-seed +
                    // pack init/dest_init below are the kept fix for the tilize inheriting stale matmul state.
                    MATH((llk_math_pack_sync_init()));
                    PACK((llk_pack_init(tilized_in0_cb_id)));
                    // A/B RESULT: disabling this moved the tilize fault EARLIER (t=4 -> t=1), so dest_init
                    // HELPS (sets up the packer DEST section Quasar tilize_init omits) — keep it. The residual
                    // fault at t=4 (first DEST bank-0 reuse after a full 4-bank rotation) is the tilize's own
                    // Quasar DEST bank rotation/release (pack_dest_section_done) not freeing banks — an LLK bug.
                    PACK((llk_pack_dest_init()));
                    // [DIAG cursor] Confirm the t=5 tilize pack OOB is the ring cursor: llk_pack writes
                    // tc_slots[tc_idx].wr_entry_idx + t into ACT_TILIZED (t up to 15). If wr_entry_idx != 0
                    // (the in-place matmul-partials ring rewind leaves ACT_TILIZED off ring-start) the +t
                    // overshoots the ring around t=5. Gated to the first tilize so it flushes pre-fault.
                    if (in0_block_w_i == 0 && in0_block_h_i == 0) {
                        // [stale-pack-BD confirm] The tilize packs into act_tilized (tilized base). If the
                        // ERROR_TRISC1 0x19 fault address (~0x37c28) lands in the OUT CB's L1 range instead of
                        // the tilized CB's range, the packer BD is still pointed at OUT (from hw_startup) --
                        // the Quasar tilize_init omits the pack hw_configure that would repoint it. esz = entry
                        // size (bytes); tilized range = [tilized_base, tilized_base + nent*esz).
                    }
#endif
                    // (TZHWCFG/TZBD/TZBDTAB/TILIZEPACK probes removed — they confirmed the tilize pack BD is

                    if constexpr (!activation_reuse) {
                        tilize_in<in0_block_w, in0_cb_id, tilized_in0_cb_id, true, !split_reader>(
                            in0_num_subblocks_read);
                    }

#ifdef SPLIT_READER
                    if constexpr (split_reader) {
                        if constexpr (!activation_reuse) {
                            tilize_in<in0_block_w, in0_cb_second_reader_id, tilized_in0_cb_id, false, true>(
                                in0_num_subblocks_read_last);
                        } else {
#ifndef ARCH_QUASAR  // activation_reuse path is off for resnet; dead on Quasar (no evil_set_*)
                            PACK((cb_tilized_in0.evil_set_write_ptr(tilized_cb_start_address)));
#endif
                            tilize_in_reuse_split_reader<
                                in0_cb_id,
                                in0_cb_second_reader_id,
                                in0_block_w,
                                in0_num_subblocks_read,
                                in0_num_subblocks_read_last,
                                tilized_in0_cb_id,
                                out_cb_tiles,
                                window_reuse_offset,
                                tilized_cb_row_offset,
                                tilized_cb_second_reader_offset,
                                image_width_in_tiles>(
                                cb_in0,
                                cb_in0_second_reader,
                                cb_tilized_in0,
                                act_cb_start_address,
                                act_cb_second_reader_start_address);
                        }
                    }
#endif

                    reconfig_data_format(in0_cb_id, in1_cb_id, in0_cb_id, mm_in0_cb_id);
                    matmul_block_init(mm_in0_cb_id, in1_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
                }

                cb_mm_in0.wait_front(in0_block_num_tiles);

                uint32_t in0_index_subblock_offset = 0;
#ifdef CHECK_SKIP_COMPUTE
                if (skip_compute) {
#ifdef ARCH_QUASAR
                    // TEN-4746 (#48552): this was a bare cb_mm_in0.wait_front (above) -> pop_front, which
                    // traps the Quasar unpacker -- POP_TILES can retire before the WAIT_TILES it follows,
                    // because nothing orders them. Every other wait/pop pair in this kernel already
                    // interposes a REAL unpack TDMA (see the matmul_partials drains at ~1019 and ~1050);
                    // this path was missed. NOP/TTI_NOP are INSUFFICIENT (LLK-team guidance +
                    // abhullar/pop-wait-fix 69014037a).
                    //
                    // Same idiom as those sites: reconfig srcA to the CB being drained, dummy copy_tile of
                    // tile 0, then restore srcA and re-init the matmul (copy_init clobbers the matmul MOP).
                    // The copy's result is discarded -- this exists only to order POP after WAIT.
                    //
                    // PR #54948 replaces this whole idiom with a dummy_unpack(dfb_id) helper (an UNPACR_NOP
                    // that reads nothing, so no reconfig/re-init and no DEST traffic). Once that lands,
                    // collapse this block to `dummy_unpack(mm_in0_cb_id);` and drop the reconfig pair.
                    reconfig_data_format_srca(in1_cb_id, mm_in0_cb_id);
                    copy_init(mm_in0_cb_id);
                    tile_regs_acquire();
                    copy_tile(mm_in0_cb_id, /*in_tile_index=*/0, /*dst_tile_index=*/0);
                    tile_regs_commit();
                    tile_regs_wait();
                    tile_regs_release();
#endif
                    cb_mm_in0.pop_front(in0_block_num_tiles);
#ifdef ARCH_QUASAR
                    reconfig_data_format_srca(mm_in0_cb_id, in1_cb_id);
                    matmul_block_init(mm_in0_cb_id, in1_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
#endif
                    continue;
                }
#endif

                cb_in1.wait_front(in1_block_num_tiles);

                if (last_inner_dim_block) {
                    if constexpr (!fuse_bias) {
                        if constexpr (pack_relu) {
                            PACK((llk_pack_relu_config(ReluConfig::zero())));
                        }
                        curr_matmul_out_cb = mm_out_cb_id;
                    }
                }

                if constexpr (packer_l1_acc) {
                    pack_reconfig_data_format(curr_matmul_out_cb);
                }
#ifdef ARCH_QUASAR
                // QSR quirk #1 (buffer descriptors are baked at op init, not recomputed per pack): the pack
                // BD was last programmed for dfb::out (compute_kernel_hw_startup) and the tilize left it
                // stale — it is NEVER repointed to the real matmul output CB. pack_block below runs the
                // init-baked MOP applying matmul_partials' tile *offset* on top of out's L1 *base* -> OOB
                // write -> PACR0_TILE_INC / ERROR_TRISC1 fault. Repoint the pack BD to the actual output CB
                // here (once per K-block; the reload path at 539+ doesn't touch pack config). WH/BH don't
                // need this (they recompute the full L1 addr from fifo_wr_ptr each pack). Mirrors
                // compute_pool_2d.cpp's llk_pack_init re-init.
                PACK((llk_pack_init(curr_matmul_out_cb)));
                // [#48552] Re-seed the MATH<->PACK DEST bank-recycle handshake for the MATMUL — the exact
                // MIRROR of the PROVEN pre-tilize re-seed (:564-570). Root cause (confirmed by removing the
                // MVMUL entirely: the hang PERSISTS byte-identical, so it is the acquire/commit/pack/
                // section_done/recycle HANDSHAKE, NOT compute — drop the earlier FPU-dvalid framing): a plain
                // matmul kernel runs this identical handshake and passes; the ONLY difference here is the
                // pretilize (datacopy->DEST->pack, 294 tiles) that ran FIRST on the SAME MATH_PACK semaphore
                // + DEST bank-parity, then the matmul reuses that machinery. The transition (reconfig /
                // matmul_block_init / the PACK llk_pack_init above) does NOT reset the MATH-side DEST section
                // pointer / MATH_PACK sem / bank-parity, so the matmul inherits the pretilize's ending phase.
                // That is self-consistent for the first two subblocks (banks 0,1 fresh) but DEADLOCKS at the
                // FIRST bank0 REUSE (out_subblock 2) -> ERROR_TRISC1 0x0119. llk_math_pack_sync_init drains the
                // pretilize's outstanding packs (cb_mm_in0.wait_front already guaranteed the data), resets
                // MATH dest parity to bank0 and re-seeds MATH_PACK to its SyncHalf init (max 2); llk_pack_dest_
                // init resets PACK parity to bank0 + re-selects the packer dest registers. Both threads restart
                // at a clean bank0/init handshake — the same clean start the standalone matmul gets from its
                // own init and the tilize gets from :564-570.
                MATH((llk_math_pack_sync_init()));
                PACK((llk_pack_dest_init()));
#endif
                // (flushes) so the LAST marker seen before the PACR0_TILE_INC fault tells whether the faulting
                // pack is the MATMUL (MMBLK) or the fuse_bias->OUT pack (BIASBLK). base is the current pack
                for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                    uint32_t in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock_i = 0; in1_subblock_i < in1_num_subblocks; ++in1_subblock_i) {
                        DPRINT_MATH(
                            "SUB k={} i0={} i1={} rl={}\n",
                            in0_block_w_i,
                            in0_subblock_i,
                            in1_subblock_i,
                            (uint32_t)enable_reload);
#ifdef ARCH_QUASAR
                        // [#55076 / 0x19 A/B PROBE] See qsr_x19_rtl_findings.md §4b. WAYPOINT is a single
                        // 32-bit store of a compile-time constant into the watcher mailbox
                        // (api/debug/waypoint.h:34-41) -- orders of magnitude lighter than DPRINT, and the
                        // cost is already being paid (the RBW/RBD we see today come from the same macro), so
                        // it perturbs the race far less than the per-subblock DPRINTs that are known to mask
                        // this bug. Watcher must be on; with watcher off every WAYPOINT compiles to nothing.
                        //
                        // MP<n> below publishes MATH_PACK *and* doubles as "MATH is at/inside
                        // tile_regs_acquire". Reading the interpretation table off the final waypoint line:
                        //
                        //   MATH last = MP2   -> blocked in tile_regs_acquire's SEMWAIT(STALL_ON_MAX) with the
                        //                        semaphore at max. STORY B: PACK is primary; PACK's own marker
                        //                        (PWAT/PPAK/PREL/PPSH) then says exactly where it is stuck.
                        //   MATH last = MP0/1 -> blocked in acquire with the semaphore NOT at max. Should be
                        //                        impossible (UnpackToDestEn is false here, so acquire is just
                        //                        the one SEMWAIT) -> would mean the RTL STALL_ON_MAX compare is
                        //                        wrong after all; re-open Q1/Q5 with a waveform.
                        //   MATH last = MACQ  -> blocked in matmul_block, i.e. SrcA/SrcB. Src bank recycle.
                        //   MATH last = MMVM  -> blocked in tile_regs_commit: STALLWAIT(-,WAIT_SFPU,MATH) on
                        //                        i_math_busy, or the SEMPOST itself. STORY A -> the shared
                        //                        tt_stall_scoreboard FIFO (findings §3a).
                        //   MATH last = MCMT  -> past commit; blocked heading into the next subblock.
                        //
                        // CAVEAT: waypoints track the RISC's program position, not Tensix retirement -- the
                        // RISC runs ahead until its instruction buffer fills, so the marker is at or slightly
                        // past the true stall. The semaphore_read below is a RISC MMIO read into the sync unit
                        // and tends to fence against the stalled pipe (that ordering is exactly why
                        // MEM_READ_NO_RESPONSE fires at all), which pulls the MATH marker back close to the
                        // real stall point. Read MP<n> as authoritative and the marker as a bound.
                        MATH({
                            const uint32_t mp = ckernel::semaphore_read(ckernel::trisc::semaphore::MATH_PACK);
                            if (mp == 0) {
                                WAYPOINT("MP0");
                            } else if (mp == 1) {
                                WAYPOINT("MP1");
                            } else if (mp == 2) {
                                WAYPOINT("MP2");
                            } else {
                                WAYPOINT("MPX");
                            }
                        });

                        // [#55076 / 0x19] "Slow but correct" fence: stop the MATH RISC running ahead.
                        //
                        // Every fix that moved the wedge so far removed one CFG access from the
                        // per-iteration path, and each bought roughly one more subblock
                        // (i0=1 -> i0=2). That is the signature of a FINITE QUEUE being filled, not a
                        // single blocking instruction: the RISC races ahead of its Tensix pipe, and a
                        // CFG access issued while _llk_math_dest_section_done_'s
                        // STALLWAIT(STALL_CFG, wait: MATH) is pending cannot retire, so the queue fills
                        // and the RISC wedges.
                        //
                        // These waiters poll csr_read<CSR::tensix_busy_status> (ckernel.h:576, :587+) --
                        // a core-internal CSR read, NOT an MMIO/CFG access, so unlike a STALLWAIT or a
                        // cfg store they cannot themselves be blocked by the stalled fabric. Spinning
                        // here holds the RISC at the top of each subblock until its pipe has fully
                        // drained, so it can never get more than one subblock ahead and the queue never
                        // fills.
                        //
                        // EXPENSIVE: this serialises the MATH RISC against its own pipe every subblock
                        // and gives up all RISC run-ahead. It is the "even if it is slow" option, not a
                        // shippable fix -- the real fix is whatever the HW team finds for why the pipe
                        // stops draining (see qsr_x19_rtl_findings.md §4j).
                        //
                        // If MATH's last waypoint is MFN0, the pipe never drains even with nothing
                        // queued behind it -- that is the cleanest possible evidence for §3a (the
                        // tt_stall_scoreboard leak) and should go straight to the HW team.
#ifdef ARCH_QUASAR
                        MATH({
                            constexpr bool kFenceMathRiscToPipePerSubblock = false;  // WAVE: off = natural run-ahead
                            if constexpr (kFenceMathRiscToPipePerSubblock) {
                                // RESULT 2026-09-03 11:52: MATH spun here forever (last waypoint MFN0)
                                // at i0=1. These are CSR polls, so they CANNOT be blocked by the
                                // fabric -- this is not a stuck access, the Tensix pipe genuinely never
                                // goes idle. Run-ahead / queue-fill is therefore eliminated as the
                                // cause. Split one marker per unit so the last one NAMES the unit whose
                                // busy bit never clears:
                                //   MFNM -> mop  never idle: the MOP sequencer never finishes issuing
                                //   MFNF -> fpu  never idle: an MVMUL never retires
                                //   MFNC -> cfg  never idle: the CFG exu never drains
                                // RESULT 2026-09-03 11:59: MFNM -- `mop` busy never clears.
                                // The matmul MOP is not flat: _llk_math_matmul_mop_config_ builds it as
                                // ckernel_template(1, FIDELITY_PHASES, TT_OP_REPLAY(0, replay_buf_len,
                                // ...), matmul_op), so the MVMUL body comes out of the REPLAY buffer.
                                // A stuck MOP is therefore either the outer sequencer or the replay
                                // unit under it. Split them:
                                //   MFNR -> replay busy never clears: the REPLAY unit is the stuck one
                                //   MFNM -> replay drains but mop does not: the outer MOP sequencer,
                                //           i.e. it cannot issue matmul_op (MVMUL blocked on SrcA/SrcB)
                                WAYPOINT("MFNR");
                                ckernel::wait_replay_idle();
                                WAYPOINT("MFNM");
                                ckernel::wait_mop_idle();
                                WAYPOINT("MFNF");
                                ckernel::wait_fpu_idle();
                                WAYPOINT("MFNC");
                                ckernel::wait_cfg_idle();
                                WAYPOINT("MFN1");
                            }
                        });
#endif
#endif
                        if (enable_reload) {
                            reconfig_data_format_srca(in1_cb_id, matmul_partials_cb);
                            copy_init(matmul_partials_cb);
                            cb_matmul_partials.wait_front(out_subblock_num_tiles);
                            tile_regs_acquire();

                            uint32_t start_dst_index = 0;
                            uint32_t start_tile_index = 0;
                            DPRINT_MATH("RLOAD_RD\n");
                            copy_block(matmul_partials_cb, start_tile_index, start_dst_index, out_subblock_num_tiles);

                            cb_matmul_partials.pop_front(out_subblock_num_tiles);
                            DPRINT_MATH("RLOAD_OK\n");
                            reconfig_data_format_srca(matmul_partials_cb, in1_cb_id);
                            matmul_block_init(
                                mm_in0_cb_id, in1_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
                        } else {
                            tile_regs_acquire();
                        }
#ifdef ARCH_QUASAR
                        MATH(WAYPOINT("MACQ"));  // [#55076 probe] past tile_regs_acquire

                        // [#55076 / 0x19 SERIALIZATION TEST] See qsr_x19_rtl_findings.md §4d-§4e.
                        // The probe showed MATH stalled INSIDE matmul_block (MACQ) while PACK was
                        // simultaneously stalled waiting for packer0 to go idle (PPAK) -- i.e. the FPU
                        // cannot finish a DEST write and the packer cannot finish a DEST read, at the same
                        // time. Not the semaphore (the SEMWAIT already passed), not dvalid (disabled), not
                        // SrcA/SrcB (UNPACK had delivered), not i_math_busy. That leaves the DEST access
                        // path both stalled clients contend for: tt_dest_prearb, whose four
                        // tt_flopped_rr_arb groups are all instantiated ENABLE_LIVELOCK_PREVENTION(0)
                        // (tt_dest_prearb.sv:786/:851/:885/:920) despite having coprime jostle periods
                        // wired up, over an all-or-nothing multi-resource acquisition (:972-978).
                        //
                        // This spin tests the TRIGGER rather than the mechanism: MATH refuses to start its
                        // MVMULs until the packer has drained every prior section, so a MATH DEST write is
                        // never in flight at the same time as a packer DEST read. If concurrency is the
                        // trigger, the hang must clear regardless of what the arbiter is doing.
                        //
                        // Why the semaphore instead of STALLWAIT(PACK0): "wait until MATH_PACK == 0" is not
                        // expressible as a SEMWAIT (STALL_ON_ZERO waits *while* it is zero), and
                        // STALLWAIT(PACK0) has the classic wait-for-something-that-has-not-started hole --
                        // it passes if the packer is momentarily idle before its next pack begins.
                        //
                        // No deadlock risk: MATH only reaches here after its own acquire, and PACK needs
                        // MATH_PACK > 0 to pack -- which MATH already posted for the previous section, so
                        // PACK always drains to 0 and releases this spin. First iteration sees 0 already.
                        //
                        // EXPENSIVE -- this serializes MATH and PACK and gives up DEST double-buffering, so
                        // it is a diagnostic, not a fix. Set to false to disable.
                        MATH({
                            // RESULT 2026-09-02: tested true -> MATH reached MDRD (MATH_PACK really did
                            // drain to 0, so the packer was provably idle and no packer DEST read was in
                            // flight) and then still stalled in the MVMUL loop. That REFUTES the
                            // concurrency trigger and with it the §4e arbiter-livelock hypothesis: MATH
                            // cannot complete a DEST write even with the DEST port entirely to itself.
                            // Left off; flip to true to reproduce.
                            constexpr bool kSerializeMathVsPackDest = false;
                            if constexpr (kSerializeMathVsPackDest) {
                                WAYPOINT("MDRN");  // stalled here => PACK never drained the prior section
                                while (ckernel::semaphore_read(ckernel::trisc::semaphore::MATH_PACK) != 0) {
                                }
                                WAYPOINT("MDRD");  // drained; no packer DEST reads should be in flight now
                            }
                        });
#endif

                        uint32_t dst_index = 0;
                        uint32_t in0_index = in0_index_subblock_offset;
                        uint32_t in1_index = in1_index_subblock_offset;
                        // prints for (i0,i1) but MMMVOK does NOT, the MATH 0x19 is in that subblock's matmul_block.
                        // Gated to the first height block (bsp1 faulted there, ~3 MMPACKs in).
                        for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; inner_dim_idx++) {
                            // hang is at idx=0 the very first MVMUL of subblock 2 stalls (SrcA/SrcB unpack for
                            // that DEST section never validated); if idx>0 it stalls mid-K accumulation. Prints
                            // the mm_in0 / in1 tile indices being read (to check for an OOB read too).
                            // NEW text => proves the JIT recompiled. On TRISC0 (UNPACK). If "UNPK i0=2 idx=0" prints
                            // (unpacker delivered SrcA/SrcB for the stalling MVMUL) yet MATH still hangs at MMK i0=2
                            // idx=0 => the fault is the DEST FPU-dvalid (a) and the per-subblock clear is
                            // ineffective/misplaced. If the last UNPK is i0=1 idx=2 (unpacker stalled ENTERING
                            // subblock 2) => it's SrcA/SrcB unpack re-arm (b), fix belongs in the unpack handshake.
                            matmul_block(
                                mm_in0_cb_id,
                                in1_cb_id,
                                in0_index,
                                in1_index,
                                dst_index,
                                false,
                                out_subblock_w,
                                out_subblock_h,
                                in0_block_w);
                            in0_index++;
                            in1_index += in1_block_w;
                        }
                        // [#48552] all MVMULs for this subblock completed (MATH survived the matmul_block loop).
#ifdef ARCH_QUASAR
                        MATH(WAYPOINT("MMVM"));  // [#55076 probe] past the MVMUL loop, entering commit
#endif

#ifdef SFPU_OP_INIT_ACTIVATION
                        if constexpr (!fuse_bias) {
                            if (last_inner_dim_block) {
                                for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                                    SFPU_OP_FUNC_ACTIVATION
                                }
                            }
                        }
#endif
                        tile_regs_commit();
#ifdef ARCH_QUASAR
                        MATH(WAYPOINT("MCMT"));  // [#55076 probe] past tile_regs_commit
#endif
                        {
                            DataflowBuffer curr_out_cb =
                                curr_matmul_out_cb == matmul_partials_cb ? cb_matmul_partials : cb_mm_out;
                            curr_out_cb.reserve_back(out_subblock_num_tiles);
                            tile_regs_wait();
#ifdef ARCH_QUASAR
                            PACK(WAYPOINT("PWAT"));  // [#55076 probe] past tile_regs_wait (SEMWAIT ON_ZERO)
#endif

                            if constexpr (packer_l1_acc) {
                                if (in0_block_w_i == 0) {
                                    pack_reconfig_l1_acc(0);
                                } else if (last_inner_dim_block) {
                                    pack_reconfig_l1_acc(fuse_bias ? 1 : 0);
                                } else {
                                    pack_reconfig_l1_acc(1);
                                }
                            }

                            uint32_t start_dst_index = 0;
#ifdef ARCH_QUASAR
                            // QSR matmul-pack DST addressing fix. The Quasar SEQUENTIAL pack
                            // (pack_block -> llk_pack_block -> get_output_tile_index<out_of_order=false>)
                            // computes l1_tile_index = tc_slots[tc_idx].wr_entry_idx + wr_entry_ptr, where
                            // wr_entry_idx advances per push_back (llk_push_tiles) AND wr_entry_ptr is a
                            // monotonic per-pack counter that reserve_back/push_back never reset. Those two
                            // DOUBLE-advance the DST address, so across the no-spill multi-height-block matmul
                            // (in0_num_blocks_h > 1) the pack drifts ~2x and walks off the OUT/partials tile
                            // boundary -> PACR0_TILE_INC OOB (ERROR_TRISC1). WH/BH don't hit this because their
                            // pack recomputes the L1 addr from the CB fifo_wr_ptr each pack. Mirror the WORKING
                            // Quasar tilize pack: out_of_order with a RELATIVE tile index (0..osnt-1). That path
                            // (get_output_tile_index<out_of_order=true>) uses ONLY wr_entry_idx + the explicit
                            // index and never touches wr_entry_ptr, so it single-advances and stays tile-aligned
                            // -- the portable "reset write ptr after push_back" sequential semantics. Each
                            // subblock reserve_back(osnt)/push_back(osnt) advances wr_entry_idx by osnt, so the
                            // relative 0..osnt-1 lands in the correct sequential OUT/partials slot for every
                            // (height-block, subblock), identical to the pre-Quasar pack_block behavior.
                            for (uint32_t t = 0; t < out_subblock_num_tiles; ++t) {
                                pack_tile<true /*out_of_order_output*/>(start_dst_index + t, curr_matmul_out_cb, t);
                            }
#else
                            pack_block(start_dst_index, curr_matmul_out_cb, out_subblock_num_tiles);
#endif

#ifdef ARCH_QUASAR
                            PACK(WAYPOINT("PPAK"));  // [#55076 probe] past the PACR0_TILE_INC MOP(s)
#endif

                            tile_regs_release();
#ifdef ARCH_QUASAR
                            // Past _llk_pack_dest_semaphore_section_done_: STALLWAIT(PACK0) + ZEROACC + the
                            // SEMGET that releases MATH. If PACK's last marker is PPAK, it is stuck in that
                            // STALLWAIT/SEMGET -- Story B with the packer as primary.
                            PACK(WAYPOINT("PREL"));
#endif
                            curr_out_cb.push_back(out_subblock_num_tiles);
#ifdef ARCH_QUASAR
                            PACK(WAYPOINT("PPSH"));  // [#55076 probe] past PUSH_TILES
#endif
                        }

                        in1_index_subblock_offset += out_subblock_w;
                    }  // for in1_num_subblocks
                    in0_index_subblock_offset += in0_subblock_num_tiles;
                }
                if (curr_matmul_out_cb == matmul_partials_cb) {
                    if constexpr (!partials_cb_uses_output) {
                        UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials);)
                        PACK(RESTORE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials);)
                    }
                }
                if constexpr (packer_l1_acc) {
                    if constexpr (fuse_bias) {
                        if (in0_block_w_i < in0_num_blocks_w - 1) {
#ifdef ARCH_QUASAR
                            // TEN-4746: a bare wait_front->pop_front traps the Quasar unpacker (POP_TILES races
                            // past WAIT_TILES). Interpose a REAL unpack TDMA (dummy copy_tile of tile 0); NOP/
                            // TTI_NOP are INSUFFICIENT (LLK-team guidance + abhullar/pop-wait-fix 69014037a).
                            // Reconfig srcA to partials for the copy, then restore srcA + re-init the matmul.
                            reconfig_data_format_srca(in1_cb_id, matmul_partials_cb);
                            copy_init(matmul_partials_cb);
#endif
                            cb_matmul_partials.wait_front(out_block_num_tiles);
#ifdef ARCH_QUASAR
                            tile_regs_acquire();
                            copy_tile(matmul_partials_cb, /*in_tile_index=*/0, /*dst_tile_index=*/0);
                            tile_regs_commit();
                            tile_regs_wait();
                            tile_regs_release();
#endif
                            cb_matmul_partials.pop_front(out_block_num_tiles);
#ifdef ARCH_QUASAR
                            reconfig_data_format_srca(matmul_partials_cb, in1_cb_id);
                            matmul_block_init(
                                mm_in0_cb_id, in1_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
#endif
                            if constexpr (spill) {
                                UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials));
                                PACK(RESTORE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials));
                            }
                        }
                        enable_reload = false;
                    } else {
                        if (in0_block_w_i < in0_num_blocks_w - 2) {
#ifdef ARCH_QUASAR
                            // TEN-4746 (see above): REAL unpack TDMA (dummy copy_tile) between the bare
                            // wait_front/pop_front; NOP/TTI_NOP insufficient.
                            reconfig_data_format_srca(in1_cb_id, matmul_partials_cb);
                            copy_init(matmul_partials_cb);
#endif
                            cb_matmul_partials.wait_front(out_block_num_tiles);
#ifdef ARCH_QUASAR
                            tile_regs_acquire();
                            copy_tile(matmul_partials_cb, /*in_tile_index=*/0, /*dst_tile_index=*/0);
                            tile_regs_commit();
                            tile_regs_wait();
                            tile_regs_release();
#endif
                            cb_matmul_partials.pop_front(out_block_num_tiles);
#ifdef ARCH_QUASAR
                            reconfig_data_format_srca(matmul_partials_cb, in1_cb_id);
                            matmul_block_init(
                                mm_in0_cb_id, in1_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
#endif
                            if constexpr (spill) {
                                UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials));
                                PACK(RESTORE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials));
                            }
                        }
                        if (in0_block_w_i == in0_num_blocks_w - 2) {
                            enable_reload = true;
                        }
                    }
                } else {
                    if constexpr (spill) {
                        enable_reload = true;

                        if constexpr (fuse_bias) {
                            if (!last_inner_dim_block) {
                                UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials));
                                PACK(RESTORE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials));
                            }
                        } else {
                            if (!last_inner_dim_block) {
                                UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials));
                            }
                            if (in0_block_w_i < in0_num_blocks_w - 2) {
                                PACK(RESTORE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials));
                            }
                        }
                    }
                }

                cb_mm_in0.pop_front(in0_block_num_tiles);
                cb_in1.pop_front(in1_block_num_tiles);
            }  // for in0_num_blocks_w
            if constexpr (matmul_partials_cb == mm_out_cb_id && partials_cb_uses_output) {
                UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials));
            }
#ifdef CHECK_SKIP_COMPUTE
            if (skip_compute) {
                continue;
            }
#endif
#ifdef FUSE_BIAS
            if constexpr (fuse_bias) {
                if constexpr (pack_relu) {
                    PACK((llk_pack_relu_config(ReluConfig::zero())));
                }
                pack_reconfig_data_format(matmul_partials_cb, untilize_mode_out_cb_id);
                if constexpr (packer_l1_acc) {
                    pack_reconfig_l1_acc(0);
                }
#ifdef ARCH_QUASAR
                // QSR quirk #1: pack_reconfig_data_format above sets only the gasket FORMAT, not the pack
                // buffer descriptor. The pack BD is still pointed at matmul_partials (from the matmul-block
                // pack_init); pack_tile below targets untilize_mode_out_cb_id -> stale base + new offset ->
                // OOB PACR0_TILE_INC / ERROR_TRISC1 (this is the fault that surfaced after the matmul-pack
                // fix, at a higher L1 addr). Repoint the pack BD to the actual pack target CB.
                PACK((llk_pack_init(untilize_mode_out_cb_id)));
#endif
                reconfig_data_format(in1_cb_id, matmul_partials_cb, mm_in0_cb_id, bias_cb_id);
                add_bcast_rows_init_short(matmul_partials_cb, bias_cb_id);

                cb_bias.wait_front(bias_ntiles_w);
                cb_matmul_partials.wait_front(out_block_num_tiles);
                for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                    uint32_t in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock_i = 0; in1_subblock_i < in1_num_subblocks; ++in1_subblock_i) {
                        tile_regs_acquire();
                        uint32_t i = 0;
                        for (uint32_t h = 0; h < out_subblock_h; ++h) {
                            uint32_t bcast_tile_i = bias_block_offset + in1_index_subblock_offset;
                            for (uint32_t w = 0; w < out_subblock_w; ++w) {
                                add_tiles_bcast_rows(matmul_partials_cb, bias_cb_id, i, bcast_tile_i, i);
                                ++bcast_tile_i;
                                ++i;
                            }
                        }

#ifdef SFPU_OP_INIT_ACTIVATION
                        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                            SFPU_OP_FUNC_ACTIVATION
                        }
#endif
                        tile_regs_commit();
                        cb_matmul_partials.pop_front(out_subblock_num_tiles);

                        cb_untilize_mode_out.reserve_back(out_subblock_num_tiles);
                        tile_regs_wait();
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
#ifdef ARCH_QUASAR
                            // Same Quasar sequential-pack double-advance fix as the matmul pack: the default
                            // pack_tile (out_of_order=false) uses get_output_tile_index<false> =
                            // wr_entry_idx (advances per push_back) + monotonic wr_entry_ptr, which DOUBLE-
                            // advances the DST across this multi-subblock / multi-height-block bias->OUT pack
                            // (fuse_bias, partials aliases OUT) and walks off the tile boundary. Use
                            // out_of_order with the RELATIVE tile index i (single-advance via wr_entry_idx),
                            // mirroring the working tilize. WH/BH keep the sequential pack (fifo_wr_ptr path).
                            pack_tile<true /*out_of_order_output*/>(i, untilize_mode_out_cb_id, i);
#else
                            pack_tile(i, untilize_mode_out_cb_id);
#endif
                        }
                        tile_regs_release();
                        cb_untilize_mode_out.push_back(out_subblock_num_tiles);

                        in1_index_subblock_offset += out_subblock_w;
                    }  // for in1_num_subblocks
                }  // in0_num_subblocks
                if constexpr (untilize_out) {
                    UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials));
                    PACK(RESTORE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials));
                }
            }
#endif  // FUSE_BIAS
            if constexpr (untilize_out) {
                if constexpr (packer_l1_acc) {
                    pack_reconfig_data_format(matmul_partials_cb, out_cb_id);
                    pack_reconfig_l1_acc(0);
                }
                if constexpr (pack_relu) {
                    PACK((llk_pack_relu_config(ReluConfig::none())));
                }
                if constexpr (!fuse_bias) {
                    reconfig_data_format_srca(in1_cb_id, matmul_partials_cb);
                }

                if constexpr (packer_untilize) {
                    pack_untilize_dest_init<out_subblock_w, out_block_w>(out_cb_id);
                    copy_init(matmul_partials_cb);
                    for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                        reblock_and_untilize<out_subblock_w, out_block_w>(
                            cb_matmul_partials, cb_out, in1_num_subblocks, out_subblock_num_tiles, out_subblock_h);
                    }
                    pack_untilize_uninit(matmul_partials_cb);
                } else {
                    compute_kernel_lib::untilize<
                        out_block_w,
                        matmul_partials_cb,
                        out_cb_id,
                        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
                        in0_num_subblocks * out_subblock_h);
                }
            }
            if constexpr ((in1_num_blocks_w > 1 || in0_num_blocks_h > 1)) {
#ifdef FUSE_BIAS
                if constexpr (fuse_bias) {
                    reconfig_data_format(matmul_partials_cb, in1_cb_id, bias_cb_id, mm_in0_cb_id);
                } else
#endif
                {
                    reconfig_data_format_srca(matmul_partials_cb, in1_cb_id);
                }
            }
        }  // for in0_num_blocks_h
#ifdef FUSE_BIAS
        if constexpr (fuse_bias) {
            bias_block_offset += in1_block_w;
        }
#endif
    }  // for in1_num_blocks_w
}  // void kernel_main()
