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
// SUNSET: delete when the legacy conv_bmm_tilize.cpp loses its last legacy consumer.

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
#define QSR_SNAPSHOT_WR(pos, dfb)                                  \
    do {                                                           \
        LocalDFBInterface& _qd = get_local_dfb_interface((dfb).get_id()); \
        for (uint8_t _qi = 0; _qi < _qd.num_tcs_to_rr; ++_qi) {    \
            (pos).entry_idx[_qi] = _qd.tc_slots[_qi].wr_entry_idx; \
            (pos).offset[_qi] = _qd.tc_slots[_qi].wr_offset;       \
        }                                                          \
        (pos).entry_ptr = _qd.wr_entry_ptr;                        \
        (pos).tc_idx = _qd.tc_idx;                                 \
    } while (0)
#define QSR_RESTORE_WR(pos, dfb)                                   \
    do {                                                           \
        LocalDFBInterface& _qd = get_local_dfb_interface((dfb).get_id()); \
        for (uint8_t _qi = 0; _qi < _qd.num_tcs_to_rr; ++_qi) {    \
            _qd.tc_slots[_qi].wr_entry_idx = (pos).entry_idx[_qi]; \
            _qd.tc_slots[_qi].wr_offset = (pos).offset[_qi];       \
        }                                                          \
        _qd.wr_entry_ptr = (pos).entry_ptr;                        \
        _qd.tc_idx = (pos).tc_idx;                                 \
    } while (0)
#define QSR_SNAPSHOT_RD(pos, dfb)                                  \
    do {                                                           \
        LocalDFBInterface& _qd = get_local_dfb_interface((dfb).get_id()); \
        for (uint8_t _qi = 0; _qi < _qd.num_tcs_to_rr; ++_qi) {    \
            (pos).entry_idx[_qi] = _qd.tc_slots[_qi].rd_entry_idx; \
            (pos).offset[_qi] = _qd.tc_slots[_qi].rd_offset;       \
        }                                                          \
        (pos).tc_idx = _qd.tc_idx;                                 \
    } while (0)
#define QSR_RESTORE_RD(pos, dfb)                                   \
    do {                                                           \
        LocalDFBInterface& _qd = get_local_dfb_interface((dfb).get_id()); \
        for (uint8_t _qi = 0; _qi < _qd.num_tcs_to_rr; ++_qi) {    \
            _qd.tc_slots[_qi].rd_entry_idx = (pos).entry_idx[_qi]; \
            _qd.tc_slots[_qi].rd_offset = (pos).offset[_qi];       \
        }                                                          \
        _qd.tc_idx = (pos).tc_idx;                                 \
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
#ifndef ARCH_QUASAR  // Quasar has no fast tilize; these helpers are only reached on the split_reader/
                     // activation_reuse path, which the resnet conv factories force OFF. Guard the
                     // raw fast_tilize_* names out so the template body parses on Quasar (dead there).
    fast_tilize_block(in_cb_id, in_block_w, out_cb_id);
#endif
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
#ifndef ARCH_QUASAR  // Quasar has no fast tilize (split_reader/activation_reuse path, off for resnet)
    fast_tilize_init_with_dt(in1_cb_id, in_block_w, out_cb_id);
#endif

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
#ifndef ARCH_QUASAR  // Quasar has no fast tilize (split_reader/activation_reuse path, off for resnet)
    fast_tilize_uninit(in2_cb_id, out_cb_id, in_block_w);
#endif
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
    [[maybe_unused]] const uint32_t tilized_cb_start_address =
        activation_reuse ? cb_tilized_in0.get_write_ptr() : 0;
#ifdef SPLIT_READER
    [[maybe_unused]] const uint32_t act_cb_second_reader_start_address =
        activation_reuse ? cb_in0_second_reader.get_read_ptr() : 0;
#endif
#endif

#ifdef CHECK_SKIP_COMPUTE
    bool skip_compute = (bool)get_arg(args::skip_compute);
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(mm_in0_cb_id, in1_cb_id, out_cb_id);
    matmul_block_init(mm_in0_cb_id, in1_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
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
                    cb_mm_in0.pop_front(in0_block_num_tiles);
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
                for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                    uint32_t in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock_i = 0; in1_subblock_i < in1_num_subblocks; ++in1_subblock_i) {
                        if (enable_reload) {
#ifndef ARCH_QUASAR
                            copy_tile_to_dst_init_short_with_dt(in1_cb_id, matmul_partials_cb);
#else
                            // QSR: copy_tile_to_dst_init_short_with_dt is WH/BH-only; expand it into its
                            // two constituent steps (identical reconfig + copy init) on Quasar.
                            reconfig_data_format_srca(in1_cb_id, matmul_partials_cb);
                            copy_tile_to_dst_init_short(matmul_partials_cb);
#endif
                            cb_matmul_partials.wait_front(out_subblock_num_tiles);
                            tile_regs_acquire();

                            uint32_t start_dst_index = 0;
                            uint32_t start_tile_index = 0;
                            copy_block_matmul_partials(
                                matmul_partials_cb, start_tile_index, start_dst_index, out_subblock_num_tiles);

                            cb_matmul_partials.pop_front(out_subblock_num_tiles);
                            reconfig_data_format_srca(matmul_partials_cb, in1_cb_id);
                            matmul_block_init(
                                mm_in0_cb_id, in1_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
                        } else {
                            tile_regs_acquire();
                        }

                        uint32_t dst_index = 0;
                        uint32_t in0_index = in0_index_subblock_offset;
                        uint32_t in1_index = in1_index_subblock_offset;
                        for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; inner_dim_idx++) {
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
                        {
                            DataflowBuffer curr_out_cb =
                                curr_matmul_out_cb == matmul_partials_cb ? cb_matmul_partials : cb_mm_out;
                            curr_out_cb.reserve_back(out_subblock_num_tiles);
                            tile_regs_wait();

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
                            pack_tile_block(start_dst_index, curr_matmul_out_cb, out_subblock_num_tiles);

                            tile_regs_release();
                            curr_out_cb.push_back(out_subblock_num_tiles);
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
                            cb_matmul_partials.wait_front(out_block_num_tiles);
                            cb_matmul_partials.pop_front(out_block_num_tiles);
                            if constexpr (spill) {
                                UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials));
                                PACK(RESTORE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials));
                            }
                        }
                        enable_reload = false;
                    } else {
                        if (in0_block_w_i < in0_num_blocks_w - 2) {
                            cb_matmul_partials.wait_front(out_block_num_tiles);
                            cb_matmul_partials.pop_front(out_block_num_tiles);
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
                            pack_tile(i, untilize_mode_out_cb_id);
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
                    copy_tile_to_dst_init_short(matmul_partials_cb);
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
