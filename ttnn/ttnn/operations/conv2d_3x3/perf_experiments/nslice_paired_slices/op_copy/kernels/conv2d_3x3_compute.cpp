// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 compute: per M-block, [untilize_footprint (TILE input)] then tilize_matmul_block
// (matmul_block with tilize as its PreKBlockFn, fp32 packer-L1 accumulation over the K walk) then
// finalize_block (row-broadcast bias add). No bias: the last K-block packs straight to the output CB
// and finalize_block is empty. ReLU applies only to the final value: the bias-add pack, or the last
// K-block's pack without bias.
//
// Strided-window im2col (stride 2): compute tilizes each unit in place at a 2-pixel row stride (window_tilize).
//
// Center-tap direct in0 (see the reader's file comment): center_mode 1 skips the tilize of K-blocks k % 3 == 1 and
// matmul_block's In0SourceFn reads them from cb_center_tiles (same data format as cb_im2col_tiles); center_mode 2
// tilizes taps 0 and 2 of every im2col row and reserves + pushes tap 1's slots unpacked (the reader wrote them).
//
// Face-direct im2col: the reader writes the in0 tiles itself, so the matmul runs without the tilize hook. With one
// K-block per M-block the reader pushes in0 in row groups and compute runs one matmul call per group as it lands,
// keeping the weights fronted across the calls, and adds the bias once per bias_stream_groups groups, so only the
// last chunk's bias add trails the gather.
//
// matmul_block leaves packer L1 accumulation on and (OutWithRelu) PACK_RELU on; every later pack
// (untilize, tilize, combine, bias add) first restores the state it needs.
//
// N-slice-outer K split (the default under a K split, when the resident in0 fits L1): see the branch below and
// conv2d_3x3_nslice_order.hpp. The K-blocked walk below remains for K splits whose [bm, K] in0 does not fit.
//
// K split (BLOCK_SHARDED, k_cols > 1): the matmul accumulates this core's channel slice into fp32
// partials [bm, all C_out] that the WRITER consumes (it scatters N-slice y to grid column y of the row),
// so compute owns that CB's reserve/push once per M-block instead of letting the helper drain it per
// K-block (the helper's drain would make compute a second consumer). The K-blocks still serialize on the
// packer: each one's tilize pushes cb_im2col_tiles between two L1-accumulating passes. Then, on a core
// that owns output columns, combine_block sums the k_cols slices the row sent it (pairwise FPU adds into
// fp32 DEST), adds the bias and packs into the output shard; ReLU is armed only for that last pack.

#include <cstdint>
#include <type_traits>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "conv2d_3x3_nslice_order.hpp"
#include "conv2d_3x3_narrow_untilize.hpp"

namespace ckl = compute_kernel_lib;

namespace {

// Strided-window tilize (see conv2d_3x3_gather_dataflow.hpp WindowGather): a cb_im2col_rm unit is one tile-row x one
// kernel row laid out as overlapping windows, im2col row i = 3 pixels (3 * c_in_tiles tile columns) from pixel slot
// 2 * i. The BH fast-tilize unpacker takes its row stride from the init's width and its read width from the chunk, so
// an init at 2 * c_in_tiles and chunks covering 3 * c_in_tiles read row i's 3 pixels at a 2-pixel stride: the same
// tiles, in the same order, as tilizing the [32, 3 * c_in_tiles] row-layout block (bit-identical).
// The unit's last 16 B hold the L1 address to tilize from (the unit itself, or a slab run on the N-slice-outer walk).
// Raw LLK, UNPACK only (capability): ckl::tilize ties the row stride to the block width and reads at the CB read
// pointer; the loop is fast_tilize_block's, with the unpacker's base taken from that address.
#ifdef TRISC_UNPACK
template <uint32_t cb_in, uint32_t block_tiles>
inline uint32_t window_unit_source() {
    invalidate_l1_cache();
    const uint32_t unit = get_local_cb_interface(get_operand_id(cb_in)).fifo_rd_ptr << 4;
    return *reinterpret_cast<volatile uint32_t*>(unit + block_tiles * 32 * 32 * 2 - 16);
}

template <uint32_t cb_in>
inline void window_unpack_chunk(uint32_t src, uint32_t chunk, uint32_t col_start) {
    const uint32_t operand_id = get_operand_id(cb_in);
    const uint32_t src_format = unpack_src_format[operand_id];
    const uint32_t base = (src >> 4) - 1 + (SCALE_DATUM_SIZE(src_format, col_start * TILE_C_DIM) >> 4);
    _llk_unpack_fast_tilize_block_(base, 0, src_format, chunk, get_operand_num_faces(operand_id));
}
#endif

template <uint32_t c_in_tiles, uint32_t cb_in, uint32_t cb_out>
ALWI void window_tilize(DataflowBuffer& in, DataflowBuffer& out, uint32_t units) {
    constexpr uint32_t row_tiles = 2 * c_in_tiles;
    constexpr uint32_t block_tiles = 3 * c_in_tiles;
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
    fast_tilize_init(cb_in, row_tiles, cb_out);
    // Every unit has the same chunk sequence: reprogram the unpack width / pack MOP only when the chunk changes.
    uint32_t prev_chunk = 0;
    for (uint32_t u = 0; u < units; ++u) {
        in.wait_front(block_tiles);
        out.reserve_back(block_tiles);
        [[maybe_unused]] uint32_t src = 0;
        UNPACK(src = window_unit_source<cb_in, block_tiles>());
        PACK((llk_pack_fast_tilize_row_begin(cb_out, 0)));
        for (uint32_t tiles_done = 0; tiles_done < block_tiles;) {
            // BH fast-tilize chunks are 2..4 tiles (fast_tilize_block's split: 5 -> 2 + 3).
            const uint32_t remaining = block_tiles - tiles_done;
            const uint32_t chunk = (remaining > 5) ? 4 : (remaining == 5) ? 2 : remaining;
            MATH((llk_math_wait_for_dest_available()));
            PACK((llk_packer_wait_for_math_done()));
            if (chunk != prev_chunk) {
                UNPACK((llk_unpack_fast_tilize_reinit_xdim(chunk)));
                PACK((llk_pack_fast_tilize_reinit_unit_dim(cb_out, chunk)));
                prev_chunk = chunk;
            }
            UNPACK((window_unpack_chunk<cb_in>(src, chunk, tiles_done)));
            MATH((llk_math_fast_tilize_block_<DST_ACCUM_MODE>(0, cb_in, 4)));
            PACK((llk_pack_fast_tilize_row_chunk(0, cb_out, chunk)));
            MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
            PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
            tiles_done += chunk;
        }
        PACK((llk_pack_fast_tilize_row_end()));
        out.push_back(block_tiles);
        in.pop_front(block_tiles);
    }
    fast_tilize_uninit(cb_in, cb_out, row_tiles);
}

// Center-tap direct modes (the descriptor's center_tap_mode).
constexpr uint32_t CENTER_K_BLOCK = 1;
constexpr uint32_t CENTER_SLOTS = 2;

// Center-tap slots: tilizes num_rows im2col rows of three taps (tap-major, each a [32, tw * 32] block), except tap 1,
// whose tw tiles the reader wrote into their in0 slots: those are reserved and pushed unpacked, and the row's empty
// tap-1 block is popped unread. One tilize init / uninit for all of it, as in the one-call form.
template <uint32_t tw, uint32_t cb_rm, uint32_t cb_tiles>
ALWI void tilize_rows_skip_center(DataflowBuffer& rm, DataflowBuffer& tiles, uint32_t num_rows) {
    namespace tc = ckl::tilize_config;
    constexpr auto NoRc = tc::ReconfigureRegisterDatatypeMode::NoReconfigure;
    for (uint32_t r = 0; r < num_rows; ++r) {
        if (r == 0) {
            ckl::tilize<tw, cb_rm, cb_tiles, tc::InitUninitMode::InitOnly>(rm, tiles, 1);
        } else {
            ckl::tilize<tw, cb_rm, cb_tiles, tc::InitUninitMode::Neither, tc::WaitMode::WaitBlock, NoRc>(rm, tiles, 1);
        }
        tiles.reserve_back(tw);
        tiles.push_back(tw);
        rm.wait_front(tw);
        rm.pop_front(tw);
        if (r + 1 == num_rows) {
            ckl::tilize<tw, cb_rm, cb_tiles, tc::InitUninitMode::UninitOnly, tc::WaitMode::WaitBlock, NoRc>(
                rm, tiles, 1);
        } else {
            ckl::tilize<tw, cb_rm, cb_tiles, tc::InitUninitMode::Neither, tc::WaitMode::WaitBlock, NoRc>(rm, tiles, 1);
        }
    }
}

// Tilizes one K-block (block_m_tiles tile-rows x block_k_tiles) of gathered im2col rows. The matmul
// restores its own state afterwards (InitMode::ShortAfterPreKBlock).
// The reader lays a tile-row out either as one [32, block_k_tiles * 32] row block or tap-major, as
// block_k_tiles / tilize_width_tiles blocks of [32, tilize_width_tiles * 32]; tilizing those blocks in order
// yields the same tiles in the same [m][k] order.
template <
    uint32_t block_k_tiles,
    uint32_t tilize_width_tiles,
    uint32_t block_m_tiles,
    uint32_t cb_im2col_rm,
    uint32_t cb_im2col_tiles,
    bool relu_on_last_matmul_pack,
    bool strided_window,
    uint32_t c_in_tiles,
    uint32_t center_mode>
struct TilizeIm2colKBlock {
    DataflowBuffer* rm;
    DataflowBuffer* tiles;

    ALWI void operator()(uint32_t block, uint32_t /*num_k_blocks*/, bool is_last) const {
        if constexpr (center_mode == CENTER_K_BLOCK) {
            // The reader copied this center-tap K-block's input tiles into cb_center_tiles (CenterTapIn0); there is
            // nothing to tilize. Never the last K-block (kw = 2 is), and the matmul sets its own L1-acc state per pack.
            if (block % 3 == 1) {
                return;
            }
        }
        // The previous K-block's pack left packer L1 accumulation on; the tilize pack must overwrite.
        pack_reconfig_l1_acc(0);
        if constexpr (relu_on_last_matmul_pack) {
            // OutWithRelu turns PACK_RELU on before this hook; activations must not be clamped.
            pack_relu_config(ReluConfig::none());
        }
        static_assert(block_k_tiles % tilize_width_tiles == 0);
        if constexpr (center_mode == CENTER_SLOTS) {
            static_assert(block_k_tiles == 3 * tilize_width_tiles && block_m_tiles >= 2);
            MaybeDeviceZoneScope("compute_tilize");
            tilize_rows_skip_center<tilize_width_tiles, cb_im2col_rm, cb_im2col_tiles>(*rm, *tiles, block_m_tiles);
        } else {
            // Includes the unpacker's per-tile-row wait on the reader's gather.
            MaybeDeviceZoneScope("compute_tilize");
            if constexpr (strided_window) {
                static_assert(block_k_tiles == 3 * c_in_tiles, "a window unit is one kernel row");
                window_tilize<c_in_tiles, cb_im2col_rm, cb_im2col_tiles>(*rm, *tiles, block_m_tiles);
            } else {
                ckl::tilize<tilize_width_tiles, cb_im2col_rm, cb_im2col_tiles>(
                    *rm, *tiles, block_m_tiles * (block_k_tiles / tilize_width_tiles));
            }
        }
        if constexpr (relu_on_last_matmul_pack) {
            if (is_last) {
                pack_relu_config(ReluConfig::zero());
            }
        }
    }
};

// Center-tap direct in0: K-block k is tap (k / 3, k % 3) (one tap per K-block); the kw = 1 ones are the reader's
// input-tile copy in cb_center_tiles, which shares cb_im2col_tiles' data format (the input dtype).
template <uint32_t center_mode, uint32_t cb_center_tiles>
struct CenterTapIn0 {
    ALWI uint32_t operator()(uint32_t block, uint32_t in0_cb_id) const {
        if constexpr (center_mode == CENTER_K_BLOCK) {
            return block % 3 == 1 ? cb_center_tiles : in0_cb_id;
        } else {
            return in0_cb_id;
        }
    }
};

#ifdef CONV2D_NSLICE_PAIR
// nslice_paired_slices experiment: a pair weight round holds two consecutive owner slices' weights, K rows interleaved
// ([K, pw] row-major, pw = 2 * out_n: slice A's out_n tiles, then slice B's); a single round holds one slice compact
// ([K, out_n]). Multiplies the resident [bm, K] in0 by round columns [c_begin, c_begin + width) (K row stride
// in1_stride) over the whole K (one DEST accumulation per sbh x sbw subblock); round column c packs to cb_lo
// (c < out_n) or cb_hi (c >= out_n) at slice column c % out_n, slice tile-row-major from offset hi_offset in cb_hi.
// The caller reserves / pushes the targets and waits / pops the round.
//   RAW LLK / HELPER BYPASS: ckl::matmul_block packs every tile of a call into ONE output CB (row-strided); a 1 x 4
// subblock whose two left tiles are the reader's slice and two right tiles the writer's cannot be expressed. This is
// the helper's LLK sequence (matmul_block_init, per-k ckernel::matmul_block over the same K order, pack_tile<true> to
// absolute offsets), with a per-tile pack target. Measured (bench, res5.1 walk): 1x2 per slice 28437 ns, paired 1x4
// 27248 ns, bit-identical.
template <uint32_t bm, uint32_t K, uint32_t out_n, uint32_t sbh, uint32_t sbw, uint32_t width, uint32_t pw>
ALWI void nslice_round_walk(
    uint32_t cb_in0, uint32_t cb_w, uint32_t c_begin, uint32_t cb_lo, uint32_t cb_hi, uint32_t hi_offset) {
    static_assert(bm % sbh == 0 && width % sbw == 0 && sbh * sbw <= 4);
    reconfig_data_format(cb_w, cb_in0);
    PACK((pack_reconfig_data_format(cb_lo)));
    ckernel::matmul_block_init(cb_in0, cb_w, false, sbw, sbh, K);
    for (uint32_t r0 = 0; r0 < bm; r0 += sbh) {
        for (uint32_t c0 = 0; c0 < width; c0 += sbw) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < K; ++k) {
                ckernel::matmul_block(cb_in0, cb_w, r0 * K + k, k * pw + c_begin + c0, 0, false, sbw, sbh, K);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t rr = 0; rr < sbh; ++rr) {
                for (uint32_t cc = 0; cc < sbw; ++cc) {
                    const uint32_t col = c_begin + c0 + cc;
                    if (col < out_n) {
                        pack_tile<true>(rr * sbw + cc, cb_lo, (r0 + rr) * out_n + col);
                    } else {
                        pack_tile<true>(rr * sbw + cc, cb_hi, hi_offset + (r0 + rr) * out_n + col - out_n);
                    }
                }
            }
            tile_regs_release();
        }
    }
}
#endif

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_im2col_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_im2col_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weights = get_compile_time_arg_val(2);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(3);
    constexpr uint32_t cb_matmul_partials = get_compile_time_arg_val(4);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t block_n_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);
    constexpr bool has_bias = get_compile_time_arg_val(12) != 0;
    constexpr bool apply_relu = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t cb_halo_slab = get_compile_time_arg_val(15);
    constexpr bool tile_input = get_compile_time_arg_val(16) != 0;
    constexpr uint32_t c_in_tiles = get_compile_time_arg_val(17);
    constexpr uint32_t slab_tile_rows = get_compile_time_arg_val(18);
    constexpr bool k_split = get_compile_time_arg_val(19) != 0;
    constexpr uint32_t k_cols = get_compile_time_arg_val(20);
    constexpr uint32_t out_n_tiles = get_compile_time_arg_val(21);
    constexpr uint32_t cb_combine_slots = get_compile_time_arg_val(22);
    constexpr uint32_t cb_combined = get_compile_time_arg_val(23);
    constexpr uint32_t combine_subblock_h = get_compile_time_arg_val(24);
    constexpr uint32_t combine_subblock_w = get_compile_time_arg_val(25);
    constexpr uint32_t im2col_tilize_width_tiles = get_compile_time_arg_val(26);
    constexpr bool face_direct = get_compile_time_arg_val(27) != 0;
    constexpr uint32_t in0_stream_rows = get_compile_time_arg_val(28);
    constexpr uint32_t bias_stream_groups = get_compile_time_arg_val(29);
    constexpr bool nslice_outer = get_compile_time_arg_val(30) != 0;
    constexpr uint32_t cb_slice_r = get_compile_time_arg_val(31);
    constexpr uint32_t slice_sb_h = get_compile_time_arg_val(32);
    constexpr uint32_t slice_sb_w = get_compile_time_arg_val(33);
    constexpr uint32_t untilize_batch_rows = get_compile_time_arg_val(34);
    constexpr bool strided_window = get_compile_time_arg_val(35) != 0;
    constexpr uint32_t center_mode = get_compile_time_arg_val(36);
    constexpr uint32_t cb_center_tiles = get_compile_time_arg_val(37);
    static_assert(center_mode == 0 || (tile_input && !face_direct && !k_split));

    const uint32_t num_m_blocks_this_core = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const bool owns_output_columns = get_arg_val<uint32_t>(1) != 0;
    [[maybe_unused]] const uint32_t my_col = get_arg_val<uint32_t>(2);

    constexpr uint32_t in0_num_subblocks = block_m_tiles / out_subblock_h;
    constexpr uint32_t in1_num_subblocks = block_n_tiles / out_subblock_w;
    static_assert(in0_num_subblocks * out_subblock_h == block_m_tiles);
    static_assert(in1_num_subblocks * out_subblock_w == block_n_tiles);
    constexpr auto mm_shape = ckl::MatmulBlockShape::of(
        in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, block_k_tiles, num_k_blocks);
    // One matmul call per in0 row group (face-direct, one K-block): the group's subblock rows x all of N.
    // The last group may be shorter; every group is whole subblocks.
    constexpr uint32_t in0_groups = (block_m_tiles + in0_stream_rows - 1) / in0_stream_rows;
    static_assert(in0_stream_rows % out_subblock_h == 0);
    static_assert(in0_groups == 1 || (face_direct && num_k_blocks == 1 && !k_split));
    static_assert(bias_stream_groups > 0);
    constexpr auto bias_shape =
        ckl::BiasAddShape::of(in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, block_n_tiles);
    static_assert(block_m_tiles % combine_subblock_h == 0 && out_n_tiles % combine_subblock_w == 0);
    constexpr auto combine_bias_shape = ckl::BiasAddShape::of(
        block_m_tiles / combine_subblock_h,
        out_n_tiles / combine_subblock_w,
        combine_subblock_h,
        combine_subblock_w,
        out_n_tiles);
    constexpr uint32_t partials_tiles = block_m_tiles * block_n_tiles;
    // The k_cols slots hold the row's contributions to this core's [bm, out_n_tiles] output block, one
    // slot after another: a k_cols-row x slot-wide block whose column sums are the outputs.
    constexpr uint32_t slot_tiles = block_m_tiles * out_n_tiles;
    // kernel_main is not a template, so the TILE-only untilize is instantiated on the ROW_MAJOR path too.
    constexpr uint32_t untilize_width_tiles = tile_input ? c_in_tiles : 1;
    // Under a K split the matmul packs partials, which ReLU must never touch.
    constexpr bool relu_on_last_matmul_pack = apply_relu && !has_bias && !k_split;
    constexpr ckl::LastBlockTarget last_block_target =
        has_bias ? ckl::LastBlockTarget::Interm
                 : (apply_relu ? ckl::LastBlockTarget::OutWithRelu : ckl::LastBlockTarget::Out);

    // Startup matches the first op (untilize for TILE input, the matmul for face-direct, else tilize); every
    // later phase reconfigures itself.
    if constexpr (tile_input) {
        compute_kernel_hw_startup(cb_input_tiles, cb_halo_slab);
    } else if constexpr (face_direct) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(
            cb_im2col_tiles, cb_weights, (has_bias || k_split) ? cb_matmul_partials : cb_output_tiles);
    } else {
        compute_kernel_hw_startup(cb_im2col_rm, cb_im2col_tiles);
    }

    DataflowBuffer input_tiles_dfb(cb_input_tiles);
    DataflowBuffer halo_slab_dfb(cb_halo_slab);
    DataflowBuffer im2col_rm_dfb(cb_im2col_rm);
    DataflowBuffer im2col_tiles_dfb(cb_im2col_tiles);
    CircularBuffer im2col_tiles(cb_im2col_tiles);
    CircularBuffer weights(cb_weights);
    CircularBuffer partials(cb_matmul_partials);
    CircularBuffer output_tiles(cb_output_tiles);
    CircularBuffer bias(cb_bias);
    CircularBuffer combined(cb_combined);
    [[maybe_unused]] CircularBuffer slice_r(nslice_outer ? cb_slice_r : cb_matmul_partials);
    DataflowBuffer combine_slots_dfb(cb_combine_slots);
    DataflowBuffer combined_dfb(cb_combined);
    DataflowBuffer output_tiles_dfb(cb_output_tiles);

    using TilizeFn = TilizeIm2colKBlock<
        block_k_tiles,
        im2col_tilize_width_tiles,
        block_m_tiles,
        cb_im2col_rm,
        cb_im2col_tiles,
        relu_on_last_matmul_pack,
        strided_window,
        c_in_tiles,
        center_mode>;
    auto make_pre_k_block = [&]() {
        if constexpr (face_direct) {
            return ckl::NoPreKBlock{};
        } else {
            return TilizeFn{&im2col_rm_dfb, &im2col_tiles_dfb};
        }
    };
    const auto pre_k_block = make_pre_k_block();
    using PreKBlockFn = std::decay_t<decltype(pre_k_block)>;

    if constexpr (has_bias) {
        bias.wait_front(out_n_tiles);
    }

    // finalize_block over `shape`'s rows of the matmul partials: + bias, ReLU on this pack only.
    auto add_bias = [&](const ckl::BiasAddShape& shape) {
        MaybeDeviceZoneScope("compute_bias_add");
        // matmul_block leaves packer L1 accumulation on after its last K-block.
        pack_reconfig_l1_acc(0);
        if constexpr (apply_relu) {
            pack_relu_config(ReluConfig::zero());
        }
        ckl::add_bias_bcast_rows<ckl::BiasBroadcast::RowBroadcast, ckl::OutputCBLayout::TileRowMajor>(
            partials, bias, output_tiles, shape);
    };

    // Sum the row's k_cols contributions per output tile and pack them; with a bias the fp32 sum goes
    // through cb_combined into the bias add, otherwise straight to the output shard.
    auto combine_block = [&]() {
        {
            // The row's slices arrive over the NoC; waiting here keeps that wait out of the sum's zone.
            MaybeDeviceZoneScope("compute_combine_wait");
            combine_slots_dfb.wait_front(k_cols * slot_tiles);
        }
        MaybeDeviceZoneScope("compute_combine");
        pack_reconfig_l1_acc(0);
        if constexpr (apply_relu && !has_bias) {
            pack_relu_config(ReluConfig::zero());
        }
        constexpr uint32_t sum_cb = has_bias ? cb_combined : cb_output_tiles;
        // The cross-core inputs are raw partial sums, so only the cross-tile adds run (Skip). No scaler
        // tile is read on this path; the scaler slot names the input CB.
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            cb_combine_slots,
            cb_combine_slots,
            sum_cb,
            ckl::ReduceInputPolicy::BulkWaitBulkPop,
            ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Fast,
            ckl::ReduceAlgorithm::AccumulateViaAdd,
            ckl::ReduceWithinTile::Skip>(
            combine_slots_dfb,
            combine_slots_dfb,
            has_bias ? combined_dfb : output_tiles_dfb,
            ckl::ReduceInputBlockShape::of(k_cols, slot_tiles));
        if constexpr (has_bias) {
            if constexpr (apply_relu) {
                pack_relu_config(ReluConfig::zero());
            }
            ckl::add_bias_bcast_rows<ckl::BiasBroadcast::RowBroadcast, ckl::OutputCBLayout::TileRowMajor>(
                combined, bias, output_tiles, combine_bias_shape);
        }
    };

    for (uint32_t m_block_idx = 0; m_block_idx < num_m_blocks_this_core; ++m_block_idx) {
        if constexpr (tile_input) {
            // untilize_footprint: the reader streams slab_tile_rows input tile-rows in slab order.
            MaybeDeviceZoneScope("compute_untilize");
            pack_reconfig_l1_acc(0);
            if constexpr (untilize_width_tiles == 1) {
                // Raw LLK on a measured win (see conv2d_3x3_narrow_untilize.hpp): the helper's 1-wide path is
                // packer-bound at one DEST section and one CB sync per tile-row.
                conv2d_3x3::narrow_untilize::untilize_1wide<cb_input_tiles, cb_halo_slab>(
                    input_tiles_dfb, halo_slab_dfb, slab_tile_rows, untilize_batch_rows);
            } else {
                ckl::untilize<untilize_width_tiles, cb_input_tiles, cb_halo_slab>(
                    input_tiles_dfb, halo_slab_dfb, slab_tile_rows);
            }
        }

        if constexpr (k_split && nslice_outer) {
            // N-slice-outer K split: the whole K of in0 is tilized once into a resident [bm, K] block (the reader
            // gathers tile-row outer, K-block inner). Then per owner N-slice, in the order of
            // conv2d_3x3_nslice_order.hpp, one matmul call runs the whole K as one K-block: DEST accumulates over K,
            // the fp32 [bm, out_n] slice is packed once, with no L1-accumulation pass, and pushed to the CB of the
            // RISC that sends it (reader: cb_slice_r; writer: cb_matmul_partials) while the next slice multiplies.
            constexpr uint32_t k_tiles = block_k_tiles * num_k_blocks;
            constexpr uint32_t num_owning_cols = (block_n_tiles + out_n_tiles - 1) / out_n_tiles;
            static_assert(block_m_tiles % slice_sb_h == 0 && out_n_tiles % slice_sb_w == 0);
            // The tilize packs must overwrite (the previous M-block's combine may have left L1 accumulation on).
            pack_reconfig_l1_acc(0);
            if constexpr (!face_direct) {
                MaybeDeviceZoneScope("compute_tilize");
                if constexpr (strided_window) {
                    window_tilize<c_in_tiles, cb_im2col_rm, cb_im2col_tiles>(
                        im2col_rm_dfb, im2col_tiles_dfb, block_m_tiles * num_k_blocks);
                } else {
                    ckl::tilize<im2col_tilize_width_tiles, cb_im2col_rm, cb_im2col_tiles>(
                        im2col_rm_dfb, im2col_tiles_dfb, block_m_tiles * (k_tiles / im2col_tilize_width_tiles));
                }
            }
#ifdef CONV2D_NSLICE_PAIR
            {
                // Pair walk (nslice_paired_slices): weight rounds per conv2d_3x3::nslice_pair_rounds over the owner
                // steps of the shared order (skips removed). A pair round's two slices are multiplied together where
                // they go to different sending RISCs, one after the other otherwise (a two-slice region of one CB could
                // wrap its ring); a single round multiplies its one slice.
                MaybeDeviceZoneScope("compute_matmul_walk");
                constexpr uint32_t pw = 2 * out_n_tiles;
                constexpr uint32_t round_tiles = k_tiles * pw;
                constexpr uint32_t slice_tiles = block_m_tiles * out_n_tiles;
                constexpr uint32_t pair_sb_h = CONV2D_NSLICE_PAIR_SB_H;
                constexpr uint32_t pair_sb_w = CONV2D_NSLICE_PAIR_SB_W;
                uint32_t steps[k_cols];
                uint32_t num_steps = 0;
                for (uint32_t j = 0; j < k_cols; ++j) {
                    if (conv2d_3x3::nslice_owner<k_cols>(my_col, j) < num_owning_cols) {
                        steps[num_steps++] = j;
                    }
                }
                uint32_t round_first[k_cols];
                uint32_t round_count[k_cols];
                const uint32_t num_rounds =
                    conv2d_3x3::nslice_pair_rounds(num_steps, CONV2D_NSLICE_PAIR_TAIL, round_first, round_count);
                im2col_tiles.wait_front(block_m_tiles * k_tiles);
                for (uint32_t r = 0; r < num_rounds; ++r) {
                    const uint32_t s = round_first[r];
                    const uint32_t cb_a =
                        conv2d_3x3::nslice_reader_sends<k_cols>(steps[s]) ? cb_slice_r : cb_matmul_partials;
                    const bool two = round_count[r] == 2;
                    const uint32_t cb_b =
                        two ? (conv2d_3x3::nslice_reader_sends<k_cols>(steps[s + 1]) ? cb_slice_r : cb_matmul_partials)
                            : cb_a;
                    cb_wait_front(cb_weights, round_tiles);
                    if (!two) {
                        // Single round: the slice's [K, out_n] weights, compact.
                        cb_reserve_back(cb_a, slice_tiles);
                        nslice_round_walk<
                            block_m_tiles,
                            k_tiles,
                            out_n_tiles,
                            slice_sb_h,
                            slice_sb_w,
                            out_n_tiles,
                            out_n_tiles>(cb_im2col_tiles, cb_weights, 0, cb_a, cb_a, 0);
                        cb_push_back(cb_a, slice_tiles);
                    } else if (cb_b != cb_a) {
                        cb_reserve_back(cb_a, slice_tiles);
                        cb_reserve_back(cb_b, slice_tiles);
                        nslice_round_walk<block_m_tiles, k_tiles, out_n_tiles, pair_sb_h, pair_sb_w, pw, pw>(
                            cb_im2col_tiles, cb_weights, 0, cb_a, cb_b, 0);
                        cb_push_back(cb_a, slice_tiles);
                        cb_push_back(cb_b, slice_tiles);
                    } else {
                        cb_reserve_back(cb_a, slice_tiles);
                        nslice_round_walk<block_m_tiles, k_tiles, out_n_tiles, slice_sb_h, slice_sb_w, out_n_tiles, pw>(
                            cb_im2col_tiles, cb_weights, 0, cb_a, cb_a, 0);
                        cb_push_back(cb_a, slice_tiles);
                        cb_reserve_back(cb_b, slice_tiles);
                        nslice_round_walk<block_m_tiles, k_tiles, out_n_tiles, slice_sb_h, slice_sb_w, out_n_tiles, pw>(
                            cb_im2col_tiles, cb_weights, out_n_tiles, cb_b, cb_b, 0);
                        cb_push_back(cb_b, slice_tiles);
                    }
                    cb_pop_front(cb_weights, round_tiles);
                }
                im2col_tiles.pop_front(block_m_tiles * k_tiles);
            }
#else
            {
                MaybeDeviceZoneScope("compute_matmul_walk");
                for (uint32_t j = 0; j < k_cols; ++j) {
                    const uint32_t y = conv2d_3x3::nslice_owner<k_cols>(my_col, j);
                    if (y >= num_owning_cols) {
                        continue;
                    }
                    CircularBuffer& slice_cb = conv2d_3x3::nslice_reader_sends<k_cols>(j) ? slice_r : partials;
                    ckl::matmul_block<
                        /*transpose=*/false,
                        /*packer_l1_acc=*/false,
                        ckl::LastBlockTarget::Out,
                        ckl::OutputCBLayout::TileRowMajor,
                        ckl::matmul_config::InitMode::Short,
                        ckl::InputPolicy::WaitAndRetainOnLastBlock,
                        ckl::InputPolicy::WaitAndPopPerKBlock>(
                        im2col_tiles,
                        weights,
                        slice_cb,
                        slice_cb,
                        ckl::MatmulBlockShape::of(
                            block_m_tiles / slice_sb_h, out_n_tiles / slice_sb_w, slice_sb_h, slice_sb_w, k_tiles, 1),
                        {},
                        {},
                        /*in1_per_core_w=*/out_n_tiles,
                        /*out_row_width=*/out_n_tiles);
                }
                im2col_tiles.pop_front(block_m_tiles * k_tiles);
            }
#endif
            if (owns_output_columns) {
                combine_block();
            }
        } else if constexpr (k_split) {
            {
                // The whole K walk, tilize hooks included (compute_tilize is the tilize share).
                MaybeDeviceZoneScope("compute_matmul_walk");
                partials.reserve_back(partials_tiles);
                ckl::matmul_block<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/true,
                    ckl::LastBlockTarget::Interm,
                    ckl::OutputCBLayout::TileRowMajor,
                    ckl::matmul_config::InitMode::ShortAfterPreKBlock,
                    ckl::InputPolicy::WaitAndPopPerKBlock,
                    ckl::InputPolicy::WaitAndPopPerKBlock,
                    ckl::NoPostCompute,
                    PreKBlockFn,
                    ckl::NoPostKBlock,
                    /*untilize_block_ct_dim=*/0,
                    ckl::NoKBlockInnerDimFn,
                    ckl::NoIn0Source,
                    ckl::NoIn1BaseOffset,
                    /*caller_owns_pack_target=*/true>(
                    im2col_tiles,
                    weights,
                    output_tiles,
                    partials,
                    mm_shape,
                    {},
                    pre_k_block,
                    /*in1_per_core_w=*/block_n_tiles,
                    /*out_row_width=*/block_n_tiles);
                partials.push_back(partials_tiles);
            }
            if (owns_output_columns) {
                combine_block();
            }
        } else if constexpr (in0_groups > 1) {
            // Matmul and bias calls interleave with the streamed in0 groups; one zone for the whole block.
            MaybeDeviceZoneScope("compute_matmul_stream");
            for (uint32_t group = 0; group < in0_groups; ++group) {
                const uint32_t group_end =
                    (group + 1) * in0_stream_rows < block_m_tiles ? (group + 1) * in0_stream_rows : block_m_tiles;
                ckl::matmul_block<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/true,
                    last_block_target,
                    ckl::OutputCBLayout::TileRowMajor,
                    ckl::matmul_config::InitMode::Short,
                    ckl::InputPolicy::WaitAndPopPerKBlock,
                    ckl::InputPolicy::WaitAndRetainOnLastBlock>(
                    im2col_tiles,
                    weights,
                    output_tiles,
                    partials,
                    ckl::MatmulBlockShape::of(
                        (group_end - group * in0_stream_rows) / out_subblock_h,
                        in1_num_subblocks,
                        out_subblock_h,
                        out_subblock_w,
                        block_k_tiles,
                        1),
                    {},
                    {},
                    /*in1_per_core_w=*/block_n_tiles,
                    /*out_row_width=*/block_n_tiles);
                if constexpr (has_bias) {
                    const uint32_t groups_done = group + 1;
                    if (groups_done % bias_stream_groups == 0 || groups_done == in0_groups) {
                        const uint32_t chunk_rows =
                            group_end - (group / bias_stream_groups) * bias_stream_groups * in0_stream_rows;
                        add_bias(ckl::BiasAddShape::of(
                            chunk_rows / out_subblock_h,
                            in1_num_subblocks,
                            out_subblock_h,
                            out_subblock_w,
                            block_n_tiles));
                        if constexpr (apply_relu) {
                            pack_relu_config(ReluConfig::none());
                        }
                    }
                }
            }
            weights.pop_front(block_k_tiles * block_n_tiles);
        } else {
            MaybeDeviceZoneScope("compute_matmul_walk");
            ckl::matmul_block<
                /*transpose=*/false,
                /*packer_l1_acc=*/true,
                last_block_target,
                ckl::OutputCBLayout::TileRowMajor,
                ckl::matmul_config::InitMode::ShortAfterPreKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::NoPostCompute,
                PreKBlockFn,
                ckl::NoPostKBlock,
                /*untilize_block_ct_dim=*/0,
                ckl::NoKBlockInnerDimFn,
                CenterTapIn0<center_mode, cb_center_tiles>>(
                im2col_tiles,
                weights,
                output_tiles,
                partials,
                mm_shape,
                {},
                pre_k_block,
                /*in1_per_core_w=*/block_n_tiles,
                /*out_row_width=*/block_n_tiles,
                {},
                {},
                CenterTapIn0<center_mode, cb_center_tiles>{});
        }

        if constexpr (has_bias && !k_split && in0_groups == 1) {
            add_bias(bias_shape);
        }
        if constexpr (apply_relu) {
            pack_relu_config(ReluConfig::none());
        }
    }

    if constexpr (has_bias) {
        bias.pop_front(out_n_tiles);
    }
}
