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
// N-slice center taps (the descriptor's NSLICE_CENTER_TAPS): the tap-direct prelude tilizes taps 0 and 2 of every unit
// (in0 [bm, 6 * c_in_tiles]) and the ring walk reads the kw = 1 taps from cb_center_tiles ([3, bm, c_in_tiles], the
// reader's copy of the input tiles), in the same K order (bit-identical).
//
// Padded slab (the descriptor's PADDED_SLAB, conv2d_3x3_padded_slab.hpp): on the N-slice center taps the untilize packs
// each image row of the slab into its own zero-gapped slot (runtime args after the masks: the core's output range), and
// padded_tilize computes every tap block's source (a slab run or the zero block) itself, after waiting for the slab's
// last pack and the reader's center push; a one-page token on cb_src_table then tells the reader the slab is dead.
//
// Tap shift (the descriptor's TAP_SHIFT, conv2d_3x3_tap_shift.hpp, on the padded-slab path with d < 32): no untilize
// and no padded tilize; the taps are built on the FPU straight from the input ring, kernel-row major, as the fill
// lands.
//
// Stride-2 taps (the descriptor's S2_TAPS, conv2d_3x3_s2_taps.hpp, on the strided-window path): no untilize and no
// window tilize; every tap of the M-block is built from the input ring (the whole slab) into in0 before the walk, in
// input-row order, as the fill lands (runtime args after the masks: the core's output range). On the K-blocked walk in0
// holds every K-block and the tilize hook does nothing.
//
// N-slice kernel-row skip: the ring walk multiplies, per row group, only the range of kernel rows that is in the image
// for some row of the group (runtime args 3..: the host's masks, see the ring walk). The others are all-zero K tiles of
// in0.
//
// Face-direct im2col: the reader writes the in0 tiles itself, so the matmul runs without the tilize hook. With one
// K-block per M-block the reader pushes in0 in row groups and compute runs one matmul call per group as it lands,
// keeping the weights fronted across the calls, and adds the bias once per bias_stream_groups groups, so only the
// last chunk's bias add trails the gather.
//
// matmul_block leaves packer L1 accumulation on and (OutWithRelu) PACK_RELU on; every later pack
// (untilize, tilize, combine, bias add) first restores the state it needs.
//
// N-slice-outer K split (the default under a K split, when the resident in0 fits L1): a ring reduce-scatter, see the
// branch below and conv2d_3x3_nslice_order.hpp. The K-blocked walk below remains for K splits whose [bm, K] in0 does
// not fit.
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
#include "api/compute/tile_move_copy.h"
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
#include "conv2d_3x3_wide_untilize.hpp"
#include "conv2d_3x3_lean_tilize.hpp"
#include "conv2d_3x3_padded_slab.hpp"
#include "conv2d_3x3_slab_direct.hpp"
#ifdef CONV2D_3X3_TAP_SHIFT
#include "conv2d_3x3_tap_shift.hpp"
#endif
#ifdef CONV2D_3X3_KW_FOLD
#include "conv2d_3x3_kw_fold.hpp"
#endif
#ifdef CONV2D_3X3_S2_TAPS
#include "conv2d_3x3_s2_taps.hpp"
#endif

namespace ckl = compute_kernel_lib;

namespace {

// Strided-window tilize (see conv2d_3x3_gather_dataflow.hpp WindowGather): a cb_im2col_rm unit is one tile-row x one
// kernel row laid out as overlapping windows, im2col row i = 3 pixels (3 * c_in_tiles tile columns) from pixel slot
// 2 * i. The BH fast-tilize unpacker takes its row stride from the init's width and its read width from the chunk, so
// an init at 2 * c_in_tiles and chunks covering 3 * c_in_tiles read row i's 3 pixels at a 2-pixel stride: the same
// tiles, in the same order, as tilizing the [32, 3 * c_in_tiles] row-layout block (bit-identical).
// The unit's last 16 B hold the L1 address to tilize from (the unit itself, or a slab run on the N-slice-outer walk).
// Raw LLK (capability): ckl::tilize ties the row stride to the block width and reads at the CB read pointer; the loop
// is fast_tilize_block's, with the unpacker's base taken from that address, and the lean pack stream of
// conv2d_3x3_lean_tilize.hpp.
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

// A runtime group height with the call shape of std::integral_constant (group_mm in the N-slice ring walk).
struct RuntimeRows {
    uint32_t v;
    uint32_t operator()() const { return v; }
};

template <uint32_t c_in_tiles, uint32_t cb_in, uint32_t cb_out>
ALWI void window_tilize(DataflowBuffer& in, DataflowBuffer& out, uint32_t units) {
    constexpr uint32_t row_tiles = 2 * c_in_tiles;
    constexpr uint32_t block_tiles = 3 * c_in_tiles;
#ifdef CONV2D_3X3_ABLATE_TILIZE
    for (uint32_t u = 0; u < units; ++u) {
        in.wait_front(block_tiles);
        out.reserve_back(block_tiles);
        out.push_back(block_tiles);
        in.pop_front(block_tiles);
    }
    return;
#endif
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
    fast_tilize_init(cb_in, row_tiles, cb_out);
    // The lean pack stream (conv2d_3x3_lean_tilize.hpp): one 1-tile MOP for every chunk width.
    PACK((conv2d_3x3::lean_tilize::pack_init()));
    // Every unit has the same chunk sequence: reprogram the unpack width only when the chunk changes.
    uint32_t prev_chunk = 0;
    for (uint32_t u = 0; u < units; ++u) {
        in.wait_front(block_tiles);
        out.reserve_back(block_tiles);
        [[maybe_unused]] uint32_t src = 0;
        UNPACK(src = window_unit_source<cb_in, block_tiles>());
        PACK((conv2d_3x3::lean_tilize::pack_row_begin(cb_out, 0)));
        for (uint32_t tiles_done = 0; tiles_done < block_tiles;) {
            // BH fast-tilize chunks are 2..4 tiles (fast_tilize_block's split: 5 -> 2 + 3).
            const uint32_t chunk = conv2d_3x3::lean_tilize::chunk_of(block_tiles - tiles_done);
            MATH((llk_math_wait_for_dest_available()));
            PACK((llk_packer_wait_for_math_done()));
            if (chunk != prev_chunk) {
                UNPACK((llk_unpack_fast_tilize_reinit_xdim(chunk)));
                prev_chunk = chunk;
            }
            UNPACK((window_unpack_chunk<cb_in>(src, chunk, tiles_done)));
            MATH((llk_math_fast_tilize_block_<DST_ACCUM_MODE>(0, cb_in, 4)));
            PACK((conv2d_3x3::lean_tilize::pack_row_chunk(chunk)));
            MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
            PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
            tiles_done += chunk;
        }
        out.push_back(block_tiles);
        in.pop_front(block_tiles);
    }
    fast_tilize_uninit(cb_in, cb_out, row_tiles);
}

// Tap-direct tilize (N-slice-outer prelude; see TapDirect in conv2d_3x3_gather_dataflow.hpp): every tap block (tw
// tiles, a [32, c_in_p] row-major block) of a unit is tilized from the L1 address the reader published in its slot's
// entry of cb_src_table: a slab run, the zero block, or the block the reader gathered into the slot. Same tiles, same
// order as ckl::tilize over the slot (bit-identical).
// Raw LLK (capability): ckl::tilize / fast_tilize_block always read at the CB read pointer; the loop is
// fast_tilize_block's (as window_tilize), with the unpacker's base taken from the published address, and the lean pack
// stream of conv2d_3x3_lean_tilize.hpp. It removes the reader's copy of most tap blocks, which paced the prelude:
// aspp.convs.1 @ LoFi 76.7 -> 74.7 us.
#ifdef TRISC_UNPACK
template <uint32_t cb_table, uint32_t taps>
inline void direct_unit_sources(uint32_t slot, uint32_t (&src)[taps]) {
    invalidate_l1_cache();
    const uint32_t table = get_local_cb_interface(get_operand_id(cb_table)).fifo_rd_ptr << 4;
    volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(table + slot * 16);
    for (uint32_t t = 0; t < taps; ++t) {
        src[t] = entry[t];
    }
}

// One-tile-wide tap block: BH tilizes it with the per-tile unpack tilize (fast_tilize_block's one-tile path); the base
// is the published address instead of the CB read pointer.
template <uint32_t cb_in>
inline void direct_unpack_tilize(uint32_t src) {
    const uint32_t id = get_operand_id(cb_in);
    llk_unpack_tilize_impl(
        (src >> 4) - 1,
        0,
        unpack_src_format[id],
        unpack_dst_format[id],
        get_operand_face_r_dim(id),
        get_operand_num_faces(id),
        get_operand_narrow_tile(id));
}
#endif

// One-tile-wide tap blocks (C_slice 32: res4.x's padded-slab prelude, tap-direct / FIFO preludes at c_in_tiles 1), up
// to NARROW_TILIZE_BATCH per DEST section (a fp32 half-sync section holds 4 tiles) with one CB reserve / push per
// section: each tile is the per-tile unpack tilize of its source into DEST tile j, then one pass packs the section's
// tiles at consecutive output offsets. Same tiles, same order as one section per tile (bit-identical). Raw LLK on a
// measured win, as the per-tile loop it replaces (compute_kernel_lib::tilize reads at the CB read pointer and, at block
// width 1, runs tilize_block: one DEST section, math/pack handshake and CB sync per tile). The BH 1-wide tilize is
// pack-bound (its pack MOP is 32 two-interface PACRs per tile); one section per tile adds the section release and
// handshake to every tile: 58.1 -> 42.2 ns/tile isolated (1 core, bf16 -> bf8_b, fp32 DEST), res4.x @ LoFi
// compute_tilize (pack) 2.49 -> 1.87 us, op 13.50 -> 12.83 us (Blackhole p150b).
constexpr uint32_t NARROW_TILIZE_BATCH = 4;
template <uint32_t cb_in, uint32_t cb_out>
struct NarrowTilizeBatch {
    DataflowBuffer& out;
    uint32_t j = 0;  // tiles in the open DEST section

    ALWI void tile([[maybe_unused]] uint32_t src) {
        if (j == 0) {
            MATH((llk_math_wait_for_dest_available()));
            PACK((llk_packer_wait_for_math_done()));
        }
        UNPACK((direct_unpack_tilize<cb_in>(src)));
        MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
            j, cb_in)));
        if (++j == NARROW_TILIZE_BATCH) {
            close();
        }
    }

    // Packs and pushes the open section (no-op when none is open). Call before any other push to cb_out.
    ALWI void close() {
        if (j == 0) {
            return;
        }
        out.reserve_back(j);
        for (uint32_t i = 0; i < j; ++i) {
            PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(i, cb_out, i)));
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
        out.push_back(j);
        j = 0;
    }
};

// skip_t (>= 0, N-slice center taps): that tap of every unit is not tilized (the matmul reads it from cb_center_tiles).
template <
    uint32_t tw,
    uint32_t taps,
    uint32_t ring,
    uint32_t cb_in,
    uint32_t cb_out,
    uint32_t cb_table,
    int32_t skip_t = -1>
ALWI void direct_tilize(DataflowBuffer& in, DataflowBuffer& out, uint32_t units, uint32_t& seq) {
    static_assert(tw != 0 && taps <= 4, "a source entry is 16 B: at most 4 taps per unit");
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
#ifdef CONV2D_3X3_ABLATE_TILIZE
    for (uint32_t u = 0; u < units; ++u, ++seq) {
        in.wait_front(tw * taps);
        for (uint32_t t = 0; t < taps; ++t) {
            if (static_cast<int32_t>(t) == skip_t) {
                continue;
            }
            out.reserve_back(tw);
            out.push_back(tw);
        }
        in.pop_front(tw * taps);
    }
    return;
#endif
    fast_tilize_init(cb_in, tw, cb_out);
    if constexpr (tw > 1) {
        // The lean pack stream (conv2d_3x3_lean_tilize.hpp): one 1-tile MOP for every chunk width.
        PACK((conv2d_3x3::lean_tilize::pack_init()));
    }
    [[maybe_unused]] uint32_t prev_chunk = 0;
    [[maybe_unused]] NarrowTilizeBatch<cb_in, cb_out> nt{out};
    for (uint32_t u = 0; u < units; ++u, ++seq) {
        in.wait_front(tw * taps);
        [[maybe_unused]] uint32_t src[taps];
        UNPACK((direct_unit_sources<cb_table, taps>(seq % ring, src)));
        for (uint32_t t = 0; t < taps; ++t) {
            if (static_cast<int32_t>(t) == skip_t) {
                continue;
            }
            if constexpr (tw == 1) {
                nt.tile(src[t]);
            } else {
                out.reserve_back(tw);
                PACK((conv2d_3x3::lean_tilize::pack_row_begin(cb_out, 0)));
                for (uint32_t tiles_done = 0; tiles_done < tw;) {
                    // BH fast-tilize chunks are 2..4 tiles (fast_tilize_block's split: 5 -> 2 + 3).
                    const uint32_t chunk = conv2d_3x3::lean_tilize::chunk_of(tw - tiles_done);
                    MATH((llk_math_wait_for_dest_available()));
                    PACK((llk_packer_wait_for_math_done()));
                    if (chunk != prev_chunk) {
                        UNPACK((llk_unpack_fast_tilize_reinit_xdim(chunk)));
                        prev_chunk = chunk;
                    }
                    UNPACK((window_unpack_chunk<cb_in>(src[t], chunk, tiles_done)));
                    MATH((llk_math_fast_tilize_block_<DST_ACCUM_MODE>(0, cb_in, 4)));
                    PACK((conv2d_3x3::lean_tilize::pack_row_chunk(chunk)));
                    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
                    tiles_done += chunk;
                }
                out.push_back(tw);
            }
        }
        in.pop_front(tw * taps);
    }
    if constexpr (tw == 1) {
        nt.close();
    }
    fast_tilize_uninit(cb_in, cb_out, tw);
}

// Padded-slab tilize (the descriptor's PADDED_SLAB, conv2d_3x3_padded_slab.hpp): every tap block of the N-slice
// prelude is one contiguous run of the padded slab or the zero block, so compute computes each source itself (output
// tile-row t, kernel row kh, tap kw: slab pixel (h_in + band_row_off[kh]) * (W + d) + w0 + kw * d, p == d) and tilizes
// it with direct_tilize's tile loop: no reader table, no gathered blocks, no cb_im2col_rm handshake. Taps 0 and 2 only
// (the center taps come from cb_center_tiles), in the unit order of direct_tilize (tile-row, kernel row, tap): the same
// in0 tiles (bit-identical). The caller orders the unpacker after the slab's last pack and the reader's center copy.
// Dead units (skip_dead): the ring walk reads unit (t, kh) only when kh is in [kh_lo[g], kh_hi[g]) of t's ring row
// group g = t / group_h (every owner's walk has the same groups, at multiples of group_h: ring_group_row0 / _rt only
// reorder them), one contiguous range, so the units it never reads are a prefix and a suffix of each tile-row's three:
// their pages are reserved and pushed unwritten (one CB advance each), keeping the walk's wait count (bit-identical).
// The caller takes skip_dead only where some row has a dead unit: elsewhere the loop is the plain one (the per-row
// range costs the 1-wide res4.x tilize +0.2 us with nothing to skip). Measured (Blackhole p150b, LoFi, zoned):
// aspp.convs.3 25085 -> 24390 ns (median of 3; the prelude tilize 3.5 -> 2.8 us, a third of its units dead),
// aspp.convs.2 -3.6 to -5%, HiFi4 BLOCK aspp_convs3 -3.1%, aspp_convs2 -1.6%, res4.x / res5.x / aspp.convs.1 flat.
// Raw LLK (capability), as direct_tilize: ckl::tilize / fast_tilize_block read at the CB read pointer.
template <
    uint32_t tw,
    uint32_t block_m_tiles,
    uint32_t cb_in,
    uint32_t cb_out,
    uint32_t output_width,
    int32_t input_height,
    int32_t input_width,
    int32_t padding,
    int32_t dilation,
    uint32_t group_h,
    uint32_t num_groups,
    bool skip_dead>
ALWI void padded_tilize(
    DataflowBuffer& out,
    uint32_t rows,
    [[maybe_unused]] uint32_t o_first,
    [[maybe_unused]] uint32_t o_end,
    [[maybe_unused]] const int32_t (&band_row_off)[3],
    [[maybe_unused]] uint32_t slab_l1,
    [[maybe_unused]] uint32_t zero_l1,
    [[maybe_unused]] const uint8_t (&kh_lo)[num_groups],
    [[maybe_unused]] const uint8_t (&kh_hi)[num_groups]) {
#ifdef CONV2D_3X3_ABLATE_TILIZE
    for (uint32_t u = 0; u < rows * 6; ++u) {
        out.reserve_back(tw);
        out.push_back(tw);
    }
    return;
#endif
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
    fast_tilize_init(cb_in, tw, cb_out);
    if constexpr (tw > 1) {
        PACK((conv2d_3x3::lean_tilize::pack_init()));
    }
    [[maybe_unused]] uint32_t prev_chunk = 0;
    [[maybe_unused]] constexpr uint32_t pixel_bytes = tw * 32 * 2;
    [[maybe_unused]] constexpr int32_t slot_px = input_width + dilation;
    [[maybe_unused]] NarrowTilizeBatch<cb_in, cb_out> nt{out};
    for (uint32_t t = 0; t < rows; ++t) {
        [[maybe_unused]] const uint32_t o = o_first + t * 32;
        [[maybe_unused]] const int32_t ho0 = o / output_width;
        [[maybe_unused]] const int32_t wo0 = o % output_width;
        int32_t live_lo = 0;
        int32_t live_hi = 3;
        if constexpr (skip_dead) {
            live_lo = kh_lo[t / group_h];
            live_hi = kh_hi[t / group_h];
            if (live_lo > 0) {
                if constexpr (tw == 1) {
                    nt.close();  // the open section's tiles precede these pages
                }
                out.reserve_back(2 * tw * live_lo);
                out.push_back(2 * tw * live_lo);
            }
        }
        for (int32_t kh = live_lo; kh < live_hi; ++kh) {
            [[maybe_unused]] uint32_t row_src = 0;
#ifdef TRISC_UNPACK
            {
                const int32_t h_in = ho0 + kh * dilation - padding;
                row_src =
                    (o >= o_end || h_in < 0 || h_in >= input_height)
                        ? 0
                        : slab_l1 + static_cast<uint32_t>((h_in + band_row_off[kh]) * slot_px + wo0) * pixel_bytes;
            }
#endif
            for (uint32_t kw = 0; kw < 3; kw += 2) {
                [[maybe_unused]] uint32_t src = 0;
                UNPACK((src = row_src == 0 ? zero_l1 : row_src + kw * dilation * pixel_bytes));
                if constexpr (tw == 1) {
                    nt.tile(src);
                } else {
                    out.reserve_back(tw);
                    PACK((conv2d_3x3::lean_tilize::pack_row_begin(cb_out, 0)));
                    for (uint32_t tiles_done = 0; tiles_done < tw;) {
                        const uint32_t chunk = conv2d_3x3::lean_tilize::chunk_of(tw - tiles_done);
                        MATH((llk_math_wait_for_dest_available()));
                        PACK((llk_packer_wait_for_math_done()));
                        if (chunk != prev_chunk) {
                            UNPACK((llk_unpack_fast_tilize_reinit_xdim(chunk)));
                            prev_chunk = chunk;
                        }
                        UNPACK((window_unpack_chunk<cb_in>(src, chunk, tiles_done)));
                        MATH((llk_math_fast_tilize_block_<DST_ACCUM_MODE>(0, cb_in, 4)));
                        PACK((conv2d_3x3::lean_tilize::pack_row_chunk(chunk)));
                        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
                        tiles_done += chunk;
                    }
                    out.push_back(tw);
                }
            }
        }
        if constexpr (skip_dead) {
            if (live_hi < 3) {
                if constexpr (tw == 1) {
                    nt.close();
                }
                out.reserve_back(2 * tw * (3 - live_hi));
                out.push_back(2 * tw * (3 - live_hi));
            }
        }
    }
    if constexpr (tw == 1) {
        nt.close();
    }
    fast_tilize_uninit(cb_in, cb_out, tw);
}

// N-slice center taps: the ring walk reads in0 rows from two CBs of different row strides (cb_im2col_tiles [bm, 6 ct],
// cb_center_tiles [3, bm, ct]). With rt > ct the BH matmul unpacker steps in0 rows by tile size x the KT_DIM GPR, which
// only matmul_block_init sets (llk_unpack_AB_matmul: MULDMAREG TMP_LO = TILE_SIZE_B x KT_DIM at every call); with
// ct >= rt it takes the call's kt_dim. Raw LLK (capability): no compute API sets the in0 row stride per call, so the
// walk rewrites the GPR between the two sources (one SETDMAREG per switch, in order on the unpack thread).
#ifdef TRISC_UNPACK
inline void set_matmul_in0_row_stride(uint32_t kt_dim) {
    TT_SETDMAREG(0, LOWER_HALFWORD(kt_dim), 0, LO_16(p_gpr_unpack::KT_DIM));
}
#endif

// Center-tap direct modes (the descriptor's center_tap_mode).
constexpr uint32_t CENTER_K_BLOCK = 1;
constexpr uint32_t CENTER_SLOTS = 2;

// Tap-direct on the K-blocked walk (TapDirectFifo in conv2d_3x3_gather_dataflow.hpp): one 16 B entry of cb_table per
// unit (tap sources, then the cb_in pages to pop after the unit); cb_in holds only the tap blocks the reader gathered,
// in unit order, each tw pages (never across the ring end). skip_center: tap 1's tiles are the reader's center slots,
// reserved and pushed unpacked. Same tiles, same order as the copying gather and tilize (bit-identical).
// Raw LLK (capability), as direct_tilize: ckl::tilize / fast_tilize_block read at the CB read pointer and tie the row
// stride to the block width. Measured (Blackhole p150b, LoFi, zoned): stem.conv2 68.6 -> 52.5 us (per-tap tilize),
// 47.6 us with window units; stem.conv3 85.4 -> 69.8; center_head.1 26.3 -> 22.4; res2 19.9 -> 17.0.
#ifdef TRISC_UNPACK
template <uint32_t cb_table, uint32_t taps>
inline uint32_t fifo_unit_sources(uint32_t (&src)[taps]) {
    invalidate_l1_cache();
    const uint32_t entry = get_local_cb_interface(get_operand_id(cb_table)).fifo_rd_ptr << 4;
    volatile uint32_t* w = reinterpret_cast<volatile uint32_t*>(entry);
    for (uint32_t t = 0; t < taps; ++t) {
        src[t] = w[t];
    }
    return w[3];
}
#endif

// windows: a unit flagged in bit 31 of its pop word is one strided window of 3 * tw tiles from src[0] (TapDirectFifo:
// three adjacent slab runs, row stride one pixel = tw tiles), tilized in fast-tilize chunks of that width, as
// window_tilize does at stride 2. UNPACK hands the flag to MATH and PACK through their mailboxes (as get_tile_address),
// since the DEST sections differ. Same tiles, same order.
template <
    uint32_t tw,
    uint32_t taps,
    bool skip_center,
    uint32_t cb_in,
    uint32_t cb_out,
    uint32_t cb_table,
    bool windows>
ALWI void fifo_tilize(DataflowBuffer& in, DataflowBuffer& out, DataflowBuffer& table, uint32_t units) {
    static_assert(tw != 0 && taps <= 3, "a source entry is 16 B: at most 3 taps and the pop count");
    static_assert(!windows || (taps == 3 && !skip_center && tw > 1), "a window unit is three adjacent taps");
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
    fast_tilize_init(cb_in, tw, cb_out);
    if constexpr (tw > 1) {
        PACK((conv2d_3x3::lean_tilize::pack_init()));
    }
    [[maybe_unused]] uint32_t prev_chunk = 0;
    [[maybe_unused]] NarrowTilizeBatch<cb_in, cb_out> nt{out};
    for (uint32_t u = 0; u < units; ++u) {
        table.wait_front(1);
        [[maybe_unused]] uint32_t src[taps];
        uint32_t pages = 0;
        UNPACK((pages = fifo_unit_sources<cb_table, taps>(src)));
#ifdef CONV2D_3X3_ABLATE_TILIZE
        UNPACK((pages &= 0x7fffffffu));
        out.reserve_back(taps * tw);
        out.push_back(taps * tw);
        for (uint32_t p = 0; p < pages; p += tw) {
            in.wait_front(tw);
            in.pop_front(tw);
        }
        table.pop_front(1);
        continue;
#endif
        if constexpr (windows) {
            uint32_t window = 0;
            UNPACK({
                window = pages >> 31;
                pages &= 0x7fffffffu;
                ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, window);
                ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, window);
            })
            MATH(window = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
            PACK(window = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
            if (window != 0) {
                constexpr uint32_t wt = 3 * tw;
                out.reserve_back(wt);
                PACK((conv2d_3x3::lean_tilize::pack_row_begin(cb_out, 0)));
                for (uint32_t tiles_done = 0; tiles_done < wt;) {
                    const uint32_t chunk = conv2d_3x3::lean_tilize::chunk_of(wt - tiles_done);
                    MATH((llk_math_wait_for_dest_available()));
                    PACK((llk_packer_wait_for_math_done()));
                    if (chunk != prev_chunk) {
                        UNPACK((llk_unpack_fast_tilize_reinit_xdim(chunk)));
                        prev_chunk = chunk;
                    }
                    UNPACK((window_unpack_chunk<cb_in>(src[0], chunk, tiles_done)));
                    MATH((llk_math_fast_tilize_block_<DST_ACCUM_MODE>(0, cb_in, 4)));
                    PACK((conv2d_3x3::lean_tilize::pack_row_chunk(chunk)));
                    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
                    tiles_done += chunk;
                }
                out.push_back(wt);
                table.pop_front(1);
                continue;
            }
        }
        for (uint32_t t = 0; t < taps; ++t) {
            if constexpr (skip_center) {
                if (t == 1) {
                    if constexpr (tw == 1) {
                        nt.close();  // the open section's tiles precede this slot
                    }
                    out.reserve_back(tw);
                    out.push_back(tw);
                    continue;
                }
            }
            if constexpr (tw == 1) {
                nt.tile(src[t]);
            } else {
                out.reserve_back(tw);
                PACK((conv2d_3x3::lean_tilize::pack_row_begin(cb_out, 0)));
                for (uint32_t tiles_done = 0; tiles_done < tw;) {
                    const uint32_t chunk = conv2d_3x3::lean_tilize::chunk_of(tw - tiles_done);
                    MATH((llk_math_wait_for_dest_available()));
                    PACK((llk_packer_wait_for_math_done()));
                    if (chunk != prev_chunk) {
                        UNPACK((llk_unpack_fast_tilize_reinit_xdim(chunk)));
                        prev_chunk = chunk;
                    }
                    UNPACK((window_unpack_chunk<cb_in>(src[t], chunk, tiles_done)));
                    MATH((llk_math_fast_tilize_block_<DST_ACCUM_MODE>(0, cb_in, 4)));
                    PACK((conv2d_3x3::lean_tilize::pack_row_chunk(chunk)));
                    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
                    tiles_done += chunk;
                }
                out.push_back(tw);
            }
        }
        // pages is set on the unpacker only, the only thread with a wait / pop.
        for (uint32_t p = 0; p < pages; p += tw) {
            in.wait_front(tw);
            in.pop_front(tw);
        }
        table.pop_front(1);
    }
    if constexpr (tw == 1) {
        nt.close();
    }
    fast_tilize_uninit(cb_in, cb_out, tw);
}

// Center-tap slots: tilizes num_rows im2col rows of three taps (tap-major, each a [32, tw * 32] block), except tap 1,
// whose tw tiles the reader wrote into their in0 slots: those are reserved and pushed unpacked, and the row's empty
// tap-1 block is popped unread. One tilize init / uninit for all of it, as in the one-call form.
template <uint32_t tw, uint32_t cb_rm, uint32_t cb_tiles>
ALWI void tilize_rows_skip_center(DataflowBuffer& rm, DataflowBuffer& tiles, uint32_t num_rows) {
    namespace tc = ckl::tilize_config;
    namespace lt = conv2d_3x3::lean_tilize;
#ifdef CONV2D_3X3_ABLATE_TILIZE
    for (uint32_t r = 0; r < 3 * num_rows; ++r) {
        rm.wait_front(tw);
        tiles.reserve_back(tw);
        tiles.push_back(tw);
        rm.pop_front(tw);
    }
    return;
#endif
    for (uint32_t r = 0; r < num_rows; ++r) {
        if (r == 0) {
            lt::tilize<tw, cb_rm, cb_tiles, tc::InitUninitMode::InitOnly>(rm, tiles, 1);
        } else {
            lt::tilize<tw, cb_rm, cb_tiles, tc::InitUninitMode::Neither, false>(rm, tiles, 1);
        }
        tiles.reserve_back(tw);
        tiles.push_back(tw);
        rm.wait_front(tw);
        rm.pop_front(tw);
        if (r + 1 == num_rows) {
            lt::tilize<tw, cb_rm, cb_tiles, tc::InitUninitMode::UninitOnly, false>(rm, tiles, 1);
        } else {
            lt::tilize<tw, cb_rm, cb_tiles, tc::InitUninitMode::Neither, false>(rm, tiles, 1);
        }
    }
}

// Tilizes one K-block (block_m_tiles tile-rows x block_k_tiles) of gathered im2col rows (lean BH fast tilize:
// conv2d_3x3_lean_tilize.hpp). The matmul
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
    uint32_t center_mode,
    bool tap_direct,
    bool windows,
    uint32_t cb_table>
struct TilizeIm2colKBlock {
    DataflowBuffer* rm;
    DataflowBuffer* tiles;
    // Tap-direct: the unit entry FIFO (cb_src_table).
    DataflowBuffer* table;

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
            if constexpr (tap_direct) {
                fifo_tilize<tilize_width_tiles, 3, true, cb_im2col_rm, cb_im2col_tiles, cb_table, false>(
                    *rm, *tiles, *table, block_m_tiles);
            } else {
                tilize_rows_skip_center<tilize_width_tiles, cb_im2col_rm, cb_im2col_tiles>(*rm, *tiles, block_m_tiles);
            }
        } else if constexpr (tap_direct) {
            // Includes the unpacker's per-unit wait on the reader.
            MaybeDeviceZoneScope("compute_tilize");
            fifo_tilize<
                tilize_width_tiles,
                block_k_tiles / tilize_width_tiles,
                false,
                cb_im2col_rm,
                cb_im2col_tiles,
                cb_table,
                windows>(*rm, *tiles, *table, block_m_tiles);
        } else {
            // Includes the unpacker's per-tile-row wait on the reader's gather.
            MaybeDeviceZoneScope("compute_tilize");
            if constexpr (strided_window) {
                static_assert(block_k_tiles == 3 * c_in_tiles, "a window unit is one kernel row");
#ifndef CONV2D_3X3_S2_TAPS
                // (S2_TAPS: every K-block's taps were built before the walk.)
                window_tilize<c_in_tiles, cb_im2col_rm, cb_im2col_tiles>(*rm, *tiles, block_m_tiles);
#endif
            } else {
#ifdef CONV2D_3X3_ABLATE_TILIZE
                for (uint32_t r = 0; r < block_m_tiles * (block_k_tiles / tilize_width_tiles); ++r) {
                    rm->wait_front(tilize_width_tiles);
                    tiles->reserve_back(tilize_width_tiles);
                    tiles->push_back(tilize_width_tiles);
                    rm->pop_front(tilize_width_tiles);
                }
#else
                conv2d_3x3::lean_tilize::tilize<tilize_width_tiles, cb_im2col_rm, cb_im2col_tiles>(
                    *rm, *tiles, block_m_tiles * (block_k_tiles / tilize_width_tiles));
#endif
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

#ifdef CONV2D_3X3_SLAB_DIRECT
// Slab-direct in0 (conv2d_3x3_slab_direct.hpp): every in0 tile is unpacked straight from the padded slab as two
// 2-face runs, K columns [0, 16) of its 32 rows at a and K columns [16, 32) at a + SD_HALF_BYTES. Each run lands in
// SrcB as 32 rows, so SrcB holds the tile's faces in column order (m0 k0, m1 k0, m0 k1, m1 k1) instead of row order
// (m0 k0, m0 k1, m1 k0, m1 k1), and the matmul MOP's two SrcB face jumps become +16 rows (B0 -> B2's slot) and
// +16 rows (B2 -> B1's slot) instead of +32 and +48: the same products in the same order (bit-identical to the
// face-direct gather's in0).
// Raw LLK (capability): ckernel::matmul_block / compute_kernel_lib::matmul_block unpack in0 from a CB read pointer at
// a tile stride, one 4-face read per tile; nothing exposes a per-tile L1 source or a two-run in0 tile. Bypassed:
// compute_kernel_lib::matmul_block (and its tilize hook) on this path; the bias is one more K-step (ones block x the
// bias tile, conv2d_3x3_slab_direct.hpp) instead of compute_kernel_lib::add_bias_bcast_rows, so the matmul packs the
// final values. Measured (stem.conv1, Blackhole p150b, LoFi, unzoned, median of 3): 22787 ns (face-direct gather on
// the reader and writer) -> 11153 ns; 14041 ns keeping the helper bias pass.
namespace slab_direct {
#ifdef TRISC_UNPACK
// Saved at the first init: the SrcB tile descriptor's x_dim and the SrcB (ch1) Z stride of one face.
static uint32_t saved_xdim = 0;
static uint32_t saved_zstride = 0;

// After matmul_block_init: in0 reads of 512 datums (2 faces), the second one Z = 1 step further, i.e. x_dim
// (SD_HALF_BYTES) further in L1 and two faces further in SrcB.
inline void unpack_init() {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    if (saved_xdim == 0) {
        saved_xdim = cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32] >> 16;
        saved_zstride = cfg[UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] & 0xffff;
    }
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 16, 0xffff0000>(SD_HALF_BYTES / 2);
    cfg_reg_rmw_tensix<UNP1_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(2 * saved_zstride);
    TTI_SETADCXX(p_setadc::UNP_B, 2 * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
}

inline void unpack_uninit() {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 16, 0xffff0000>(saved_xdim);
    cfg_reg_rmw_tensix<UNP1_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(saved_zstride);
}

// One k-step of an (rt x ct) subblock, ct >= rt (in0 reused across the ct in1 tiles, the LLK's reuse_a order):
// a[t] = in0 tile t's first run, b = in1's first tile (16 B units, minus 1). Mirrors _llk_unpack_AB_matmul_.
template <uint32_t ct_dim, uint32_t rt_dim>
inline void unpack_k_step(const uint32_t* a, uint32_t b) {
    static_assert(ct_dim >= rt_dim);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    for (uint32_t t = 0; t < rt_dim; ++t) {
        wait_for_next_context(2);
        _llk_unpack_configure_addresses_(b, a[t], cfg);
        semaphore_post(semaphore::UNPACK_SYNC);
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        TTI_UNPACR(SrcB, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcB, 0b00010001, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101);
        TT_MOP(0, ct_dim - 1, unp_cfg_context == 0 ? 0 : 0xff);
        t6_semaphore_get(semaphore::UNPACK_SYNC);
        switch_config_context(unp_cfg_context);
    }
}
#endif

#ifdef TRISC_MATH
// matmul_configure_addrmod's full-tile, non-transposed ADDR_MOD_2 / ADDR_MOD_4 with the column-order SrcB jumps.
inline void math_patch_addrmods() {
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 1},
        .srcb = {.incr = 16, .clr = 0, .cr = 1},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2);
    addr_mod_t{
        .srca = {.incr = 32, .clr = 0, .cr = 1},
        .srcb = {.incr = 16, .clr = 0, .cr = 1},
        .dest = {.incr = 0, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_4);
}

// matmul_block's math half (with its dynamic throttle), re-patching the address mods after every MOP re-init.
template <uint32_t cb_in0, uint32_t cb_in1, uint32_t ct_dim, uint32_t rt_dim>
inline void math_k_step(uint32_t idst) {
    volatile uint32_t mm_throttle_en = *(ckernel::throttle_ptr) % 2;
    if (mm_throttle_en) {
        if (ckernel::throttled_mop_status != 1) {
            llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE_MAX>(cb_in0, cb_in1, 0, ct_dim, rt_dim);
            math_patch_addrmods();
            ckernel::throttled_mop_status = 1;
        }
        llk_math_matmul<MATH_FIDELITY, MM_THROTTLE_MAX>(idst, ct_dim, rt_dim);
    } else {
        if (ckernel::throttled_mop_status != 0) {
            llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(cb_in0, cb_in1, 0, ct_dim, rt_dim);
            math_patch_addrmods();
            ckernel::throttled_mop_status = 0;
        }
        llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim);
    }
}
#endif
}  // namespace slab_direct
#endif

}  // namespace

// N-slice ring: this core's arrival count (semaphore RING_ARRIVAL_SEM_ID; column my_col - 1's reader writes it behind
// each row group it forwards here). Dataflow firmware exposes the semaphore base as sem_l1_base; TRISC firmware does
// not, so the packer reads it from the launch message (BRISC advances launch_msg_rd_ptr only after every RISC of the
// core finished, so the pointer names this launch throughout).
inline volatile tt_l1_ptr uint32_t* ring_arrival_count() {
    const uint32_t rd = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    const volatile tt_l1_ptr kernel_config_msg_t* cfg = &GET_MAILBOX_ADDRESS_DEV(launch[rd])->kernel_config;
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        cfg->kernel_config_base[ProgrammableCoreType::TENSIX] + cfg->sem_offset[ProgrammableCoreType::TENSIX] +
        RING_ARRIVAL_SEM_ID * L1_ALIGNMENT);
}

void kernel_main() {
#ifdef CONV2D_3X3_KW_FOLD
    // kw-fold + output-shift path (conv2d_3x3_kw_fold.hpp); runtime args 3, 4: the core's first output tile-row and
    // its valid tile-rows; 0: its M-blocks; 5, 6: halo exchange with the upper / lower neighbour core.
    conv2d_3x3::kw_fold::run<
        get_compile_time_arg_val(14),  // cb_input_tiles: the slab
        get_compile_time_arg_val(2),   // cb_weights
        get_compile_time_arg_val(3),   // cb_bias (kw-fold: the writer's scratch)
        get_compile_time_arg_val(5),   // cb_output_tiles
        get_compile_time_arg_val(4),   // cb_matmul_partials: the fp32 accumulator
        get_compile_time_arg_val(23),  // cb_combined: the fp32 bias rows (srcB of the final pass)
        get_compile_time_arg_val(40),  // cb_src_table: ACC-ready token (pack -> unpack)
        KWF_CB_HALO_OUT,               // halo exchange: pack -> writer tokens
        KWF_CB_HALO_IN,                // halo exchange: writer -> unpack tokens
        get_compile_time_arg_val(21),  // N
        get_compile_time_arg_val(17),  // C_in tiles
        get_compile_time_arg_val(6),   // bm
        KWF_D,
        KWF_TPR,
        KWF_SBW,
        KWF_SBH,
        KWF_GW,
        get_compile_time_arg_val(13) != 0,                            // relu
        get_compile_time_arg_val(18) * get_compile_time_arg_val(17),  // slab pages
        KWF_M_TILES,
        KWF_KBLOCKED != 0,
        KWF_PLANE_BYTES,
        KWF_NOINLINE != 0,
        KWF_XCHG != 0,
        KWF_FUSED != 0>(
        get_arg_val<uint32_t>(3),
        get_arg_val<uint32_t>(4),
        get_arg_val<uint32_t>(0),
        get_arg_val<uint32_t>(5) != 0,
        get_arg_val<uint32_t>(6) != 0);
#else
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
    // N-slice ring: one page per packed row group, packer -> reader (see the N-slice branch).
    constexpr uint32_t cb_ring_token = get_compile_time_arg_val(31);
    // N-slice ring subblock: slice_sb_h rows (the ring's row group; the last group of a slice may be shorter, see
    // conv2d_3x3::ring_group_row0) x slice_sb_w columns.
    constexpr uint32_t slice_sb_h = get_compile_time_arg_val(32);
    constexpr uint32_t slice_sb_w = get_compile_time_arg_val(33);
    constexpr uint32_t untilize_batch_rows = get_compile_time_arg_val(34);
    constexpr bool strided_window = get_compile_time_arg_val(35) != 0;
    constexpr uint32_t center_mode = get_compile_time_arg_val(36);
    constexpr uint32_t cb_center_tiles = get_compile_time_arg_val(37);
    // Ring slab (see the reader): M-block m untilizes runtime arg 3 + 2m tile-rows and pops runtime arg 4 + 2m unread
    // tail rows of its last cb_input_tiles batch.
    constexpr bool ring_slab = get_compile_time_arg_val(38) != 0;
    // Tap-direct N-slice prelude: tap block sources from cb_src_table (one entry per cb_im2col_rm unit slot).
    constexpr bool tap_direct = get_compile_time_arg_val(39) != 0;
    constexpr uint32_t cb_src_table = get_compile_time_arg_val(40);
    constexpr uint32_t im2col_rm_depth_rows = get_compile_time_arg_val(41);
    // Tap-direct on the K-blocked walk: cb_src_table is a FIFO of unit entries (fifo_tilize), with window units.
    constexpr bool kb_tap_direct = get_compile_time_arg_val(42) != 0;
    constexpr bool tap_direct_windows = get_compile_time_arg_val(43) != 0;
    // N-slice center taps (see the file comment).
    constexpr bool nslice_center = get_compile_time_arg_val(44) != 0;
    // Work-balanced M split (the descriptor's balanced_block_ranges): on the N-slice walk the untilize runs a
    // host-given slab row count, and the tilize (rt_prelude: not under the split row gather, whose unit sequence is CT)
    // and the ring walk run over the M-block's valid tile-rows only. Per M-block runtime args (valid tile-rows, slab
    // rows) after the masks.
    constexpr bool rt_valid = get_compile_time_arg_val(45) != 0;
    constexpr bool rt_prelude = get_compile_time_arg_val(46) != 0;
    // The ring walk's group matmuls with the group height as a template constant (see group_mm).
    constexpr bool ct_group_rows = get_compile_time_arg_val(47) != 0;
    static_assert(!ct_group_rows || (k_split && nslice_outer && slice_sb_w == 1 && slice_sb_h == 4));
    static_assert(!rt_valid || (k_split && nslice_outer));
    static_assert(!rt_prelude || rt_valid);
    static_assert(!nslice_center || (tap_direct && block_k_tiles == 3 * c_in_tiles));
    // Padded slab (the descriptor's PADDED_SLAB, conv2d_3x3_padded_slab.hpp): N-slice center taps on a slab whose rows
    // sit in zero-gapped slots; the prelude tilizes every tap straight from it (padded_tilize), the reader only fills.
#ifdef CONV2D_3X3_PADDED_SLAB
    constexpr bool padded_slab = true;
    constexpr int32_t pad_in_h = PAD_SLAB_IN_H;
    constexpr int32_t pad_in_w = PAD_SLAB_IN_W;
    constexpr uint32_t pad_out_w = PAD_SLAB_OUT_W;
    constexpr int32_t pad_p = PAD_SLAB_PAD;
    constexpr int32_t pad_d = PAD_SLAB_DIL;
    constexpr int32_t pad_num_in_px = PAD_SLAB_NUM_IN_PX;
    constexpr uint32_t pad_num_out_px = PAD_SLAB_NUM_OUT_PX;
    constexpr uint32_t pad_slots_max = PAD_SLAB_SLOTS_MAX;
    constexpr uint32_t cb_zero_src = PAD_SLAB_CB_ZERO;
#else
    constexpr bool padded_slab = false;
    constexpr int32_t pad_in_h = 1, pad_in_w = 32, pad_p = 1, pad_d = 1, pad_num_in_px = 32;
    constexpr uint32_t pad_out_w = 32, pad_num_out_px = 32, pad_slots_max = 1, cb_zero_src = 0;
#endif
    static_assert(!padded_slab || (nslice_center && k_split && nslice_outer && tile_input && pad_p == pad_d));
    // Stride-2 taps from DEST (the descriptor's S2_TAPS, conv2d_3x3_s2_taps.hpp).
#ifdef CONV2D_3X3_S2_TAPS
    constexpr bool s2_taps = true;
    [[maybe_unused]] constexpr int32_t s2_in_h = S2_IN_H;
    [[maybe_unused]] constexpr int32_t s2_in_w = S2_IN_W;
    [[maybe_unused]] constexpr uint32_t s2_out_w = S2_OUT_W;
    [[maybe_unused]] constexpr int32_t s2_num_in_px = S2_NUM_IN_PX;
    [[maybe_unused]] constexpr uint32_t s2_num_out_px = S2_NUM_OUT_PX;
#else
    constexpr bool s2_taps = false;
    [[maybe_unused]] constexpr int32_t s2_in_h = 1, s2_in_w = 64, s2_num_in_px = 64;
    [[maybe_unused]] constexpr uint32_t s2_out_w = 32, s2_num_out_px = 32;
#endif
    static_assert(!s2_taps || (strided_window && tile_input && !face_direct && !ring_slab && center_mode == 0));
    // Slab-direct (the descriptor's SLAB_DIRECT): face-direct whole-K, in0 unpacked straight from the padded slab.
#ifdef CONV2D_3X3_SLAB_DIRECT
    constexpr bool slab_direct = true;
#else
    constexpr bool slab_direct = false;
#endif
    static_assert(!slab_direct || (face_direct && num_k_blocks == 1 && !k_split && !tile_input));
    static_assert(!tap_direct || (!face_direct && !strided_window));
    static_assert(kb_tap_direct == (tap_direct && !(k_split && nslice_outer)));
    static_assert(!ring_slab || (tile_input && !k_split && !face_direct && c_in_tiles > 1));
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
        center_mode,
        kb_tap_direct,
        tap_direct_windows,
        cb_src_table>;
    DataflowBuffer src_table_dfb(cb_src_table);
    auto make_pre_k_block = [&]() {
        if constexpr (face_direct) {
            return ckl::NoPreKBlock{};
        } else {
            return TilizeFn{&im2col_rm_dfb, &im2col_tiles_dfb, &src_table_dfb};
        }
    };
    const auto pre_k_block = make_pre_k_block();
    using PreKBlockFn = std::decay_t<decltype(pre_k_block)>;

    // The bias is waited for at its first use, not at kernel start: every core reads the same few bias pages from DRAM
    // at kernel start (3-4 us), and an early wait would hold the unpacker, so the first untilize and the fill behind
    // its 2-batch input CB, for that long. It stays fronted until the kernel end, so every later wait returns at once.
    // finalize_block over `shape`'s rows of the matmul partials: + bias, ReLU on this pack only.
    auto add_bias = [&](const ckl::BiasAddShape& shape) {
        {
            MaybeDeviceZoneScope("compute_bias_wait");
            bias.wait_front(out_n_tiles);
        }
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
            {
                // First use of the bias on this path (see add_bias).
                MaybeDeviceZoneScope("compute_combine_bias_wait");
                bias.wait_front(out_n_tiles);
            }
            if constexpr (apply_relu) {
                pack_relu_config(ReluConfig::zero());
            }
            ckl::add_bias_bcast_rows<ckl::BiasBroadcast::RowBroadcast, ckl::OutputCBLayout::TileRowMajor>(
                combined, bias, output_tiles, combine_bias_shape);
        }
    };

    // Untilize of a >= 2-tile-wide slab. Raw LLK on a measured win (see conv2d_3x3_wide_untilize.hpp): the helper's
    // fast_untilize packs every slab row of a > 4-tile row as its own L1 stream and syncs per tile-row. ring_rows > 0:
    // cb_halo_slab is the ring slab, and a section group never crosses its end.
    [[maybe_unused]] uint32_t untilize_ring_pos = 0;
    // Tap-direct: units tilized so far (the reader's direct_seq).
    [[maybe_unused]] uint32_t direct_seq = 0;
    [[maybe_unused]] auto untilize_wide_rows = [&](uint32_t rows, uint32_t ring_rows) {
        // kernel_main is not a template: this is instantiated on ROW_MAJOR paths too (untilize_width_tiles = 1).
        constexpr uint32_t wide_ct = untilize_width_tiles < 2 ? 2 : untilize_width_tiles;
        if constexpr (ckl::can_use_fast_untilize<untilize_width_tiles, cb_input_tiles, cb_halo_slab>()) {
            wide_untilize::untilize_wide<wide_ct, cb_input_tiles, cb_halo_slab>(
                input_tiles_dfb, halo_slab_dfb, rows, untilize_batch_rows, ring_rows, untilize_ring_pos);
        } else {
            ckl::untilize<untilize_width_tiles, cb_input_tiles, cb_halo_slab>(input_tiles_dfb, halo_slab_dfb, rows);
        }
    };

    // Padded slab: the block's output range (runtime args after the kernel-row masks and the rt_valid per-M-block
    // args), the slab and zero-block bases.
    [[maybe_unused]] uint32_t pad_m_tile_start = 0;
    [[maybe_unused]] uint32_t pad_actual_m_tiles = 0;
    [[maybe_unused]] uint32_t pad_slab_l1 = 0;
    [[maybe_unused]] uint32_t pad_zero_l1 = 0;
    if constexpr (padded_slab) {
        constexpr uint32_t kh_words = ((block_m_tiles + slice_sb_h - 1) / slice_sb_h + 9) / 10;
        constexpr uint32_t rt_words = rt_valid ? 2 : 0;
        pad_m_tile_start = get_arg_val<uint32_t>(3 + num_m_blocks_this_core * (kh_words + rt_words));
        pad_actual_m_tiles = get_arg_val<uint32_t>(4 + num_m_blocks_this_core * (kh_words + rt_words));
        // The unpacker never pops either CB: their read pointers stay at the base.
        UNPACK((pad_slab_l1 = get_local_cb_interface(get_operand_id(cb_halo_slab)).fifo_rd_ptr << 4));
        UNPACK((pad_zero_l1 = get_local_cb_interface(get_operand_id(cb_zero_src)).fifo_rd_ptr << 4));
        PACK((pad_slab_l1 = (get_local_cb_interface(get_output_id(cb_halo_slab)).fifo_wr_ptr) << 4));
    }
    // S2_TAPS: the core's output range (runtime args after the kernel-row masks and the rt_valid per-M-block args).
    [[maybe_unused]] uint32_t s2_m_tile_start = 0;
    [[maybe_unused]] uint32_t s2_actual_m_tiles = 0;
    if constexpr (s2_taps) {
        constexpr uint32_t kh_words =
            (k_split && nslice_outer) ? ((block_m_tiles + slice_sb_h - 1) / slice_sb_h + 9) / 10 : 0;
        constexpr uint32_t rt_words = rt_valid ? 2 : 0;
        s2_m_tile_start = get_arg_val<uint32_t>(3 + num_m_blocks_this_core * (kh_words + rt_words));
        s2_actual_m_tiles = get_arg_val<uint32_t>(4 + num_m_blocks_this_core * (kh_words + rt_words));
    }
    [[maybe_unused]] conv2d_3x3::padded_slab::Plan pad_plan{};
    [[maybe_unused]] uint32_t pad_o_first = 0;
    [[maybe_unused]] uint32_t pad_o_end = 0;

    // rt_valid: the per-M-block args follow the kernel-row masks (ceil(ring groups / 10) words per M-block).
    constexpr uint32_t rt_ring_groups = (block_m_tiles + slice_sb_h - 1) / slice_sb_h;
    [[maybe_unused]] const uint32_t rt_arg_base = 3 + ((rt_ring_groups + 9) / 10) * num_m_blocks_this_core;
    // N-slice ring: incoming row groups packed onto so far, over all M-blocks (the arrival count each waits for).
    [[maybe_unused]] uint32_t ring_need = 0;
    for (uint32_t m_block_idx = 0; m_block_idx < num_m_blocks_this_core; ++m_block_idx) {
        [[maybe_unused]] const uint32_t valid_rows =
            rt_valid ? get_arg_val<uint32_t>(rt_arg_base + 2 * m_block_idx) : block_m_tiles;
        [[maybe_unused]] const uint32_t slab_rows =
            rt_valid ? get_arg_val<uint32_t>(rt_arg_base + 2 * m_block_idx + 1) : slab_tile_rows;
        [[maybe_unused]] const uint32_t prelude_rows = rt_prelude ? valid_rows : block_m_tiles;
        [[maybe_unused]] uint32_t s2_o_first = 0;
        [[maybe_unused]] uint32_t s2_o_end = 0;
        if constexpr (s2_taps) {
            const uint32_t done = m_block_idx * block_m_tiles;
            const uint32_t valid = s2_actual_m_tiles - done < block_m_tiles ? s2_actual_m_tiles - done : block_m_tiles;
            s2_o_first = (s2_m_tile_start + done) * 32;
            const uint32_t o_end_tiles = s2_o_first + valid * 32;
            s2_o_end = o_end_tiles < s2_num_out_px ? o_end_tiles : s2_num_out_px;
        }
        if constexpr (padded_slab) {
            const uint32_t done = m_block_idx * block_m_tiles;
            const uint32_t valid =
                pad_actual_m_tiles - done < block_m_tiles ? pad_actual_m_tiles - done : block_m_tiles;
            pad_o_first = (pad_m_tile_start + done) * 32;
            const uint32_t o_end_tiles = pad_o_first + valid * 32;
            pad_o_end = o_end_tiles < pad_num_out_px ? o_end_tiles : pad_num_out_px;
            conv2d_3x3::padded_slab::plan<pad_out_w, pad_in_w, pad_p, pad_d, pad_num_in_px>(
                pad_o_first, pad_o_end, pad_plan);
        }
        if constexpr (tile_input) {
            // untilize_footprint: the reader streams slab_tile_rows input tile-rows in slab order.
            MaybeDeviceZoneScope("compute_untilize");
            pack_reconfig_l1_acc(0);
            if constexpr (s2_taps) {
                // Stride-2 taps (conv2d_3x3_s2_taps.hpp): no untilize; the prelude builds every tap from the input
                // ring.
            } else
#ifdef CONV2D_3X3_TAP_SHIFT
                if constexpr (padded_slab) {
                // Tap shift (conv2d_3x3_tap_shift.hpp): no untilize; the prelude builds every tap from the input ring.
            } else
#endif
#ifdef CONV2D_3X3_ABLATE_UNTILIZE
                if constexpr (true) {
                {
                    uint32_t rows = slab_rows;
                    uint32_t tail_rows = 0;
                    uint32_t pushed_tail = 0;
                    if constexpr (padded_slab) {
                        rows = pad_plan.rows;
                        pushed_tail = slab_rows - rows;
                    } else if constexpr (ring_slab) {
                        rows = get_arg_val<uint32_t>(3 + 2 * m_block_idx);
                        tail_rows = get_arg_val<uint32_t>(4 + 2 * m_block_idx);
                    }
                    for (uint32_t r = 0; r < rows; ++r) {
                        input_tiles_dfb.wait_front(untilize_width_tiles);
                        halo_slab_dfb.reserve_back(untilize_width_tiles);
                        halo_slab_dfb.push_back(untilize_width_tiles);
                        input_tiles_dfb.pop_front(untilize_width_tiles);
                    }
                    if (tail_rows > 0) {
                        input_tiles_dfb.wait_front(tail_rows * c_in_tiles);
                        input_tiles_dfb.pop_front(tail_rows * c_in_tiles);
                    }
                    if (pushed_tail > 0) {
                        input_tiles_dfb.wait_front(pushed_tail * c_in_tiles);
                        input_tiles_dfb.pop_front(pushed_tail * c_in_tiles);
                        halo_slab_dfb.reserve_back(pushed_tail * c_in_tiles);
                        halo_slab_dfb.push_back(pushed_tail * c_in_tiles);
                    }
                }
            } else
#endif
                if constexpr (padded_slab) {
                // Padded slab: each of the plan's tile-rows to its row slot (one M-block per core: the slab CB's
                // pointers stay at the base). Every thread builds the table: the wide untilize's section split depends
                // on it. The fill's unread tail rows are popped, not untilized, and their slab pages pushed unwritten.
                [[maybe_unused]] uint32_t px_bytes[slab_tile_rows > 0 ? slab_tile_rows : 1];
                const uint32_t rows = pad_plan.rows;
                for (uint32_t k = 0; k < rows; ++k) {
                    px_bytes[k] = pad_slab_l1 +
                                  conv2d_3x3::padded_slab::tile_row_px<pad_in_w, pad_d, pad_slots_max>(pad_plan, k) *
                                      c_in_tiles * 64;
                }
                if constexpr (untilize_width_tiles == 1) {
                    conv2d_3x3::narrow_untilize::untilize_1wide_at<cb_input_tiles, cb_halo_slab>(
                        input_tiles_dfb, halo_slab_dfb, rows, untilize_batch_rows, px_bytes);
                } else {
                    constexpr uint32_t wide_ct = untilize_width_tiles < 2 ? 2 : untilize_width_tiles;
                    wide_untilize::untilize_wide<wide_ct, cb_input_tiles, cb_halo_slab>(
                        input_tiles_dfb, halo_slab_dfb, rows, untilize_batch_rows, 0, untilize_ring_pos, px_bytes);
                }
                const uint32_t tail = slab_rows - rows;
                if (tail > 0) {
                    input_tiles_dfb.wait_front(tail * c_in_tiles);
                    input_tiles_dfb.pop_front(tail * c_in_tiles);
                    halo_slab_dfb.reserve_back(tail * c_in_tiles);
                    halo_slab_dfb.push_back(tail * c_in_tiles);
                }
            } else if constexpr (ring_slab) {
                const uint32_t rows = get_arg_val<uint32_t>(3 + 2 * m_block_idx);
                const uint32_t tail_rows = get_arg_val<uint32_t>(4 + 2 * m_block_idx);
                if (rows > 0) {
                    untilize_wide_rows(rows, slab_tile_rows);
                }
                if (tail_rows > 0) {
                    input_tiles_dfb.wait_front(tail_rows * c_in_tiles);
                    input_tiles_dfb.pop_front(tail_rows * c_in_tiles);
                }
            } else if constexpr (untilize_width_tiles == 1) {
                // Raw LLK on a measured win (see conv2d_3x3_narrow_untilize.hpp): the helper's 1-wide path is
                // packer-bound at one DEST section and one CB sync per tile-row.
                conv2d_3x3::narrow_untilize::untilize_1wide<cb_input_tiles, cb_halo_slab>(
                    input_tiles_dfb, halo_slab_dfb, slab_rows, untilize_batch_rows);
            } else {
                untilize_wide_rows(slab_rows, 0);
            }
        }

        if constexpr (k_split && nslice_outer) {
            // N-slice-outer K split as a ring reduce-scatter (conv2d_3x3_nslice_order.hpp): the whole K of in0 is
            // tilized once into a resident [bm, K] block (the reader gathers tile-row outer, K-block inner). Step j
            // multiplies owner my_col - 1 - j's slice over the whole K (DEST accumulates over K) and packs it into
            // slot k_cols - 1 - j of cb_combine_slots with packer L1 accumulation onto the partial column my_col - 1
            // sent for that owner (step 0 starts the chain: plain pack). The slots are reserved whole, so every pack
            // is at an absolute offset. Each row group (one matmul subblock row) of a step with an incoming partial
            // (j >= 1) waits for this core's arrival count, which the reader of column my_col - 1 writes right
            // behind the group's data, and is pushed as a ring token (4 pages) after its pack, which tells the reader
            // the group is ready to forward to my_col + 1. The last step (own slice) leaves the row's fp32 sum in
            // slot 0, the front of cb_combine_slots: only the bias add (or a copy) and ReLU remain, packed into the
            // output shard.
            // Raw LLK (matmul_block / pack_tile): compute_kernel_lib::matmul_block cannot pack its only K-block
            // with L1 accumulation onto a partial already in L1 (its L1-accumulating pack is "K-block > 0"), nor
            // gate each subblock row's pack on another core's arrival count, nor run a shorter last subblock: it
            // takes one subblock shape per call, which must divide the block. A bm 7 x out_n 2 slice (res5.x) then
            // only gets 1 x 2 subblocks, and the BH matmul unpacker reads rt + rt * ct tiles per k-step (ct + ct * rt
            // when rt > ct): 1.5 unpacks per tile-MAC, 19.5 cycles per tile-MAC. 4 x 1 subblocks with a 3-row last
            // group cost 1.25 / 1.33: 17.5 cycles per tile-MAC, res5.x @ LoFi 45.4 -> 42.2 us (Blackhole p150b).
            constexpr uint32_t k_tiles = block_k_tiles * num_k_blocks;
            // in0's K: all of it, or (N-slice center taps) taps 0 and 2 of every kernel row.
            constexpr uint32_t in0_k_tiles = k_tiles - (nslice_center ? 3 * c_in_tiles : 0);
            CircularBuffer center(cb_center_tiles);
            constexpr uint32_t num_owning_cols = (block_n_tiles + out_n_tiles - 1) / out_n_tiles;
            static_assert(out_n_tiles % slice_sb_w == 0 && (slice_sb_w == 1 || block_m_tiles % slice_sb_h == 0));
            constexpr uint32_t ring_groups = (block_m_tiles + slice_sb_h - 1) / slice_sb_h;
            // Kernel-row skip: ring row group g (rows [g * slice_sb_h, ...): every owner's walk has the same
            // groups, only their order differs, so a group is r0 / slice_sb_h) multiplies kernel rows
            // [kh_lo[g], kh_hi[g]) only, the range that is in the image for some row of the group. A kernel row
            // outside the image for all of a group's rows is all zeros in in0 (the reader publishes the zero block and
            // zero center rows for it; the padded-slab prelude does not tilize those units at all, see padded_tilize,
            // while the gather-paced preludes still do: skipping there measured flat). Only exact zero products are
            // removed: bit-identical. The host sends one 3-bit kernel-row mask per group, 10 per word,
            // ceil(ring_groups / 10) words per M-block (_nslice_kh_masks in the descriptor), always a contiguous range
            // of at least one kernel row. The loops keep the base loops' form: a per-kernel-row mask test measured
            // +2.7% on a 4 + 1-row group walk with nothing to skip.
            // Raw LLK (capability), as the rest of this walk: compute_kernel_lib::matmul_block multiplies the whole K
            // of every subblock; it has no per-subblock K range. aspp.convs.3 @ LoFi (d = 18: 1/3 of every core's
            // tile-MACs are out-of-image kernel rows) 33729 -> 27297 ns (Blackhole p150b, median of 3).
            uint8_t kh_lo[ring_groups];
            uint8_t kh_hi[ring_groups];
            {
                constexpr uint32_t words = (ring_groups + 9) / 10;
                for (uint32_t g = 0; g < ring_groups; ++g) {
                    const uint32_t word = get_arg_val<uint32_t>(3 + m_block_idx * words + g / 10);
                    const uint32_t mask = (word >> (3 * (g % 10))) & 7;
                    kh_lo[g] = static_cast<uint8_t>((mask & 1) ? 0 : ((mask & 2) ? 1 : 2));
                    kh_hi[g] = static_cast<uint8_t>((mask & 4) ? 3 : ((mask & 2) ? 2 : 1));
                }
            }
            // The tilize packs must overwrite (the previous M-block's combine may have left L1 accumulation on).
            pack_reconfig_l1_acc(0);
            if constexpr (!face_direct) {
                MaybeDeviceZoneScope("compute_tilize");
                if constexpr (strided_window) {
#ifdef CONV2D_3X3_S2_TAPS
                    // Stride-2 taps (conv2d_3x3_s2_taps.hpp): in0 [rows, K] (unit (t, kh) at t * K + kh * 3 ct) built
                    // from the input ring in input-row order; units of kernel rows the walk skips are not built. Then
                    // the ring is popped. Raw LLK (capability), see the header: no helper decimates rows.
                    im2col_tiles_dfb.reserve_back(prelude_rows * in0_k_tiles);
                    conv2d_3x3::s2_taps::
                        build<c_in_tiles, cb_input_tiles, cb_im2col_tiles, s2_out_w, s2_in_h, s2_in_w, s2_num_in_px>(
                            input_tiles_dfb,
                            s2_o_first,
                            s2_o_end,
                            [](uint32_t t, int32_t kh, uint32_t ch) {
                                return t * in0_k_tiles + static_cast<uint32_t>(kh) * 3 * c_in_tiles + ch;
                            },
                            [&](uint32_t t, int32_t kh) {
                                const uint32_t g = t / slice_sb_h;
                                return kh < kh_lo[g] || kh >= kh_hi[g];
                            });
                    im2col_tiles_dfb.push_back(prelude_rows * in0_k_tiles);
                    input_tiles_dfb.wait_front(slab_rows * c_in_tiles);
                    input_tiles_dfb.pop_front(slab_rows * c_in_tiles);
#else
                    window_tilize<c_in_tiles, cb_im2col_rm, cb_im2col_tiles>(
                        im2col_rm_dfb, im2col_tiles_dfb, prelude_rows * num_k_blocks);
#endif
                } else if constexpr (padded_slab) {
#ifdef CONV2D_3X3_TAP_SHIFT
                    // Tap shift (the descriptor's TAP_SHIFT, conv2d_3x3_tap_shift.hpp): the input ring no longer shares
                    // in0's L1 (it sits in the slab / combine-slots region), so the taps are built as their slab rows
                    // land, with no wait for the reader's center push (the walk waits for it). Then the ring is popped
                    // and the slab pages pushed unwritten (the reader's slab handshake), and the slab-dead token
                    // follows. Raw LLK (capability), see the header: no helper builds a tile from row ranges of two
                    // tiles.
                    conv2d_3x3::tap_shift::run<
                        c_in_tiles,
                        in0_k_tiles,
                        cb_input_tiles,
                        cb_im2col_tiles,
                        pad_out_w,
                        pad_in_h,
                        pad_in_w,
                        pad_p,
                        pad_d,
                        slice_sb_h,
                        ring_groups>(
                        input_tiles_dfb,
                        im2col_tiles_dfb,
                        prelude_rows,
                        pad_o_first,
                        pad_o_end,
                        pad_plan,
                        kh_lo,
                        kh_hi);
                    input_tiles_dfb.wait_front(slab_rows * c_in_tiles);
                    input_tiles_dfb.pop_front(slab_rows * c_in_tiles);
                    halo_slab_dfb.reserve_back(slab_rows * c_in_tiles);
                    halo_slab_dfb.push_back(slab_rows * c_in_tiles);
#else
                    {
                        // The slab's last pack landed (the push is stored after the packer drained), and the reader's
                        // center copy, with the slab gaps' zeroing, landed: it read the input ring, which shares in0's
                        // L1 (the walk waits on the same pages). Tilizing the tile-rows past the ring before the
                        // center push measured flat (the center copy ends with the untilize) and would need the gaps
                        // zeroed earlier.
                        MaybeDeviceZoneScope("compute_tilize_wait");
                        CircularBuffer(cb_halo_slab).wait_front(slab_rows * c_in_tiles);
                        center.wait_front(3 * block_m_tiles * c_in_tiles);
                    }
                    // Dead units (see padded_tilize): skipped only where some prelude row has one.
                    auto run_padded_tilize = [&](auto skip_dead) {
                        padded_tilize<
                            im2col_tilize_width_tiles,
                            block_m_tiles,
                            cb_im2col_rm,
                            cb_im2col_tiles,
                            pad_out_w,
                            pad_in_h,
                            pad_in_w,
                            pad_p,
                            pad_d,
                            slice_sb_h,
                            ring_groups,
                            decltype(skip_dead)::value>(
                            im2col_tiles_dfb,
                            prelude_rows,
                            pad_o_first,
                            pad_o_end,
                            pad_plan.band_row_off,
                            pad_slab_l1,
                            pad_zero_l1,
                            kh_lo,
                            kh_hi);
                    };
                    bool any_dead = false;
                    for (uint32_t g = 0; g * slice_sb_h < prelude_rows; ++g) {
                        any_dead |= kh_lo[g] > 0 || kh_hi[g] < 3;
                    }
                    if (any_dead) {
                        run_padded_tilize(std::true_type{});
                    } else {
                        run_padded_tilize(std::false_type{});
                    }
#endif
                    // The slab is dead: tell the reader (cb_src_table is a one-page token on this path).
                    CircularBuffer done_token(cb_src_table);
                    done_token.reserve_back(1);
                    done_token.push_back(1);
                } else if constexpr (tap_direct) {
                    direct_tilize<
                        im2col_tilize_width_tiles,
                        block_k_tiles / im2col_tilize_width_tiles,
                        im2col_rm_depth_rows,
                        cb_im2col_rm,
                        cb_im2col_tiles,
                        cb_src_table,
                        nslice_center ? 1 : -1>(
                        im2col_rm_dfb, im2col_tiles_dfb, prelude_rows * num_k_blocks, direct_seq);
                } else {
#ifdef CONV2D_3X3_ABLATE_TILIZE
                    for (uint32_t r = 0; r < prelude_rows * (k_tiles / im2col_tilize_width_tiles); ++r) {
                        im2col_rm_dfb.wait_front(im2col_tilize_width_tiles);
                        im2col_tiles_dfb.reserve_back(im2col_tilize_width_tiles);
                        im2col_tiles_dfb.push_back(im2col_tilize_width_tiles);
                        im2col_rm_dfb.pop_front(im2col_tilize_width_tiles);
                    }
#else
                    conv2d_3x3::lean_tilize::tilize<im2col_tilize_width_tiles, cb_im2col_rm, cb_im2col_tiles>(
                        im2col_rm_dfb, im2col_tiles_dfb, prelude_rows * (k_tiles / im2col_tilize_width_tiles));
#endif
                }
            }
            CircularBuffer ring_token(cb_ring_token);
            CircularBuffer slots(cb_combine_slots);
            {
                MaybeDeviceZoneScope("compute_matmul_walk");
                slots.reserve_back(k_cols * slot_tiles);
                const uint32_t first = conv2d_3x3::ring_next_real<k_cols, num_owning_cols>(my_col, 0);
                reconfig_data_format(cb_weights, cb_im2col_tiles);
                pack_reconfig_data_format(cb_combine_slots);
                ckernel::matmul_block_init(cb_im2col_tiles, cb_weights, false, slice_sb_w, slice_sb_h, k_tiles);
                uint32_t init_rows = slice_sb_h;
                // Wide short group (RING_WIDE_SHORT_GROUP, CT group heights only): a 1-row group of a slice 2..4 tiles
                // wide runs one 1 x out_n subblock (1 + out_n unpacks per k-step) instead of out_n 1 x 1 ones (2 each).
                // The spare-column groups' 5-row slices (4 + 1) need it: res5.0 / res5.1 on 13 groups 33.7 -> 29.3 /
                // 33.7 -> 29.1 us (zoned); with 1 x 1 short groups the 5-row walk is as slow as the 6-row one (4 + 2).
                // Each output tile still sums its k-tiles in the same order in DEST: bit-identical. Raw LLK as the rest
                // of this walk (compute_kernel_lib::matmul_block: one subblock shape per call, dividing the block).
#ifdef RING_WIDE_SHORT_GROUP
                constexpr bool wide_short = ct_group_rows && out_n_tiles >= 2 && out_n_tiles <= 4;
#else
                constexpr bool wide_short = false;
#endif
                uint32_t init_w = slice_sb_w;
#ifdef NSLICE_WEIGHT_COL_RUNS
                // nslice_step_shape: column-major weight rounds (the writer's NSLICE_WEIGHT_COL_RUNS): in1 tile (k, c)
                // at c * k_tiles + k; a wide subblock steps its in1 columns by k_tiles tiles (the unpacker's SrcA
                // address step, restored after the walk). Raw LLK (capability): no compute API sets the in1 column
                // stride.
                constexpr uint32_t in1_kstep = 1;
                constexpr uint32_t in1_cstride = k_tiles;
                UNPACK((llk_unpack_AB_matmul_set_in1_column_stride(cb_weights, k_tiles)));
#else
                constexpr uint32_t in1_kstep = out_n_tiles;
                constexpr uint32_t in1_cstride = 1;
#endif
                // rt_valid: the ring row groups over the block's valid tile-rows (the reader forwards the same ones).
                const uint32_t walk_groups = (valid_rows + slice_sb_h - 1) / slice_sb_h;
                im2col_tiles.wait_front(prelude_rows * in0_k_tiles);
                if constexpr (nslice_center) {
                    center.wait_front(3 * block_m_tiles * c_in_tiles);
                }
                for (uint32_t j = first; j < k_cols;
                     j = conv2d_3x3::ring_next_real<k_cols, num_owning_cols>(my_col, j + 1)) {
                    const uint32_t owner = conv2d_3x3::nslice_owner<k_cols>(my_col, j);
                    weights.wait_front(k_tiles * out_n_tiles);
#ifdef NSS_PROBE_NO_ACC_TOGGLE
                    // Measurement only: L1 accumulation set once (wrong sums on the first step).
                    if (j == first) {
                        pack_reconfig_l1_acc(1);
                    }
#else
                    pack_reconfig_l1_acc(j >= 1 ? 1 : 0);
#endif
                    const uint32_t slot_base = (k_cols - 1 - j) * slot_tiles;
                    for (uint32_t g = 0; g < walk_groups; ++g) {
                        const uint32_t r0 = rt_valid ? conv2d_3x3::ring_group_row0_rt<slice_sb_h>(valid_rows, owner, g)
                                                     : conv2d_3x3::ring_group_row0<block_m_tiles, slice_sb_h>(owner, g);
                        const uint32_t rows = valid_rows - r0 < slice_sb_h ? valid_rows - r0 : slice_sb_h;
                        const uint32_t lo = kh_lo[r0 / slice_sb_h];
                        const uint32_t hi = kh_hi[r0 / slice_sb_h];
                        const uint32_t group_w = (wide_short && rows == 1) ? out_n_tiles : slice_sb_w;
                        if (rows != init_rows || group_w != init_w) {
                            ckernel::matmul_block_init(cb_im2col_tiles, cb_weights, false, group_w, rows, k_tiles);
                            init_rows = rows;
                            init_w = group_w;
                        }
#ifdef NSS_PROBE_EXTRA_INIT
                        // Measurement only: one redundant init per group (same shape).
                        ckernel::matmul_block_init(cb_im2col_tiles, cb_weights, false, init_w, init_rows, k_tiles);
#endif
#ifdef RING_SB_UNITS
                        // nslice_step_shape: one ring unit (token, forward, arrival count) per matmul subblock of the
                        // group instead of per group, so a wide slice's hop stays one subblock's bytes.
                        constexpr bool sb_units = true;
#else
                        constexpr bool sb_units = false;
#endif
                        auto ring_unit_wait = [&]() {
                            ring_token.reserve_back(1);
                            if (j >= 1) {
                                // The unit's incoming partial landed: column my_col - 1 wrote this core's arrival
                                // count right behind it. The packer waits for it itself; the matmul runs ahead.
                                ++ring_need;
#ifndef NSS_PROBE_NO_ARRIVAL_WAIT
                                PACK(({
                                    volatile tt_l1_ptr uint32_t* arrived = ring_arrival_count();
                                    do {
                                        invalidate_l1_cache();
                                    } while (*arrived < ring_need);
                                }));
#endif
                            }
                        };
                        if constexpr (!sb_units) {
                            ring_unit_wait();
                        }
                        // The group's matmuls and packs. ct_group_rows: the group height is a template constant (one
                        // instance per height 1..slice_sb_h), so the r x 1 subblock's unpack / math loops are
                        // specialized (on the balanced split's ranges; with runtime dims a 1-row group costs nearly a
                        // full one: res5.2 @ LoFi, a 4 + 1-row walk 26.4 us vs 24.9 us with template heights).
                        auto group_mm = [&](auto rows_c, auto w_c) {
                            const uint32_t group_rows = rows_c();
                            constexpr uint32_t sb_w = decltype(w_c)::value;
                            for (uint32_t c0 = 0; c0 < out_n_tiles; c0 += sb_w) {
                                tile_regs_acquire();
                                uint32_t in1_index = c0 * in1_cstride;
#ifndef CONV2D_3X3_ABLATE_MATMUL
                                if constexpr (nslice_center) {
                                    // K order unchanged (kh, kw, c). One loop per source, each with constant CB ids (a
                                    // single loop choosing the CB per k measured res5.1 +5.6%).
                                    constexpr uint32_t ct = c_in_tiles;
                                    uint32_t in0_index = r0 * in0_k_tiles + lo * 2 * ct;
                                    uint32_t ctr_index = r0 * ct + lo * block_m_tiles * ct;
                                    in1_index += lo * 3 * ct * in1_kstep;
                                    UNPACK((set_matmul_in0_row_stride(in0_k_tiles)));
                                    for (uint32_t kh = lo; kh < hi; ++kh) {
                                        for (uint32_t c = 0; c < ct; ++c) {
                                            ckernel::matmul_block(
                                                cb_im2col_tiles,
                                                cb_weights,
                                                in0_index++,
                                                in1_index,
                                                0,
                                                false,
                                                sb_w,
                                                group_rows,
                                                in0_k_tiles);
                                            in1_index += in1_kstep;
                                        }
                                        UNPACK((set_matmul_in0_row_stride(ct)));
                                        for (uint32_t c = 0; c < ct; ++c) {
                                            ckernel::matmul_block(
                                                cb_center_tiles,
                                                cb_weights,
                                                ctr_index++,
                                                in1_index,
                                                0,
                                                false,
                                                sb_w,
                                                group_rows,
                                                ct);
                                            in1_index += in1_kstep;
                                        }
                                        ctr_index += (block_m_tiles - 1) * ct;
                                        UNPACK((set_matmul_in0_row_stride(in0_k_tiles)));
                                        for (uint32_t c = 0; c < ct; ++c) {
                                            ckernel::matmul_block(
                                                cb_im2col_tiles,
                                                cb_weights,
                                                in0_index++,
                                                in1_index,
                                                0,
                                                false,
                                                sb_w,
                                                group_rows,
                                                in0_k_tiles);
                                            in1_index += in1_kstep;
                                        }
                                    }
                                } else {
                                    // K order (kh, kw, c): kernel rows [lo, hi) are K tiles [lo * kkh, hi * kkh).
                                    static_assert(k_tiles % 3 == 0);
                                    constexpr uint32_t kkh = k_tiles / 3;
                                    const uint32_t k_lo = lo * kkh;
                                    const uint32_t k_hi = hi * kkh;
                                    uint32_t in0_index = r0 * k_tiles + k_lo;
                                    in1_index += k_lo * in1_kstep;
                                    for (uint32_t k = k_lo; k < k_hi; ++k) {
                                        ckernel::matmul_block(
                                            cb_im2col_tiles,
                                            cb_weights,
                                            in0_index,
                                            in1_index,
                                            0,
                                            false,
                                            sb_w,
                                            group_rows,
                                            k_tiles);
                                        ++in0_index;
                                        in1_index += in1_kstep;
                                    }
                                }
#endif
                                tile_regs_commit();
                                tile_regs_wait();
                                if constexpr (sb_units) {
                                    ring_unit_wait();
                                }
                                ckl::pack_subblock_row_strided(
                                    0,
                                    cb_combine_slots,
                                    slot_base + r0 * out_n_tiles + c0,
                                    out_n_tiles,
                                    group_rows,
                                    sb_w);
                                tile_regs_release();
                                if constexpr (sb_units) {
                                    ring_token.push_back(1);
                                }
                            }
                        };
                        using SbW = std::integral_constant<uint32_t, slice_sb_w>;
                        if constexpr (ct_group_rows) {
                            switch (rows) {
                                case 1:
#ifdef NSS_PROBE_NO_SHORT
                                    // Measurement only (wrong sums): the 1-row short group's matmul is skipped; its
                                    // DEST tiles are still packed (whatever DEST holds) and its ring unit still runs.
                                    if (true) {
                                        constexpr uint32_t w1 = wide_short ? out_n_tiles : slice_sb_w;
                                        for (uint32_t c0 = 0; c0 < out_n_tiles; c0 += w1) {
                                            tile_regs_acquire();
                                            tile_regs_commit();
                                            tile_regs_wait();
                                            if constexpr (sb_units) {
                                                ring_unit_wait();
                                            }
                                            ckl::pack_subblock_row_strided(
                                                0,
                                                cb_combine_slots,
                                                slot_base + r0 * out_n_tiles + c0,
                                                out_n_tiles,
                                                1,
                                                w1);
                                            tile_regs_release();
                                            if constexpr (sb_units) {
                                                ring_token.push_back(1);
                                            }
                                        }
                                        break;
                                    }
#endif
                                    if constexpr (wide_short) {
                                        group_mm(
                                            std::integral_constant<uint32_t, 1>{},
                                            std::integral_constant<uint32_t, out_n_tiles>{});
                                    } else {
                                        group_mm(std::integral_constant<uint32_t, 1>{}, SbW{});
                                    }
                                    break;
                                case 2: group_mm(std::integral_constant<uint32_t, 2>{}, SbW{}); break;
                                case 3: group_mm(std::integral_constant<uint32_t, 3>{}, SbW{}); break;
                                default: group_mm(std::integral_constant<uint32_t, 4>{}, SbW{}); break;
                            }
                        } else {
                            group_mm(RuntimeRows{rows}, SbW{});
                        }
                        if constexpr (!sb_units) {
                            ring_token.push_back(1);
                        }
                    }
                    weights.pop_front(k_tiles * out_n_tiles);
                }
                im2col_tiles.pop_front(prelude_rows * in0_k_tiles);
                if constexpr (nslice_center) {
                    center.pop_front(3 * block_m_tiles * c_in_tiles);
                }
#ifdef NSLICE_WEIGHT_COL_RUNS
                UNPACK((llk_unpack_AB_matmul_set_in1_column_stride(cb_weights, 1)));
#endif
                pack_reconfig_l1_acc(0);
                slots.push_back(k_cols * slot_tiles);
            }
            {
                MaybeDeviceZoneScope("compute_combine");
                if (owns_output_columns) {
                    if constexpr (apply_relu) {
                        pack_relu_config(ReluConfig::zero());
                    }
                    if constexpr (has_bias) {
                        {
                            // First use of the bias on this path (see add_bias).
                            MaybeDeviceZoneScope("compute_combine_bias_wait");
                            bias.wait_front(out_n_tiles);
                        }
                        ckl::add_bias_bcast_rows<ckl::BiasBroadcast::RowBroadcast, ckl::OutputCBLayout::TileRowMajor>(
                            slots, bias, output_tiles, combine_bias_shape);
                    } else {
                        reconfig_data_format_srca(cb_combine_slots);
                        pack_reconfig_data_format(cb_output_tiles);
                        copy_init(cb_combine_slots);
                        slots.wait_front(slot_tiles);
                        output_tiles.reserve_back(slot_tiles);
                        for (uint32_t t = 0; t < slot_tiles; ++t) {
                            tile_regs_acquire();
                            copy_tile(cb_combine_slots, t, 0);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_tile(0, cb_output_tiles);
                            tile_regs_release();
                        }
                        output_tiles.push_back(slot_tiles);
                        slots.pop_front(slot_tiles);
                    }
                    // The other slots carried partials that moved on.
                    slots.wait_front((k_cols - 1) * slot_tiles);
                    slots.pop_front((k_cols - 1) * slot_tiles);
                } else {
                    slots.wait_front(k_cols * slot_tiles);
                    slots.pop_front(k_cols * slot_tiles);
                }
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
        } else if constexpr (slab_direct) {
#ifdef CONV2D_3X3_SLAB_DIRECT
            // Row groups of in0_stream_rows tile-rows, the bias added once per bias_stream_groups groups (as the
            // face-direct stream below); every in0 tile unpacked from the slab (slab_direct:: above).
            MaybeDeviceZoneScope("compute_matmul_stream");
            using SdGeom = conv2d_3x3::slab_direct::Geom<
                SD_IN_H,
                SD_IN_W,
                SD_OUT_W,
                SD_STRIDE,
                SD_PAD,
                SD_DIL,
                SD_GAP,
                SD_PITCH,
                SD_PIXEL_BYTES,
                SD_HALF_BYTES>;
            constexpr uint32_t kkh = SD_KKH;
            constexpr uint32_t sb_h = out_subblock_h;
            constexpr uint32_t sb_w = out_subblock_w;
            // in0 is always the reused operand (the LLK's reuse_a order): a subblock taller than wide runs one row per
            // k-step call.
            constexpr uint32_t mm_rt = sb_w >= sb_h ? sb_h : 1;
#ifdef SD_FUSED_BIAS
            // The bias is one more K-step (conv2d_3x3_slab_direct.hpp): the matmul packs the final values.
            constexpr bool fused_bias = has_bias;
#else
            constexpr bool fused_bias = false;
#endif
            constexpr bool relu_on_mm_pack = apply_relu && (fused_bias || !has_bias);
            constexpr uint32_t target_cb = (has_bias && !fused_bias) ? cb_matmul_partials : cb_output_tiles;
            CircularBuffer target(target_cb);
            CircularBuffer slab(cb_halo_slab);
            const uint32_t done = m_block_idx * block_m_tiles;
            const uint32_t sd_m_tile_start = get_arg_val<uint32_t>(3);
            const uint32_t sd_actual = get_arg_val<uint32_t>(4);
            const uint32_t valid_m = sd_actual - done < block_m_tiles ? sd_actual - done : block_m_tiles;
            const uint32_t o_first = (sd_m_tile_start + done) * 32;
            [[maybe_unused]] const int32_t ih_lo = SdGeom::row_lo(o_first);
#ifdef SD_STREAM
            // Streamed fill (conv2d_3x3_slab_stream.hpp): the weights first, then each subblock row waits for the slab
            // pages its in0 runs read (the first wait is the startup one).
            constexpr bool sd_stream = true;
#else
            constexpr bool sd_stream = false;
            {
                MaybeDeviceZoneScope("compute_slab_wait");
                slab.wait_front(SD_SLAB_PX);
            }
#endif
#ifdef SD_BIAS_IN_WEIGHTS
            // The bias tiles ride at the tail of the weight round (the writer's SD_BIAS_IN_WEIGHTS).
            constexpr uint32_t round_tiles = (block_k_tiles + 1) * block_n_tiles;
            constexpr bool bias_in_weights = true;
#else
            constexpr uint32_t round_tiles = block_k_tiles * block_n_tiles;
            constexpr bool bias_in_weights = false;
#endif
            if constexpr (sd_stream) {
                MaybeDeviceZoneScope("compute_sd_weight_wait");
                weights.wait_front(round_tiles);
            } else {
                weights.wait_front(round_tiles);
            }
            [[maybe_unused]] uint32_t sd_waited = 0;
            [[maybe_unused]] uint32_t slab_l1 = 0, zero_l1 = 0, w_base = 0, w_tile = 0;
            UNPACK((slab_l1 = get_local_cb_interface(get_operand_id(cb_halo_slab)).fifo_rd_ptr << 4));
            UNPACK((zero_l1 = get_local_cb_interface(get_operand_id(cb_im2col_tiles)).fifo_rd_ptr << 4));
            UNPACK((w_base = get_local_cb_interface(get_operand_id(cb_weights)).fifo_rd_ptr - 1));
            UNPACK((w_tile = get_local_cb_interface(get_operand_id(cb_weights)).fifo_page_size));
            [[maybe_unused]] uint32_t ones_a[sb_h] = {}, b_base = 0;
            if constexpr (fused_bias) {
                UNPACK(({
                    const uint32_t ones =
                        zero_l1 + (get_local_cb_interface(get_operand_id(cb_im2col_tiles)).fifo_page_size << 4);
                    for (uint32_t t = 0; t < sb_h; ++t) {
                        ones_a[t] = (ones >> 4) - 1;
                    }
                }));
            }
            // Fused bias: the bias is waited for at its first K-step (the weight K-steps of the first subblock need
            // only the slab and the weights); it stays fronted until the kernel end.
            bool bias_waited = bias_in_weights;
            if constexpr (bias_in_weights) {
                UNPACK((b_base = w_base + w_tile * block_k_tiles * block_n_tiles));
            }
            bool need_init = true;
            for (uint32_t group = 0; group < in0_groups; ++group) {
                const uint32_t row0 = group * in0_stream_rows;
                const uint32_t group_end =
                    row0 + in0_stream_rows < block_m_tiles ? row0 + in0_stream_rows : block_m_tiles;
                if (need_init) {
                    reconfig_data_format(cb_weights, cb_im2col_tiles);
                    pack_reconfig_data_format(target_cb);
                    ckernel::matmul_block_init(cb_im2col_tiles, cb_weights, 0, sb_w, mm_rt, block_k_tiles);
                    UNPACK((slab_direct::unpack_init()));
                    MATH((slab_direct::math_patch_addrmods()));
                    pack_reconfig_l1_acc(0);
                    if constexpr (relu_on_mm_pack) {
                        pack_relu_config(ReluConfig::zero());
                    }
                    need_init = false;
                }
                for (uint32_t r = row0; r < group_end; r += sb_h) {
                    if constexpr (sd_stream) {
                        // The subblock row's last valid tile-row reads furthest into the slab.
                        UNPACK(({
                            const uint32_t last = r + sb_h - 1 < valid_m ? r + sb_h - 1 : valid_m - 1;
                            const uint32_t need = r < valid_m ? SdGeom::need_px<kkh>(ih_lo, o_first + last * 32) : 0;
                            if (need > sd_waited) {
                                if (sd_waited == 0) {
                                    MaybeDeviceZoneScope("compute_slab_wait");
                                    slab.wait_front(need);
                                } else {
                                    slab.wait_front(need);
                                }
                                sd_waited = need;
                            }
                        }));
                    }
                    target.reserve_back(sb_h * block_n_tiles);
                    // in0 tile of row r + t, K-tile k: its first run's L1 address (16 B units, minus 1).
                    [[maybe_unused]] uint32_t a[block_k_tiles][sb_h];
                    UNPACK(({
                        for (uint32_t t = 0; t < sb_h; ++t) {
                            const uint32_t o = o_first + (r + t) * 32;
                            for (uint32_t k = 0; k < block_k_tiles; ++k) {
                                const uint32_t addr =
                                    r + t < valid_m
                                        ? SdGeom::in0_addr(
                                              slab_l1, zero_l1, ih_lo, o, static_cast<int32_t>(k / kkh), k % kkh)
                                        : zero_l1;
                                a[k][t] = (addr >> 4) - 1;
                            }
                        }
                    }));
                    for (uint32_t c = 0; c < block_n_tiles; c += sb_w) {
                        tile_regs_acquire();
                        for (uint32_t t0 = 0; t0 < sb_h; t0 += mm_rt) {
                            for (uint32_t k = 0; k < block_k_tiles; ++k) {
                                UNPACK((slab_direct::unpack_k_step<sb_w, mm_rt>(
                                    &a[k][t0], w_base + w_tile * (k * block_n_tiles + c))));
                                MATH((slab_direct::math_k_step<cb_im2col_tiles, cb_weights, sb_w, mm_rt>(t0 * sb_w)));
                            }
                            if constexpr (fused_bias) {
                                if (!bias_waited) {
                                    MaybeDeviceZoneScope("compute_bias_wait");
                                    bias.wait_front(out_n_tiles);
                                    UNPACK((b_base = get_local_cb_interface(get_operand_id(cb_bias)).fifo_rd_ptr - 1));
                                    bias_waited = true;
                                }
                                UNPACK((slab_direct::unpack_k_step<sb_w, mm_rt>(ones_a, b_base + w_tile * c)));
                                MATH((slab_direct::math_k_step<cb_im2col_tiles, cb_weights, sb_w, mm_rt>(t0 * sb_w)));
                            }
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        ckl::pack_subblock_row_strided(0, target_cb, c, block_n_tiles, sb_h, sb_w);
                        tile_regs_release();
                    }
                    target.push_back(sb_h * block_n_tiles);
                }
                if constexpr (has_bias && !fused_bias) {
                    const uint32_t groups_done = group + 1;
                    if (groups_done % bias_stream_groups == 0 || groups_done == in0_groups) {
                        UNPACK((slab_direct::unpack_uninit()));
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
                        need_init = true;
                    }
                }
            }
            if (!need_init) {
                UNPACK((slab_direct::unpack_uninit()));
            }
            weights.pop_front(round_tiles);
            if constexpr (sd_stream) {
                // Every page pushed before the pop (the reader pushes the whole slab per block).
                slab.wait_front(SD_SLAB_PX);
            }
            slab.pop_front(SD_SLAB_PX);
#endif
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
#ifdef CONV2D_3X3_S2_TAPS
            {
                // Stride-2 taps (conv2d_3x3_s2_taps.hpp): every K-block's in0 ([bm, bk] per kernel row, K-block
                // major) built from the input ring in input-row order before the walk (behind the fill and the first
                // weight round); then the ring is popped. The walk's tilize hook does nothing. Raw LLK (capability),
                // see the header: no helper decimates rows.
                MaybeDeviceZoneScope("compute_tilize");
                static_assert(num_k_blocks == 3 && block_k_tiles == 3 * c_in_tiles);
                pack_reconfig_l1_acc(0);
                im2col_tiles_dfb.reserve_back(block_m_tiles * block_k_tiles * num_k_blocks);
                conv2d_3x3::s2_taps::
                    build<c_in_tiles, cb_input_tiles, cb_im2col_tiles, s2_out_w, s2_in_h, s2_in_w, s2_num_in_px>(
                        input_tiles_dfb,
                        s2_o_first,
                        s2_o_end,
                        [](uint32_t t, int32_t kh, uint32_t ch) {
                            return static_cast<uint32_t>(kh) * block_m_tiles * block_k_tiles + t * block_k_tiles + ch;
                        },
                        [](uint32_t, int32_t) { return false; });
                im2col_tiles_dfb.push_back(block_m_tiles * block_k_tiles * num_k_blocks);
                input_tiles_dfb.wait_front(slab_rows * c_in_tiles);
                input_tiles_dfb.pop_front(slab_rows * c_in_tiles);
            }
#endif
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

        if constexpr (has_bias && !k_split && in0_groups == 1 && !slab_direct) {
            add_bias(bias_shape);
        }
        if constexpr (apply_relu) {
            pack_relu_config(ReluConfig::none());
        }
    }

#ifdef SD_BIAS_IN_WEIGHTS
    constexpr bool bias_cb_used = false;  // the bias came with the weight rounds
#else
    constexpr bool bias_cb_used = has_bias;
#endif
    if constexpr (bias_cb_used) {
        // A core that owns no output columns never waited for it.
        bias.wait_front(out_n_tiles);
        bias.pop_front(out_n_tiles);
    }
#endif
}
