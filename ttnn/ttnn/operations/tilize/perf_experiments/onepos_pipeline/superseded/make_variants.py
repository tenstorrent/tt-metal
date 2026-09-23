"""Perf 2 onepos_pipeline: generate kernels_op/ from the op's CURRENT kernels plus the sub-block pipeline.

The generated kernels compile to the op's kernels unless the host wrapper (the harness
tests/ttnn/unit_tests/operations/tilize/test_tilize_perf2_onepos_pipeline.py) adds the defines:
  ONEPOS_NSB=<n>         compute + writer (+ reader): cut a one-position tile-row into ~n column
                         sub-blocks (onepos_sub_blocks.hpp); compute pushes each sub-block's tiles,
                         the writer writes each sub-block as soon as it is packed.
  ONEPOS_SPLIT_READS=1   reader: on a one-position streamed stick walk, read the tile-row as n
                         column parts (one transaction id each) and push each part's pages once it
                         lands, so compute starts on part 0 while the others are in flight.
  ONEPOS_NO_ROTATE       sub-blocks in column order, writes rotated inside each sub-block (first variant);
                         default: production order starts at the writer's rotated first tile.
  compute RT arg 2       (appended by the wrapper) stick_rotation, for that start sub-block.
  ONEPOS_NO_COMPUTE / ONEPOS_NO_WRITER   ablations: keep the other side's sub-block path only.
kernels_op/ is git-ignored: regenerate with `python3 make_variants.py` (host-only script).
"""
import os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "../../kernels")
DST = os.path.join(HERE, "kernels_op")


def once(s, needle, repl):
    assert s.count(needle) == 1, needle
    return s.replace(needle, repl)


COMPUTE_FNS = r"""
#if defined(ONEPOS_NSB) && defined(ARCH_WORMHOLE)  // WH LLK signatures (BH fast tilize differs: port needed)
#include "onepos_sub_blocks.hpp"
namespace onepos {

// RAW LLK (helper bypass, measured justification in perf_experiments/onepos_pipeline/README.md):
// compute_kernel_lib::tilize<block_width> (and the compute API fast_tilize_block / tilize_block
// under it) tie the unpacker's row stride to the width being tilized (full_dim == block). A
// resident input shard's tile-row is block_width tiles wide, so tilizing a COLUMN SUB-BLOCK of it
// needs the stride of the whole row with the width of the sub-block. The WH unpack LLK already takes
// them separately (llk_unpack_fast_tilize_block(icb, tile_index, unit_dim, num_units, full_dim):
// tile_index is a column offset inside the row, full_dim the row stride; llk_unpack_tilize(icb,
// tile_index, block_ct_dim) likewise). These are the WH branches of fast_tilize_block / tilize_block
// (tt_metal/hw/inc/api/compute/tilize.h) with full_dim decoupled from the sub-block width.
ALWI void fast_tilize_cols(uint32_t icb, uint32_t block, uint32_t full_dim, uint32_t ocb, uint32_t in_col) {
    uint32_t packed_tiles = 0;
    uint32_t remaining_tiles = block;
    constexpr uint32_t dest_size = DST_ACCUM_MODE ? 4 : 8;
    const uint32_t unit_dim = full_dim == 1 ? 1 : 2;  // what fast_tilize_init(icb, full_dim, ocb) programmed
    uint32_t num_units = dest_size / unit_dim;
    while (packed_tiles < block) {
        const uint32_t read_tile_index = in_col + packed_tiles;
        const uint32_t write_tile_index = packed_tiles;
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));
        if (remaining_tiles > 2 * dest_size) {
            UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
            MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
            PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));
            packed_tiles += dest_size;
            remaining_tiles -= dest_size;
        } else if (remaining_tiles > dest_size) {
            const uint32_t even_remainder = remaining_tiles / 2 + ((remaining_tiles / 2) % 2);
            num_units = even_remainder / unit_dim;
            UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
            MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
            PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));
            packed_tiles += even_remainder;
            remaining_tiles -= even_remainder;
        } else {
            if (remaining_tiles % 2 == 0 || unit_dim == 1) {
                num_units = remaining_tiles / unit_dim;
                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
                MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
                PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));
            } else if (remaining_tiles == 3) {
                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, 3, 1, full_dim)));
                MATH((llk_math_fast_tilize_block_(0, icb, 3, 1)));
                PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, 3, 1)));
            } else {
                num_units = (remaining_tiles - 3) / unit_dim;
                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
                MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
                PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));
                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index + remaining_tiles - 3, 3, 1, full_dim)));
                MATH((llk_math_fast_tilize_block_(remaining_tiles - 3, icb, 3, 1)));
                PACK((llk_pack_fast_tilize_block(remaining_tiles - 3, ocb, write_tile_index + remaining_tiles - 3, 3, 1)));
            }
            packed_tiles += remaining_tiles;
            remaining_tiles = 0;
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

// The slow (lossless / non-fast-format) tilize of a column sub-block: tilize_block's WH body with
// llk_unpack_tilize(icb, column, full_dim) per tile instead of llk_unpack_tilize_block(icb, block).
ALWI void slow_tilize_cols(uint32_t icb, uint32_t block, uint32_t full_dim, uint32_t ocb, uint32_t in_col) {
    for (uint32_t t = 0; t < block; ++t) {
        UNPACK((llk_unpack_tilize(icb, in_col + t, full_dim)));
    }
    for (uint32_t t = 0; t < block; ++t) {
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));
        MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
            0, icb)));
        PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(0, ocb, t)));
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

// One tile-row (block_width pages in cb_in) tilized as SubBlocks<block_width, nsb> column
// sub-blocks: wait for the sub-block's pages (cumulative), pack it, push its tiles. Same init /
// reconfig / uninit as the helper's InitAndUninit + UnpackAndPackReconfigure call, same fast-vs-slow
// selection (can_use_fast_tilize && !lossless fp32).
template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out, compute_kernel_lib::tilize_config::Fp32Mode fp32_mode, uint32_t nsb>
ALWI void tilize_row_sub_blocked(uint32_t j0) {
    using SB = SubBlocks<block_width, nsb>;
    constexpr bool lossless = fp32_mode == compute_kernel_lib::tilize_config::Fp32Mode::Lossless &&
                              compute_kernel_lib::is_fp32_input_format<cb_in>();
    constexpr bool use_fast = compute_kernel_lib::can_use_fast_tilize<block_width, cb_in, cb_out>() && !lossless;
    reconfig_data_format_srca(cb_in);
#ifdef ARCH_WORMHOLE
    if constexpr (use_fast) {
        reconfig_data_format_srcb(cb_in);
    }
#endif
    pack_reconfig_data_format(cb_out);
    if constexpr (use_fast) {
        fast_tilize_init(cb_in, block_width, cb_out);
    } else {
        tilize_init(cb_in, block_width, cb_out);
    }
    uint32_t published = 0;  // input pages published through this production step
    for (uint32_t p = 0; p < SB::n; ++p) {
        const uint32_t k = SB::at(j0, p);
        const uint32_t first = SB::first(k);
        const uint32_t w = SB::width(k);
        published += w;
        cb_wait_front(cb_in, published);
        cb_reserve_back(cb_out, w);
        if constexpr (use_fast) {
            fast_tilize_cols(cb_in, w, block_width, cb_out, first);
        } else {
            slow_tilize_cols(cb_in, w, block_width, cb_out, first);
        }
        cb_push_back(cb_out, w);
    }
    cb_pop_front(cb_in, block_width);
    if constexpr (use_fast) {
        fast_tilize_uninit(cb_in, cb_out, block_width);
    } else {
        tilize_uninit(cb_in, cb_out);
    }
}

}  // namespace onepos
#endif  // ONEPOS_NSB

void kernel_main() {"""

COMPUTE_CALL = """#if defined(ONEPOS_NSB) && defined(ARCH_WORMHOLE) && !defined(ONEPOS_NO_COMPUTE)
    if constexpr (!split_reader && onepos::SubBlocks<block_width, ONEPOS_NSB>::n > 1) {
        if (num_blocks == 1) {  // one-position walk: tilize the tile-row in column sub-blocks
            // RT arg 2 (onepos only): stick_rotation, so compute derives the writer's first sub-block.
            const uint32_t valid_width = core_col_tiles < block_width ? core_col_tiles : block_width;
            onepos::tilize_row_sub_blocked<block_width, cb_input_sticks, cb_output_tiles, fp32_mode, ONEPOS_NSB>(
                onepos::SubBlocks<block_width, ONEPOS_NSB>::start(get_arg_val<uint32_t>(2), valid_width));
            return;
        }
    }
#endif
    MaybeDeviceZoneScope("compute_tilize");"""

WRITER_CALL = """// ARCH_WORMHOLE: the compute side (raw WH LLK) only sub-blocks on WH, and the production order
// (rotated start sub-block) must agree between compute and this RISC-V.
#if defined(ONEPOS_NSB) && defined(ARCH_WORMHOLE) && !defined(ONEPOS_NO_WRITER)
    if constexpr (!output_resident && !split_reader && onepos::SubBlocks<block_width, ONEPOS_NSB>::n > 1) {
        tilize_dataflow::Walker<block_width> w(row_start, core_row_tiles, col_start, core_col_tiles, row_rotation);
        if (w.num_positions() == 1) {
            // One-position walk: write each column sub-block as soon as compute has packed it.
            using SB = onepos::SubBlocks<block_width, ONEPOS_NSB>;
            const auto out_acc = TensorAccessor(output_args, dst_addr, out_tile_bytes);
            const uint32_t row_tile_idx = w.row() * tiles_per_row + w.first_col();
            const uint32_t valid_width = w.valid_width();
            const uint32_t base = get_read_ptr(cb_output_tiles);
            // The op's store_rows order (t0, t0 + 1, ..., wrap) regrouped by sub-block: sub-block j0
            // (holding t0) is produced first and its columns [t0, end) written at once; the other
            // sub-blocks follow in column order (wrapping); j0's columns [first, t0) go last.
            const uint32_t t0 = stick_rotation % valid_width;
            const uint32_t j0 = SB::start(stick_rotation, valid_width);
            auto write_cols = [&](uint32_t cb_off, uint32_t first, uint32_t lo, uint32_t hi) {
                for (uint32_t c = lo; c < hi; ++c) {
                    noc_async_write(
                        base + (cb_off + c - first) * out_tile_bytes,
                        out_acc.get_noc_addr(row_tile_idx + c),
                        out_tile_bytes);
                }
            };
            uint32_t cb_off = 0;  // CB page of the current sub-block's first tile (production order)
            for (uint32_t p = 0; p < SB::n; ++p) {
                const uint32_t k = SB::at(j0, p);
                const uint32_t first = SB::first(k);
                const uint32_t end = first + SB::width(k);
                if (p == 0) {
                    MaybeDeviceZoneScope("writer_wait");  // starved on compute: the first sub-block
                    cb_wait_front(cb_output_tiles, cb_off + SB::width(k));
                } else {
                    MaybeDeviceZoneScope("writer_wait_next");  // starved on compute: later sub-blocks
                    cb_wait_front(cb_output_tiles, cb_off + SB::width(k));
                }
                const uint32_t stop = end < valid_width ? end : valid_width;
#ifdef ONEPOS_NO_ROTATE
                if (first < stop) {
                    const uint32_t count = stop - first;  // per-sub-block rotation (the first variant)
                    uint32_t t = stick_rotation % count;
                    for (uint32_t i = 0; i < count; ++i) {
                        write_cols(cb_off, first, first + t, first + t + 1);
                        if (++t == count) {
                            t = 0;
                        }
                    }
                }
#else
                write_cols(cb_off, first, p == 0 ? t0 : first, stop);
#endif
                cb_off += SB::width(k);
            }
#ifndef ONEPOS_NO_ROTATE
            write_cols(0, SB::first(j0), SB::first(j0), t0);
#endif
            noc_async_writes_flushed();
            cb_pop_front(cb_output_tiles, block_width);
            noc_async_write_barrier();
            return;
        }
    }
#endif
    if constexpr (output_resident) {"""

READER_CALL = """    const auto input_accessor = TensorAccessor(input_args, src_addr, stick_page_bytes);
#if defined(ONEPOS_NSB) && defined(ONEPOS_SPLIT_READS) && defined(ARCH_WORMHOLE)
    if constexpr (
        coalesce_depth == 0 && !padded && !split_reader && co_read == 0 && pages_per_stick == 1 &&
        onepos::SubBlocks<block_width, ONEPOS_NSB>::n > 1) {
        tilize_dataflow::Walker<block_width> w(row_start, core_row_tiles, col_start, core_col_tiles, row_rotation);
        if (w.num_positions() == 1) {
            // One-position walk: read the tile-row as column parts (one transaction id each), and
            // publish each part's pages as soon as it has landed.
            using SB = onepos::SubBlocks<block_width, ONEPOS_NSB>;
            static_assert(SB::n <= 14, "one transaction id per column part");
            constexpr uint32_t block_stick_bytes = block_width * tile_col_bytes;
            cb_reserve_back(cb_input_sticks, block_width);
            const uint32_t base = get_write_ptr(cb_input_sticks);
            const uint32_t first_stick = w.row() * tile_h;
            const uint32_t valid_width = w.valid_width();
            const uint32_t j0 = SB::start(stick_rotation, valid_width);
            for (uint32_t p = 0; p < SB::n; ++p) {
                const uint32_t k = SB::at(j0, p);
                const uint32_t first = SB::first(k);
                const uint32_t end = first + SB::width(k);
                const uint32_t stop = end < valid_width ? end : valid_width;
                if (first < stop) {
                    noc_async_read_set_trid(1 + p);
                    tilize_dataflow::read_tile_row_sticks<tile_h, block_stick_bytes, page_bytes, pages_per_stick>(
                        input_accessor,
                        first_stick,
                        base + first * tile_col_bytes,
                        (w.first_col() + first) * tile_col_bytes,
                        (stop - first) * tile_col_bytes,
                        stick_rotation & (tile_h - 1),
                        0,
                        tile_h);
                }
            }
            for (uint32_t p = 0; p < SB::n; ++p) {
                const uint32_t k = SB::at(j0, p);
                if (SB::first(k) < valid_width) {
                    noc_async_read_barrier_with_trid(1 + p);
                }
                cb_push_back(cb_input_sticks, SB::width(k));
            }
            noc_async_read_set_trid(0);
            return;
        }
    }
#endif"""


def gen():
    shutil.rmtree(DST, ignore_errors=True)
    shutil.copytree(SRC, DST)
    shutil.copy(os.path.join(HERE, "onepos_sub_blocks.hpp"), DST)
    p = os.path.join(DST, "tilize_compute.cpp")
    c = open(p).read()
    c = once(c, "void kernel_main() {", COMPUTE_FNS)
    c = once(c, '    MaybeDeviceZoneScope("compute_tilize");', COMPUTE_CALL)
    open(p, "w").write(c)
    p = os.path.join(DST, "tilize_writer.cpp")
    w = open(p).read()
    w = once(
        w, '#include "tilize_stick_reads.hpp"', '#include "tilize_stick_reads.hpp"\n#include "onepos_sub_blocks.hpp"'
    )
    w = once(w, "    if constexpr (output_resident) {", WRITER_CALL)
    open(p, "w").write(w)
    p = os.path.join(DST, "tilize_reader.cpp")
    r = open(p).read()
    r = once(
        r, '#include "tilize_stick_reads.hpp"', '#include "tilize_stick_reads.hpp"\n#include "onepos_sub_blocks.hpp"'
    )
    r = once(r, "    const auto input_accessor = TensorAccessor(input_args, src_addr, stick_page_bytes);", READER_CALL)
    open(p, "w").write(r)
    print("wrote", DST)


if __name__ == "__main__":
    gen()
