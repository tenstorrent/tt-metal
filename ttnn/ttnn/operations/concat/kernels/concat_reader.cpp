// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// concat reader: gather_block (op_design.md, Blocking Model). Fills cb_output_tiles with this cluster's output
// pages [start_tile, start_tile + num_tiles), in output page order, straight from the input pages: no staging
// buffer, no CPU copy, every input byte that lands in the output read once.
//
// Regime page_copy (every seam on a tile boundary): an output tile is exactly one input page. Implicit sync: one
// read tagged NocOptions::TXN_ID per ring entry, the DM0 interrupt handler posts the credits.
//
// Regime run_assembly (some seam inside a tile): an output tile is a set of byte runs from the input pages that
// cover its 32-wide window along `dim`. Explicit sync (the host opts the ring out of implicit sync): per block of
// block_tiles output pages, reserve_back, issue every run of every page as a sub-page read (trid 0) into its
// entry, one barrier on trid 0, push_back. The run generator below is the block operation's internal walk.
//
// Regime row_major_copy (ROW_MAJOR, dim=-2): an output stick is exactly one input stick. Implicit sync, as
// page_copy, with page = stick: the piece is the one whose row range holds the output row.
//
// Regime row_major_runs (ROW_MAJOR, dim=-1): output stick q is input stick q of every piece, side by side: one
// sub-page read per piece (ext_k elements from offset 0 into the entry at element offset off_k; Quasar's NoC moves
// any byte alignment). Explicit sync, as run_assembly: per block of block_tiles sticks, reserve, count reads per
// stick tagged trid 0, one barrier, push.
//
// No helper: the op has no compute phase and every step is a NoC read (op_design.md, API Mapping, "Helpers
// considered and rejected"). Reader threads: page_copy deals output tiles round-robin over R threads;
// run_assembly is built for the plan's R = 1 (a later R > 1 is a perf lamp of the design).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t TILE = 32;
constexpr uint32_t FACE = 16;
constexpr uint32_t REGIME_PAGE_COPY = 0;
constexpr uint32_t REGIME_RUN_ASSEMBLY = 1;
constexpr uint32_t REGIME_ROW_MAJOR_COPY = 2;
constexpr uint32_t REGIME_ROW_MAJOR_RUNS = 3;
constexpr uint32_t MAX_INPUTS = 5;  // in_0 .. in_4: TARGET's largest count

// The accessor of input K, or a placeholder when this program binds fewer than K + 1 inputs (never used then).
template <uint32_t K, uint32_t count>
auto piece_accessor() {
    if constexpr (K >= count) {
        return 0u;
    } else if constexpr (K == 0) {
        return TensorAccessor(*tensor::get_token_if_present<"in_0">());
    } else if constexpr (K == 1) {
        return TensorAccessor(*tensor::get_token_if_present<"in_1">());
    } else if constexpr (K == 2) {
        return TensorAccessor(*tensor::get_token_if_present<"in_2">());
    } else if constexpr (K == 3) {
        return TensorAccessor(*tensor::get_token_if_present<"in_3">());
    } else {
        return TensorAccessor(*tensor::get_token_if_present<"in_4">());
    }
}

// Byte offset of element (r, c) inside a tile of four 16x16 faces.
template <uint32_t elem_bytes>
constexpr uint32_t elem_off(uint32_t r, uint32_t c) {
    return (((r / FACE) * 2 + c / FACE) * FACE * FACE + (r % FACE) * FACE + c % FACE) * elem_bytes;
}

FORCE_INLINE uint32_t min3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t m = a < b ? a : b;
    return m < c ? m : c;
}

// The output tile cursor: (image, tile row, tile col) in output page order, advanced one page at a time.
struct OutTileCursor {
    uint32_t img, row, col, rows, cols;
    void seek(uint32_t page) {
        const uint32_t per_image = rows * cols;
        img = page / per_image;
        const uint32_t rest = page - img * per_image;
        row = rest / cols;
        col = rest - row * cols;
    }
    void advance() {
        if (++col == cols) {
            col = 0;
            if (++row == rows) {
                row = 0;
                ++img;
            }
        }
    }
};

}  // namespace

void kernel_main() {
    constexpr uint32_t count = get_arg(args::count);
    constexpr bool concat_is_w = get_arg(args::concat_is_w) != 0;
    constexpr uint32_t regime = get_arg(args::regime);
    constexpr uint32_t block_tiles = get_arg(args::block_tiles);
    constexpr uint32_t page_bytes = get_arg(args::page_bytes);  // one ring entry: a tile, or an aligned stick
    constexpr uint32_t elem_bytes = get_arg(args::elem_bytes);
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t face_row_bytes = FACE * elem_bytes;
    constexpr bool explicit_sync = regime == REGIME_RUN_ASSEMBLY || regime == REGIME_ROW_MAJOR_RUNS;
    static_assert(count >= 1 && count <= MAX_INPUTS, "concat binds in_0 .. in_{count-1}, count <= MAX_INPUTS");
    static_assert(!explicit_sync || R == 1, "the explicit-sync regimes are built for one reader thread");

    const uint32_t start_tile = get_arg(args::start_tile);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t cross_tiles = get_arg(args::cross_tiles);
    const uint32_t cross_extent = get_arg(args::cross_extent);
    const uint32_t concat_tiles_out = get_arg(args::concat_tiles_out);
    const auto piece_extent = get_arg(args::piece_extent);
    const auto piece_offset = get_arg(args::piece_offset);

    DataflowBuffer cb_output_tiles(dfb::cb_output_tiles);
    Noc noc;
    const auto acc_0 = piece_accessor<0, count>();
    const auto acc_1 = piece_accessor<1, count>();
    const auto acc_2 = piece_accessor<2, count>();
    const auto acc_3 = piece_accessor<3, count>();
    const auto acc_4 = piece_accessor<4, count>();
    // Calls f with the accessor of input k (k < count).
    auto with_piece = [&](uint32_t k, auto&& f) {
        if constexpr (count > 4) {
            if (k == 4) {
                f(acc_4);
                return;
            }
        }
        if constexpr (count > 3) {
            if (k == 3) {
                f(acc_3);
                return;
            }
        }
        if constexpr (count > 2) {
            if (k == 2) {
                f(acc_2);
                return;
            }
        }
        if constexpr (count > 1) {
            if (k == 1) {
                f(acc_1);
                return;
            }
        }
        f(acc_0);
    };

    // Source page of input k: tile j along `dim`, tile x across it, image img.
    auto src_page = [&](uint32_t k, uint32_t img, uint32_t j, uint32_t x) -> uint32_t {
        const uint32_t tiles_k = (piece_extent[k] + TILE - 1) / TILE;
        return img * cross_tiles * tiles_k + (concat_is_w ? x * tiles_k + j : j * cross_tiles + x);
    };

    OutTileCursor cur{0, 0, 0, concat_is_w ? cross_tiles : concat_tiles_out, concat_is_w ? concat_tiles_out : cross_tiles};

    if constexpr (regime == REGIME_PAGE_COPY) {
        // Every piece but the last is a whole number of tiles along `dim`, so piece k starts at tile
        // piece_offset[k] / 32 and output tile p is page p - that of the last piece starting at or before p.
        cur.seek(start_tile + get_my_thread_id());
        for (uint32_t i = get_my_thread_id(); i < num_tiles; i += R) {
            const uint32_t p = concat_is_w ? cur.col : cur.row;
            const uint32_t x = concat_is_w ? cur.row : cur.col;
            uint32_t k = count - 1;
            while (k > 0 && piece_offset[k] / TILE > p) {
                --k;
            }
            const uint32_t page = src_page(k, cur.img, p - piece_offset[k] / TILE, x);
            with_piece(k, [&](const auto& acc) {
                noc.async_read<NocOptions::TXN_ID>(acc, cb_output_tiles, {.page_id = page}, {});
            });
            for (uint32_t s = 0; s < R; ++s) {
                cur.advance();
            }
        }
        // No barrier: firmware credits a last partial batch and drains the ring when the kernel returns. Hand the
        // tag register back with the user id (it is sticky; a tt-metal program that runs next issues untagged reads).
        noc_async_read_set_trid(0, noc.get_noc_id());
    } else if constexpr (regime == REGIME_ROW_MAJOR_COPY) {
        // Output stick q is row h of image img of the output; the piece holding row h gives its stick.
        const uint32_t rows_out = piece_offset[count - 1] + piece_extent[count - 1];
        const uint32_t first = start_tile + get_my_thread_id();
        uint32_t img = first / rows_out;
        uint32_t h = first - img * rows_out;
        for (uint32_t i = get_my_thread_id(); i < num_tiles; i += R) {
            uint32_t k = count - 1;
            while (k > 0 && piece_offset[k] > h) {
                --k;
            }
            const uint32_t page = img * piece_extent[k] + (h - piece_offset[k]);
            with_piece(k, [&](const auto& acc) {
                noc.async_read<NocOptions::TXN_ID>(acc, cb_output_tiles, {.page_id = page}, {});
            });
            for (h += R; h >= rows_out; h -= rows_out) {
                ++img;
            }
        }
        // As page_copy: hand the sticky tag back.
        noc_async_read_set_trid(0, noc.get_noc_id());
    } else if constexpr (regime == REGIME_ROW_MAJOR_RUNS) {
        constexpr NocOptVals user_txn = {.trid = 0};
        for (uint32_t done = 0; done < num_tiles;) {
            const uint32_t block_extent = (num_tiles - done) < block_tiles ? (num_tiles - done) : block_tiles;
            cb_output_tiles.reserve_back(block_extent);
            for (uint32_t e = 0; e < block_extent; ++e) {
                const uint32_t q = start_tile + done + e;  // the same stick index in every piece
                for (uint32_t k = 0; k < count; ++k) {
                    with_piece(k, [&](const auto& acc) {
                        noc.async_read<NocOptions::TXN_ID>(
                            acc,
                            cb_output_tiles,
                            piece_extent[k] * elem_bytes,
                            {.page_id = q},
                            {.offset_bytes = e * page_bytes + piece_offset[k] * elem_bytes},
                            user_txn);
                    });
                }
            }
            noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
            cb_output_tiles.push_back(block_extent);
            done += block_extent;
        }
        // Every transfer carried the user id, so the tag register is already handed back.
    } else {
        constexpr NocOptVals user_txn = {.trid = 0};

        // One pending run, extended while the next run continues it in the same source page and the same entry.
        struct Run {
            uint32_t piece, page, src_off, dst_off, len;
        } pending{0, 0, 0, 0, 0};
        auto issue = [&](const Run& run) {
            with_piece(run.piece, [&](const auto& acc) {
                noc.async_read<NocOptions::TXN_ID>(
                    acc,
                    cb_output_tiles,
                    run.len,
                    {.page_id = run.page, .offset_bytes = run.src_off},
                    {.offset_bytes = run.dst_off},
                    user_txn);
            });
        };
        auto emit = [&](uint32_t k, uint32_t page, uint32_t src_off, uint32_t dst_off, uint32_t len) {
            if (pending.len != 0 && pending.piece == k && pending.page == page &&
                pending.src_off + pending.len == src_off && pending.dst_off + pending.len == dst_off) {
                pending.len += len;
                return;
            }
            if (pending.len != 0) {
                issue(pending);
            }
            pending = Run{k, page, src_off, dst_off, len};
        };

        // Walks the pieces in the window [32p, 32p + 32) of output tile p along `dim`, as segments on which neither
        // the destination coordinate u nor the source in-tile coordinate v crosses a multiple of 16 (so no run
        // leaves a face, and none leaves its source page). seg(k, page, u, v, n) is called per segment.
        auto for_each_segment = [&](uint32_t img, uint32_t p, uint32_t x, auto&& seg) {
            const uint32_t win_lo = p * TILE;
            const uint32_t win_hi = win_lo + TILE;
            for (uint32_t k = 0; k < count; ++k) {
                const uint32_t off = piece_offset[k];
                const uint32_t end = off + piece_extent[k];
                const uint32_t a = off > win_lo ? off : win_lo;
                const uint32_t b = end < win_hi ? end : win_hi;
                uint32_t n = 0;
                for (uint32_t g = a; g < b; g += n) {
                    const uint32_t u = g - win_lo;
                    const uint32_t l = g - off;
                    const uint32_t v = l % TILE;
                    n = min3(b - g, FACE - u % FACE, FACE - v % FACE);
                    seg(k, src_page(k, img, l / TILE, x), u, v, n);
                }
            }
        };

        cur.seek(start_tile);
        for (uint32_t done = 0; done < num_tiles;) {
            const uint32_t block_extent = (num_tiles - done) < block_tiles ? (num_tiles - done) : block_tiles;
            cb_output_tiles.reserve_back(block_extent);
            for (uint32_t e = 0; e < block_extent; ++e) {
                const uint32_t base = e * page_bytes;
                const uint32_t p = concat_is_w ? cur.col : cur.row;
                const uint32_t x = concat_is_w ? cur.row : cur.col;
                // Elements of the cross axis inside this tile; beyond them is tile padding, not read.
                const uint32_t cross_left = cross_extent - x * TILE;
                const uint32_t cross_valid = cross_left < TILE ? cross_left : TILE;
                if constexpr (concat_is_w) {
                    // Columns join: per row a run of n elements. Row half outermost, so whole faces merge.
                    for (uint32_t rh = 0; rh * FACE < cross_valid; ++rh) {
                        const uint32_t r_end = (rh + 1) * FACE < cross_valid ? (rh + 1) * FACE : cross_valid;
                        for_each_segment(cur.img, p, x, [&](uint32_t k, uint32_t page, uint32_t u, uint32_t v, uint32_t n) {
                            for (uint32_t r = rh * FACE; r < r_end; ++r) {
                                emit(k, page, elem_off<elem_bytes>(r, v), base + elem_off<elem_bytes>(r, u), n * elem_bytes);
                            }
                        });
                    }
                } else {
                    // Rows join: per column half a run of n face rows.
                    for_each_segment(cur.img, p, x, [&](uint32_t k, uint32_t page, uint32_t u, uint32_t v, uint32_t n) {
                        for (uint32_t h = 0; h * FACE < cross_valid; ++h) {
                            emit(k, page, elem_off<elem_bytes>(v, h * FACE), base + elem_off<elem_bytes>(u, h * FACE),
                                 n * face_row_bytes);
                        }
                    });
                }
                cur.advance();
            }
            if (pending.len != 0) {
                issue(pending);
                pending.len = 0;
            }
            noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
            cb_output_tiles.push_back(block_extent);
            done += block_extent;
        }
        // Every transfer carried the user id, so the tag register is already handed back.
    }
}
