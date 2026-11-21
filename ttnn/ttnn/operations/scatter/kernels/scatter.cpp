// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// scatter, regime resident_line_block (op_design.md): one data-movement kernel whose `threads` harts read,
// scatter and write. No compute kernel: every output element is a bitwise copy of an `input` or `src` element.
//
// Helper deviation (recorded as the design's API Mapping does): every block operation is raw NoC + scratchpad
// code. No kernel_lib helper moves an element to a row that is data (`index`): copy_tile_column copies with
// dst row = src row over a DFB, the eltwise helpers run on NEOs in tile order. The element-offset formula is the
// one tile_column_dataflow.hpp encodes for a 32x32 tile of four 16x16 faces.
//
// Schedule per cluster (all threads run it together; dealing is inside each block operation):
//   for each block (<= block_width_faces face-columns of one image, the whole output line):
//       load_line_block    input faces            -> out_column   (no barrier of its own)
//       for each chunk (<= block_src_tile_rows source tile-rows):
//           load_src_chunk src faces + index rows -> src_rows, index_rows
//           scatter_chunk  element moves stages   -> out_column, source rows dealt in contiguous ranges
//       store_line_block   out_column faces       -> output
//
// Cache / thread protocol (Quasar DM harts have write-back caches that NoC traffic does not see):
//   NoC reads land -> read barrier -> sync_threads -> scoped_lock acquire (L2 invalidate) -> sync_threads ->
//   stores -> scoped_lock release (L2 flush) -> sync_threads -> NoC writes -> write barrier -> sync_threads.
// Every thread calls every sync_threads, also with nothing dealt to it.
//
// Every NoC transfer carries the user transaction id 0, and the kernel hands trid 0 back as its last statements.

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace {

// --- Plan and blocking knobs (host: scatter_program_artifacts.py) -------------------------------------------
constexpr uint32_t kThreads = get_arg(args::threads);
constexpr uint32_t kSkipThread = get_arg(args::probe_skip_thread);  // == kThreads: no thread skips
constexpr uint32_t kElemBytes = get_arg(args::elem_bytes);
constexpr uint32_t kTileBytes = get_arg(args::tile_bytes);
constexpr uint32_t kOutRows = get_arg(args::out_rows);           // H, the real line length
constexpr uint32_t kSrcRows = get_arg(args::src_rows);           // Hs
constexpr uint32_t kWidth = get_arg(args::width);                // W
constexpr uint32_t kOutTileRows = get_arg(args::out_tile_rows);  // ceil(H / 32)
constexpr uint32_t kSrcTileRows = get_arg(args::src_tile_rows);  // ceil(Hs / 32)
constexpr uint32_t kTileCols = get_arg(args::tile_cols);         // ceil(W / 32)
constexpr uint32_t kWidthFaces = get_arg(args::width_faces);     // face-columns per image
constexpr uint32_t kBlockWidthFaces = get_arg(args::block_width_faces);
constexpr uint32_t kBlockSrcTileRows = get_arg(args::block_src_tile_rows);
constexpr uint32_t kSrcStageDepth = get_arg(args::src_stage_depth);
constexpr uint32_t kLine = get_arg(args::cache_line_bytes);

// --- Tile geometry --------------------------------------------------------------------------------------------
constexpr uint32_t kTileDim = 32;
constexpr uint32_t kFaceDim = 16;
constexpr uint32_t kFaceElems = kFaceDim * kFaceDim;
constexpr uint32_t kTileElems = kTileDim * kTileDim;
constexpr uint32_t kFaceBytes = kTileBytes / 4;
constexpr uint32_t kIndexElemBytes = 4;                         // INT32
constexpr uint32_t kIndexFaceBytes = kFaceDim * kIndexElemBytes;  // one face-column of one index row
constexpr uint32_t kIndexStageRowElems = kBlockWidthFaces * kFaceDim;
constexpr uint32_t kChunkRows = kBlockSrcTileRows * kTileDim;
constexpr uint32_t kNumChunks = (kSrcTileRows + kBlockSrcTileRows - 1) / kBlockSrcTileRows;

static_assert(kSrcStageDepth == 1, "src_stage_depth > 1 needs the two-slot stage ping-pong, not built yet");
static_assert(kElemBytes == 2 || kElemBytes == 4, "bfloat16 or float32 elements");
static_assert(kTileBytes == kTileElems * kElemBytes, "a tile page is 32x32 elements");
using Elem = std::conditional_t<kElemBytes == 2, uint16_t, uint32_t>;

constexpr NocOptVals kTxn = {.trid = 0};

// Element offset inside a 32x32 tile of four 16x16 faces.
inline uint32_t tile_elem(uint32_t r, uint32_t c) {
    return ((r / kFaceDim) * 2 + c / kFaceDim) * kFaceElems + (r % kFaceDim) * kFaceDim + c % kFaceDim;
}

// One block: face-columns [f0, f0 + nf) of image `image`, touching tile-columns [tc0, tc0 + ntc).
struct Block {
    uint32_t image;
    uint32_t f0;
    uint32_t nf;
    uint32_t tc0;
    uint32_t ntc;
};

// Byte offset of the first cache line that starts inside a scratchpad (the host gives one line of slack).
template <typename T>
inline uint32_t line_start(const Scratchpad<T>& pad) {
    const uint32_t addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&pad[0]));
    return ((addr + kLine - 1) / kLine) * kLine - addr;
}

// Move the faces of tile-column `tc` that lie in the block, between a tile page and a scratchpad tile slot:
// the whole tile when both of its face-columns are in the block, else faces {h, h + 2} of its half h (another
// cluster owns the other half: neither reads nor writes it here).
template <bool kRead, typename Acc, typename Pad>
inline void move_tile(
    const Noc& noc, const Acc& acc, const Pad& pad, uint32_t page_id, uint32_t pad_offset, const Block& b, uint32_t tc) {
    const uint32_t lo = (b.f0 > 2 * tc) ? b.f0 : 2 * tc;
    const uint32_t hi = (b.f0 + b.nf < 2 * tc + 2) ? b.f0 + b.nf : 2 * tc + 2;
    auto one = [&](uint32_t offset, uint32_t size) {
        if constexpr (kRead) {
            noc.async_read<NocOptions::TXN_ID>(
                acc, pad, size, {.page_id = page_id, .offset_bytes = offset}, {.offset_bytes = pad_offset + offset}, kTxn);
        } else {
            noc.async_write<NocOptions::TXN_ID>(
                pad, acc, size, {.offset_bytes = pad_offset + offset}, {.page_id = page_id, .offset_bytes = offset}, kTxn);
        }
    };
    if (hi - lo == 2) {
        one(0, kTileBytes);
    } else {
        const uint32_t half = lo % 2;
        one(half * kFaceBytes, kFaceBytes);
        one((half + 2) * kFaceBytes, kFaceBytes);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t unit_start = get_arg(args::unit_start);
    const uint32_t num_units = get_arg(args::num_units);
    const uint32_t tid = get_my_thread_id();

    Noc noc;
    const auto input_acc = TensorAccessor(tensor::input);
    const auto index_acc = TensorAccessor(tensor::index);
    const auto src_acc = TensorAccessor(tensor::src);
    const auto output_acc = TensorAccessor(tensor::output);

    Scratchpad<Elem> out_column(scratch::out_column);
    Scratchpad<Elem> src_rows(scratch::src_rows);
    Scratchpad<uint32_t> index_rows(scratch::index_rows);
    // Each region starts on a cache line of its own (bytes, then elements).
    const uint32_t slab0 = line_start(out_column);
    const uint32_t src0 = line_start(src_rows);
    const uint32_t idx0 = line_start(index_rows);
    const uint32_t slab0_e = slab0 / kElemBytes;
    const uint32_t src0_e = src0 / kElemBytes;
    const uint32_t idx0_e = idx0 / kIndexElemBytes;

    // ---- load_line_block: the block's faces of every output tile-row -> out_column. Tiles dealt round-robin.
    auto load_line_block = [&](const Block& b) {
        const uint32_t n = kOutTileRows * b.ntc;
        for (uint32_t k = tid; k < n; k += kThreads) {
            const uint32_t tr = k / b.ntc;
            const uint32_t tc = b.tc0 + k % b.ntc;
            const uint32_t page = (b.image * kOutTileRows + tr) * kTileCols + tc;
            move_tile<true>(noc, input_acc, out_column, page, slab0 + k * kTileBytes, b, tc);
        }
    };

    // ---- store_line_block: the same faces out_column -> output.
    auto store_line_block = [&](const Block& b) {
        const uint32_t n = kOutTileRows * b.ntc;
        for (uint32_t k = tid; k < n; k += kThreads) {
            const uint32_t tr = k / b.ntc;
            const uint32_t tc = b.tc0 + k % b.ntc;
            const uint32_t page = (b.image * kOutTileRows + tr) * kTileCols + tc;
            move_tile<false>(noc, output_acc, out_column, page, slab0 + k * kTileBytes, b, tc);
        }
    };

    // Source rows [row0, row0 + rows) of chunk `chunk`.
    auto chunk_row0 = [](uint32_t chunk) { return chunk * kChunkRows; };
    auto chunk_rows = [](uint32_t chunk) {
        const uint32_t row0 = chunk * kChunkRows;
        return (kSrcRows - row0 < kChunkRows) ? kSrcRows - row0 : kChunkRows;
    };

    // ---- load_src_chunk: the block's faces of the chunk's src tile-rows -> src_rows, and the block's columns
    // of the chunk's index rows -> index_rows. Tiles and rows dealt round-robin. No barrier here.
    auto load_src_chunk = [&](const Block& b, uint32_t chunk) {
        const uint32_t str0 = chunk * kBlockSrcTileRows;
        const uint32_t nstr = (kSrcTileRows - str0 < kBlockSrcTileRows) ? kSrcTileRows - str0 : kBlockSrcTileRows;
        const uint32_t n = nstr * b.ntc;
        for (uint32_t k = tid; k < n; k += kThreads) {
            const uint32_t tr = str0 + k / b.ntc;
            const uint32_t tc = b.tc0 + k % b.ntc;
            const uint32_t page = (b.image * kSrcTileRows + tr) * kTileCols + tc;
            move_tile<true>(noc, src_acc, src_rows, page, src0 + k * kTileBytes, b, tc);
        }
        const uint32_t row0 = chunk_row0(chunk);
        const uint32_t rows = chunk_rows(chunk);
        const uint32_t piece_bytes = b.nf * kIndexFaceBytes;
        for (uint32_t r = tid; r < rows; r += kThreads) {
            noc.async_read<NocOptions::TXN_ID>(
                index_acc,
                index_rows,
                piece_bytes,
                {.page_id = b.image * kSrcRows + row0 + r, .offset_bytes = b.f0 * kIndexFaceBytes},
                {.offset_bytes = idx0 + r * kIndexStageRowElems * kIndexElemBytes},
                kTxn);
        }
    };

    // ---- scatter_chunk: this thread's contiguous range of the chunk's source rows, ascending, columns
    // innermost. A position outside [0, H) (a negative INT32 reads as a large unsigned) is skipped (R6).
    auto scatter_chunk = [&](const Block& b, uint32_t chunk) {
        const uint32_t rows = chunk_rows(chunk);
        const uint32_t q = rows / kThreads;
        const uint32_t rem = rows % kThreads;
        const uint32_t my_rows = q + (tid < rem ? 1 : 0);
        const uint32_t my_first = tid * q + (tid < rem ? tid : rem);
        if (tid == kSkipThread) {
            return;  // probe: this thread's source rows are left undone
        }
        for (uint32_t r = my_first; r < my_first + my_rows; ++r) {
            const uint32_t src_tr = r / kTileDim;  // tile-row inside the chunk's stage
            const uint32_t src_r = r % kTileDim;
            const uint32_t idx_row = idx0_e + r * kIndexStageRowElems;
            for (uint32_t fl = 0; fl < b.nf; ++fl) {
                const uint32_t f = b.f0 + fl;
                const uint32_t tcl = f / 2 - b.tc0;
                const uint32_t c_base = (f % 2) * kFaceDim;  // column of the face inside its tile
                const uint32_t src_tile = src0_e + (src_tr * b.ntc + tcl) * kTileElems;
                const uint32_t col0 = f * kFaceDim;
                const uint32_t ncols = (kWidth - col0 < kFaceDim) ? kWidth - col0 : kFaceDim;
                for (uint32_t c = 0; c < ncols; ++c) {
                    const uint32_t p = index_rows[idx_row + fl * kFaceDim + c];
                    if (p < kOutRows) {
                        const uint32_t out = slab0_e + ((p / kTileDim) * b.ntc + tcl) * kTileElems +
                                             tile_elem(p % kTileDim, c_base + c);
                        out_column[out] = src_rows[src_tile + tile_elem(src_r, c_base + c)];
                    }
                }
            }
        }
    };

    // Blocks of this cluster's unit range: at most kBlockWidthFaces face-columns, never across an image.
    auto block_at = [&](uint32_t u, uint32_t end) {
        Block b;
        b.image = u / kWidthFaces;
        b.f0 = u % kWidthFaces;
        uint32_t nf = kBlockWidthFaces;
        if (end - u < nf) {
            nf = end - u;
        }
        if (kWidthFaces - b.f0 < nf) {
            nf = kWidthFaces - b.f0;
        }
        b.nf = nf;
        b.tc0 = b.f0 / 2;
        b.ntc = (b.f0 + nf - 1) / 2 - b.tc0 + 1;
        return b;
    };

    const uint32_t end = unit_start + num_units;
    if (num_units > 0) {
        load_src_chunk(block_at(unit_start, end), 0);
    }
    for (uint32_t u = unit_start; u < end;) {
        const Block b = block_at(u, end);
        const uint32_t slab_elems = kOutTileRows * b.ntc * kTileElems;
        const uint32_t stage_tile_elems = kBlockSrcTileRows * b.ntc * kTileElems;
        const uint32_t stage_index_elems = kChunkRows * kIndexStageRowElems;

        load_line_block(b);  // its reads fly with chunk 0's, issued before
        noc.async_read_barrier<NocOptions::TXN_ID>(kTxn);
        sync_threads();  // the slab and chunk 0 have landed, every thread's share
        {
            // L2 invalidate of the slab (NoC-written); released (flushed) after the last chunk's stores.
            auto slab_lock = out_column.scoped_lock(slab0_e, slab_elems);
            for (uint32_t chunk = 0; chunk < kNumChunks; ++chunk) {
                if (chunk > 0) {
                    load_src_chunk(b, chunk);
                    noc.async_read_barrier<NocOptions::TXN_ID>(kTxn);
                    sync_threads();  // the chunk has landed
                }
                {
                    auto src_lock = src_rows.scoped_lock(src0_e, stage_tile_elems);
                    auto idx_lock = index_rows.scoped_lock(idx0_e, stage_index_elems);
                    // An invalidate drops dirty lines: no thread stores before every thread is past its own.
                    sync_threads();
                    scatter_chunk(b, chunk);
                }  // stage locks released
                sync_threads();  // nobody reads the stages any more: they may be refilled
            }
        }  // slab lock released: every store reaches L1
        sync_threads();  // every thread's flush is done before any NoC write reads the slab
        store_line_block(b);
        const uint32_t next = u + b.nf;
        if (next < end) {
            load_src_chunk(block_at(next, end), 0);  // independent of the drain below
        }
        noc.async_write_barrier<NocOptions::TXN_ID>(kTxn);
        sync_threads();  // the slab may be refilled
        u = next;
    }

    // Hand the hart back with the user id in its tag registers.
    noc_async_read_set_trid(0, noc.get_noc_id());
    noc_async_write_set_trid(0, noc.get_noc_id());
}
