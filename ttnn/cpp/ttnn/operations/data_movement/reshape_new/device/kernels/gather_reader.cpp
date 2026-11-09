// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gather reader: for each output unit in [unit_begin, unit_end), walks the input element runs that make it up and
// stages them, with their placements, into stage DFB entries (layout in gather_common.hpp).
//
// Both tensors are viewed as batch x height x width logical elements (dims -2 and -1; everything before folded
// into the batch). An output element at flat logical index f comes from the input element at the same f.
//   ROW_MAJOR: unit u is chunk (u % chunks_per_page) of output page (u / chunks_per_page), or, when
//              pages_per_unit > 1, output pages [u * pages_per_unit, (u + 1) * pages_per_unit); a page holds
//              page_width elements of one row.
//   TILE:      unit u is output tile u; tiles are stored face by face, face rows contiguous.
// A run never crosses an input row, an input page or face row, or an output face row, so each run is one
// contiguous source range and one contiguous placement. Runs of one input page share one read while each starts
// inside the bytes read so far or at most kMaxReadGapBytes past them (a tile's face rows interleave two faces, so
// one read usually covers a whole input tile); runs that also continue each other in the unit share one record.
//
// A read starts at its source address rounded down to source_alignment and lands on a source_alignment boundary
// of L1: a DRAM read must keep its source's offset within that alignment (64 bytes on Blackhole). The record
// points past the lead-in bytes.
//
// Entries alternate between the two stage buffers, each with its own transaction id, so the reads of one entry
// are in flight while the previous entry's are waited for.
//
// Compile-time args: see ReshapeNewGatherProgramFactory
// Runtime args:      unit_begin, unit_end
// Bindings:          dfb::stage0, dfb::stage1 (producer), tensor::src

#include <stdint.h>

#include "api/alignment.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/reshape_new/device/kernels/gather_common.hpp"

namespace {

using namespace reshape_new;

constexpr bool tiled = get_arg(args::tiled) != 0;
constexpr uint32_t element_size = get_arg(args::element_size);
constexpr uint32_t page_bytes = get_arg(args::page_bytes);
constexpr uint32_t unit_bytes = get_arg(args::unit_bytes);
constexpr uint32_t chunks_per_page = get_arg(args::chunks_per_page);
constexpr uint32_t pages_per_unit = get_arg(args::pages_per_unit);
constexpr uint32_t page_slot_bytes = get_arg(args::page_slot_bytes);
constexpr uint32_t in_width = get_arg(args::in_width);
constexpr uint32_t in_height = get_arg(args::in_height);
constexpr uint32_t in_padded_height = get_arg(args::in_padded_height);
constexpr uint32_t in_page_width = get_arg(args::in_page_width);
constexpr uint32_t in_pages_per_row = get_arg(args::in_pages_per_row);
constexpr uint32_t out_width = get_arg(args::out_width);
constexpr uint32_t out_height = get_arg(args::out_height);
constexpr uint32_t out_padded_height = get_arg(args::out_padded_height);
constexpr uint32_t out_page_width = get_arg(args::out_page_width);
constexpr uint32_t out_pages_per_row = get_arg(args::out_pages_per_row);
constexpr uint32_t out_num_pages = get_arg(args::out_num_pages);
constexpr uint32_t tile_height = get_arg(args::tile_height);
constexpr uint32_t tile_width = get_arg(args::tile_width);
constexpr uint32_t face_height = get_arg(args::face_height);
constexpr uint32_t face_width = get_arg(args::face_width);
constexpr uint32_t source_alignment = get_arg(args::source_alignment);
constexpr uint32_t records_per_entry = get_arg(args::records_per_entry);
constexpr uint32_t header_bytes = get_arg(args::header_bytes);
constexpr uint32_t stage_entry_bytes = get_arg(args::stage_entry_bytes);

static_assert(is_power_of_2(source_alignment), "source_alignment must be a power of two");

constexpr uint32_t faces_per_tile_row = tile_width / face_width;
constexpr uint32_t face_elements = face_height * face_width;
constexpr uint32_t in_tile_rows = in_padded_height / tile_height;
constexpr uint32_t out_tile_rows = out_padded_height / tile_height;

// Transaction ids of the two stage buffers.
constexpr uint32_t kStageTrids[2] = {1, 2};

// Unneeded bytes a read may take in to cover a later run of its page: cheaper than another NOC transaction.
constexpr uint32_t kMaxReadGapBytes = 1024;

FORCE_INLINE uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Byte offset of element (row, col) of a tile.
FORCE_INLINE uint32_t tile_offset(uint32_t row, uint32_t col) {
    const uint32_t face = (row / face_height) * faces_per_tile_row + col / face_width;
    return (face * face_elements + (row % face_height) * face_width + col % face_width) * element_size;
}

// NOC address of a byte of an input page. The page's base is looked up once per page change, so consecutive runs
// of one page skip the accessor's bank arithmetic.
template <typename Accessor>
class PageAddresses {
public:
    explicit PageAddresses(const Accessor& src) : src_(src) {}

    FORCE_INLINE uint64_t get(uint32_t page, uint32_t offset) {
        if (page != page_) {
            page_ = page;
            base_ = src_.get_noc_addr(page);
        }
        return base_ + offset;
    }

private:
    const Accessor& src_;
    uint32_t page_ = 0xFFFFFFFFu;
    uint64_t base_ = 0;
};

// Input position of a flat logical element index, advanced run by run.
struct InputCursor {
    uint32_t batch;
    uint32_t row;
    uint32_t col;

    FORCE_INLINE void seek(uint32_t flat) {
        const uint32_t per_batch = in_height * in_width;
        batch = flat / per_batch;
        const uint32_t rest = flat % per_batch;
        row = rest / in_width;
        col = rest % in_width;
    }

    FORCE_INLINE void advance(uint32_t elements) {
        col += elements;
        if (col == in_width) {
            col = 0;
            if (++row == in_height) {
                row = 0;
                ++batch;
            }
        }
    }
};

// Fills stage entries for one unit at a time. Entries are opened lazily and closed when full or at the end of the
// unit. A closed entry is pushed once the next one is closed (or by finish()), when its reads are waited for.
template <typename Stage0, typename Stage1, typename Pages>
class Stager {
public:
    Stager(Stage0& stage0, Stage1& stage1, Pages& pages, Noc& noc) :
        stage0_(stage0), stage1_(stage1), pages_(pages), noc_(noc) {}

    FORCE_INLINE void begin_unit(uint32_t page, uint32_t page_offset, uint32_t bytes, uint32_t pages, bool padded) {
        page_ = page;
        page_offset_ = page_offset;
        unit_bytes_ = bytes;
        pages_count_ = pages;
        flags_ = kFlagFirst | (padded ? kFlagPadded : 0);
        open();
    }

    // Stages `bytes` bytes at `offset` of input page `page` for placement at `unit_offset` of the unit's slot.
    FORCE_INLINE void add(uint32_t page, uint32_t offset, uint32_t unit_offset, uint32_t bytes) {
        const uint32_t end = offset + bytes;
        const bool covered = read_open_ && page == read_page_ && offset >= read_begin_ &&
                             offset <= read_end_ + kMaxReadGapBytes &&
                             read_dst_ + read_lead_ + ((end > read_end_ ? end : read_end_) - read_begin_) <= entry_end_;
        const uint32_t staged_in_read = read_dst_ + read_lead_ + (offset - read_begin_) - entry_;
        const bool extends_record =
            covered && count_ != 0 && last_unit_end_ == unit_offset && last_staged_end_ == staged_in_read;
        if (!covered || (!extends_record && count_ == records_per_entry)) {
            flush_read();
            const uint32_t lead = static_cast<uint32_t>(pages_.get(page, offset)) & (source_alignment - 1);
            if (count_ == records_per_entry || cursor_ + lead + bytes > entry_end_) {
                close(false);
                flags_ &= ~kFlagFirst;
                open();
            }
            read_open_ = true;
            read_page_ = page;
            read_begin_ = offset;
            read_end_ = end;
            read_lead_ = lead;
            read_dst_ = cursor_;
        } else if (end > read_end_) {
            read_end_ = end;
        }
        const uint32_t staged = read_dst_ + read_lead_ + (offset - read_begin_) - entry_;
        if (count_ != 0 && last_unit_end_ == unit_offset && last_staged_end_ == staged) {
            record(count_ - 1)[2] += bytes;
        } else {
            volatile tt_l1_ptr uint32_t* rec = record(count_);
            rec[0] = unit_offset;
            rec[1] = staged;
            rec[2] = bytes;
            ++count_;
        }
        last_unit_end_ = unit_offset + bytes;
        last_staged_end_ = staged + bytes;
    }

    FORCE_INLINE void end_unit() { close(true); }

    // Hands over the last closed entry.
    FORCE_INLINE void finish() {
        if (pending_) {
            wait_and_push(current_ ^ 1);
            pending_ = false;
        }
    }

private:
    FORCE_INLINE volatile tt_l1_ptr uint32_t* record(uint32_t r) const {
        return header_ + kHeaderWords + kRecordWords * r;
    }

    FORCE_INLINE void open() {
        if (current_ == 0) {
            stage0_.reserve_back(1);
            entry_ = stage0_.get_write_ptr();
        } else {
            stage1_.reserve_back(1);
            entry_ = stage1_.get_write_ptr();
        }
        header_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(entry_));
        entry_end_ = entry_ + stage_entry_bytes;
        cursor_ = align_power_of_2(entry_ + header_bytes, source_alignment);
        count_ = 0;
        read_open_ = false;
    }

    FORCE_INLINE void flush_read() {
        if (!read_open_) {
            return;
        }
        const uint32_t bytes = read_lead_ + (read_end_ - read_begin_);
        const uint64_t src_noc = pages_.get(read_page_, read_begin_) - read_lead_;
        const NocOptVals opts{.trid = kStageTrids[current_]};
        if (current_ == 0) {
            noc_.async_read<NocOptions::TXN_ID>(
                PrecomposedUnicastEndpoint{},
                stage0_,
                bytes,
                {.noc_addr = src_noc},
                {.offset_bytes = read_dst_ - entry_},
                opts);
        } else {
            noc_.async_read<NocOptions::TXN_ID>(
                PrecomposedUnicastEndpoint{},
                stage1_,
                bytes,
                {.noc_addr = src_noc},
                {.offset_bytes = read_dst_ - entry_},
                opts);
        }
        cursor_ = align_power_of_2(read_dst_ + bytes, source_alignment);
        read_open_ = false;
    }

    FORCE_INLINE void wait_and_push(uint32_t stage) {
        noc_.async_read_barrier<NocOptions::TXN_ID>({.trid = kStageTrids[stage]});
        if (stage == 0) {
            stage0_.push_back(1);
        } else {
            stage1_.push_back(1);
        }
    }

    FORCE_INLINE void close(bool last) {
        flush_read();
        header_[kHeaderPage] = page_;
        header_[kHeaderPageOffset] = page_offset_;
        header_[kHeaderUnitBytes] = unit_bytes_;
        header_[kHeaderFlags] = flags_ | (last ? kFlagLast : 0);
        header_[kHeaderCount] = count_;
        header_[kHeaderPageCount] = pages_count_;
        // The previous entry's reads were issued before this entry's: wait for them only, then hand it over.
        if (pending_) {
            wait_and_push(current_ ^ 1);
        }
        pending_ = true;
        current_ ^= 1;
    }

    Stage0& stage0_;
    Stage1& stage1_;
    Pages& pages_;
    Noc& noc_;
    volatile tt_l1_ptr uint32_t* header_ = nullptr;
    uint32_t current_ = 0;
    bool pending_ = false;
    uint32_t entry_ = 0;
    uint32_t entry_end_ = 0;
    uint32_t cursor_ = 0;
    uint32_t count_ = 0;
    // The read being built: bytes [read_begin_, read_end_) of input page read_page_, landing at read_dst_ after
    // read_lead_ alignment bytes.
    bool read_open_ = false;
    uint32_t read_page_ = 0;
    uint32_t read_begin_ = 0;
    uint32_t read_end_ = 0;
    uint32_t read_lead_ = 0;
    uint32_t read_dst_ = 0;
    uint32_t last_unit_end_ = 0;
    uint32_t last_staged_end_ = 0;
    uint32_t page_ = 0;
    uint32_t page_offset_ = 0;
    uint32_t unit_bytes_ = 0;
    uint32_t pages_count_ = 0;
    uint32_t flags_ = 0;
};

template <typename StagerT>
FORCE_INLINE void stage_tile(StagerT& stager, uint32_t tile) {
    const uint32_t tiles_per_batch = out_tile_rows * out_pages_per_row;
    const uint32_t batch = tile / tiles_per_batch;
    const uint32_t rest = tile % tiles_per_batch;
    const uint32_t row0 = (rest / out_pages_per_row) * tile_height;
    const uint32_t col0 = (rest % out_pages_per_row) * tile_width;
    const uint32_t rows = row0 < out_height ? min_u32(tile_height, out_height - row0) : 0;
    const uint32_t cols = col0 < out_width ? min_u32(tile_width, out_width - col0) : 0;
    stager.begin_unit(tile, 0, page_bytes, 1, rows < tile_height || cols < tile_width);

    InputCursor in;
    for (uint32_t r = 0; r < rows && cols != 0; ++r) {
        in.seek(((batch * out_height) + row0 + r) * out_width + col0);
        for (uint32_t c = 0; c < cols;) {
            const uint32_t in_tile_row = in.row % tile_height;
            const uint32_t in_tile_col = in.col % tile_width;
            uint32_t run = min_u32(cols - c, face_width - c % face_width);
            run = min_u32(run, face_width - in_tile_col % face_width);
            run = min_u32(run, in_width - in.col);
            const uint32_t in_page =
                (in.batch * in_tile_rows + in.row / tile_height) * in_pages_per_row + in.col / tile_width;
            stager.add(in_page, tile_offset(in_tile_row, in_tile_col), tile_offset(r, c), run * element_size);
            c += run;
            in.advance(run);
        }
    }
    stager.end_unit();
}

// Where the bytes [page_offset, page_offset + bytes) of a ROW_MAJOR output page sit in the logical tensor, and how
// many of their elements are logical (the rest is padding).
struct RowPiece {
    uint32_t batch;
    uint32_t height;
    uint32_t col0;
    uint32_t valid;
};

FORCE_INLINE RowPiece row_piece(uint32_t page, uint32_t page_offset, uint32_t bytes) {
    const uint32_t row = page / out_pages_per_row;
    RowPiece piece;
    piece.batch = row / out_padded_height;
    piece.height = row % out_padded_height;
    piece.col0 = (page % out_pages_per_row) * out_page_width + page_offset / element_size;
    const uint32_t elements = bytes / element_size;
    piece.valid = (piece.height < out_height && piece.col0 < out_width) ? min_u32(elements, out_width - piece.col0) : 0;
    return piece;
}

template <typename StagerT>
FORCE_INLINE void stage_row_unit(StagerT& stager, uint32_t unit) {
    uint32_t first_page;
    uint32_t page_offset;
    uint32_t bytes;
    uint32_t page_count;
    if constexpr (pages_per_unit > 1) {
        first_page = unit * pages_per_unit;
        page_offset = 0;
        bytes = page_bytes;
        page_count = min_u32(pages_per_unit, out_num_pages - first_page);
    } else {
        first_page = unit / chunks_per_page;
        page_offset = (unit % chunks_per_page) * unit_bytes;
        bytes = min_u32(unit_bytes, page_bytes - page_offset);
        page_count = 1;
    }
    bool padded = false;
    for (uint32_t p = 0; p < page_count && !padded; ++p) {
        padded = row_piece(first_page + p, page_offset, bytes).valid * element_size < bytes;
    }
    stager.begin_unit(first_page, page_offset, bytes, page_count, padded);

    for (uint32_t p = 0; p < page_count; ++p) {
        const RowPiece piece = row_piece(first_page + p, page_offset, bytes);
        if (piece.valid == 0) {
            continue;
        }
        const uint32_t slot_offset = p * page_slot_bytes;
        InputCursor in;
        in.seek(((piece.batch * out_height) + piece.height) * out_width + piece.col0);
        for (uint32_t e = 0; e < piece.valid;) {
            const uint32_t in_page_col = in.col % in_page_width;
            uint32_t run = min_u32(piece.valid - e, in_page_width - in_page_col);
            run = min_u32(run, in_width - in.col);
            const uint32_t in_page = (in.batch * in_padded_height + in.row) * in_pages_per_row + in.col / in_page_width;
            stager.add(in_page, in_page_col * element_size, slot_offset + e * element_size, run * element_size);
            e += run;
            in.advance(run);
        }
    }
    stager.end_unit();
}

}  // namespace

void kernel_main() {
    const uint32_t unit_begin = get_arg(args::unit_begin);
    const uint32_t unit_end = get_arg(args::unit_end);

    DataflowBuffer stage0(dfb::stage0);
    DataflowBuffer stage1(dfb::stage1);
    Noc noc;
    const auto src = TensorAccessor(tensor::src);
    PageAddresses pages(src);
    Stager stager(stage0, stage1, pages, noc);

    for (uint32_t unit = unit_begin; unit < unit_end; ++unit) {
        if constexpr (tiled) {
            stage_tile(stager, unit);
        } else {
            stage_row_unit(stager, unit);
        }
    }
    stager.finish();
}
