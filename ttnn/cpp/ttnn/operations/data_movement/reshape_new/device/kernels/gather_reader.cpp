// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gather reader: for each output unit in [unit_begin, unit_end), walks the input element runs that make it up and
// stages them, with their placements, into stage DFB entries (layout in gather_common.hpp).
//
// Both tensors are viewed as batch x height x width logical elements (dims -2 and -1; everything before folded
// into the batch). An output element at flat logical index f comes from the input element at the same f.
//   ROW_MAJOR: unit u is chunk (u % chunks_per_page) of output page (u / chunks_per_page); a page holds
//              page_width elements of one row.
//   TILE:      unit u is output tile u; tiles are stored face by face, face rows contiguous.
// A run never crosses an input row, an input page or face row, or an output face row, so each run is one
// contiguous read and one contiguous placement.
//
// A run is read from its source address rounded down to source_alignment into a 16-byte aligned slot, which
// keeps the read legal for any source offset; the record points past the lead-in bytes.
//
// Compile-time args: see ReshapeNewGatherProgramFactory
// Runtime args:      unit_begin, unit_end
// Bindings:          dfb::stage (producer), tensor::src

#include <stdint.h>

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
constexpr uint32_t tile_height = get_arg(args::tile_height);
constexpr uint32_t tile_width = get_arg(args::tile_width);
constexpr uint32_t face_height = get_arg(args::face_height);
constexpr uint32_t face_width = get_arg(args::face_width);
constexpr uint32_t source_alignment = get_arg(args::source_alignment);
constexpr uint32_t records_per_entry = get_arg(args::records_per_entry);
constexpr uint32_t header_bytes = get_arg(args::header_bytes);
constexpr uint32_t stage_entry_bytes = get_arg(args::stage_entry_bytes);

constexpr uint32_t faces_per_tile_row = tile_width / face_width;
constexpr uint32_t face_elements = face_height * face_width;
constexpr uint32_t in_tile_rows = in_padded_height / tile_height;
constexpr uint32_t out_tile_rows = out_padded_height / tile_height;

FORCE_INLINE uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Byte offset of element (row, col) of a tile.
FORCE_INLINE uint32_t tile_offset(uint32_t row, uint32_t col) {
    const uint32_t face = (row / face_height) * faces_per_tile_row + col / face_width;
    return (face * face_elements + (row % face_height) * face_width + col % face_width) * element_size;
}

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

// Fills stage entries for one unit. Entries are opened lazily and closed when full or at the end of the unit.
class Stager {
public:
    Stager(DataflowBuffer& stage, Noc& noc) : stage_(stage), noc_(noc) {}

    FORCE_INLINE void begin_unit(uint32_t page, uint32_t page_offset, uint32_t bytes, bool padded) {
        page_ = page;
        page_offset_ = page_offset;
        unit_bytes_ = bytes;
        flags_ = kFlagFirst | (padded ? kFlagPadded : 0);
        open();
    }

    template <typename Accessor>
    FORCE_INLINE void add(const Accessor& src, uint32_t page, uint32_t offset, uint32_t unit_offset, uint32_t bytes) {
        const uint64_t src_noc = src.get_noc_addr(page, offset);
        const uint32_t lead = static_cast<uint32_t>(src_noc) & (source_alignment - 1);
        const uint32_t span = (lead + bytes + kNocWordBytes - 1) & ~(kNocWordBytes - 1);
        if (count_ == records_per_entry || cursor_ + span > stage_entry_bytes) {
            close(false);
            flags_ &= ~kFlagFirst;
            open();
        }
        noc_.async_read(
            PrecomposedUnicastEndpoint{},
            stage_,
            lead + bytes,
            {.noc_addr = src_noc - lead},
            {.offset_bytes = cursor_});
        volatile tt_l1_ptr uint32_t* record = header_ + kHeaderWords + kRecordWords * count_;
        record[0] = unit_offset;
        record[1] = cursor_ + lead;
        record[2] = bytes;
        ++count_;
        cursor_ += span;
    }

    FORCE_INLINE void end_unit() { close(true); }

private:
    FORCE_INLINE void open() {
        stage_.reserve_back(1);
        header_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(stage_.get_write_ptr()));
        count_ = 0;
        cursor_ = header_bytes;
    }

    FORCE_INLINE void close(bool last) {
        header_[kHeaderPage] = page_;
        header_[kHeaderPageOffset] = page_offset_;
        header_[kHeaderUnitBytes] = unit_bytes_;
        header_[kHeaderFlags] = flags_ | (last ? kFlagLast : 0);
        header_[kHeaderCount] = count_;
        noc_.async_read_barrier();
        stage_.push_back(1);
    }

    DataflowBuffer& stage_;
    Noc& noc_;
    volatile tt_l1_ptr uint32_t* header_ = nullptr;
    uint32_t page_ = 0;
    uint32_t page_offset_ = 0;
    uint32_t unit_bytes_ = 0;
    uint32_t flags_ = 0;
    uint32_t count_ = 0;
    uint32_t cursor_ = 0;
};

template <typename Accessor>
FORCE_INLINE void stage_tile(Stager& stager, const Accessor& src, uint32_t tile) {
    const uint32_t tiles_per_batch = out_tile_rows * out_pages_per_row;
    const uint32_t batch = tile / tiles_per_batch;
    const uint32_t rest = tile % tiles_per_batch;
    const uint32_t row0 = (rest / out_pages_per_row) * tile_height;
    const uint32_t col0 = (rest % out_pages_per_row) * tile_width;
    const uint32_t rows = row0 < out_height ? min_u32(tile_height, out_height - row0) : 0;
    const uint32_t cols = col0 < out_width ? min_u32(tile_width, out_width - col0) : 0;
    stager.begin_unit(tile, 0, page_bytes, rows < tile_height || cols < tile_width);

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
            stager.add(src, in_page, tile_offset(in_tile_row, in_tile_col), tile_offset(r, c), run * element_size);
            c += run;
            in.advance(run);
        }
    }
    stager.end_unit();
}

template <typename Accessor>
FORCE_INLINE void stage_row_chunk(Stager& stager, const Accessor& src, uint32_t unit) {
    const uint32_t page = unit / chunks_per_page;
    const uint32_t page_offset = (unit % chunks_per_page) * unit_bytes;
    const uint32_t bytes = min_u32(unit_bytes, page_bytes - page_offset);
    const uint32_t row = page / out_pages_per_row;
    const uint32_t batch = row / out_padded_height;
    const uint32_t height = row % out_padded_height;
    const uint32_t col0 = (page % out_pages_per_row) * out_page_width + page_offset / element_size;
    const uint32_t elements = bytes / element_size;
    const uint32_t valid = (height < out_height && col0 < out_width) ? min_u32(elements, out_width - col0) : 0;
    stager.begin_unit(page, page_offset, bytes, valid * element_size < bytes);

    if (valid != 0) {
        InputCursor in;
        in.seek(((batch * out_height) + height) * out_width + col0);
        for (uint32_t e = 0; e < valid;) {
            const uint32_t in_page_col = in.col % in_page_width;
            uint32_t run = min_u32(valid - e, in_page_width - in_page_col);
            run = min_u32(run, in_width - in.col);
            const uint32_t in_page = (in.batch * in_padded_height + in.row) * in_pages_per_row + in.col / in_page_width;
            stager.add(src, in_page, in_page_col * element_size, e * element_size, run * element_size);
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

    DataflowBuffer stage(dfb::stage);
    Noc noc;
    const auto src = TensorAccessor(tensor::src);
    Stager stager(stage, noc);

    for (uint32_t unit = unit_begin; unit < unit_end; ++unit) {
        if constexpr (tiled) {
            stage_tile(stager, src, unit);
        } else {
            stage_row_chunk(stager, src, unit);
        }
    }
}
