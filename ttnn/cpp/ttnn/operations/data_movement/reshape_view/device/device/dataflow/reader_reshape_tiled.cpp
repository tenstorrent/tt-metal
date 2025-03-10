// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <tuple>
#include <type_traits>

#include "cpp/ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"
#include "dataflow_api.h"
#include "dprint.h"

template <uint32_t C, uint32_t H, uint32_t W, uint32_t Tilesz_H, uint32_t Tilesz_W>
struct TileDims {
    static constexpr uint32_t w = (W + Tilesz_W - 1) / Tilesz_W;
    static constexpr uint32_t h = (H + Tilesz_H - 1) / Tilesz_H;
    static constexpr uint32_t c = w * h;
};

template <
    uint32_t C,
    uint32_t H,
    uint32_t W,
    uint32_t Tilesz_H,
    uint32_t Tilesz_W,
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> class TileDimsT>
inline std::tuple<uint32_t, uint32_t, uint32_t> page_index_to_idxs(const uint32_t& tile_offset) {
    using tile_dims_t = TileDimsT<C, H, W, Tilesz_H, Tilesz_W>;

    const uint32_t c = tile_offset / tile_dims_t::c;

    // TODO could be done without mod with persistent indices
    const uint32_t hw_tile_offset = tile_offset % tile_dims_t::c;
    const uint32_t h_tile = hw_tile_offset / tile_dims_t::w;
    const uint32_t w_tile = hw_tile_offset % tile_dims_t::w;

    return std::make_tuple(c, h_tile * Tilesz_H, w_tile * Tilesz_W);
}

template <uint32_t C1, uint32_t H1, uint32_t W1, uint32_t C2, uint32_t H2, uint32_t W2>
inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> idxs_to_reshaped_idxs(
    const uint32_t c1, const uint32_t h1, const uint32_t w1) {
    // TODO avoid recomputing intermediates here
    const uint32_t offset = c1 * H1 * W1 + h1 * W1 + w1;

    // TODO and here
    const uint32_t c2 = offset / (H2 * W2);

    const uint32_t hw2 = offset % (H2 * W2);
    const uint32_t h2 = hw2 / H2;
    const uint32_t w2 = hw2 % H2;

    return std::make_tuple(c2, h2, w2, hw2);
}

template <
    uint32_t C,
    uint32_t H,
    uint32_t W,
    uint32_t Tilesz_H,
    uint32_t Tilesz_W,
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> class TileDimsT>
inline uint32_t idxs_to_page_index(const uint32_t c, const uint32_t h, const uint32_t w) {
    using tile_dims_t = TileDimsT<C, H, W, Tilesz_H, Tilesz_W>;

    // TODO probably more intermediates that could be maintained
    return c * tile_dims_t::c + h / Tilesz_H * tile_dims_t::w + w / Tilesz_W;
}

template <uint32_t H, uint32_t Tilesz_H>
struct TileIteratorEndPaddedH {
    const uint32_t tile_end_h;

    TileIteratorEndPaddedH(const uint32_t& h_start) : tile_end_h(min(H - h_start, Tilesz_H) - 1) {}
};

template <uint32_t W, uint32_t Tilesz_W>
struct TileIteratorEndPaddedW {
    const uint32_t tile_end_w;

    TileIteratorEndPaddedW(const uint32_t w_start) : tile_end_w(min(W - w_start, Tilesz_W) - 1) {}
};

template <uint32_t H, uint32_t Tilesz_H>
struct TileIteratorEndAlignedH {
    TileIteratorEndAlignedH(const uint32_t&) {}
    static constexpr uint32_t tile_end_h = Tilesz_H - 1;
};

template <uint32_t W, uint32_t Tilesz_W>
struct TileIteratorEndAlignedW {
    TileIteratorEndAlignedW(const uint32_t&) {}
    static constexpr uint32_t tile_end_w = Tilesz_W - 1;
};

template <uint32_t H, uint32_t Tilesz_H>
using TileIteratorEndH =
    std::conditional_t<H % Tilesz_H == 0, TileIteratorEndAlignedH<H, Tilesz_H>, TileIteratorEndPaddedH<H, Tilesz_H>>;

template <uint32_t W, uint32_t Tilesz_W>
using TileIteratorEndW =
    std::conditional_t<W % Tilesz_W == 0, TileIteratorEndAlignedW<W, Tilesz_W>, TileIteratorEndPaddedW<W, Tilesz_W>>;

template <uint32_t H, uint32_t W, uint32_t Tilesz_H, uint32_t Tilesz_W>
struct TileIterator : TileIteratorEndH<H, Tilesz_H>, TileIteratorEndW<W, Tilesz_W> {
    TileIterator(const uint32_t& in_start_h, const uint32_t& in_start_w) :
        TileIteratorEndH<H, Tilesz_H>(in_start_h),
        TileIteratorEndW<W, Tilesz_W>(in_start_w),
        start_h(in_start_h),
        start_w(in_start_w),
        tile_idx_h(0),
        tile_idx_w(0),
        offset(0),
        first(true) {};

    inline bool next() {
        if (first) {
            first = false;
            return true;
        }
        if (tile_idx_w < tile_end_w) {
            ++offset;
            ++tile_idx_w;
            return true;
        } else if (tile_idx_h < tile_end_h) {
            ++offset;
            tile_idx_w = 0;
            ++tile_idx_h;
            return true;
        } else {
            return false;
        }
    }

    std::tuple<uint32_t, uint32_t, const uint32_t&> operator*() {
        return std::make_tuple(this->h(), this->w(), offset);
    }

private:
    using TileIteratorEndH<H, Tilesz_H>::tile_end_h;
    using TileIteratorEndW<W, Tilesz_W>::tile_end_w;
    const uint32_t& start_h;
    const uint32_t& start_w;
    uint32_t tile_idx_h;
    uint32_t tile_idx_w;
    uint32_t offset;
    bool first;

    inline const uint32_t h() { return start_h + tile_idx_h; }

    inline const uint32_t w() { return start_w + tile_idx_w; }
};

using ttnn::operations::data_movement::reshape::ReshapeMapping;

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_output_page = get_arg_val<uint32_t>(1);
    const uint32_t end_output_page = get_arg_val<uint32_t>(2);

    constexpr uint32_t Tilesz_h = get_compile_time_arg_val(0);
    constexpr uint32_t Tilesz_w = get_compile_time_arg_val(1);

    // input tensor dims
    constexpr uint32_t Csz_i = get_compile_time_arg_val(2);
    constexpr uint32_t Hsz_i = get_compile_time_arg_val(3);
    constexpr uint32_t Wsz_i = get_compile_time_arg_val(4);

    // reshaped output tensor dims
    constexpr uint32_t Csz_o = get_compile_time_arg_val(5);
    constexpr uint32_t Hsz_o = get_compile_time_arg_val(6);
    constexpr uint32_t Wsz_o = get_compile_time_arg_val(7);
    constexpr bool input_is_dram = get_compile_time_arg_val(8);

    constexpr uint32_t cb_id_mapping = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_input = tt::CBIndex::c_1;

    const uint32_t page_size_bytes = get_tile_size(cb_id_input);
    const DataFormat input_data_format = get_dataformat(cb_id_input);

    //  TODO sharded
    const InterleavedAddrGenFast<input_is_dram> input_addrgen = {
        .bank_base_address = src_addr, .page_size = page_size_bytes, .data_format = input_data_format};

    typedef TileIterator<Hsz_o, Wsz_o, Tilesz_h, Tilesz_w> output_tile_iter_t;

    // loop over output (reshaped) pages this core is responsible for
    for (uint32_t page_idx = start_output_page; page_idx < end_output_page; ++page_idx) {
        DPRINT << "Output page: " << page_idx;

        // calculate tensor index start (top left corner of tile) from page index of output page
        const auto [co_0, ho_0, wo_0] = page_index_to_idxs<Csz_o, Hsz_o, Wsz_o, Tilesz_h, Tilesz_w, TileDims>(page_idx);
        output_tile_iter_t iter(ho_0, wo_0);

        // iterate over every element in the output page tile
        volatile tt_l1_ptr ReshapeMapping* map_write_ptr = nullptr;
        while (iter.next()) {
            const auto [ho, wo, hwo] = *iter;

            // calculate corresponding tensor index of input tensor
            const auto [ci, hi, wi, hwi] =
                idxs_to_reshaped_idxs<Csz_o, Hsz_o, Wsz_o, Csz_i, Hsz_i, Wsz_i>(co_0, ho, wo);

            // calculate input tensor page index from input tensor index
            const auto input_page_index =
                idxs_to_page_index<Csz_o, Hsz_o, Wsz_o, Tilesz_h, Tilesz_w, TileDims>(co_0, ho, wo);
            DPRINT << "Tensor Indices " << co_0 << ", " << ho << "," << wo << " input page: " << input_page_index
                   << "\n";

            // if this element is contiguous with the previous element just bump the num_elements for the reader.
            // otherwise push and move on to the next page.
            if (map_write_ptr == nullptr || !map_write_ptr->increment_contiguous(input_page_index, hwi, hwo)) {
                {
                    cb_reserve_back(cb_id_input, 1);
                    uint32_t tile_write_addr = get_write_ptr(cb_id_input);
                    // do the read first so we can continue while it does stuff async
                    noc_async_read_tile(input_page_index, input_addrgen, tile_write_addr);
                }
                {
                    cb_reserve_back(cb_id_mapping, 1);
                    uint32_t map_addr = get_write_ptr(cb_id_mapping);

                    map_write_ptr = reinterpret_cast<volatile tt_l1_ptr ReshapeMapping*>(map_addr);
                    map_write_ptr->input_page_index = input_page_index;
                    map_write_ptr->input_tile_offset = hwi;
                    map_write_ptr->num_elements = 1;
                    map_write_ptr->output_tile_offset = hwo;
                }
            } else {
                noc_async_read_barrier();
                cb_push_back(cb_id_mapping, 1);
                cb_push_back(cb_id_input, 1);

                DPRINT << "PUSH BACK";
            }
        }

        noc_async_read_barrier();
        cb_push_back(cb_id_mapping, 1);
        cb_push_back(cb_id_input, 1);
        DPRINT << "PUSH BACK";
    }
}
