// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute leaves a site as column 1 + r of a row-tile's result; the consumers want it
// as column 0 of a tile of its own, site-major. That is a gather of 32 strided values,
// which costs the RISC a few hundred cycles per output tile and costs a permute a
// dispatch and a full pass over every tile, so it is done here.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_scores/device/kernels/attn_res_scores_items.hpp"

namespace {

// An fp32 tile is four 16x16 faces, row-major within a face and face-major across the
// tile: (row, col) lives at face (row / 16) * 2 + col / 16.
constexpr uint32_t kFaceDim = 16;
constexpr uint32_t kFaceElems = kFaceDim * kFaceDim;

inline uint32_t tile_index(uint32_t row, uint32_t col) {
    return ((row / kFaceDim) * 2 + col / kFaceDim) * kFaceElems + (row % kFaceDim) * kFaceDim + col % kFaceDim;
}

// Column `col` of `src` into column 0 of `dst`. The rest of `dst` is left as it is.
inline void gather_column(const volatile uint32_t* src, volatile uint32_t* dst, uint32_t col) {
    for (uint32_t row = 0; row < 32; ++row) {
        dst[tile_index(row, 0)] = src[tile_index(row, col)];
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t num_candidates = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t num_sites = get_compile_time_arg_val(2);
    constexpr uint32_t kSplitArgs = 3;
    constexpr uint32_t sites_per_group = get_compile_time_arg_val(kSplitArgs + 4);
    constexpr auto exp_args = TensorAccessorArgs<kSplitArgs + 5>();
    constexpr auto shift_args = TensorAccessorArgs<exp_args.next_compile_time_args_offset()>();
    constexpr auto mass_args = TensorAccessorArgs<shift_args.next_compile_time_args_offset()>();

    const auto exp_addr = get_common_arg_val<uint32_t>(0);
    const auto shift_addr = get_common_arg_val<uint32_t>(1);
    const auto mass_addr = get_common_arg_val<uint32_t>(2);

    constexpr uint32_t cb_shift_out = 16;
    constexpr uint32_t cb_exp_out = 17;
    constexpr uint32_t cb_mass_out = 18;
    constexpr uint32_t cb_staging = 24;
    constexpr uint32_t tile_bytes = get_tile_size(cb_exp_out);
    constexpr uint32_t tile_words = tile_bytes / sizeof(uint32_t);

    Noc noc;
    CircularBuffer cb_shift_obj(cb_shift_out);
    CircularBuffer cb_exp_obj(cb_exp_out);
    CircularBuffer cb_mass_obj(cb_mass_out);
    // Never pushed, so both pointers stay at its base and it can be addressed directly.
    CircularBuffer cb_staging_obj(cb_staging);
    const uint32_t staging_base = cb_staging_obj.get_write_ptr();

    auto exp_accessor = TensorAccessor(exp_args, exp_addr);
    auto shift_accessor = TensorAccessor(shift_args, shift_addr);
    auto mass_accessor = TensorAccessor(mass_args, mass_addr);

    // Zero once: each output tile only ever has its column 0 rewritten, so the padding
    // columns stay zero for every tile this core writes.
    {
        volatile uint32_t* staging = reinterpret_cast<volatile uint32_t*>(staging_base);
        for (uint32_t i = 0; i < sites_per_group * tile_words; ++i) {
            staging[i] = 0;
        }
    }

    // Spread one result tile's site columns over the staging tiles and write each to
    // the page `page_of(site)` names. One barrier per result tile: the staging tiles are
    // reused by the next one.
    auto scatter = [&](CircularBuffer& src_obj,
                       uint32_t src_offset,
                       const auto& accessor,
                       auto page_of,
                       const AttnResScoresItem& item) {
        const auto* src = reinterpret_cast<const volatile uint32_t*>(src_obj.get_read_ptr() + src_offset);
        for (uint32_t s = 0; s < item.num_sites; ++s) {
            auto* dst = reinterpret_cast<volatile uint32_t*>(staging_base + s * tile_bytes);
            gather_column(src, dst, 1 + item.first_site + s);
        }
        for (uint32_t s = 0; s < item.num_sites; ++s) {
            noc.async_write(
                cb_staging_obj,
                accessor,
                tile_bytes,
                {.offset_bytes = s * tile_bytes},
                {.page_id = page_of(item.first_site + s)});
        }
        noc.async_write_barrier();
    };

    for_each_attn_res_scores_item<kSplitArgs, num_sites>([&](const AttnResScoresItem& item) {
        // Shift and mass are [R, 1, N, 1]: a site's plane is Ht pages.
        const auto scalar_page = [&](uint32_t site) { return site * Ht + item.row; };

        cb_shift_obj.wait_front(1);
        scatter(cb_shift_obj, 0, shift_accessor, scalar_page, item);
        cb_shift_obj.pop_front(1);

        // Exponentials are [R, C, N, 1]: a site's plane is C * Ht pages.
        cb_exp_obj.wait_front(num_candidates);
        for (uint32_t c = 0; c < num_candidates; ++c) {
            scatter(
                cb_exp_obj,
                c * tile_bytes,
                exp_accessor,
                [&](uint32_t site) { return (site * num_candidates + c) * Ht + item.row; },
                item);
        }
        cb_exp_obj.pop_front(num_candidates);

        cb_mass_obj.wait_front(1);
        scatter(cb_mass_obj, 0, mass_accessor, scalar_page, item);
        cb_mass_obj.pop_front(1);
    });
}
