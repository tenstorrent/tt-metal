// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// V1 reader for fused multi-scale deformable attention: sampling locations are
// already materialized, so the location source is a plain lookup in the staged
// arena and there is no secondary operand. Every other responsibility — the
// hand-off of the geometry to the SFPU, boundary handling, the NoC gather of the
// four bilinear neighbours, the tile scatter and the input-tile contract — is in
// fused_msda_reader_common.hpp and identical to V2's.
//
// The coordinate space (MSDA [0, 1] vs grid_sample [-1, 1]) and align_corners
// are folded into the per-level scale and bias the compute kernel applies, so
// nothing here depends on either.

#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/dataflow/fused_msda_reader_common.hpp"

constexpr auto value_args = TensorAccessorArgs<MSDA_TENSOR_ACCESSOR_ARG_BASE>();
constexpr auto attn_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
constexpr auto loc_args = TensorAccessorArgs<attn_args.next_compile_time_args_offset()>();

static_assert(!FROM_OFFSETS, "V1 reader must not be built with the from-offsets frontend");

namespace {

// Staged sampling_locations, handed on as bf16 bit patterns.
struct MaterializedLocations {
    decltype(TensorAccessor(loc_args, 0u, 0u)) acc;
    uint32_t arena_l1;

    void stage(Noc& noc, uint32_t b, uint32_t q_start, uint32_t head, uint32_t v_rows) {
        fused_msda::stage_location_tensor(noc, acc, arena_l1, b, q_start, head, v_rows);
    }

    void primary(uint32_t r, uint32_t l, uint32_t p, uint16_t& x, uint16_t& y) const {
        CoreLocalMem<volatile uint16_t> ptr(fused_msda::staged_loc_addr(arena_l1, r, l, p));
        x = ptr[0];
        y = ptr[1];
    }
};

}  // namespace

void kernel_main() {
    const uint32_t value_addr = get_arg_val<uint32_t>(0);
    const uint32_t attn_addr = get_arg_val<uint32_t>(1);
    const uint32_t loc_addr = get_arg_val<uint32_t>(2);

    const auto value_acc = TensorAccessor(value_args, value_addr, value_page_nbytes);
    const auto attn_acc = TensorAccessor(attn_args, attn_addr, attn_stick_nbytes);

    CircularBuffer loc_scratch_cb(loc_scratch_cb_index);
    loc_scratch_cb.reserve_back(TILE_MAX_ROWS * LOC_STICKS_PER_ROW);

    MaterializedLocations loc_src{
        .acc = TensorAccessor(loc_args, loc_addr, loc_stick_nbytes),
        .arena_l1 = loc_scratch_cb.get_write_ptr(),
    };

    fused_msda::reader_main(value_acc, attn_acc, loc_src);
}
