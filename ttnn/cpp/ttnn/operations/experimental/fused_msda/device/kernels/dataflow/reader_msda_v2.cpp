// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// V2 reader for fused multi-scale deformable attention: the sampling grid is
// never materialized. Instead of reading a location, the reader forms it from a
// reference point and a raw sampling offset:
//
//     loc = reference_points[b, q, r(l, p)] + sampling_offsets[b, q, h, l, p] / [W_l, H_l]
//
// r(l, p) depends on REF_MODE:
//   0 (level):  R == L, r = l         — DINO-family deformable decoders
//   1 (pillar): P % R == 0, r = p % R — BEVFormer, where the point axis is
//                                       laid out as (P / R, R) over z-anchors
//
// Everything after that step is fused_msda_reader_common.hpp, byte-identical to
// V1 — which is what makes V2 a pure frontend and not a different operator.

#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/dataflow/fused_msda_reader_common.hpp"

constexpr auto value_args = TensorAccessorArgs<MSDA_TENSOR_ACCESSOR_ARG_BASE>();
constexpr auto attn_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
constexpr auto off_args = TensorAccessorArgs<attn_args.next_compile_time_args_offset()>();
constexpr auto ref_args = TensorAccessorArgs<off_args.next_compile_time_args_offset()>();

static_assert(NUM_REFS > 0, "V2 reader requires R > 0");

namespace {

struct ReferencePlusOffset {
    decltype(TensorAccessor(off_args, 0u, 0u)) off_acc;
    decltype(TensorAccessor(ref_args, 0u, 0u)) ref_acc;
    uint32_t off_arena_l1;
    uint32_t ref_arena_l1;

    void stage(Noc& noc, uint32_t b, uint32_t q_start, uint32_t head, uint32_t v_rows) {
        // Offsets share the (B, Q, H, L, P, 2) / packed layout with V1's
        // locations, so the same staging routine applies.
        fused_msda::stage_location_tensor(noc, off_acc, off_arena_l1, b, q_start, head, v_rows);
        // reference_points is (B, Q, R, 2): one page per (b, q, r), head-invariant.
        for (uint32_t r = 0; r < v_rows; ++r) {
            const uint32_t base = (b * Q + (q_start + r)) * NUM_REFS;
            for (uint32_t k = 0; k < NUM_REFS; ++k) {
                CoreLocalMem<uint32_t> dst(ref_arena_l1 + (r * NUM_REFS + k) * ref_stick_nbytes);
                noc.async_read(ref_acc, dst, ref_stick_nbytes, {.page_id = base + k}, {.offset_bytes = 0});
            }
        }
    }

    void location(uint32_t r, uint32_t l, uint32_t p, const fused_msda::LevelGeom& g, float& x, float& y) const {
        const uint32_t ref_idx = (REF_MODE == 0) ? l : (p % NUM_REFS);
        CoreLocalMem<volatile uint16_t> ref(ref_arena_l1 + (r * NUM_REFS + ref_idx) * ref_stick_nbytes);
        CoreLocalMem<volatile uint16_t> off(fused_msda::staged_loc_addr(off_arena_l1, r, l, p));
        // Offsets are raw, in feature-map pixel units; inv_width / inv_height
        // are precomputed per level so this stays a multiply.
        x = fused_msda::bf16_to_float(ref[0]) + fused_msda::bf16_to_float(off[0]) * g.inv_width;
        y = fused_msda::bf16_to_float(ref[1]) + fused_msda::bf16_to_float(off[1]) * g.inv_height;
    }
};

}  // namespace

void kernel_main() {
    const uint32_t value_addr = get_arg_val<uint32_t>(0);
    const uint32_t attn_addr = get_arg_val<uint32_t>(1);
    const uint32_t off_addr = get_arg_val<uint32_t>(2);
    const uint32_t ref_addr = get_arg_val<uint32_t>(3);

    const auto value_acc = TensorAccessor(value_args, value_addr, value_page_nbytes);
    const auto attn_acc = TensorAccessor(attn_args, attn_addr, attn_stick_nbytes);

    CircularBuffer off_scratch_cb(loc_scratch_cb_index);
    off_scratch_cb.reserve_back(TILE_MAX_ROWS * LOC_STICKS_PER_ROW);
    CircularBuffer ref_scratch_cb(ref_scratch_cb_index);
    ref_scratch_cb.reserve_back(TILE_MAX_ROWS * NUM_REFS);

    ReferencePlusOffset loc_src{
        .off_acc = TensorAccessor(off_args, off_addr, loc_stick_nbytes),
        .ref_acc = TensorAccessor(ref_args, ref_addr, ref_stick_nbytes),
        .off_arena_l1 = off_scratch_cb.get_write_ptr(),
        .ref_arena_l1 = ref_scratch_cb.get_write_ptr(),
    };

    fused_msda::reader_main(value_acc, attn_acc, loc_src);
}
