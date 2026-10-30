// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// V2 reader for fused multi-scale deformable attention: the sampling grid is
// never materialized. Instead of reading a location, the op forms it from a
// reference point and a raw sampling offset:
//
//     loc = reference_points[b, q, r(l, p)]
//         + sampling_offsets[b, q, h, l, p] * scale_l
//
// scale_l is the pixel spacing of the normalized -> pixel mapping (README.md
// §2.2), so one offset unit is one feature-map pixel.
//
// r(l, p) depends on REF_MODE:
//   0 (level):  R == L, r = l         — DINO-family deformable decoders
//   1 (pillar): P % R == 0, r = p % R — BEVFormer, where the point axis is
//                                       laid out as (P / R, R) over z-anchors
//
// The addition, the / [W_l, H_l] normalization and the normalized -> pixel
// mapping all happen on compute in one expression:
//
//     px = ref_x * primary_scale + off_x * secondary_scale + bias
//
// with the three constants folded per level on the host — on the SFPU per
// point, or as matrix entries on the FPU with the tiled geometry. This reader
// only decides *which* bf16 pair is the reference and which is the offset (and,
// tiled, where they land in the block's tiles); it does no arithmetic on
// either. reference_points are bf16, or float32 when REF_FP32 is defined
// (per-point path only; REF_FP32 changes the reference operand's tile format
// to float32).
// Everything else is fused_msda_reader_common.hpp, shared with V1.

#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/dataflow/fused_msda_reader_common.hpp"

constexpr auto value_args = TensorAccessorArgs<MSDA_TENSOR_ACCESSOR_ARG_BASE>();
constexpr auto attn_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
constexpr auto off_args = TensorAccessorArgs<attn_args.next_compile_time_args_offset()>();
constexpr auto ref_args = TensorAccessorArgs<off_args.next_compile_time_args_offset()>();

static_assert(NUM_REFS > 0, "V2 reader requires R > 0");
static_assert(FROM_OFFSETS, "V2 reader must be built with the from-offsets frontend");

namespace {

struct ReferencePlusOffset {
    decltype(TensorAccessor(off_args, 0u, 0u)) off_acc;
    decltype(TensorAccessor(ref_args, 0u, 0u)) ref_acc;
    uint32_t off_arena_l1;
    uint32_t ref_arena_l1;
    // First query row (b * Q + q_start) whose reference points the ref arena
    // holds. It starts invalid on every launch, so the cache never outlives one
    // program run and a new reference_points tensor is always re-read. Only
    // stage() writes the ref arena, so a skipped stage leaves the previous
    // block's rows intact.
    uint32_t staged_ref_row = UINT32_MAX;

    // reference_points is (B, Q, R, 2): one page per (b, q, r), head-invariant,
    // so consecutive heads of one query block reuse the staged rows. v_rows is
    // a function of q_start, so the first row identifies the block. Returns
    // whether the rows were re-read.
    bool stage_refs(Noc& noc, uint32_t b, uint32_t q_start, uint32_t v_rows) {
        const uint32_t ref_row = b * Q + q_start;
        if (ref_row == staged_ref_row) {
            return false;
        }
        staged_ref_row = ref_row;
        for (uint32_t r = 0; r < v_rows; ++r) {
            const uint32_t base = (b * Q + (q_start + r)) * NUM_REFS;
            for (uint32_t k = 0; k < NUM_REFS; ++k) {
                CoreLocalMem<uint32_t> dst(ref_arena_l1 + (r * NUM_REFS + k) * ref_stick_nbytes);
                noc.async_read(ref_acc, dst, ref_stick_nbytes, {.page_id = base + k}, {.offset_bytes = 0});
            }
        }
        return true;
    }

    void stage(Noc& noc, uint32_t b, uint32_t q_start, uint32_t head, uint32_t v_rows) {
        // Offsets share the (B, Q, H, L, P, 2) / packed layout with V1's
        // locations, so the same staging routine applies.
        fused_msda::stage_location_tensor(noc, off_acc, off_arena_l1, b, q_start, head, v_rows);
        stage_refs(noc, b, q_start, v_rows);
    }

    bool stage_tiled(
        Noc& noc,
        uint32_t b,
        uint32_t q_start,
        uint32_t head,
        uint32_t v_rows,
        uint32_t off_l1,
        uint32_t off_row_nbytes) {
        fused_msda::stage_location_tensor(noc, off_acc, off_l1, b, q_start, head, v_rows, off_row_nbytes);
        return stage_refs(noc, b, q_start, v_rows);
    }

    // Reference k of row r to (r, k) and (r, 16 + k); see
    // msda_tiled_geometry.hpp. Plain loads and stores, fenced on both sides
    // here; the caller has barriered the arena's NoC reads.
    void fill_ref_tile(uint32_t tile_l1, uint32_t v_rows) const {
        asm volatile("" ::: "memory");
        constexpr uint32_t STICK_WORDS = ref_stick_nbytes / sizeof(uint32_t);
        for (uint32_t r = 0; r < v_rows; ++r) {
            const auto off = fused_msda_tile_layout::tile_row_offsets(r);
            uint16_t* x = &CoreLocalMem<uint16_t>(tile_l1 + off.lo)[0];
            uint16_t* y = &CoreLocalMem<uint16_t>(tile_l1 + off.hi)[0];
            const uint32_t* src = &CoreLocalMem<uint32_t>(ref_arena_l1 + r * NUM_REFS * ref_stick_nbytes)[0];
#pragma GCC unroll 4
            for (uint32_t k = 0; k < NUM_REFS; ++k) {
                const uint32_t xy = src[k * STICK_WORDS];  // (x, y) bf16, little-endian
                x[k] = static_cast<uint16_t>(xy);
                y[k] = static_cast<uint16_t>(xy >> 16);
            }
        }
        asm volatile("" ::: "memory");
    }

#ifdef REF_FP32
    // Two float32 bit patterns. The SFPU multiplies them; converting here would
    // round the point to bf16, or pay for soft-float on a core with no FPU.
    void primary(uint32_t r, uint32_t l, uint32_t p, uint32_t& x, uint32_t& y) const {
        const uint32_t ref_idx = (REF_MODE == 0) ? l : (p % NUM_REFS);
        CoreLocalMem<volatile uint32_t> ref(ref_arena_l1 + (r * NUM_REFS + ref_idx) * ref_stick_nbytes);
        x = ref[0];
        y = ref[1];
    }
#else
    void primary(uint32_t r, uint32_t l, uint32_t p, uint16_t& x, uint16_t& y) const {
        const uint32_t ref_idx = fused_msda::ref_index(l, p);
        CoreLocalMem<volatile uint16_t> ref(ref_arena_l1 + (r * NUM_REFS + ref_idx) * ref_stick_nbytes);
        x = ref[0];
        y = ref[1];
    }
#endif

    void secondary(uint32_t r, uint32_t l, uint32_t p, uint16_t& x, uint16_t& y) const {
        CoreLocalMem<volatile uint16_t> off(fused_msda::staged_loc_addr(off_arena_l1, r, l, p));
        x = off[0];
        y = off[1];
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

    // Tiled geometry stages the offsets into its row-major block instead, and
    // the factory allocates no offset arena.
    uint32_t off_arena_l1 = 0;
    if constexpr (!TILED_GEOMETRY) {
        CircularBuffer off_scratch_cb(loc_scratch_cb_index);
        off_scratch_cb.reserve_back(TILE_MAX_ROWS * LOC_STICKS_PER_ROW);
        off_arena_l1 = off_scratch_cb.get_write_ptr();
    }
    CircularBuffer ref_scratch_cb(ref_scratch_cb_index);
    ref_scratch_cb.reserve_back(TILE_MAX_ROWS * NUM_REFS);

    ReferencePlusOffset loc_src{
        .off_acc = TensorAccessor(off_args, off_addr, loc_stick_nbytes),
        .ref_acc = TensorAccessor(ref_args, ref_addr, ref_stick_nbytes),
        .off_arena_l1 = off_arena_l1,
        .ref_arena_l1 = ref_scratch_cb.get_write_ptr(),
    };

    fused_msda::reader_main(value_acc, attn_acc, loc_src);
}
