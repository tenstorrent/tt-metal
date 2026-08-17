// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Canonical host-compatible BF8_B/BF4_B packer for Wormhole.
//
// The Tensix packer and tt-metal's host pack_as_bfp*_tiles disagree on exact
// ties (upstream issue #17237).  Data-movement results are compared against the
// host encoder bit-for-bit, so representation-changing casts finish here on a
// dataflow RISC with the same shared-exponent + round-to-nearest-even algorithm
// as blockfloat_common.cpp.  Input and output pages use TT's face-major TILE
// order: four 16x16 faces, with one shared exponent per 16-value face row.
//
// CT args:
//   [0] cb_in
//   [1] cb_scratch
//   [2] input element bytes (2=BF16, 4=FP32)
//   [3] output magnitude bits (7=BF8_B, 3=BF4_B)
//   [4..] destination TensorAccessorArgs
// RT args:
//   [0] dst_addr, [1] num_tiles, [2] start_tile
//   [3] Ht, [4] Wt, [5] logical_H, [6] logical_W,
//   [7] scrub_source_padding

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

FORCE_INLINE uint32_t source_fp32_bits(
    const uint32_t src_addr, const uint32_t index) {
    constexpr uint32_t INPUT_ELEMENT_BYTES = get_compile_time_arg_val(2);
    if constexpr (INPUT_ELEMENT_BYTES == 2) {
        CoreLocalMem<const volatile uint16_t> src(src_addr);
        return static_cast<uint32_t>(src[index]) << 16;
    } else {
        CoreLocalMem<const volatile uint32_t> src(src_addr);
        return src[index];
    }
}

FORCE_INLINE uint8_t rne_mantissa(
    const uint32_t bits, const uint32_t shared_exp) {
    constexpr uint32_t MAGNITUDE_BITS = get_compile_time_arg_val(3);
    constexpr uint32_t SHIFT = 24 - MAGNITUDE_BITS;
    constexpr uint32_t MAX_MAGNITUDE = (1u << MAGNITUDE_BITS) - 1u;
    constexpr uint32_t ROUND_MASK = (1u << SHIFT) - 1u;
    constexpr uint32_t TIE = 1u << (SHIFT - 1u);

    const uint32_t exp = (bits >> 23) & 0xffu;
    if (exp == 0u) {
        return 0u;  // +/- zero and FP32/BF16 denormals
    }

    const uint32_t sign = bits >> 31;
    uint32_t magnitude = (1u << 23) | (bits & 0x007fffffu);
    const uint32_t exp_delta = shared_exp - exp;
    magnitude = exp_delta >= 24u ? 0u : (magnitude >> exp_delta);

    const uint32_t remainder = magnitude & ROUND_MASK;
    magnitude >>= SHIFT;
    if (remainder > TIE || (remainder == TIE && (magnitude & 1u) != 0u)) {
        ++magnitude;
    }
    if (magnitude > MAX_MAGNITUDE) {
        magnitude = MAX_MAGNITUDE;
    }
    if (magnitude == 0u) {
        return 0u;
    }
    return static_cast<uint8_t>((sign << MAGNITUDE_BITS) | magnitude);
}

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t INPUT_ELEMENT_BYTES = get_compile_time_arg_val(2);
    constexpr uint32_t MAGNITUDE_BITS = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();
    static_assert(
        INPUT_ELEMENT_BYTES == 2 || INPUT_ELEMENT_BYTES == 4,
        "canonical BFP pack input must be BF16 or FP32");
    static_assert(
        MAGNITUDE_BITS == 7 || MAGNITUDE_BITS == 3,
        "canonical BFP pack output must be BF8_B or BF4_B");

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    const uint32_t Ht = get_arg_val<uint32_t>(3);
    const uint32_t Wt = get_arg_val<uint32_t>(4);
    const uint32_t logical_H = get_arg_val<uint32_t>(5);
    const uint32_t logical_W = get_arg_val<uint32_t>(6);
    const bool scrub_source_padding = get_arg_val<uint32_t>(7) != 0u;

    constexpr uint32_t EXPONENT_BYTES = 64;
    constexpr uint32_t DATA_BYTES = MAGNITUDE_BITS == 7 ? 1024 : 512;
    constexpr uint32_t OUTPUT_PAGE_BYTES = EXPONENT_BYTES + DATA_BYTES;
    const auto d = TensorAccessor(
        dst_args, dst_addr, dst_args.get_aligned_page_size());

    Noc noc;
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_scratch_obj(cb_scratch);

    // This kernel owns the scratch slot for its whole lifetime.  It has no
    // downstream consumer, so complete the reserve/push handshake up front:
    // a bare reserve with no matching push is an un-flushed CB (the consumer
    // signal is never posted, hanging its cb_wait_front on silicon).  Pushing
    // the depth-1 slot leaves it occupied-but-flushed; packed_addr (captured
    // before the push) stays valid and is reused for every tile.
    cb_scratch_obj.reserve_back(1);
    const uint32_t packed_addr = cb_scratch_obj.get_write_ptr();
    cb_scratch_obj.push_back(1);

    for (uint32_t tile_offset = 0; tile_offset < num_tiles; ++tile_offset) {
        cb_in_obj.wait_front(1);
        const uint32_t src_addr = cb_in_obj.get_read_ptr();
        CoreLocalMem<volatile uint8_t> packed(packed_addr);

        const uint32_t tile_id = start_tile + tile_offset;
        const uint32_t tile_row = (tile_id / Wt) % Ht;
        const uint32_t tile_col = tile_id % Wt;

        for (uint32_t face = 0; face < 4; ++face) {
            const uint32_t face_h = (face >> 1) * 16u;
            const uint32_t face_w = (face & 1u) * 16u;
            for (uint32_t row = 0; row < 16; ++row) {
                const uint32_t group = face * 16u + row;
                const uint32_t source_base = face * 256u + row * 16u;
                const uint32_t local_h = face_h + row;
                const uint32_t global_h = tile_row * 32u + local_h;

                uint32_t shared_exp = 0;
                for (uint32_t col = 0; col < 16; ++col) {
                    const uint32_t local_w = face_w + col;
                    const bool invalid = scrub_source_padding &&
                        (global_h >= logical_H ||
                         tile_col * 32u + local_w >= logical_W);
                    const uint32_t bits = invalid
                        ? 0u : source_fp32_bits(src_addr, source_base + col);
                    const uint32_t exp = (bits >> 23) & 0xffu;
                    shared_exp = exp > shared_exp ? exp : shared_exp;
                }
                packed[group] = static_cast<uint8_t>(shared_exp);

                if constexpr (MAGNITUDE_BITS == 7) {
                    const uint32_t output_base = EXPONENT_BYTES + group * 16u;
                    for (uint32_t col = 0; col < 16; ++col) {
                        const uint32_t local_w = face_w + col;
                        const bool invalid = scrub_source_padding &&
                            (global_h >= logical_H ||
                             tile_col * 32u + local_w >= logical_W);
                        const uint32_t bits = invalid
                            ? 0u : source_fp32_bits(src_addr, source_base + col);
                        packed[output_base + col] =
                            rne_mantissa(bits, shared_exp);
                    }
                } else {
                    const uint32_t output_base = EXPONENT_BYTES + group * 8u;
                    for (uint32_t pair = 0; pair < 8; ++pair) {
                        const uint32_t col0 = pair * 2u;
                        const uint32_t col1 = col0 + 1u;
                        const uint32_t local_w0 = face_w + col0;
                        const uint32_t local_w1 = face_w + col1;
                        const bool invalid0 = scrub_source_padding &&
                            (global_h >= logical_H ||
                             tile_col * 32u + local_w0 >= logical_W);
                        const bool invalid1 = scrub_source_padding &&
                            (global_h >= logical_H ||
                             tile_col * 32u + local_w1 >= logical_W);
                        const uint32_t bits0 = invalid0
                            ? 0u : source_fp32_bits(src_addr, source_base + col0);
                        const uint32_t bits1 = invalid1
                            ? 0u : source_fp32_bits(src_addr, source_base + col1);
                        packed[output_base + pair] = static_cast<uint8_t>(
                            rne_mantissa(bits0, shared_exp) |
                            (rne_mantissa(bits1, shared_exp) << 4));
                    }
                }
            }
        }

        noc.async_write(packed, d, OUTPUT_PAGE_BYTES,
                        {.offset_bytes = 0},
                        {.page_id = tile_id, .offset_bytes = 0});
        noc.async_write_barrier();
        cb_in_obj.pop_front(1);
    }
}
