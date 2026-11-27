// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace g6_concat {

// Inputs of one call may sit in different buffer types. The interleaved accessor's buffer type is a
// template parameter, so a reader holds one DRAM and one L1 accessor chain and picks per input by
// bit k of the CT mask (mask_lo: inputs 0..31, mask_hi: 32..63). A placement absent from the call
// repeats the present chain's args; its is_dram flag then tells the kernel that placement is
// absent, and the selection folds at compile time.
template <uint32_t l1_mask_lo, uint32_t l1_mask_hi, typename DramArgs, typename L1Args>
struct InputPlacement {
    static constexpr bool has_dram_inputs = DramArgs::is_dram;
    static constexpr bool has_l1_inputs = !L1Args::is_dram;
    static_assert(has_dram_inputs || has_l1_inputs, "every input is in DRAM or in L1");
    static_assert(!has_dram_inputs || !has_l1_inputs || (l1_mask_lo | l1_mask_hi) != 0, "mixed needs a mask");

    FORCE_INLINE static bool is_l1(uint32_t input_idx) {
        if constexpr (!has_l1_inputs) {
            return false;
        } else if constexpr (!has_dram_inputs) {
            return true;
        } else {
            const uint32_t word = input_idx < 32 ? l1_mask_lo : l1_mask_hi;
            return ((word >> (input_idx % 32)) & 1u) != 0;
        }
    }

    // NoC read alignment of input k's buffer: source and destination must agree modulo this.
    FORCE_INLINE static uint32_t read_alignment(uint32_t input_idx) {
        return is_l1(input_idx) ? NOC_L1_READ_ALIGNMENT_BYTES : NOC_DRAM_READ_ALIGNMENT_BYTES;
    }

    // Calls f(accessor) with input k's accessor rebased on base_address. Call once per run of pages
    // of one input, not per page: the mixed case branches here.
    template <typename F>
    FORCE_INLINE static void with_accessor(uint32_t input_idx, uint32_t base_address, uint32_t page_bytes, F&& f) {
        if constexpr (!has_l1_inputs) {
            f(TensorAccessor(DramArgs{}, base_address, page_bytes));
        } else if constexpr (!has_dram_inputs) {
            f(TensorAccessor(L1Args{}, base_address, page_bytes));
        } else if (is_l1(input_idx)) {
            f(TensorAccessor(L1Args{}, base_address, page_bytes));
        } else {
            f(TensorAccessor(DramArgs{}, base_address, page_bytes));
        }
    }
};

}  // namespace g6_concat
