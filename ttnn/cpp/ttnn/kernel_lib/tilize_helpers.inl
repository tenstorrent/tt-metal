// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file tilize_helpers.inl
 * @brief Implementation of tilize helper functions
 *
 * This file contains the implementation details for the tilize() function.
 * It should only be included by tilize_helpers.hpp.
 */
#include "ttnn/cpp/ttnn/kernel_lib/dfb_helpers_compute.hpp"
#include "api/dataflow/dataflow_buffer.h"

// JIT generates chlkc_descriptors.h (not per-variable files), included via chlkc_list.h.
// The arrays are available in scope but guarded by TRISC type:
//   - unpack_src_format[] / unpack_dst_format[]   : UNPACK and MATH TRISCs (not PACK)
//   - unpack_tile_r/c_dim[]                       : UNPACK and MATH TRISCs (not PACK)
//   - pack_src_format[] / pack_dst_format[]       : PACK TRISC only
//   - pack_tile_r/c_dim[]                         : PACK TRISC only
// Note: unpack_src_format[cb] == pack_dst_format[cb] (both are L1 format, equalized by JIT).
// Note: unpack_tile_r/c_dim[cb] == pack_tile_r/c_dim[cb] (both from desc.buf_tile_r/c_dim_arr).
namespace compute_kernel_lib {

// =============================================================================
// Internal Helper Implementations
// =============================================================================

template <uint32_t input_dfb>
constexpr bool has_supported_fast_tilize_format() {
    constexpr auto format = dfb_l1_format<input_dfb>();
    return format == static_cast<uint32_t>(DataFormat::Float32) ||
           format == static_cast<uint32_t>(DataFormat::Float16_b);
}

template <uint32_t input_dfb>
constexpr bool is_fp32_input_format() {
    return dfb_l1_format<input_dfb>() == static_cast<uint32_t>(DataFormat::Float32);
}

template <uint32_t output_dfb>
constexpr bool is_fp32_output_format() {
    return dfb_l1_format<output_dfb>() == static_cast<uint32_t>(DataFormat::Float32);
}

template <uint32_t input_dfb, bool pack_default>
constexpr bool has_unpack_to_dest_fp32() {
    // Detects whether the CB was configured with UnpackToDestMode::UnpackToDestFp32 in the
    // program factory. The JIT folds that host-side enum into unpack_dst_format[]: when set,
    // a Float32 CB keeps Dest-side format Float32 (0); with Default mode it is downgraded to
    // Tf32 (4). So comparing unpack_src_format[cb] == unpack_dst_format[cb] is a reliable
    // compile-time signal on the TRISCs that can see both arrays (UNPACK and MATH; PACK cannot).
    //
    // pack_default controls the value returned on PACK (where the check is not observable).
    // Callers pick it so the surrounding static_assert passes on PACK, deferring enforcement
    // to UNPACK/MATH.
#if defined(UCK_CHLKC_PACK)
    return pack_default;
#else
    return unpack_src_format[input_dfb] == unpack_dst_format[input_dfb];
#endif
}

template <uint32_t input_dfb, bool pack_default>
constexpr bool has_unpack_to_dest_fp32() {
    // Detects whether the CB was configured with UnpackToDestMode::UnpackToDestFp32 in the
    // program factory. The JIT folds that host-side enum into unpack_dst_format[]: when set,
    // a Float32 CB keeps Dest-side format Float32 (0); with Default mode it is downgraded to
    // Tf32 (4). So comparing unpack_src_format[cb] == unpack_dst_format[cb] is a reliable
    // compile-time signal on the TRISCs that can see both arrays (UNPACK and MATH; PACK cannot).
    //
    // pack_default controls the value returned on PACK (where the check is not observable).
    // Callers pick it so the surrounding static_assert passes on PACK, deferring enforcement
    // to UNPACK/MATH.
#if defined(UCK_CHLKC_PACK)
    return pack_default;
#else
    return unpack_src_format[input_dfb] == unpack_dst_format[input_dfb];
#endif
}

template <uint32_t block_width_tiles, uint32_t input_dfb, uint32_t output_dfb>
constexpr bool can_use_fast_tilize() {
#ifdef ARCH_QUASAR
    // Quasar has no fast-tilize LLK (and per the LLK team it never will) — only regular tilize.
    // Always take the regular tilize_init/tilize_block/tilize_uninit path below.
    return false;
#else
    // Float32 OUTPUT is unsupported: fast-tilize's pack path uses Read_32b=0
    // (bf16-stride stepping through DEST), which truncates fp32 DEST to bf16.
    // That truncation is acceptable for bf16/bfp output but destroys precision
    // for fp32 output, producing garbage results in downstream fp32 consumers
    // (see attn_matmul_fp32 regression).
    return block_width_tiles < 256 && dfb_has_32x32_tiles<output_dfb>() && !get_dst_full_sync_enabled() &&
           has_supported_fast_tilize_format<input_dfb>() && !is_fp32_output_format<output_dfb>();
#endif
}

// =============================================================================
// Main Function Implementation
// =============================================================================

template <
    uint32_t block_width_tiles,
    uint32_t input_dfb,
    uint32_t output_dfb,
    tilize_config::InitUninitMode init_uninit_mode,
    tilize_config::WaitMode wait_mode,
    tilize_config::ReconfigureRegisterDatatypeMode reconfig_mode,
    tilize_config::Fp32Mode fp32_mode,
    tilize_config::RemapMode remap_mode>
ALWI void tilize(uint32_t num_blocks, std::optional<uint32_t> total_input_pages) {
    // Compile-time validation
    static_assert(block_width_tiles > 0, "block_width_tiles must be greater than 0");
    static_assert(input_dfb != output_dfb, "Tilize cannot be done in-place: input_dfb and output_dfb must be different");
    static_assert(input_dfb < 32, "Invalid input_dfb: must be less than 32");
    static_assert(output_dfb < 32, "Invalid output_dfb: must be less than 32");

    // Runtime parameter validation
    ASSERT(num_blocks > 0);

    // Determine if we're using fast tilize mode (automatic detection based on tile size, sync mode, and data format).
    // Fp32Mode::Lossless disables fast tilize only for fp32 inputs to preserve exact values
    // (fast tilize truncates fp32 → tf32). Has no effect on non-fp32 formats.
    constexpr bool lossless_fp32_override =
        (fp32_mode == tilize_config::Fp32Mode::Lossless) && is_fp32_input_format<input_dfb>();
    constexpr bool use_fast = can_use_fast_tilize<block_width_tiles, input_dfb, output_dfb>() && !lossless_fp32_override;

    // Lossless fp32 requires BOTH fp32 Dest AND the input CB configured with UnpackToDestFp32.
    // Without these, the slow tilize path still round-trips fp32 through tf32 in Dest and the
    // "lossless" promise is silently broken. Gate with if constexpr so the asserts only fire
    // when the input is actually fp32 and Lossless mode is requested.
    if constexpr (lossless_fp32_override) {
        static_assert(DST_ACCUM_MODE,
            "Fp32Mode::Lossless requires fp32_dest_acc_en=true in the ComputeConfig: "
            "Dest must hold fp32, otherwise the slow tilize path still downgrades to tf32.");
        static_assert(has_unpack_to_dest_fp32<input_dfb, /*pack_default=*/true>(),
            "Fp32Mode::Lossless requires UnpackToDestMode::UnpackToDestFp32 on the input CB. "
            "Set unpack_to_dest_mode[input_cb] = UnpackToDestMode::UnpackToDestFp32 in the "
            "program factory; otherwise the unpacker truncates fp32 to tf32 on its way to Dest.");
    }

    // Conversely, the fast tilize path requires UnpackToDestMode::Default on the input CB.
    // Combining fast_tilize with UnpackToDestFp32 silently corrupts output (the unpacker
    // writes 32-bit fp32 payloads into Dest slots that fast_tilize's pack stage reads as
    // tf32). Only applicable to fp32 inputs — Float16_b is unaffected because its
    // unpack_dst_format equals its unpack_src_format regardless of UnpackToDestMode.
    if constexpr (use_fast && is_fp32_input_format<input_dfb>()) {
        static_assert(!has_unpack_to_dest_fp32<input_dfb, /*pack_default=*/false>(),
            "Fast tilize on fp32 input requires UnpackToDestMode::Default on the input CB. "
            "Combining fast_tilize with UnpackToDestFp32 corrupts output. Either leave "
            "unpack_to_dest_mode[input_cb] as Default in the program factory, or request "
            "Fp32Mode::Lossless to force the slow (bit-exact) tilize path.");
    }

    // Determine if we're doing data type reconfiguration
    constexpr bool use_unpack_reconfig =
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure) ||
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    constexpr bool use_pack_reconfig =
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::PackReconfigure) ||
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    const bool asymmetric_dfb_pages = total_input_pages.has_value();
    if (asymmetric_dfb_pages) {
        ASSERT(*total_input_pages > (num_blocks - 1) * 32);  // at least one row in the last block
        ASSERT(*total_input_pages <= num_blocks * 32);       // rows fit within num_blocks tile-rows
    }

    // Tilize input must not be a block float format (Bfp8/4/2 and _b variants).
    // Block floats have shared exponents that break row-major-to-tile reinterpretation.
    UNPACK(ASSERT(!is_block_float_format(unpack_src_format[input_dfb])));

    // Reconfigure register datatypes if requested
    if constexpr (use_unpack_reconfig) {
        // Reconfigure srcA for unpack
        reconfig_data_format_srca(input_dfb);

#ifndef ARCH_BLACKHOLE
        if constexpr (use_fast) {
            // WH fast-tilize uses both SrcA and SrcB; reconfigure SrcB to match input.
            // BH fast-tilize only uses SrcA — SrcB must not be touched so matmul
            // weights stay configured correctly.
            reconfig_data_format_srcb(input_dfb);
        }
#endif
    }

    if constexpr (use_pack_reconfig) {
        // Reconfigure output for pack
        pack_reconfig_data_format(output_dfb);
    }

    // Compile-time initialization based on InitUninitMode
    if constexpr (
        init_uninit_mode == tilize_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == tilize_config::InitUninitMode::InitOnly) {
        if constexpr (use_fast) {
#ifdef ARCH_BLACKHOLE
            if constexpr (remap_mode == tilize_config::RemapMode::AssumeConfigured) {
                fast_tilize_init_skip_remap(input_dfb, block_width_tiles, output_dfb);
            } else
#endif
            {
#ifndef ARCH_QUASAR  // Quasar has no fast tilize (use_fast is always false here); keep the name out of the parse
                fast_tilize_init(input_dfb, block_width_tiles, output_dfb);
#else
                // Unreachable: can_use_fast_tilize() returns false on Quasar so use_fast is always false.
                // Trap (watcher/runtime assert) in case this path is ever reached.
                ASSERT(false);
#endif
            }
        } else {
            tilize_init(input_dfb, block_width_tiles, output_dfb);
        }
    }

    // Validate DFB capacity
    if (asymmetric_dfb_pages) {
        uint32_t max_in = (*total_input_pages < 32) ? *total_input_pages : 32;
        UNPACK(ASSERT(get_dfb_num_pages(input_dfb) >= max_in));
    } else {
        UNPACK(ASSERT(get_dfb_num_pages(input_dfb) >= block_width_tiles));
    }
    PACK(ASSERT(get_dfb_num_pages(output_dfb) >= block_width_tiles));

    // Construct DataflowBuffer objects for sync operations
    DataflowBuffer in_dfb(input_dfb);
    DataflowBuffer out_dfb(output_dfb);

    // Upfront wait (when requested)
    if constexpr (wait_mode == tilize_config::WaitMode::WaitUpfront) {
        uint32_t total_wait = asymmetric_dfb_pages ? *total_input_pages : (block_width_tiles * num_blocks);
        in_dfb.wait_front(total_wait);
    }

    // Main loop
    uint32_t pages_left = total_input_pages.value_or(0);
    uint32_t input_pages = block_width_tiles;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Determine input pages for this block
        if (asymmetric_dfb_pages) {
            // Asymmetric: min(32, pages_left)
            input_pages = (pages_left < 32) ? pages_left : 32;
        }

        if constexpr (wait_mode == tilize_config::WaitMode::WaitBlock) {
            in_dfb.wait_front(input_pages);
        }

        out_dfb.reserve_back(block_width_tiles);

        if constexpr (use_fast) {
#ifndef ARCH_QUASAR  // Quasar has no fast tilize (use_fast is always false here); keep the name out of the parse
            fast_tilize_block(input_dfb, block_width_tiles, output_dfb);
#else
            // Unreachable: can_use_fast_tilize() returns false on Quasar so use_fast is always false.
            // Trap (watcher/runtime assert) in case this path is ever reached.
            ASSERT(false);
#endif
        } else {
            tilize_block(input_dfb, block_width_tiles, output_dfb);
        }

        out_dfb.push_back(block_width_tiles);
        in_dfb.pop_front(input_pages);

        if (asymmetric_dfb_pages) {
            pages_left -= input_pages;
        }
    }

    // Compile-time cleanup based on InitUninitMode
    if constexpr (
        init_uninit_mode == tilize_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == tilize_config::InitUninitMode::UninitOnly) {
        if constexpr (use_fast) {
#ifndef ARCH_QUASAR  // Quasar has no fast tilize (use_fast is always false here); keep the name out of the parse
            fast_tilize_uninit(input_dfb, output_dfb, block_width_tiles);
#else
            // Unreachable: can_use_fast_tilize() returns false on Quasar so use_fast is always false.
            // Trap (watcher/runtime assert) in case this path is ever reached.
            ASSERT(false);
#endif
        } else {
            tilize_uninit(input_dfb, output_dfb);
        }
    }
}

}  // namespace compute_kernel_lib
