// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

namespace ckernel::packer
{
using DataFormatType = std::underlying_type_t<DataFormat>;

constexpr std::uint32_t replay_buf_offset = 16; // split replay buffer usage between fpu/sfpu
                                                // fist 16 for sfpu, next 16 for fpu
constexpr std::uint32_t NUM_PACKERS = 1;        // Number of packers

// Pack config
struct pack_config_t
{
    // word 0
    std::uint32_t row_ptr_section_size : 16;
    std::uint32_t exp_section_size     : 16;
    // word 1
    std::uint32_t l1_dest_addr : 32;
    // word 2
    std::uint32_t uncompress              : 1;
    std::uint32_t add_l1_dest_addr_offset : 1;
    std::uint32_t disable_pack_zero_flag  : 1;
    std::uint32_t reserved_0              : 1;
    std::uint32_t out_data_format : DATA_FORMAT_BIT_COUNT;
    std::uint32_t in_data_format : DATA_FORMAT_BIT_COUNT;
    std::uint32_t dis_shared_exp_assembler            : 1;
    std::uint32_t auto_set_last_pacr_intf_sel         : 1;
    std::uint32_t enable_out_fifo                     : 1;
    std::uint32_t sub_l1_tile_header_size             : 1;
    std::uint32_t src_if_sel                          : 1;
    std::uint32_t pack_start_intf_pos                 : 4;
    std::uint32_t all_pack_disable_zero_compress_ovrd : 1;
    std::uint32_t add_tile_header_size                : 1;
    std::uint32_t pack_dis_y_pos_start_offset         : 1;
    std::uint32_t l1_src_addr                         : 8;
    //   The bit unp_lf8_4b_exp is configured in the unpack, remove word 3 to avoid potential race condition
    //   //word 3
    //   uint32_t downsample_mask : 16;
    //   uint32_t downsample_shift_count  : 3;
    //   uint32_t pack_l1_acc : 1; //Not new to BH, but moved
    //   //uint32_t read_mode : 1; //Removed in BH
    //   uint32_t exp_threshold_en  : 1;
    //   uint32_t reserved_2 : 1;
    //   uint32_t unp_lf8_4b_exp: 1;
    //   uint32_t pac_lf8_4b_exp: 1;
    //   uint32_t exp_threshold : 8;
};

static_assert(sizeof(pack_config_t) == (sizeof(std::uint32_t) * 3));

union pack_config_u
{
    std::uint32_t val[3];
    pack_config_t f;
};

// Relu Config
struct relu_config_t
{
    std::uint32_t ALU_ACC_CTRL_Zero_Flag_disabled_src      : 1;
    std::uint32_t ALU_ACC_CTRL_Zero_Flag_disabled_dst      : 1;
    std::uint32_t STACC_RELU_ApplyRelu                     : 4;
    std::uint32_t STACC_RELU_ReluThreshold                 : 16;
    std::uint32_t DISABLE_RISC_BP_Disable_main             : 1;
    std::uint32_t DISABLE_RISC_BP_Disable_trisc            : 3;
    std::uint32_t DISABLE_RISC_BP_Disable_ncrisc           : 1;
    std::uint32_t DISABLE_RISC_BP_Disable_bmp_clear_main   : 1;
    std::uint32_t DISABLE_RISC_BP_Disable_bmp_clear_trisc  : 3;
    std::uint32_t DISABLE_RISC_BP_Disable_bmp_clear_ncrisc : 1;
};

static_assert(sizeof(relu_config_t) == (sizeof(std::uint32_t)));

union relu_config_u
{
    std::uint32_t val[1];
    relu_config_t r;
};

// Dest rd control
struct dest_rd_ctrl_t
{
    std::uint32_t PCK_DEST_RD_CTRL_Read_32b_data  : 1;
    std::uint32_t PCK_DEST_RD_CTRL_Read_unsigned  : 1;
    std::uint32_t PCK_DEST_RD_CTRL_Read_int8      : 1;
    std::uint32_t PCK_DEST_RD_CTRL_Round_10b_mant : 1;
    std::uint32_t PCK_DEST_RD_CTRL_Reserved       : 28;
};

static_assert(sizeof(dest_rd_ctrl_t) == (sizeof(std::uint32_t)));

union dest_rd_ctrl_u
{
    std::uint32_t val;
    dest_rd_ctrl_t f;
};

// RMW mask over every non-reserved field of dest_rd_ctrl_t. A packer data-format reconfig writes the
// whole field set with it, as the full configure does with a word write.
constexpr std::uint32_t PCK_DEST_RD_CTRL_FIELDS_MASK =
    PCK_DEST_RD_CTRL_Read_32b_data_MASK | PCK_DEST_RD_CTRL_Read_unsigned_MASK | PCK_DEST_RD_CTRL_Read_int8_MASK | PCK_DEST_RD_CTRL_Round_10b_mant_MASK;
// dest_rd_ctrl_t is four 1-bit fields at consecutive bit positions followed by the reserved bits; the
// mask must be exactly that run of bits. (Checked arithmetically: clang cannot constant-evaluate a
// bit cast of a bit-field struct.)
static_assert(
    PCK_DEST_RD_CTRL_Read_32b_data_MASK == (1u << PCK_DEST_RD_CTRL_Read_32b_data_SHAMT) &&
        PCK_DEST_RD_CTRL_Read_unsigned_MASK == (1u << PCK_DEST_RD_CTRL_Read_unsigned_SHAMT) &&
        PCK_DEST_RD_CTRL_Read_int8_MASK == (1u << PCK_DEST_RD_CTRL_Read_int8_SHAMT) &&
        PCK_DEST_RD_CTRL_Round_10b_mant_MASK == (1u << PCK_DEST_RD_CTRL_Round_10b_mant_SHAMT),
    "PCK_DEST_RD_CTRL fields must be single bits");
static_assert(
    PCK_DEST_RD_CTRL_Read_unsigned_SHAMT == PCK_DEST_RD_CTRL_Read_32b_data_SHAMT + 1 &&
        PCK_DEST_RD_CTRL_Read_int8_SHAMT == PCK_DEST_RD_CTRL_Read_unsigned_SHAMT + 1 &&
        PCK_DEST_RD_CTRL_Round_10b_mant_SHAMT == PCK_DEST_RD_CTRL_Read_int8_SHAMT + 1,
    "PCK_DEST_RD_CTRL fields must be consecutive bits, as laid out in dest_rd_ctrl_t");
// The constant stays on the right-hand side: the cross-thread config checker (infra/check_cross_thread_cfg.py)
// reads `<name>_MASK =` as a definition and would resolve the mask to this assert instead.
static_assert(
    (((1u << (PCK_DEST_RD_CTRL_Round_10b_mant_SHAMT - PCK_DEST_RD_CTRL_Read_32b_data_SHAMT + 1)) - 1) << PCK_DEST_RD_CTRL_Read_32b_data_SHAMT) ==
        PCK_DEST_RD_CTRL_FIELDS_MASK,
    "PCK_DEST_RD_CTRL_FIELDS_MASK must cover every non-reserved field of dest_rd_ctrl_t");
static_assert(
    PCK_DEST_RD_CTRL_Read_32b_data_ADDR32 == PCK_DEST_RD_CTRL_Read_unsigned_ADDR32 &&
        PCK_DEST_RD_CTRL_Read_32b_data_ADDR32 == PCK_DEST_RD_CTRL_Read_int8_ADDR32 &&
        PCK_DEST_RD_CTRL_Read_32b_data_ADDR32 == PCK_DEST_RD_CTRL_Round_10b_mant_ADDR32,
    "PCK_DEST_RD_CTRL fields must share one ADDR32 for a combined RMW");

// PACK_EDGE_OFFSET_SEC[0:3] register structure
//
// Lower 16b represent a mask that is applied on a single row of one face on the packer output
// Higher 16b contain information about which TILE_ROW_SET_MAPPING register is used for each packer (only in PACK_EDGE_OFFSET_SEC0)
//
// There are 4 PACK_EDGE_OFFSET_SEC[0:3] registers and 4 TILE_ROW_SET_MAPPING[0:3] registers.
// TILE_ROW_SET_MAPPING[0:3] have 2 bits for each row inside a face that determine which PACK_EDGE_OFFSET_SEC[0:3] mask is used.
// Only PACK_EDGE_OFFSET_SEC0 register has higher 16b configured to determine TILE_ROW_SET_MAPPING[0:3] registers used for each packer.
// Other PACK_EDGE_OFFSET_SEC[1:3] registers are used only for the masks in the lower 16b.
struct pck_edge_offset_t
{
    std::uint32_t mask                      : 16;
    std::uint32_t mode                      : 1;
    std::uint32_t tile_row_set_select_pack0 : 2;
    std::uint32_t tile_row_set_select_pack1 : 2;
    std::uint32_t tile_row_set_select_pack2 : 2;
    std::uint32_t tile_row_set_select_pack3 : 2;
    std::uint32_t reserved                  : 7;
};

static_assert(sizeof(pck_edge_offset_t) == (sizeof(std::uint32_t)));

union pck_edge_offset_u
{
    std::uint32_t val;
    pck_edge_offset_t f;
};

// Pack counters
struct pack_counters_t
{
    std::uint32_t pack_per_xy_plane        : 8;
    std::uint32_t pack_reads_per_xy_plane  : 8;
    std::uint32_t pack_xys_per_til         : 7;
    std::uint32_t pack_yz_transposed       : 1;
    std::uint32_t pack_per_xy_plane_offset : 8;
};

static_assert(sizeof(pack_counters_t) == (sizeof(std::uint32_t)));

union pack_counters_u
{
    std::uint32_t val;
    pack_counters_t f;
};

// Set unpacker offsets to 0, except for unpacker 0, channel 1, X, which is the tile X dimension
inline void packer_addr_counter_init()
{
    TTI_SETADCXY(0b100, 0, 0, 0, 0, 0b1011);
    TTI_SETADCZW(0b100, 0, 0, 0, 0, 0b1111);
}

/**
 * \brief Returns true if `out_l1` is a valid output of the FP32 late-conversion column.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_packer_fp32_late_column_output(const DataFormat out_l1)
{
    switch (out_l1)
    {
        case DataFormat::Float32:
        case DataFormat::Float16_b:
        case DataFormat::Float16:
        case DataFormat::Lf8:
        case DataFormat::Bfp8:
        case DataFormat::Bfp4:
        case DataFormat::Bfp2:
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp2_b:
            return true;
        default:
            return false;
    }
}

/**
 * \brief Returns true if `out_l1` is a valid output of the combined late-conversion column.
 *
 * Combined column: "From TF32 or BF16 or E8M6 or FP16 or E5M7 or E5M6 or FP8".
 * This is exactly the FP32 column plus Tf32.
 */
__attribute__((noinline)) bool is_packer_combined_late_column_output(const DataFormat out_l1)
{
    return out_l1 == DataFormat::Tf32 || is_packer_fp32_late_column_output(out_l1);
}

/**
 * \brief Returns true if conversion is supported by EARLY packer conversion stage.
 *
 * Validates supported dst-register to intermediate-format pairs for Blackhole's early conversion
 * stage. For this API, `out_l1` is interpreted as the requested intermediate format code.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_packer_to_L1_early_conversion_supported(const DataFormat in_reg, const DataFormat out_l1)
{
    switch (in_reg)
    {
        case DataFormat::Float32:
            return out_l1 == DataFormat::Float32 ||   // FP32 (identity)
                   out_l1 == DataFormat::Tf32 ||      // TF32
                   out_l1 == DataFormat::Float16_b || // BF16
                   out_l1 == DataFormat::Bfp8_b ||    // E8M6 (encoded as BFP8)
                   out_l1 == DataFormat::Int32 ||     // INT32 (bitcast)
                   out_l1 == DataFormat::Int8 ||      // INT8
                   out_l1 == DataFormat::UInt8;       // UINT8

        case DataFormat::Float16_b:
            return out_l1 == DataFormat::Tf32 ||      // TF32
                   out_l1 == DataFormat::Float16_b || // BF16
                   out_l1 == DataFormat::Bfp8_b ||    // E8M6 (encoded as BFP8)
                   out_l1 == DataFormat::Int8;        // INT8

        case DataFormat::Float16:
            return out_l1 == DataFormat::Float16 || // FP16
                   out_l1 == DataFormat::Bfp8 ||    // E5M7/E5M6 (encoded as BFP8a)
                   out_l1 == DataFormat::Lf8 ||     // FP8 (e5m2)
                   out_l1 == DataFormat::Int8;      // INT8

        case DataFormat::Int32:
            return out_l1 == DataFormat::Float32 ||   // FP32 (bitcast)
                   out_l1 == DataFormat::Tf32 ||      // TF32 (bitcast+round)
                   out_l1 == DataFormat::Float16_b || // BF16 (top 16b bitcast)
                   out_l1 == DataFormat::Int32 ||     // INT32 (identity)
                   out_l1 == DataFormat::Int8 ||      // INT8
                   out_l1 == DataFormat::UInt8;       // UINT8

        case DataFormat::UInt16: // INT16 identity path
            return out_l1 == DataFormat::UInt16;

        default:
            return false;
    }
}

/**
 * \brief Returns true if conversion is supported by LATE packer conversion stage.
 *
 * Validates supported intermediate (LateFromFormat) to L1 pairs for Blackhole's late conversion
 * stage.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_packer_to_L1_late_conversion_supported(const DataFormat in_reg, const DataFormat out_l1)
{
    switch (in_reg)
    {
        case DataFormat::Float32: // From FP32 column
            return is_packer_fp32_late_column_output(out_l1);

        // Combined column (TF32 / BF16 / E8M6 / FP16 / E5M7 / E5M6 / FP8).
        // Bfp4/Bfp2 and Bfp4_b/Bfp2_b are kept for backward compatibility with existing code.
        case DataFormat::Tf32:
        case DataFormat::Float16_b:
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp2_b:
        case DataFormat::Float16:
        case DataFormat::Bfp8:
        case DataFormat::Bfp4:
        case DataFormat::Bfp2:
        case DataFormat::Lf8:
            return is_packer_combined_late_column_output(out_l1);

        case DataFormat::Int32: // From INT32 column
            return out_l1 == DataFormat::Int32;

        case DataFormat::UInt16: // From INT16 column
            return out_l1 == DataFormat::UInt16;

        case DataFormat::Int8: // From INT8/UINT8 column
        case DataFormat::UInt8:
            return out_l1 == DataFormat::Int8 || out_l1 == DataFormat::UInt8;

        case DataFormat::UInt32:
            return out_l1 == DataFormat::UInt32;

        default:
            return false;
    }
}

/**
 * \brief Returns true if either EARLY or LATE packer conversion stage supports the conversion.
 */
__attribute__((noinline)) bool is_packer_to_L1_conversion_supported(const DataFormat in_reg, const DataFormat out_l1)
{
    return is_packer_to_L1_early_conversion_supported(in_reg, out_l1) || is_packer_to_L1_late_conversion_supported(in_reg, out_l1);
}

// The block packs' state in one word, so set_packer_strides resets it with one store: byte 0 last output + 1 (with
// LLK_PACK_STATE_W if it took W), byte 1 reconfigured, bytes 2-3 the channel-1 W stride; unread, its stores drop.
[[maybe_unused]] alignas(4) static std::uint8_t pack_state[4] = {0, 0, 0, 0};
constexpr std::uint8_t LLK_PACK_STATE_FU_OPEN = 0xff;
constexpr std::uint8_t LLK_PACK_STATE_W       = 0x80;
// The output (id + 1) whose configuration the last llk_pack_reconfig_data_format wrote, 0 when unknown: a reconfigure to
// the same output then writes nothing. Every other writer of the fields it writes forgets it; unread, its stores drop.
[[maybe_unused]] static std::uint8_t pack_live_output = 0;
#define LLK_PACK_LIVE_OUTPUT 1


// The next default per-tile pack's destination, predicted by the compute API's per-tile pack; a half-sync release writes it
// inside the packer drain and sets LLK_PACK_PRE_WRITTEN, and a pack to it then writes none. 0: none. Every pack, per tile or
// block, replaces or clears it.
[[maybe_unused]] static std::uint32_t pack_pre_next = 0;
// The last release drained the packer and no pack has run since the next section's wait: the push needs no stall of its own.
[[maybe_unused]] static bool pack_push_drained = false;
// SDPA kernels: tiles of the last push (llk_push_tiles), which picks the reconfigure form.
[[maybe_unused]] static std::uint32_t pack_sdpa_last_push = 0;
// SDPA kernels: tiles of the last reserve (llk_wait_for_free_tiles), the block now being packed.
[[maybe_unused]] static std::uint32_t pack_sdpa_reserve = 0;
constexpr std::uint32_t LLK_PACK_PRE_WRITTEN        = 0x80000000u;
inline void llk_pack_live_forget()
{
    pack_live_output = 0;
}
inline std::uint32_t llk_pack_state_half()
{
    std::uint16_t half;
    __builtin_memcpy(&half, pack_state, sizeof(half));
    return half;
}
inline void llk_pack_state_set_half(const std::uint16_t half)
{
    __builtin_memcpy(pack_state, &half, sizeof(half));
}
// A pack program's state store: the channel-1 W stride bytes reset only where it zeroed channel 1, so a W or closed block
// after an init or reconfigure keeps its stride.
template <bool ch1_zeroed>
inline void llk_pack_state_store(const std::uint16_t half)
{
    if constexpr (ch1_zeroed)
    {
        const std::uint32_t state = half;
        __builtin_memcpy(pack_state, &state, sizeof(state));
    }
    else
    {
        llk_pack_state_set_half(half);
    }
}
inline std::uint32_t pack_ch1_w_stride()
{
    std::uint16_t stride;
    __builtin_memcpy(&stride, pack_state + 2, sizeof(stride));
    return stride;
}
inline void pack_ch1_w_stride_set(const std::uint32_t tile_stride)
{
    const std::uint16_t stride = tile_stride;
    __builtin_memcpy(pack_state + 2, &stride, sizeof(stride));
}

// The mask context in a halfword of its own, so a kernel that stores none folds the select: byte 0 matmul (1) or
// other, byte 1 a pack-thread SFPU init ran (sticky).
[[maybe_unused]] alignas(2) static std::uint8_t pack_ctx[2] = {0, 0};
inline void llk_pack_math_context(const std::uint8_t context)
{
    pack_ctx[0] = context;
}
inline void llk_pack_sfpu_context()
{
    pack_ctx[1] = 1;
}

// Specialized by the matmul API headers (1 matmul.h, 2 experimental/2_0/matmul.h, 3 experimental/matmul_custom.h).
template <int api>
struct llk_pack_matmul_api
{
    static constexpr bool value = false;
};

// Specialized by layernorm.h: the norm kernels, which include matmul.h without being bound by matmul.
template <int api>
struct llk_pack_norm_api
{
    static constexpr bool value = false;
};

// Specialized by the compute API (pack_math_context.h) with the kernel's MATH_FIDELITY, which a tt-llk test defines only when it
// names a fidelity.
template <int api>
struct llk_pack_lofi_api
{
    static constexpr bool value = false;
};

// Specialized from the kernel's DEST mode and CB formats (llk_pack_common_api.h): fp32 DEST, in0 at 32 bit and an in1 that is not.
template <int api>
struct llk_pack_in0_only32
{
    static constexpr bool value = false;
};

// Only a kernel with a matmul API reaches the matmul contexts; dep, a template parameter of the caller, delays the lookup.
template <std::uint32_t dep>
constexpr bool llk_pack_matmul_kernel()
{
    return llk_pack_matmul_api<dep * 0 + 1>::value || llk_pack_matmul_api<dep * 0 + 2>::value || llk_pack_matmul_api<dep * 0 + 3>::value;
}

// A LoFi fp32 DEST matmul whose in0 alone is 32 bit binds on its pack (a 32-bit output at the 32-bit rate, mm_r3_1), and
// unpacker A, which an early release costs L1 grants, reads a 16-bit in1.
template <std::uint32_t dep>
constexpr bool llk_pack_matmul_pack_bound()
{
    return llk_pack_lofi_api<dep * 0 + 1>::value && llk_pack_in0_only32<dep * 0 + 1>::value && llk_pack_matmul_kernel<dep>() &&
           !llk_pack_norm_api<dep * 0 + 1>::value;
}

// A matmul kernel other than a norm one, whose unpackers bound it: at half sync with fp32 DEST or at LoFi an earlier pack start
// costs them L1 grants.
template <std::uint32_t dep>
constexpr bool llk_pack_late_start(const bool is_fp32_dest_acc_en)
{
    return (is_fp32_dest_acc_en || llk_pack_lofi_api<dep * 0 + 1>::value) && llk_pack_matmul_kernel<dep>() && !llk_pack_norm_api<dep * 0 + 1>::value;
}

// Specialized by the headers of the pack-thread SFPU inits (compute_kernel_api.h and seven eltwise_unary headers).
template <int api>
struct llk_pack_sfpu_api
{
    static constexpr bool value = false;
};

// Specialized by the SDPA SFPU header (ckernel_sfpu_sdpa.h): SDPA compute kernels.
template <int api>
struct llk_pack_sdpa_api
{
    static constexpr bool value = false;
};

template <std::uint32_t dep>
constexpr bool llk_pack_sfpu_kernel()
{
    return llk_pack_sfpu_api<dep * 0 + 1>::value || llk_pack_sfpu_api<dep * 0 + 2>::value || llk_pack_sfpu_api<dep * 0 + 3>::value ||
           llk_pack_sfpu_api<dep * 0 + 4>::value || llk_pack_sfpu_api<dep * 0 + 5>::value || llk_pack_sfpu_api<dep * 0 + 6>::value ||
           llk_pack_sfpu_api<dep * 0 + 7>::value || llk_pack_sfpu_api<dep * 0 + 8>::value;
}

// 0 other, 1 matmul, 2 matmul with a pack-thread SFPU: without a matmul API 0 at compile time, without a pack-thread
// SFPU header never 2.
template <std::uint32_t matmul_kt>
inline std::uint32_t llk_pack_context()
{
    if constexpr (!llk_pack_matmul_kernel<matmul_kt>())
    {
        return 0;
    }
    else
    {
        std::uint16_t half;
        __builtin_memcpy(&half, pack_ctx, sizeof(half));
        const std::uint32_t sfpu = llk_pack_sfpu_kernel<matmul_kt>() && (half >> 8) != 0 ? 1 : 0;
        return (half & 0xff) != 0 ? 1 + sfpu : 0;
    }
}

// W offset state: base and stride last programmed, and the tile the next block starts at; anything that writes
// L1_Dest_addr or the channel-1 W stride makes it stale (LLK_PACK_W_INVALIDATE).
[[maybe_unused]] static bool pack_w_valid               = false;
[[maybe_unused]] static std::uint32_t pack_w_page       = 0;
[[maybe_unused]] static std::uint32_t pack_w_next       = 0;
[[maybe_unused]] static std::uint32_t pack_w_tiles      = 0;
// An init forgets the last output, not the base: a pack init before every block then never programs the W stride.
#define LLK_PACK_W_INVALIDATE() (pack_w_valid = false)
// A fast untilize left open is a last output of its own, forgotten by every set_packer_strides.
#define LLK_PACK_FAST_UNTILIZE_MARK_OPEN() (pack_state[0] = LLK_PACK_STATE_FU_OPEN)
#define LLK_PACK_FAST_UNTILIZE_OPEN() (pack_state[0] == LLK_PACK_STATE_FU_OPEN)

// Another pack program replaced the default MOP: llk_pack_block packs per tile until a default pack init. Only those
// inits write pack_other_program_seen, so kernels without one fold the test, and without llk_pack_block keep no flag.
[[maybe_unused]] static bool pack_other_program_seen = false;
[[maybe_unused]] static bool pack_other_program      = false;
// A default init with a MOP other than hw_configure's (zero output, more than one tile) wrote it: inits rewrite it from then on.
[[maybe_unused]] static bool pack_mop_dirty = false;
// The packer input width (bytes) whose default strides are live, 0 for another program's: read only by the default inits of a
// kernel whose pack CBs differ in width (llk_pack_width_single), so elsewhere its stores drop.
[[maybe_unused]] static std::uint8_t pack_strides_width = 0;
#define LLK_PACK_OTHER_PROGRAM_SET() (pack_other_program_seen = true, pack_other_program = true)
#define LLK_PACK_OTHER_PROGRAM_CLEAR() (pack_other_program = false)
#define LLK_PACK_OTHER_PROGRAM_ACTIVE() (pack_other_program_seen && pack_other_program)

// Set by the fast untilize init, forgotten by set_packer_strides; the block packs close an open one first. Only
// llk_pack_fast_untilize.h sets pack_fast_untilize_linked, so other kernels compile no close.
template <DstSync>
inline constexpr bool pack_fast_untilize_linked = false;
[[maybe_unused]] static bool pack_fast_untilize_seen         = false;
[[maybe_unused]] static std::uint32_t pack_fast_untilize_src = 0;

// Some CB of the kernel unpacks to DEST, as pack_reconfig_u2d (llk_pack_block_tile.h sets it again): set_packer_strides
// keeps its guard NOPs in default mode there.
#if defined(LLK_UNPACK_TO_DEST_ANY)
template <PackMode>
inline constexpr bool pack_strides_u2d = LLK_UNPACK_TO_DEST_ANY != 0;
#else
template <PackMode>
inline constexpr bool pack_strides_u2d = true;
#endif
// PackMode::Default through a caller's template argument, so pack_strides_u2d is read at the caller's instantiation.
template <auto>
inline constexpr PackMode llk_pack_default_mode = PackMode::Default;

// reconfigured: the reconfigure's state, stored by the caller of the noinline reconfigure (store_state false there),
// so that call writes no memory and the caller keeps its registers across it. zero_ch1: hw_configure and the fast
// untilize close; elsewhere a channel-1 stride left by the strided fast untilize (Y) or the W forms (W) is idle, as
// they reset their channel-1 counter at the end of every block and no other pack program moves it. ch1_u2d: with an unpack
// to DEST the inits keep them, and the reconfigures only with fp32 DEST.
template <PackMode pack_mode = PackMode::Default, bool reconfigured = false, bool store_state = true, bool zero_ch1 = false, bool ch1_u2d = true>
inline void set_packer_strides(const std::uint32_t pack_src_format, const std::uint32_t tile_c_dim)
{
    llk_pack_live_forget();
    std::uint32_t x_stride = datum_size_in_bytes(pack_src_format);
    // The state stores first: behind the config writes they wait for them.
    if constexpr (store_state)
    {
        llk_pack_state_store<zero_ch1 || (ch1_u2d && pack_strides_u2d<pack_mode>)>(reconfigured ? 0x100 : 0);
        pack_strides_width = pack_mode == PackMode::Default ? x_stride : 0;
    }
    std::uint32_t y_stride = FACE_C_DIM * x_stride;
    std::uint32_t w_stride = TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * x_stride;

    // Untilize mode has 2 packer interfaces active, so z counter needs to jump by 2
    // faces, since z counter is only 1 bit (can't be programmed to inc by 2)
    const std::uint32_t z_stride = ((pack_mode != PackMode::Default) && (tile_c_dim == TILE_C_DIM)) ? 2 * FACE_R_DIM * y_stride : FACE_R_DIM * y_stride;

    TT_SETDMAREG(0, LOWER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, LO_16(p_gpr_pack::TMP0)); // x-stride not used!
    TT_SETDMAREG(0, UPPER_HALFWORD((y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, HI_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, LOWER_HALFWORD((z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT)), 0, LO_16(p_gpr_pack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD((w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT)), 0, HI_16(p_gpr_pack::TMP1));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
    // Like the NOPs, they delay the pack start, which a u2d loop's init and an fp32 reconfigure need.
    if constexpr (zero_ch1 || (ch1_u2d && pack_strides_u2d<pack_mode>))
    {
        TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);
        TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    }
    // A default pack writes its destination before its first PACR, which spaces the stride writes from it; with an unpack
    // to DEST the NOPs stay, as the earlier pack start contends with the unpacker for L1.
    if constexpr (pack_mode != PackMode::Default || pack_strides_u2d<pack_mode>)
    {
        TTI_NOP;
        TTI_NOP;
    }
}

// The packer accumulates in L1 (Pack_L1_Acc): under half sync llk_pack_block packs per tile, as main. Only a runtime
// enable writes pack_l1_acc_seen, so a kernel that never accumulates folds the test away.
[[maybe_unused]] static bool pack_l1_acc_seen = false;
[[maybe_unused]] static bool pack_l1_acc_on   = false;
#define LLK_PACK_L1_ACC_ACTIVE() (pack_l1_acc_seen && pack_l1_acc_on)

inline void reconfigure_packer_l1_acc(const std::uint32_t pack_l1_acc)
{
    if (pack_l1_acc != 0)
    {
        pack_l1_acc_seen = true;
    }
    pack_l1_acc_on = pack_l1_acc != 0;
    // Stall to avoid clobbering current packer configuration, then enable/disable L1 accumulation
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pack_L1_Acc_RMW>(pack_l1_acc);
    // Accumulating writes are bank atomics; with the fast tile end drain the lockstep packer ports can settle into a slow rate.
#if !defined(TT_METAL_TTSIM) // ttsim does not model PACK_GLOBAL_CFG_CTL and aborts on any access to it
    cfg_reg_rmw_tensix<PACK_GLOBAL_CFG_CTL_pack_disable_fast_tile_end_drain_RMW>(pack_l1_acc ? 1 : 0);
#endif
}

/**
 * @brief Configure packer exponent thresholding for the current destination and output formats.
 *
 * We use packer exponent thresholding to zero out datums that are too small to be repesentable
 * This is specifically done in FP32 (Dest) -> BFPxA (L1) case.
 *
 * This implies that exponent thresholding should be reconfigured whenever packer formats change.
 *
 * @tparam is_fp32_dest_acc_en True when Dest register is FP32.
 * @param pack_dst_format Pack output data format.
 */
// The exponent threshold word for an output format (reconfigure_exp_threshold).
template <bool is_fp32_dest_acc_en>
constexpr std::uint32_t pack_exp_threshold_bits(const std::uint32_t pack_dst_format)
{
    // Workaround for HW bug: tenstorrent/budabackend#1394
    // After early format conversion the exponent is EXP_B; packing to BFP-A uses EXP_A.
    // Zero out values too small to represent in EXP_A.
    //
    // For a given EXP_B number to be representable in EXP_A, it must satisfy:
    // EXP_BIAS_B = 127, EXP_BIAS_A = 15, and EXP_MIN_A = 1 - EXP_BIAS_A = -14
    //   EXP_B - EXP_BIAS_B >= EXP_MIN_A
    //   EXP_B - 127 >= -14
    //   EXP_B >= 113
    // The packer compares against threshold 113 and forces datums with EXP_B < 113 to zero.
    const bool enable             = is_fp32_dest_acc_en && IS_BFP_A_FORMAT(pack_dst_format);
    const std::uint32_t threshold = enable ? 113 : 0;
    return (threshold << THCON_SEC0_REG1_Exp_threshold_SHAMT) | (static_cast<std::uint32_t>(enable) << THCON_SEC0_REG1_Exp_threshold_en_SHAMT);
}

template <bool is_fp32_dest_acc_en>
inline void reconfigure_exp_threshold(const std::uint32_t pack_dst_format)
{
    static_assert(
        THCON_SEC0_REG1_Exp_threshold_en_ADDR32 == THCON_SEC0_REG1_Exp_threshold_ADDR32,
        "THCON_SEC0_REG1_Exp_threshold_en and Exp_threshold must share ADDR32 for combined RMW");

    constexpr std::uint32_t THRESHOLD_RMW_MASK = THCON_SEC0_REG1_Exp_threshold_en_MASK | THCON_SEC0_REG1_Exp_threshold_MASK;

    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Exp_threshold_ADDR32, 0, THRESHOLD_RMW_MASK>(pack_exp_threshold_bits<is_fp32_dest_acc_en>(pack_dst_format));
}

// PCK_DEST_RD_CTRL for a packer input and output format (set_packer_config, reconfig_packer_data_format): bit 0 Read_32b_data,
// 1 Read_unsigned, 2 Read_int8, 3 Round_10b_mant.
template <bool is_fp32_dest_acc_en>
constexpr std::uint32_t pack_dest_rd_ctrl_bits(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format)
{
    const bool is_32b_format = pack_src_format == to_underlying(DataFormat::Int32) || pack_src_format == to_underlying(DataFormat::UInt32) ||
                               pack_src_format == to_underlying(DataFormat::Float32);
    const bool is_int8_format = pack_src_format == to_underlying(DataFormat::Int8) || pack_src_format == to_underlying(DataFormat::UInt8);
    const bool read_32b       = is_32b_format || is_fp32_dest_acc_en;
    const bool read_int8      = !(is_fp32_dest_acc_en || is_32b_format) && is_int8_format;
    const bool read_unsigned  = pack_dst_format == to_underlying(DataFormat::UInt8);
    // Round to 10 bit mantissa from fp32 dest
    const bool round_10b = (is_fp32_dest_acc_en && (pack_src_format == to_underlying(DataFormat::Float16))) ||
                           (!IS_A_FORMAT(pack_src_format) && (pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3));
    return static_cast<std::uint32_t>(read_32b) | (static_cast<std::uint32_t>(read_unsigned) << 1) | (static_cast<std::uint32_t>(read_int8) << 2) |
           (static_cast<std::uint32_t>(round_10b) << 3);
}

constexpr bool pack_is_fp8_e4m3(const std::uint32_t pack_dst_format)
{
    return (pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3);
}

// ALU_FORMAT_SPEC_REG2_Dstacc for a packer input and output format (configure_pack, reconfig_packer_data_format).
constexpr std::uint32_t pack_dstacc_format(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format)
{
    return pack_is_fp8_e4m3(pack_dst_format) ? to_underlying(DataFormat::Float16) : masked_data_format(pack_src_format);
}

// The reconfigure's global writes a kernel can drop (llk_pack_reconfig_data_format): those whose value is the same for every
// pack CB, so hw_configure already wrote it and no reconfigure can change it. dest_rd_ctrl is not one: the run-time FP32 DEST
// switch and fast tilize write it too.
constexpr std::uint32_t LLK_PACK_RECONFIG_SKIP_PAC_LF8       = 2;
constexpr std::uint32_t LLK_PACK_RECONFIG_SKIP_EXP_THRESHOLD = 4;
constexpr std::uint32_t LLK_PACK_RECONFIG_SKIP_DSTACC        = 8;
constexpr std::uint32_t LLK_PACK_RECONFIG_SKIP_STRIDES       = 16;
// Every write, the drain included: the reconfigure is a no-op (llk_pack_reconfig_skip_all).
constexpr std::uint32_t LLK_PACK_RECONFIG_SKIP_ALL = 32;
// A header with stride writes of its own (other pack programs, custom matmul and SDPA packs) specializes its tag, so the
// reconfigure keeps its stride write in that kernel.
template <int tag>
inline constexpr bool pack_strides_writer = false;
// Fast tilize writes the strides too: llk_pack_fast_tilize.h sets this.
template <DstSync>
inline constexpr bool pack_fast_tilize_linked = false;
// Some CB of the kernel unpacks to DEST (LLK_UNPACK_TO_DEST_ANY, the JIT's pack-thread define); without it, assume one does.
// The test shim defines it after this header, so llk_pack_block_tile.h sets it again; the use reads it through its template
// argument, which takes that later value.
#if defined(LLK_UNPACK_TO_DEST_ANY)
template <DstSync>
inline constexpr bool pack_reconfig_u2d = LLK_UNPACK_TO_DEST_ANY != 0;
#else
template <DstSync>
inline constexpr bool pack_reconfig_u2d = true;
#endif

// A kernel bound by the kernel config buffer (LLK_SIZE_MODE): the metal API sets it, for the calls a kernel may wrap and so
// cannot take a template argument.
template <DstSync>
inline constexpr bool pack_size_mode = false;

// Half sync, fp32 DEST and an unpack to DEST: a reconfigure whose writes the skip mask drops lets the pack start into the
// unpack to DEST of the other half, so the reconfigure helpers gate it (gate = true) unless the caller waits for the unpacker.
template <bool is_fp32_dest_acc_en, DstSync sync>
constexpr bool llk_pack_reconfig_u2d_half()
{
    return is_fp32_dest_acc_en && sync == DstSync::SyncHalf && pack_reconfig_u2d<sync>;
}

constexpr bool llk_pack_out32(const unsigned char f)
{
    return f == to_underlying(DataFormat::Float32) || f == to_underlying(DataFormat::Tf32) || f == to_underlying(DataFormat::Int32) ||
           f == to_underlying(DataFormat::UInt32);
}

// Some CB of the kernel has a block-float format.
template <std::size_t n>
constexpr bool llk_pack_any_bfp(const unsigned char (&pack_dst_format)[n])
{
    for (const unsigned char f : pack_dst_format)
    {
        if (f != to_underlying(DataFormat::Invalid) && IS_BFP_FORMAT(f))
        {
            return true;
        }
    }
    return false;
}

// 0 no pack CB is 32 bit, 1 every one is, 2 both kinds.
template <std::size_t n>
constexpr int llk_pack_out32_kind(const unsigned char (&pack_dst_format)[n])
{
    bool w = false, o = false;
    for (const unsigned char f : pack_dst_format)
    {
        if (f != to_underlying(DataFormat::Invalid))
        {
            (llk_pack_out32(f) ? w : o) = true;
        }
    }
    return w ? (o ? 2 : 1) : 0;
}

// The reconfigure's global writes that a kernel's pack formats keep constant (LLK_PACK_RECONFIG_SKIP_*), from the descriptor
// arrays (constant in the JIT): a field with the same value for every pack CB was written by hw_configure, and no reconfigure
// can change it. The strides stay where another stride writer is linked (fast tilize and untilize, pack_strides_writer tags).
template <bool is_fp32_dest_acc_en, DstSync sync, std::size_t n, bool gate = true, typename S>
constexpr std::uint32_t llk_pack_reconfig_skip(const S& pack_src_format, const unsigned char (&pack_dst_format)[n])
{
    std::uint32_t skip = LLK_PACK_RECONFIG_SKIP_PAC_LF8 | LLK_PACK_RECONFIG_SKIP_EXP_THRESHOLD | LLK_PACK_RECONFIG_SKIP_DSTACC | LLK_PACK_RECONFIG_SKIP_STRIDES;
    bool any           = false;
    std::uint32_t lf8 = 0, thr = 0, acc = 0, src0 = 0;
    for (std::size_t i = 0; i < n; i++)
    {
        const std::uint32_t src = pack_src_format[i];
        const std::uint32_t dst = pack_dst_format[i];
        if (src == to_underlying(DataFormat::Invalid) || dst == to_underlying(DataFormat::Invalid))
        {
            continue;
        }
        const std::uint32_t v_lf8 = pack_is_fp8_e4m3(dst);
        const std::uint32_t v_thr = pack_exp_threshold_bits<is_fp32_dest_acc_en>(masked_data_format(dst));
        const std::uint32_t v_acc = pack_dstacc_format(src, dst);
        if (!any)
        {
            any  = true;
            lf8  = v_lf8;
            thr  = v_thr;
            acc  = v_acc;
            src0 = masked_data_format(src);
            continue;
        }
        if (v_lf8 != lf8)
        {
            skip &= ~LLK_PACK_RECONFIG_SKIP_PAC_LF8;
        }
        if (v_thr != thr)
        {
            skip &= ~LLK_PACK_RECONFIG_SKIP_EXP_THRESHOLD;
        }
        if (v_acc != acc)
        {
            skip &= ~LLK_PACK_RECONFIG_SKIP_DSTACC;
        }
        // Default-mode strides follow the packer input format alone.
        if (masked_data_format(src) != src0)
        {
            skip &= ~LLK_PACK_RECONFIG_SKIP_STRIDES;
        }
    }
    // Half sync, fp32 DEST and an unpack to DEST: the pack restarts sooner after a skipping reconfigure and loses L1 to the unpacker.
    if (!any || (gate && llk_pack_reconfig_u2d_half<is_fp32_dest_acc_en, sync>()))
    {
        return 0;
    }
    // Fast tilize's fp32 path reconfigures through a Float16_b input with this mask, so Dstacc is written.
    if (is_fp32_dest_acc_en && pack_fast_tilize_linked<sync>)
    {
        skip &= ~LLK_PACK_RECONFIG_SKIP_DSTACC;
    }
    if (pack_fast_tilize_linked<sync> || pack_fast_untilize_linked<sync> || pack_strides_writer<1> || pack_strides_writer<2> || pack_strides_writer<3> ||
        pack_strides_writer<4> || pack_strides_writer<5> || pack_strides_writer<6> || pack_strides_writer<7> || pack_strides_writer<8> ||
        pack_strides_writer<9> || pack_strides_writer<10>)
    {
        skip &= ~LLK_PACK_RECONFIG_SKIP_STRIDES;
    }
    return skip;
}

// The whole reconfigure, its drain included, where every pack output has one format pair and face count and no partial
// face: hw_configure wrote the same values and the other writers of those fields are other programs, which the stride
// writers already exclude. Fast tilize writes a Float16_b input; the tile header register has no reader on Blackhole.
template <bool is_fp32_dest_acc_en, DstSync sync, std::size_t n, bool gate = true, typename S>
constexpr std::uint32_t llk_pack_reconfig_skip_all(
    const S& pack_src_format,
    const unsigned char (&pack_dst_format)[n],
    const std::uint8_t (&pack_tile_num_faces)[n],
    const std::uint8_t (&pack_partial_face)[n])
{
    constexpr std::uint32_t every =
        LLK_PACK_RECONFIG_SKIP_PAC_LF8 | LLK_PACK_RECONFIG_SKIP_EXP_THRESHOLD | LLK_PACK_RECONFIG_SKIP_DSTACC | LLK_PACK_RECONFIG_SKIP_STRIDES;
    const std::uint32_t skip = llk_pack_reconfig_skip<is_fp32_dest_acc_en, sync, n, gate>(pack_src_format, pack_dst_format);
    if (skip != every || pack_fast_tilize_linked<sync>)
    {
        return skip;
    }
    bool any                 = false;
    std::uint32_t src0       = 0;
    std::uint32_t dst0       = 0;
    std::uint32_t num_faces0 = 0;
    for (std::size_t i = 0; i < n; i++)
    {
        const std::uint32_t src = pack_src_format[i];
        const std::uint32_t dst = pack_dst_format[i];
        if (src == to_underlying(DataFormat::Invalid) || dst == to_underlying(DataFormat::Invalid))
        {
            continue;
        }
        if (pack_partial_face[i] != 0)
        {
            return skip;
        }
        if (!any)
        {
            any        = true;
            src0       = src;
            dst0       = dst;
            num_faces0 = pack_tile_num_faces[i];
            continue;
        }
        if (src != src0 || dst != dst0 || pack_tile_num_faces[i] != num_faces0)
        {
            return skip;
        }
    }
    return any ? (skip | LLK_PACK_RECONFIG_SKIP_ALL) : skip;
}

// A header that writes the pack address modes or X counter without the other-program mark specializes its tag.
template <int tag>
inline constexpr bool pack_init_writer = false;

// No other pack program, stride or init writer linked. dep: a template argument of the caller, so the tags are read at its
// instantiation.
template <std::uint32_t dep, DstSync sync>
constexpr bool llk_pack_only_default_program()
{
    constexpr int t = static_cast<int>(dep * 0);
    return !(pack_fast_tilize_linked<sync> || pack_fast_untilize_linked<sync> || pack_strides_writer<t + 1> || pack_strides_writer<t + 2> ||
             pack_strides_writer<t + 3> || pack_strides_writer<t + 4> || pack_strides_writer<t + 5> || pack_strides_writer<t + 6> ||
             pack_strides_writer<t + 7> || pack_strides_writer<t + 8> || pack_strides_writer<t + 9> || pack_strides_writer<t + 10> ||
             pack_init_writer<t + 1> || pack_init_writer<t + 2> || pack_init_writer<t + 3>);
}

// A block-float or 32-bit output: a 16-bit one drains too fast for the release to hide the write.
constexpr bool llk_pack_pre_format(const unsigned char f)
{
    return llk_pack_out32(f) || IS_BFP_FORMAT(f);
}

// Marks the output CB of each compute API pack_tile call as the kernel is parsed (pack.h): completing llk_pack_out_set<N> makes
// llk_pack_out_used(llk_pack_out_tag<N>) callable. N: the CB id when a constant, 0xFF for a run-time one, 0xFE for a pack_block.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend"
template <int>
struct llk_pack_out_tag
{
    friend auto llk_pack_out_used(llk_pack_out_tag);
};
#pragma GCC diagnostic pop
template <int N>
struct llk_pack_out_set
{
    friend auto llk_pack_out_used(llk_pack_out_tag<N>)
    {
        return true;
    }
};
template <int N, std::uint32_t dep, typename = decltype(llk_pack_out_used(llk_pack_out_tag<N + 0 * dep>{}))>
constexpr bool llk_pack_out_marked(int)
{
    return true;
}
template <int N, std::uint32_t dep>
constexpr bool llk_pack_out_marked(...)
{
    return false;
}
template <std::uint32_t dep, int... n>
constexpr int llk_pack_one_output_of(std::integer_sequence<int, n...>)
{
    constexpr bool m[] = {llk_pack_out_marked<n, dep>(0)...};
    int found          = -1;
    for (int i = 0; i < static_cast<int>(sizeof...(n)); i++)
    {
        if (m[i])
        {
            if (found >= 0)
            {
                return -1;
            }
            found = i;
        }
    }
    return (llk_pack_out_marked<0xFF, dep>(0) || llk_pack_out_marked<0xFE, dep>(0)) ? -1 : found;
}
// The one output CB the kernel's pack_tile calls name, or -1 (none, several, a run-time one, or a pack_block). dep: a template
// argument of the caller, so the marks of the whole kernel are read.
template <std::uint32_t dep>
constexpr int llk_pack_one_output()
{
    return llk_pack_one_output_of<dep>(std::make_integer_sequence<int, 64> {});
}

// Specialized by the metal pack API (llk_copy_kernel.h): a half-sync 16-bit DEST copy kernel with no math between unpack and pack.
template <int api>
struct llk_pack_copy_api
{
    static constexpr bool value = false;
};

// Such a kernel keeps main's late pack start: the word-form destination write and the DEST offset after the drain, as its
// pack runs while its unpacker reads the next tile (sharded identity +0.25 %).
template <std::uint32_t dep>
constexpr bool llk_pack_copy(const bool is_fp32_dest_acc_en)
{
    return !is_fp32_dest_acc_en && llk_pack_copy_api<static_cast<int>(dep * 0 + 1)>::value;
}

// Specialized where the kernel's formats are known (llk_pack_block_tile.h): whether output cb is block float or 32 bit.
template <int api>
struct llk_pack_pre_format_api
{
    static constexpr bool of(int)
    {
        return false;
    }
};

// The kernel's one output (llk_pack_one_output) is block float or 32 bit.
template <std::uint32_t dep>
constexpr bool llk_pack_pre_output()
{
    constexpr int one = llk_pack_one_output<dep>();
    return one >= 0 && llk_pack_pre_format_api<static_cast<int>(dep * 0 + 1)>::of(one);
}

// Half sync, no other pack program linked, not in size mode, one block-float or 32-bit output: the release may write the next
// destination (pack_pre_next). Its callers add 16-bit DEST: with fp32 DEST no row gained and some paid (+1.4 to +2.8).
template <DstSync sync>
constexpr bool llk_pack_pre_dst_sync()
{
    return sync == DstSync::SyncHalf && !pack_size_mode<sync> && llk_pack_only_default_program<0, sync>() &&
           llk_pack_pre_output<static_cast<std::uint32_t>(sync)>();
}

// No other writer of the default program's address modes, strides or X counter linked: hw_configure leaves the address
// modes and X counter live, and a default init writes the strides only where they differ (llk_pack_width_single, else
// pack_strides_width). Programs that mark themselves (LLK_PACK_OTHER_PROGRAM_SET) are checked at run time. dep: a template argument of the caller, so the tags are read at
// its instantiation.
template <auto dep, DstSync sync, std::size_t n, typename S>
constexpr bool llk_pack_init_live(const S& pack_src_format, const unsigned char (&pack_dst_format)[n])
{
    if (pack_fast_tilize_linked<sync> || pack_fast_untilize_linked<sync> || pack_strides_writer<1> || pack_strides_writer<2> ||
        pack_strides_writer<3> || pack_strides_writer<4> || pack_strides_writer<5> || pack_strides_writer<6> || pack_strides_writer<7> ||
        pack_strides_writer<8> || pack_strides_writer<9> || pack_strides_writer<10> || pack_init_writer<1> || pack_init_writer<2>)
    {
        return false;
    }
    for (std::size_t i = 0; i < n; i++)
    {
        if (pack_src_format[i] != to_underlying(DataFormat::Invalid) && pack_dst_format[i] != to_underlying(DataFormat::Invalid))
        {
            return true;
        }
    }
    return false;
}

// Every pack CB of one packer input width: the strides hw_configure wrote stay those of every default init.
template <std::size_t n, typename S>
constexpr bool llk_pack_width_single(const S& pack_src_format, const unsigned char (&pack_dst_format)[n])
{
    bool any            = false;
    std::uint32_t size0 = 0;
    for (std::size_t i = 0; i < n; i++)
    {
        const std::uint32_t src = pack_src_format[i];
        if (src == to_underlying(DataFormat::Invalid) || pack_dst_format[i] == to_underlying(DataFormat::Invalid))
        {
            continue;
        }
        if (!any)
        {
            any   = true;
            size0 = datum_size_in_bytes(src);
        }
        else if (datum_size_in_bytes(src) != size0)
        {
            return false;
        }
    }
    return any;
}

// llk_pack_init_live and every pack CB of one face height and face count, with no other MOP writer linked (pack_init_writer<3>,
// the contiguous block MOP): hw_configure's default MOP stays live too, so a default init of one tile writes no MOP.
template <auto dep, DstSync sync, std::size_t n, typename F, typename N, typename S>
constexpr bool llk_pack_mop_live(const S& pack_src_format, const unsigned char (&pack_dst_format)[n], const F (&face_r_dim)[n], const N (&num_faces)[n])
{
    if (!llk_pack_init_live<dep, sync>(pack_src_format, pack_dst_format) || pack_init_writer<3>)
    {
        return false;
    }
    bool any            = false;
    std::uint32_t face0 = 0, faces0 = 0;
    for (std::size_t i = 0; i < n; i++)
    {
        if (pack_src_format[i] == to_underlying(DataFormat::Invalid) || pack_dst_format[i] == to_underlying(DataFormat::Invalid))
        {
            continue;
        }
        if (!any)
        {
            any    = true;
            face0  = face_r_dim[i];
            faces0 = num_faces[i];
        }
        else if (face_r_dim[i] != face0 || num_faces[i] != faces0)
        {
            return false;
        }
    }
    return any;
}

template <bool is_fp32_dest_acc_en>
inline void set_packer_config(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t num_faces = 4, const bool partial_face = false)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    const std::uint32_t pack_output_src_format = masked_data_format(pack_src_format);
    const std::uint32_t pack_output_dst_format = masked_data_format(pack_dst_format);
    // Gasket converts Float16_b -> Float16 before the packer, so hardware in_data_format must be Float16 for Fp8 output.
    const std::uint32_t pack_hw_src_format =
        ((pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3)) ? to_underlying(DataFormat::Float16) : pack_output_src_format;

    // Set packer config
    pack_config_u config;
    for (std::uint32_t i = 0; i < 3; i++)
    {
        config.val[i] = 0;
    }

    config.f.exp_section_size =
        ((pack_output_dst_format == to_underlying(DataFormat::Lf8)) || (pack_output_dst_format == to_underlying(DataFormat::Int8)))
            ? 0
            : (partial_face ? 1 : num_faces); // set to num_faces as exp section size is not used for non-bfp formats except for lf8/int8

    config.f.uncompress      = 1;
    config.f.out_data_format = pack_output_dst_format;
    config.f.in_data_format  = pack_hw_src_format;

    reconfigure_exp_threshold<is_fp32_dest_acc_en>(pack_output_dst_format);

    // Program:
    // THCON_SEC0_REG1_Row_start_section_size = cfg_reg_array[1][0 +: 16];
    // THCON_SEC0_REG1_Exp_section_size = cfg_reg_array[1][16 +: 16];
    // This is filled with garbage, and will be set up on every pack:
    //           THCON_SEC0_REG1_L1_Dest_addr = cfg_reg_array[1][32 +: 32];
    // THCON_SEC0_REG1_Disable_zero_compress = cfg_reg_array[1][64 +: 1];
    // THCON_SEC0_REG1_Add_l1_dest_addr_offset = cfg_reg_array[1][65 +: 1];
    // THCON_SEC0_REG1_Unused0 = cfg_reg_array[1][66 +: 2];
    // THCON_SEC0_REG1_Out_data_format = cfg_reg_array[1][68 +: 4];
    // THCON_SEC0_REG1_In_data_format = cfg_reg_array[1][72 +: 4];
    // THCON_SEC0_REG1_Unused00 = cfg_reg_array[1][76 +: 4];
    // THCON_SEC0_REG1_Source_interface_selection = cfg_reg_array[1][80 +: 1];
    // THCON_SEC0_REG1_Packs_per_xy_plane = cfg_reg_array[1][81 +: 7];
    // THCON_SEC0_REG1_L1_source_addr = cfg_reg_array[1][88 +: 8];
    // THCON_SEC0_REG1_Downsample_mask = cfg_reg_array[1][96 +: 16];
    // THCON_SEC0_REG1_Downsample_shift_count = cfg_reg_array[1][112 +: 3];
    // THCON_SEC0_REG1_Read_mode = cfg_reg_array[1][115 +: 1];
    // THCON_SEC0_REG1_Exp_threshold_en = cfg_reg_array[1][116 +: 1];
    // THCON_SEC0_REG1_Unused1 = cfg_reg_array[1][117 +: 3];
    // THCON_SEC0_REG1_Exp_threshold = cfg_reg_array[1][120 +: 8];
    // for (uint i=0; i<4; i++) cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32+i]=config.val[i];
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2] = config.val[2];
    // cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32+3]=config.val[3];

    // Reset L1 accumulation flag
    reconfigure_packer_l1_acc(0);

    dest_rd_ctrl_u dest_rd_ctrl;
    dest_rd_ctrl.val = 0;

    bool is_32b_format = pack_src_format == to_underlying(DataFormat::Int32) || pack_src_format == to_underlying(DataFormat::UInt32) ||
                         pack_src_format == to_underlying(DataFormat::Float32);
    bool is_int8_format = pack_src_format == to_underlying(DataFormat::Int8) || pack_src_format == to_underlying(DataFormat::UInt8);

    dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_32b_data = is_32b_format || is_fp32_dest_acc_en;
    dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_int8     = !(is_fp32_dest_acc_en || is_32b_format) && is_int8_format;

    if (pack_dst_format == to_underlying(DataFormat::UInt8))
    {
        dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_unsigned = 1;
    }

    // Round to 10 bit mantissa from fp32 dest
    if ((is_fp32_dest_acc_en && (pack_src_format == to_underlying(DataFormat::Float16))) ||
        (!IS_A_FORMAT(pack_src_format) && (pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3)))
    {
        dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Round_10b_mant = 1;
    }
    cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32] = dest_rd_ctrl.val;

    // Save to GPR for quick data format reconfig
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    sync_regfile_write(p_gpr_pack::EXP0_SEC_SIZE_BFP);
}

// Forward declaration: defined later in this file. Needed here so that the non-dependent call
// inside `reconfig_packer_data_format`'s LLK_ASSERT is found via unqualified lookup at
// the template's first-phase name lookup (ADL cannot find it from a std::uint32_t argument).
__attribute__((noinline)) bool is_pack_reads_per_xy_plane(const std::uint32_t expected, const std::uint32_t nop_count = 10);

// impl: the body, inlined both into the out-of-line form below and into a caller that needs no out-of-line copy.
// size: LLK_SIZE_MODE, main's dest_rd_ctrl branches and the strides through hw_configure's out-of-line copy.
template <bool is_fp32_dest_acc_en, std::uint32_t skip = 0, bool size = false>
__attribute__((always_inline)) inline void reconfig_packer_data_format_impl(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const bool partial_face)
{
    llk_pack_live_forget();
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    const std::uint32_t pack_output_src_format = masked_data_format(pack_src_format);
    const std::uint32_t pack_output_dst_format = masked_data_format(pack_dst_format);
    // Gasket converts Float16_b -> Float16 before the packer, so hardware in_data_format must be Float16 for Fp8 output.
    const std::uint32_t pack_hw_src_format =
        ((pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3)) ? to_underlying(DataFormat::Float16) : pack_output_src_format;

    LLK_ASSERT(
        is_packer_to_L1_conversion_supported(static_cast<DataFormat>(pack_output_src_format), static_cast<DataFormat>(pack_output_dst_format)),
        "Unsupported packer to L1 conversion.");

    // Configure packers
    pack_config_u config;
    config.val[2] = 0; // Only need to modify word[2][15:0]

    config.f.uncompress      = 1;
    config.f.out_data_format = pack_output_dst_format;
    config.f.in_data_format  = pack_hw_src_format;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[2]), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK | p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP_LO, p_cfg::WRCFG_32b, THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2);

    LLK_ASSERT(
        is_pack_reads_per_xy_plane(1),
        "reconfig_packer_data_format: pack_reads_per_xy_plane counter must be 1 before packer reconfig "
        "(invariant violated; reduce mask should have been cleared via _llk_pack_reduce_mask_clear_). Please uncomment "
        "DEVICE_PRINT #2111 for debugging.");

    // Read_int8 is what makes an Int8/UInt8 pack read 16-bit Dest datums as 8-bit integers, so a
    // reconfig onto or away from Int8/UInt8 must write it like the full configure does: the RMW
    // covers every dest_rd_ctrl field.
    if constexpr (skip == 0 || size)
    {
        // Mask 0 (the u2d gate among them) keeps main's branches: the branch-free word costs its serial path a cycle.
        dest_rd_ctrl_u dest_rd_ctrl;
        dest_rd_ctrl.val = 0;

        bool is_32b_format = pack_src_format == to_underlying(DataFormat::Int32) || pack_src_format == to_underlying(DataFormat::UInt32) ||
                             pack_src_format == to_underlying(DataFormat::Float32);
        bool is_int8_format = pack_src_format == to_underlying(DataFormat::Int8) || pack_src_format == to_underlying(DataFormat::UInt8);

        dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_32b_data = is_32b_format || is_fp32_dest_acc_en;
        dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_int8     = !(is_fp32_dest_acc_en || is_32b_format) && is_int8_format;

        if (pack_dst_format == to_underlying(DataFormat::UInt8))
        {
            dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_unsigned = 1;
        }
        // Round to 10 bit mantissa from fp32 dest
        if ((is_fp32_dest_acc_en && (pack_src_format == to_underlying(DataFormat::Float16))) ||
            (!IS_A_FORMAT(pack_src_format) && (pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3)))
        {
            dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Round_10b_mant = 1;
        }
        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_ADDR32, PCK_DEST_RD_CTRL_Read_32b_data_SHAMT, PCK_DEST_RD_CTRL_FIELDS_MASK>(dest_rd_ctrl.val);
    }
    else
    {
        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_ADDR32, PCK_DEST_RD_CTRL_Read_32b_data_SHAMT, PCK_DEST_RD_CTRL_FIELDS_MASK>(
            pack_dest_rd_ctrl_bits<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format));
    }

    if (IS_BFP_FORMAT(pack_output_dst_format))
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG1_Exp_section_size_RMW>((partial_face ? 1 : num_faces));
    }
    else if ((pack_output_dst_format == to_underlying(DataFormat::Lf8)) || (pack_output_dst_format == to_underlying(DataFormat::Int8)))
    {
        TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC0_REG1_Row_start_section_size_ADDR32);
    }

    if constexpr (!(skip & LLK_PACK_RECONFIG_SKIP_PAC_LF8))
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pac_LF8_4b_exp_RMW>(pack_is_fp8_e4m3(pack_dst_format));
    }

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_pack::TILE_HEADER));

    if constexpr (!(skip & LLK_PACK_RECONFIG_SKIP_EXP_THRESHOLD))
    {
        reconfigure_exp_threshold<is_fp32_dest_acc_en>(pack_output_dst_format);
    }

    if constexpr (!(skip & LLK_PACK_RECONFIG_SKIP_DSTACC))
    {
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(pack_dstacc_format(pack_src_format, pack_dst_format));
    }

    // Set packer strides
    if constexpr (!(skip & LLK_PACK_RECONFIG_SKIP_STRIDES) && size)
    {
        set_packer_strides<PackMode::Default, false, true, false, is_fp32_dest_acc_en>(pack_output_src_format, tile_c_dim);
    }
    else if constexpr (!(skip & LLK_PACK_RECONFIG_SKIP_STRIDES))
    {
        set_packer_strides<PackMode::Default, true, false, false, is_fp32_dest_acc_en>(pack_output_src_format, tile_c_dim);
    }

    // Program the packer X (datum) counter. _llk_pack_init_ owns this counter ("inits own SETADCXX"),
    // but reconfig also programs it here: a reconfig is not always followed by an init, and on Blackhole
    // the value is a mode-independent constant (single row, FACE_C_DIM - 1), so it is safe to set here.
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
}

template <bool is_fp32_dest_acc_en, std::uint32_t skip = 0, bool size = false>
__attribute__((noinline)) inline void reconfig_packer_data_format(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const bool partial_face)
{
    reconfig_packer_data_format_impl<is_fp32_dest_acc_en, skip, size>(pack_src_format, pack_dst_format, tile_size, tile_c_dim, num_faces, partial_face);
}

// The packer reconfigure with its pack state store, which set_packer_strides makes in every other call.
template <bool is_fp32_dest_acc_en, std::uint32_t skip = 0, bool inl = false, bool size = false>
inline void reconfig_packer_data_format_and_state(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const bool partial_face)
{
    if constexpr (inl)
    {
        reconfig_packer_data_format_impl<is_fp32_dest_acc_en, skip, size>(pack_src_format, pack_dst_format, tile_size, tile_c_dim, num_faces, partial_face);
    }
    else
    {
        reconfig_packer_data_format<is_fp32_dest_acc_en, skip, size>(pack_src_format, pack_dst_format, tile_size, tile_c_dim, num_faces, partial_face);
    }
    if constexpr (!(skip & LLK_PACK_RECONFIG_SKIP_STRIDES))
    {
        pack_strides_width = datum_size_in_bytes(pack_src_format);
    }
    llk_pack_state_store<!(skip & LLK_PACK_RECONFIG_SKIP_STRIDES) && is_fp32_dest_acc_en && pack_strides_u2d<llk_pack_default_mode<skip>>>(0x100);
}

// The words the reconfigure leaves for one output, as set_packer_config, reconfig_packer_data_format and
// set_packer_strides<Default> write them.
constexpr std::uint32_t pack_cfg_format_word(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format)
{
    const std::uint32_t hw_src = pack_is_fp8_e4m3(pack_dst_format) ? to_underlying(DataFormat::Float16) : masked_data_format(pack_src_format);
    return (1u << THCON_SEC0_REG1_Disable_zero_compress_SHAMT) | (masked_data_format(pack_dst_format) << THCON_SEC0_REG1_Out_data_format_SHAMT) |
           (hw_src << THCON_SEC0_REG1_In_data_format_SHAMT);
}

constexpr bool pack_cfg_no_exp_section(const std::uint32_t pack_dst_format)
{
    return masked_data_format(pack_dst_format) == to_underlying(DataFormat::Lf8) || masked_data_format(pack_dst_format) == to_underlying(DataFormat::Int8);
}

constexpr std::uint32_t pack_cfg_ystride_word(const std::uint32_t pack_src_format)
{
    return (FACE_C_DIM * datum_size_in_bytes(pack_src_format)) << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT;
}

constexpr std::uint32_t pack_cfg_zwstride_word(const std::uint32_t pack_src_format)
{
    const std::uint32_t x_stride = datum_size_in_bytes(pack_src_format);
    return ((FACE_R_DIM * FACE_C_DIM * x_stride) << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT) |
           ((TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * x_stride) << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT);
}

// Per config word, the bits the kernel's CBs set differently; the other stride writers leave strides no CB has (strides_full:
// written whole, as the full reconfigure does). No ok under the unpack-to-DEST gate.
struct llk_pack_reconfig_imm_t
{
    bool ok;
    bool strides_full;
    std::uint32_t format, section, lf8, threshold, dstacc, ystride, zwstride;
};

// compute_mpwi's CB format signature: UInt16 CB 3, one index format (UInt16 or UInt32) on CBs 6 and 8-12, Float16_b CBs 0, 4, 7
template <std::size_t n>
constexpr bool llk_pack_mpwi_cbs(const unsigned char (&f)[n])
{
    if (n < 13)
    {
        return false;
    }
    constexpr std::uint32_t f16b = to_underlying(DataFormat::Float16_b);
    const std::uint32_t idx      = f[6];
    bool ok = f[3] == to_underlying(DataFormat::UInt16) && (idx == to_underlying(DataFormat::UInt16) || idx == to_underlying(DataFormat::UInt32)) &&
              f[0] == f16b && f[4] == f16b && f[7] == f16b;
    for (std::size_t i = 8; i <= 12; i++)
    {
        ok = ok && f[i] == idx;
    }
    return ok;
}

template <bool is_fp32_dest_acc_en, DstSync sync, std::size_t n, bool gate = true, typename S>
constexpr llk_pack_reconfig_imm_t llk_pack_reconfig_imm(
    const S& pack_src_format, const unsigned char (&pack_dst_format)[n], const unsigned char (&pack_num_faces)[n], const unsigned char (&pack_partial_face)[n])
{
    llk_pack_reconfig_imm_t m {false, false, 0, 0, 0, 0, 0, 0, 0};
    bool any = false;
    std::uint32_t f0 = 0, s0 = 0, l0 = 0, t0 = 0, a0 = 0, y0 = 0, z0 = 0;
    // Fast tilize and hw_cleanup leave configs no CB has: fast tilize's fp32 path reconfigures each output through a
    // Float16_b input, hw_cleanup configures Float16_b to Float16_b. Those enter the masks as extra entries after the CBs.
    constexpr std::uint32_t f16b  = to_underlying(DataFormat::Float16_b);
    const bool ftz_fp32           = is_fp32_dest_acc_en && pack_fast_tilize_linked<sync>;
    const bool cleanup            = pack_strides_writer<4>;
    const std::size_t entries     = n + (ftz_fp32 ? n : 0) + (cleanup ? 1 : 0);
    for (std::size_t e = 0; e < entries; e++)
    {
        const std::size_t i     = e < n ? e : (e < 2 * n && ftz_fp32 ? e - n : 0);
        const bool canonical    = e == entries - 1 && cleanup;
        const std::uint32_t src = canonical ? f16b : (e < n ? pack_src_format[i] : f16b);
        const std::uint32_t dst = canonical ? f16b : pack_dst_format[i];
        if (src == to_underlying(DataFormat::Invalid) || dst == to_underlying(DataFormat::Invalid) || (e >= n && !canonical && pack_src_format[i] == to_underlying(DataFormat::Invalid)))
        {
            continue;
        }
        const std::uint32_t faces   = canonical ? 4 : pack_num_faces[i];
        const std::uint32_t partial = canonical ? 0 : pack_partial_face[i];
        const std::uint32_t f = pack_cfg_format_word(src, dst);
        // Word 0 after hw_configure on this output, and after a reconfigure to it (which writes it for block-float,
        // Lf8 and Int8 outputs only).
        const std::uint32_t s_hw = pack_cfg_no_exp_section(dst) ? 0 : (partial ? 1u : faces) << THCON_SEC0_REG1_Exp_section_size_SHAMT;
        const std::uint32_t s_rc = IS_BFP_FORMAT(masked_data_format(dst)) ? faces << THCON_SEC0_REG1_Exp_section_size_SHAMT : s_hw;
        const std::uint32_t l = std::uint32_t(pack_is_fp8_e4m3(dst)) << THCON_SEC0_REG1_Pac_LF8_4b_exp_SHAMT;
        const std::uint32_t t = pack_exp_threshold_bits<is_fp32_dest_acc_en>(masked_data_format(dst));
        const std::uint32_t a = pack_dstacc_format(src, dst) << ALU_FORMAT_SPEC_REG2_Dstacc_SHAMT;
        const std::uint32_t y = pack_cfg_ystride_word(src);
        const std::uint32_t z = pack_cfg_zwstride_word(src);
        if (!any)
        {
            any = e < n;
            f0 = f, s0 = s_hw, l0 = l, t0 = t, a0 = a, y0 = y, z0 = z;
        }
        m.format |= f ^ f0;
        m.section |= (s_hw ^ s0) | (s_rc ^ s0);
        m.lf8 |= l ^ l0;
        m.threshold |= t ^ t0;
        m.dstacc |= a ^ a0;
        m.ystride |= y ^ y0;
        m.zwstride |= z ^ z0;
    }
    m.ok = any;
    // compute_mpwi keeps the full reconfigure (4x8 window +0.13 % with the immediates); its 3x3 rows lose 5-9 % with it, but the
    // window size is a kernel compile arg the LLK cannot see
    if ((gate && llk_pack_reconfig_u2d_half<is_fp32_dest_acc_en, sync>()) || ((pack_fast_tilize_linked<sync> || pack_strides_writer<4>) && llk_pack_mpwi_cbs(pack_dst_format)))
    {
        m.ok = false;
        return m;
    }
    if (pack_fast_untilize_linked<sync> || pack_strides_writer<1> || pack_strides_writer<2> || pack_strides_writer<3> || pack_strides_writer<5> ||
        pack_strides_writer<6> || pack_strides_writer<7> || pack_strides_writer<8> || pack_strides_writer<9> || pack_strides_writer<10>)
    {
        m.strides_full = true;
    }
    // Fast tilize and hw_cleanup write the strides.
    if (pack_fast_tilize_linked<sync> || pack_strides_writer<4>)
    {
        m.strides_full = true;
    }
    return m;
}

// The kernel's distinct (packer input, output, face count) triples: a run-time output's reconfigure picks its immediate
// sequence among them (llk_pack_reconfig_data_format's table path); n above the capacity means too many to list.
struct llk_pack_reconfig_profiles_t
{
    std::uint32_t n;
    std::uint8_t src[8], dst[8], faces[8];
};

template <std::size_t n, typename S>
constexpr llk_pack_reconfig_profiles_t llk_pack_reconfig_profiles(
    const S& pack_src_format, const unsigned char (&pack_dst_format)[n], const unsigned char (&pack_num_faces)[n])
{
    llk_pack_reconfig_profiles_t p {0, {}, {}, {}};
    for (std::size_t i = 0; i < n; i++)
    {
        if (pack_src_format[i] == to_underlying(DataFormat::Invalid) || pack_dst_format[i] == to_underlying(DataFormat::Invalid))
        {
            continue;
        }
        bool seen = false;
        for (std::uint32_t k = 0; k < p.n && k < 8; k++)
        {
            seen = seen || (p.src[k] == pack_src_format[i] && p.dst[k] == pack_dst_format[i] && p.faces[k] == pack_num_faces[i]);
        }
        if (!seen)
        {
            if (p.n < 8)
            {
                p.src[p.n]   = pack_src_format[i];
                p.dst[p.n]   = pack_dst_format[i];
                p.faces[p.n] = pack_num_faces[i];
            }
            p.n++;
        }
    }
    return p;
}

// The bits of mask in one config word, set to val's: one RMWCIB per byte the mask touches.
template <std::uint32_t addr, std::uint32_t mask, std::uint32_t val>
TT_ALWAYS_INLINE void pack_cfg_rmw_imm()
{
    if constexpr ((mask & 0xff) != 0)
    {
        TTI_RMWCIB0(mask & 0xff, val & mask & 0xff, addr);
    }
    if constexpr (((mask >> 8) & 0xff) != 0)
    {
        TTI_RMWCIB1((mask >> 8) & 0xff, ((val & mask) >> 8) & 0xff, addr);
    }
    if constexpr (((mask >> 16) & 0xff) != 0)
    {
        TTI_RMWCIB2((mask >> 16) & 0xff, ((val & mask) >> 16) & 0xff, addr);
    }
    if constexpr ((mask >> 24) != 0)
    {
        TTI_RMWCIB3(mask >> 24, (val & mask) >> 24, addr);
    }
}

// reconfig_packer_data_format_and_state for a constant output: only the bits the kernel's formats change, as immediates.
// dest_rd_ctrl is always written (the run-time FP32 DEST switch writes it too); the tile header GPR has no Blackhole reader.
template <
    bool is_fp32_dest_acc_en,
    bool strides_full,
    std::uint32_t format_mask,
    std::uint32_t section_mask,
    std::uint32_t lf8_mask,
    std::uint32_t threshold_mask,
    std::uint32_t dstacc_mask,
    std::uint32_t ystride_mask,
    std::uint32_t zwstride_mask,
    std::uint32_t pack_src_format,
    std::uint32_t pack_dst_format,
    std::uint32_t num_faces>
TT_ALWAYS_INLINE void reconfig_packer_data_format_imm_and_state()
{
    llk_pack_live_forget();
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        is_packer_to_L1_conversion_supported(
            static_cast<DataFormat>(masked_data_format(pack_src_format)), static_cast<DataFormat>(masked_data_format(pack_dst_format))),
        "Unsupported packer to L1 conversion.");
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK | p_stall::THCON);
    pack_cfg_rmw_imm<THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2, format_mask, pack_cfg_format_word(pack_src_format, pack_dst_format)>();
    LLK_ASSERT(
        is_pack_reads_per_xy_plane(1),
        "reconfig_packer_data_format: pack_reads_per_xy_plane counter must be 1 before packer reconfig "
        "(invariant violated; reduce mask should have been cleared via _llk_pack_reduce_mask_clear_).");
    pack_cfg_rmw_imm<
        PCK_DEST_RD_CTRL_Read_32b_data_ADDR32,
        PCK_DEST_RD_CTRL_FIELDS_MASK,
        pack_dest_rd_ctrl_bits<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format) << PCK_DEST_RD_CTRL_Read_32b_data_SHAMT>();
    if constexpr (IS_BFP_FORMAT(masked_data_format(pack_dst_format)))
    {
        pack_cfg_rmw_imm<
            THCON_SEC0_REG1_Row_start_section_size_ADDR32,
            section_mask & THCON_SEC0_REG1_Exp_section_size_MASK,
            num_faces << THCON_SEC0_REG1_Exp_section_size_SHAMT>();
    }
    else if constexpr (pack_cfg_no_exp_section(pack_dst_format))
    {
        pack_cfg_rmw_imm<THCON_SEC0_REG1_Row_start_section_size_ADDR32, section_mask, 0>();
    }
    pack_cfg_rmw_imm<THCON_SEC0_REG1_Pac_LF8_4b_exp_ADDR32, lf8_mask, std::uint32_t(pack_is_fp8_e4m3(pack_dst_format)) << THCON_SEC0_REG1_Pac_LF8_4b_exp_SHAMT>();
    pack_cfg_rmw_imm<THCON_SEC0_REG1_Exp_threshold_ADDR32, threshold_mask, pack_exp_threshold_bits<is_fp32_dest_acc_en>(masked_data_format(pack_dst_format))>();
    pack_cfg_rmw_imm<ALU_FORMAT_SPEC_REG2_Dstacc_ADDR32, dstacc_mask, pack_dstacc_format(pack_src_format, pack_dst_format) << ALU_FORMAT_SPEC_REG2_Dstacc_SHAMT>();
    if constexpr (strides_full)
    {
        set_packer_strides<PackMode::Default, true, false, false, is_fp32_dest_acc_en>(pack_src_format, TILE_C_DIM);
    }
    else
    {
        pack_cfg_rmw_imm<PCK0_ADDR_CTRL_XY_REG_0_Ystride_ADDR32, ystride_mask, pack_cfg_ystride_word(pack_src_format)>();
        pack_cfg_rmw_imm<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32, zwstride_mask, pack_cfg_zwstride_word(pack_src_format)>();
    }
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
    if constexpr (strides_full || ystride_mask != 0 || zwstride_mask != 0)
    {
        pack_strides_width = datum_size_in_bytes(pack_src_format);
    }
    llk_pack_state_store<strides_full && is_fp32_dest_acc_en && pack_strides_u2d<llk_pack_default_mode<format_mask>>>(0x100);
}

// size: LLK_SIZE_MODE, main's strides (channel 1 untouched), the one copy every pack program's strides then share.
template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool size = false>
inline void configure_pack(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t tile_c_dim                  = TILE_C_DIM,
    const std::uint32_t num_faces                   = 4,
    const bool partial_face                         = false,
    const std::uint32_t relu_config                 = 0)
{
    llk_pack_live_forget();
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        is_packer_to_L1_conversion_supported(static_cast<DataFormat>(pack_src_format & 0xF), static_cast<DataFormat>(pack_dst_format & 0xF)),
        "Unsupported packer to L1 conversion.");
    // Get pointer to registers for current state ID
    volatile std::uint32_t* cfg = get_cfg_pointer();

    // Drain the packer before any of the configuration below is rewritten, whatever the formats. The raw
    // cfg[] stores in this function and in set_packer_config are RISC MMIO, which no STALLWAIT can hold
    // back: the STALLWAIT only keeps new packer instructions out, and tensix_sync() blocks this RISC until
    // the packer has gone idle, which is what orders those stores behind a pack still in flight. Without
    // it the pack samples the new format/width mid-tile.
    TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK);
    tensix_sync();

    const std::uint32_t pack_output_src_format = masked_data_format(pack_src_format);

    set_packer_strides<pack_mode, false, true, !size>(pack_src_format, tile_c_dim);

    // NOTE: the packer X (datum) counter (SETADCXX PAC) is intentionally NOT programmed here. It is
    // owned by _llk_pack_init_, which runs after hw-configure and sets its own value per the
    // "inits own SETADCXX" contract.

    t6_mutex_acquire(mutex::REG_RMW);

    // Set Fp8 E4M3 mode for packer
    bool is_fp8_e4m3 = (pack_dst_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3);
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pac_LF8_4b_exp_RMW>(is_fp8_e4m3);

    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(is_fp8_e4m3 ? to_underlying(DataFormat::Float16) : pack_output_src_format);

    // Establish the no-override baseline for the Dstacc ALU format-select fields (ADDR32=0), consumed
    // by the packer: clear Dstacc_val/override so the base REG2_Dstacc format programmed above is
    // used. The SrcA/SrcB override fields in the same word are owned by the math thread
    // (_llk_math_hw_configure_); the two writers touch disjoint bits and rely on per-byte RMWCIB
    // atomicity, so this write does not depend on the surrounding REG_RMW mutex for cross-thread safety.
    constexpr std::uint32_t dstacc_fmt_override_mask = ALU_FORMAT_SPEC_REG_Dstacc_val_MASK | ALU_FORMAT_SPEC_REG_Dstacc_override_MASK;
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_Dstacc_val_ADDR32, 0, dstacc_fmt_override_mask>(0);

    // Config RELU
    relu_config_u hw_relu_config;
    hw_relu_config.r.STACC_RELU_ApplyRelu     = relu_config & 0xffff;
    hw_relu_config.r.STACC_RELU_ReluThreshold = (relu_config >> 16) & 0xffff;

    static_assert(STACC_RELU_ApplyRelu_ADDR32 == STACC_RELU_ReluThreshold_ADDR32, "STACC_RELU_ApplyRelu and ReluThreshold must share ADDR32 for combined RMW");

    constexpr std::uint32_t hw_relu_mask = STACC_RELU_ApplyRelu_MASK | STACC_RELU_ReluThreshold_MASK;
    cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32, 0, hw_relu_mask>(hw_relu_config.val[0]);

    t6_mutex_release(mutex::REG_RMW);

    set_packer_config<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, num_faces, partial_face);

    // PACK_COUNTERS_SEC0_pack_per_xy_plane = cfg_reg_array[3][0 +: 8];
    // PACK_COUNTERS_SEC0_pack_reads_per_xy_plane = cfg_reg_array[3][8 +: 8];
    // PACK_COUNTERS_SEC0_pack_xys_per_tile = cfg_reg_array[3][16 +: 7];
    // PACK_COUNTERS_SEC0_pack_yz_transposed = cfg_reg_array[3][23 +: 1];
    pack_counters_u pack_counters;
    pack_counters.val = 0;
    // Number of reads per face used for resetting tile position generator for edge masks.
    pack_counters.f.pack_reads_per_xy_plane = 1;
    for (std::uint32_t i = 0; i < NUM_PACKERS; i++)
    {
        cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = pack_counters.val; // disable auto last generation
    }

    pck_edge_offset_u pck_edge_offset;
    pck_edge_offset.val    = 0;
    pck_edge_offset.f.mask = 0xffff;

    cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32]                = pck_edge_offset.val;
    cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x0; // All packers use row set mapping 0, edge offset 0 mask
    cfg[PCK_EDGE_TILE_FACE_SET_SELECT_select_ADDR32]     = 0x0;

    regfile[p_gpr_pack::TILE_HEADER]     = tile_size;
    regfile[p_gpr_pack::TILE_HEADER + 1] = 0;
    regfile[p_gpr_pack::TILE_HEADER + 2] = 0;
    regfile[p_gpr_pack::TILE_HEADER + 3] = 0;
    sync_regfile_write(p_gpr_pack::TILE_HEADER + 3);
}

inline std::uint8_t get_packer_dest_offset_index()
{
    return (dest_offset_id ? p_gpr_pack::DEST_OFFSET_HI : p_gpr_pack::DEST_OFFSET_LO);
}

inline std::uint32_t get_packer_dest_offset()
{
    return (dest_offset_id ? DEST_REGISTER_HALF_SIZE : 0x0);
}

inline void flip_packer_dest_offset_id()
{
    dest_offset_id = 1 - dest_offset_id;
}

// Flip packer dest register offset to 0 or DEST_REGISTER_HALF_SIZE
// flip-flopping between two halves
template <DstSync Dst>
inline void select_packer_dest_registers()
{
    if constexpr (Dst == DstSync::SyncFull)
    {
        TTI_WRCFG(p_gpr_pack::DEST_OFFSET_LO, p_cfg::WRCFG_32b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
    }
    else
    {
        TT_WRCFG(get_packer_dest_offset_index(), p_cfg::WRCFG_32b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
    }
    TTI_DMANOP;
    TTI_DMANOP;
}

// Program packer destination addresses from GPRs
// keep_output_addr: leave OUTPUT_ADDR holding the plain address, which fast tilize advances from.
// w_invalidate: false for llk_pack_block's own packs, which mark the state instead.
template <bool keep_output_addr = true, bool w_invalidate = true>
inline void program_packer_destination_t(std::uint32_t addr)
{
    LLK_ASSERT(is_valid_L1_address(addr), "L1 address must be in valid L1 memory region");
    if constexpr (w_invalidate)
    {
        LLK_PACK_W_INVALIDATE();
    }
    // No packer-drain STALLWAIT is needed before reprogramming the destination: the pack thread issues its
    // instruction stream in order, so each tile's PACR has already started -- and latched L1_Dest_addr,
    // sampled at PACR start -- before the next call's WRCFG reprograms it, and the Last=1 PACR that ends each
    // pack MOP drains the packer and forces the next pack to re-sample L1_Dest_addr. The STALLWAIT below is
    // only the GPR-producer fence: it ensures the SETDMAREG write to OUTPUT_ADDR retires before WRCFG reads it.
    // Packs issue the byte-wise write before set_dst_write_addr, so its RMWCIBs finish during the pack's SEMWAIT; the WRCFG
    // write stays after it, as its STALLWAIT would hold SETADC.
    if constexpr (!keep_output_addr)
    {
        // Byte 3 stays as it is: the Blackhole packer reads only L1_Dest_addr[16:0] (bit 31 is Wormhole's packer-0 offset).
        TT_RMWCIB0(0xff, addr & 0xff, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
        TT_RMWCIB1(0xff, (addr >> 8) & 0xff, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
        // Same word as Data = (addr >> 16) & 0xff, reusing byte 1's 0xff00 mask.
        TT_INSN(TT_OP_RMWCIB2(0xff, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32) + ((addr >> 8) & 0xff00));
        return;
    }
    std::uint32_t new_l1_addr = (1 << 31) | addr;
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);

    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
    TTI_DMANOP; // One NOP should be enough for WRCFG due to SETDMAREG above.
}

// main's name and signature, so a kernel can still wrap it (the forms above take another name).
inline void program_packer_destination(std::uint32_t addr)
{
    program_packer_destination_t(addr);
}

// RT: If multiple contexts are used, for issue #https://github.com/tenstorrent/tt-llk-bh/issues/20
// then this function needs to be re-written
template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim, bool diagonal = false>
inline void program_packer_untilized_destination(const std::uint32_t addr, const std::uint32_t pack_dst_format)
{
    // const uint32_t block_size = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * TILE_C_DIM * (TILE_R_DIM/4));
    // constexpr uint32_t offset0 = 0;
    // const uint32_t offset1 = (1*block_size)/16;
    // const uint32_t offset2 = (2*block_size)/16;
    // const uint32_t offset3 = (3*block_size)/16;

    // TT_SETDMAREG(0, LOWER_HALFWORD(addr+offset0), 0, LO_16(p_gpr_pack::OUTPUT_ADDR+0));
    // TT_SETDMAREG(0, UPPER_HALFWORD(addr+offset0), 0, HI_16(p_gpr_pack::OUTPUT_ADDR+0));
    // TT_SETDMAREG(0, LOWER_HALFWORD(addr+offset1), 0, LO_16(p_gpr_pack::OUTPUT_ADDR+1));
    // TT_SETDMAREG(0, UPPER_HALFWORD(addr+offset1), 0, HI_16(p_gpr_pack::OUTPUT_ADDR+1));
    // TT_SETDMAREG(0, LOWER_HALFWORD(addr+offset2), 0, LO_16(p_gpr_pack::OUTPUT_ADDR+2));
    // TT_SETDMAREG(0, UPPER_HALFWORD(addr+offset2), 0, HI_16(p_gpr_pack::OUTPUT_ADDR+2));
    // TT_SETDMAREG(0, LOWER_HALFWORD(addr+offset3), 0, LO_16(p_gpr_pack::OUTPUT_ADDR+3));
    // TT_SETDMAREG(0, UPPER_HALFWORD(addr+offset3), 0, HI_16(p_gpr_pack::OUTPUT_ADDR+3));
    // TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);

    // TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
    // TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+1, 0, THCON_SEC0_REG8_L1_Dest_addr_ADDR32);
    // TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+2, 0, THCON_SEC1_REG1_L1_Dest_addr_ADDR32);
    // TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR+3, 0, THCON_SEC1_REG8_L1_Dest_addr_ADDR32);
    // TTI_NOP; TTI_NOP;
}

inline void program_packer_dest_offset_registers(std::uint32_t dest_tile_offset)
{
    TT_SETDMAREG(0, LOWER_HALFWORD(dest_tile_offset), 0, LO_16(p_gpr_pack::TEMP_TILE_OFFSET));
    TT_SETDMAREG(0, UPPER_HALFWORD(dest_tile_offset), 0, HI_16(p_gpr_pack::TEMP_TILE_OFFSET));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON | p_stall::PACK);
    TTI_WRCFG(p_gpr_pack::TEMP_TILE_OFFSET, p_cfg::WRCFG_32b, PCK0_ADDR_BASE_REG_0_Base_ADDR32);
    TTI_DMANOP;
    TTI_DMANOP;
}

// READERS FOR CONFIG STRUCTS

inline pack_config_t read_pack_config_helper(std::uint32_t reg_addr, const volatile std::uint32_t tt_reg_ptr* cfg)
{
    pack_config_u config = {.val = 0};

    config.val[0] = cfg[reg_addr];
    config.val[1] = cfg[reg_addr + 1];
    config.val[2] = cfg[reg_addr + 2];

    return config.f;
}

__attribute__((noinline)) std::array<pack_config_t, NUM_PACKERS> read_pack_config()
{
    std::array<pack_config_t, NUM_PACKERS> config_vec;

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    config_vec[0] = read_pack_config_helper(THCON_SEC0_REG1_Row_start_section_size_ADDR32, cfg);

    return config_vec;
}

inline relu_config_t read_relu_config()
{
    relu_config_u config;

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    config.val[0]                          = cfg[ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32];

    return config.r;
}

inline dest_rd_ctrl_t read_dest_rd_ctrl()
{
    dest_rd_ctrl_u dest;

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    dest.val = cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32];

    return dest.f;
}

inline pck_edge_offset_t read_pack_edge_offset_helper(std::uint32_t reg_addr, const volatile std::uint32_t tt_reg_ptr* cfg)
{
    pck_edge_offset_u edge = {.val = 0};

    edge.val = cfg[reg_addr];

    return edge.f;
}

inline std::array<pck_edge_offset_t, NUM_PACKERS> read_pack_edge_offset()
{
    std::array<pck_edge_offset_t, NUM_PACKERS> edge_vec;

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    edge_vec[0] = read_pack_edge_offset_helper(PCK_EDGE_OFFSET_SEC0_mask_ADDR32, cfg);

    return edge_vec;
}

inline pack_counters_t read_pack_counters_helper(std::uint32_t reg_addr, const volatile std::uint32_t tt_reg_ptr* cfg)
{
    pack_counters_u counters = {.val = 0};
    counters.val             = cfg[reg_addr];

    return counters.f;
}

__attribute__((noinline)) std::array<pack_counters_t, NUM_PACKERS> read_pack_counters()
{
    std::array<pack_counters_t, NUM_PACKERS> config_vec;

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    config_vec[0] = read_pack_counters_helper(PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32, cfg);

    return config_vec;
}

/**
 * Validates that all packers' config matches the expected formats.
 * On mismatch, issues DEVICE_PRINT (when enabled) and LLK_ASSERT. Typically invoked via
 * `LLK_ASSERT_BLOCK(are_packers_configured_correctly(...))` in llk_pack_tile_api.h.
 *
 * The pack_reads_per_xy_plane counter is validated separately via is_pack_reads_per_xy_plane
 * (invoked at reconfig time), since its expected value depends on whether a reduce mask is
 * currently programmed, not on the per-pack call site.
 *
 * @param pack_src_format   Expected input data format for all packers
 * @param pack_dst_format   Expected output data format for all packers
 * @param nop_count         Number of nop operations to ensure configuration writes complete (default 10)
 */
__attribute__((noinline)) void are_packers_configured_correctly(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t nop_count = 10)
{
    // Ensure configuration writes complete before subsequent operations
    tensix_sync();
    for (std::uint32_t i = 0; i < nop_count; i++)
    {
        asm volatile("nop");
    }

    static_assert(NUM_PACKERS == 1, "NUM_PACKERS must be 1");
    const pack_config_t config = read_pack_config()[0];
    // Must match set_packer_config / reconfig_packer_data_format: gasket feeds Float16 into the packer for Fp8_e4m3 L1 output.
    const std::uint32_t pack_output_src_format = masked_data_format(pack_src_format);
    const std::uint32_t pack_output_dst_format = masked_data_format(pack_dst_format);

    // Special case for Fp8_e4m3: Gasket feeds Float16 into the packer for L1 output.
    const std::uint32_t pack_hw_src_format =
        ((pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3)) ? to_underlying(DataFormat::Float16) : pack_output_src_format;

    if (config.in_data_format != pack_hw_src_format)
    {
        // DEVICE_PRINT("#1101 are_packers_configured_correctly: pack_src_format mismatch. expected: {}, actual: {}\n", pack_hw_src_format,
        // config.in_data_format);
        LLK_ASSERT(
            (config.in_data_format == pack_hw_src_format),
            "are_packers_configured_correctly: pack_src_format mismatch. Uncomment DEVICE_PRINT #1101 to inspect "
            "expected/actual.");
    }
    if (config.out_data_format != pack_output_dst_format)
    {
        // DEVICE_PRINT("#1102 are_packers_configured_correctly: pack_dst_format mismatch. expected: {}, actual: {}\n", pack_output_dst_format,
        // config.out_data_format);
        LLK_ASSERT(
            (config.out_data_format == pack_output_dst_format),
            "are_packers_configured_correctly: pack_dst_format mismatch. Uncomment DEVICE_PRINT #1102 to inspect "
            "expected/actual.");
    }
}

/**
 * Validates that the packer pack_reads_per_xy_plane counter matches the expected value.
 * Intended to be called from reconfig paths to assert the invariant: between non-reduce
 * operations the counter is 1; the reduce layer owns transitions to FACE_R_DIM via
 * `_llk_pack_reduce_mask_config_` and back to 1 via `_llk_pack_reduce_mask_clear_`.
 *
 * @param expected   Expected value of pack_reads_per_xy_plane
 * @param nop_count  Number of nop operations to ensure configuration writes complete (default 10)
 * @return true if the packer counter matches expected, false otherwise.
 */
__attribute__((noinline)) bool is_pack_reads_per_xy_plane(const std::uint32_t expected, const std::uint32_t nop_count)
{
    // Ensure configuration writes complete before reading back
    tensix_sync();
    for (std::uint32_t i = 0; i < nop_count; i++)
    {
        asm volatile("nop");
    }

    static_assert(NUM_PACKERS == 1, "NUM_PACKERS must be 1");
    const pack_counters_t counters = read_pack_counters()[0];
    if (counters.pack_reads_per_xy_plane != expected)
    {
        // Debug: print the mismatched packer counter value.
        // DEVICE_PRINT(
        //     "#2111 is_pack_reads_per_xy_plane: mismatch on packer 0 (expected={}) actual [P0={}]\n",
        //     expected,
        //     counters.pack_reads_per_xy_plane);
        return false;
    }
    return true;
}

// A one-tile pack_block ran. Stored only there, so a kernel without one folds the reads to false and keeps the gated
// 32-bit-output reconfigure.
[[maybe_unused]] static bool pack_short_block = false;

} // namespace ckernel::packer
