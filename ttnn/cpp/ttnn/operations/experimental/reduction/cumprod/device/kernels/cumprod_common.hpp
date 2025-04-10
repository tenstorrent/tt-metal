// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"

#include "dataflow_api.h"
#include "tt-metalium/tt_backend_api_types.hpp"

using namespace tt;

constexpr uint32_t ONE_TILE{1};
constexpr uint32_t FIRST_TILE{0};
constexpr uint32_t WORKING_REG{0};

union Scaler {
    float f;
    uint32_t u;

    // TODO(jbbieniekTT): finish this
    Scaler(const DataFormat& dtype) {
        switch (dtype) {
            case DataFormat::Float32:
            case DataFormat::Float16:
            case DataFormat::Bfp8:
            case DataFormat::Bfp4:
            case DataFormat::Bfp2:
            case DataFormat::Float16_b:
            case DataFormat::Bfp8_b:
            case DataFormat::Bfp4_b:
            case DataFormat::Bfp2_b:
            case DataFormat::Lf8:
            case DataFormat::Fp8_e4m3:
            case DataFormat::Tf32: f = 1.0f;
            case DataFormat::Int8:
            case DataFormat::UInt8:
            case DataFormat::UInt16:
            case DataFormat::Int32:
            case DataFormat::UInt32:
            case DataFormat::RawUInt8:
            case DataFormat::RawUInt16:
            case DataFormat::RawUInt32: u = 1;
            case DataFormat::Invalid: break;  // TODO(jbbieniekTT): TT_FATAL
        }
    }
};

enum class CumprodArgEnum : uint8_t {
    IS_INPUT_DRAM,
    IS_OUTPUT_DRAM,
    CB_INPUT,
    CB_ACC,
    CB_OUTPUT,
    BATCHES,
    CHANNELS,
    WIDTH_TILES,
    HEIGHT_TILES,
    SRC_ADDR,
    DST_ADDR,
    ARG_COUNT
};

struct CumprodCompileTimeArgs {
    const uint32_t is_input_dram;
    const uint32_t is_output_dram;
    const uint32_t cb_input;
    const uint32_t cb_acc;
    const uint32_t cb_output;
    const uint32_t batches;
    const uint32_t channels;
    const uint32_t width_tiles;
    const uint32_t height_tiles;
    const uint32_t src_addr;
    const uint32_t dst_addr;

    constexpr CumprodCompileTimeArgs(const CumprodCompileTimeArgs& args) = default;

    constexpr CumprodCompileTimeArgs(
        const uint32_t& is_input_dram,
        const uint32_t& is_output_dram,
        const uint32_t& cb_input,
        const uint32_t& cb_acc,
        const uint32_t& cb_output,
        const uint32_t& batches,
        const uint32_t& channels,
        const uint32_t& width_tiles,
        const uint32_t& height_tiles,
        const uint32_t& src_addr,
        const uint32_t& dst_addr) :
        is_input_dram{is_input_dram},
        is_output_dram{is_output_dram},
        cb_input{cb_input},
        cb_acc{cb_acc},
        cb_output{cb_output},
        batches{batches},
        channels{channels},
        width_tiles{width_tiles},
        height_tiles{height_tiles},
        src_addr{src_addr},
        dst_addr{dst_addr} {}
};

constexpr CumprodCompileTimeArgs get_compile_time_args() {
    return {
        get_compile_time_arg_val(CumprodArgEnum::IS_INPUT_DRAM),
        get_compile_time_arg_val(CumprodArgEnum::IS_OUTPUT_DRAM),
        get_compile_time_arg_val(CumprodArgEnum::CB_INPUT),
        get_compile_time_arg_val(CumprodArgEnum::CB_ACC),
        get_compile_time_arg_val(CumprodArgEnum::CB_OUTPUT),
        get_compile_time_arg_val(CumprodArgEnum::BATCHES),
        get_compile_time_arg_val(CumprodArgEnum::CHANNELS),
        get_compile_time_arg_val(CumprodArgEnum::WIDTH_TILES),
        get_compile_time_arg_val(CumprodArgEnum::HEIGHT_TILES),
        get_compile_time_arg_val(CumprodArgEnum::SRC_ADDR),
        get_compile_time_arg_val(CumprodArgEnum::DST_ADDR)};
}

uint32_t select_tile(
    const uint32_t& batch,
    const uint32_t& channel,
    const uint32_t& ht,
    const uint32_t& wt,
    const CumprodCompileTimeArgs& args) {
    return args.channels * args.height_tiles * args.width_tiles * batch +
           args.height_tiles * args.width_tiles * channel + args.width_tiles * ht + wt;
}

void for_each_tile_grouped_by_channels(
    const CumprodCompileTimeArgs& compile_time_args,
    auto&& tile_handler,  // assumes a lambda with a self-contained reference to `addr_gtor`
    std::optional<auto&&> acc_handler = std::nullopt,
    std::optional<auto&&> acc_end_handler = std::nullopt) {
    for (uint32_t b{0}; b < compile_time_args.batches; ++b) {
        for (uint32_t ht{0}; ht < compile_time_args.height_tiles; ++ht) {
            for (uint32_t wt{0}; wt < compile_time_args.width_tiles; ++wt) {
                if (acc_handler.has_value()) {
                    acc_handler.value()(compile_time_args);
                }
                for (uint32_t c{0}; c < compile_time_args.channels; ++c) {
                    tile_handler(b, c, ht, wt, compile_time_args);
                }
                if (acc_end_handler.has_value()) {
                    acc_end_handler.value()(compile_time_args);
                }
            }
        }
    }
}
