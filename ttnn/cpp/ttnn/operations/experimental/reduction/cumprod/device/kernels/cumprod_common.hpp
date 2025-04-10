// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr uint32_t ONE_TILE{1};
constexpr uint32_t FIRST_TILE{0};
constexpr uint32_t WORKING_REG{0};
constexpr float ACC_START_VALUE{1.f};  // fill the acc tile with this beforehand

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
        const uint32_t& height_tiles) :
        is_input_dram{is_input_dram},
        is_output_dram{is_output_dram},
        cb_input{cb_input},
        cb_acc{cb_acc},
        cb_output{cb_output},
        batches{batches},
        channels{channels},
        width_tiles{width_tiles},
        height_tiles{height_tiles} {}
};

FORCE_INLINE constexpr CumprodCompileTimeArgs get_compile_time_args() {
    return {
        get_compile_time_arg_val(static_cast<uint32_t>(CumprodArgEnum::IS_INPUT_DRAM)),
        get_compile_time_arg_val(static_cast<uint32_t>(CumprodArgEnum::IS_OUTPUT_DRAM)),
        get_compile_time_arg_val(static_cast<uint32_t>(CumprodArgEnum::CB_INPUT)),
        get_compile_time_arg_val(static_cast<uint32_t>(CumprodArgEnum::CB_ACC)),
        get_compile_time_arg_val(static_cast<uint32_t>(CumprodArgEnum::CB_OUTPUT)),
        get_compile_time_arg_val(static_cast<uint32_t>(CumprodArgEnum::BATCHES)),
        get_compile_time_arg_val(static_cast<uint32_t>(CumprodArgEnum::CHANNELS)),
        get_compile_time_arg_val(static_cast<uint32_t>(CumprodArgEnum::WIDTH_TILES)),
        get_compile_time_arg_val(static_cast<uint32_t>(CumprodArgEnum::HEIGHT_TILES))};
}

FORCE_INLINE uint32_t select_tile(
    const uint32_t& batch,
    const uint32_t& channel,
    const uint32_t& ht,
    const uint32_t& wt,
    const CumprodCompileTimeArgs& args) {
    return args.channels * args.height_tiles * args.width_tiles * batch +
           args.height_tiles * args.width_tiles * channel + args.width_tiles * ht + wt;
}
