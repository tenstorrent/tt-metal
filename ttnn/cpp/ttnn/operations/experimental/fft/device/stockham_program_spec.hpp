// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

namespace ttnn::experimental::prim::stockham_spec {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

inline constexpr uint32_t kTileElems = 1024u;
inline constexpr uint32_t kTileBytesFp32 = 4096u;
inline constexpr uint32_t kTileBytesBf16 = 2048u;

inline const KernelSpecName READER{"reader"};
inline const KernelSpecName WRITER{"writer"};
inline const KernelSpecName COMPUTE{"compute"};

inline const TensorParamName IN_R{"in_real"};
inline const TensorParamName IN_I{"in_imag"};
inline const TensorParamName TW_R{"tw_real"};
inline const TensorParamName TW_I{"tw_imag"};
inline const TensorParamName PT_R{"post_tw_real"};
inline const TensorParamName PT_I{"post_tw_imag"};
inline const TensorParamName OUT_R{"out_real"};
inline const TensorParamName OUT_I{"out_imag"};

inline const DFBSpecName EVEN_R{"even_r"};
inline const DFBSpecName EVEN_I{"even_i"};
inline const DFBSpecName ODD_R{"odd_r"};
inline const DFBSpecName ODD_I{"odd_i"};
inline const DFBSpecName TWIDDLE_R{"twiddle_r"};
inline const DFBSpecName TWIDDLE_I{"twiddle_i"};
inline const DFBSpecName OUT0_R{"out0_r"};
inline const DFBSpecName OUT0_I{"out0_i"};
inline const DFBSpecName OUT1_R{"out1_r"};
inline const DFBSpecName OUT1_I{"out1_i"};
inline const DFBSpecName TMP_R{"tmp_r"};
inline const DFBSpecName TMP_I{"tmp_i"};
inline const DFBSpecName TW_ODD_R{"tw_odd_r"};
inline const DFBSpecName TW_ODD_I{"tw_odd_i"};
inline const DFBSpecName STATE_R{"state_r"};
inline const DFBSpecName STATE_I{"state_i"};
inline const DFBSpecName SYNC{"sync"};
inline const DFBSpecName IN_R_BF16{"in_r_bf16"};
inline const DFBSpecName IN_I_BF16{"in_i_bf16"};
inline const DFBSpecName OUT_R_BF16{"out_r_bf16"};
inline const DFBSpecName OUT_I_BF16{"out_i_bf16"};
inline const DFBSpecName POST_TW_R{"post_twiddle_r"};
inline const DFBSpecName POST_TW_I{"post_twiddle_i"};

inline Group<DataflowBufferSpec> make_dataflow_buffers(bool is_bf16, bool apply_post_twiddle) {
    const auto fp32 = [](const DFBSpecName& name, uint32_t entries) {
        return DataflowBufferSpec{
            .unique_id = name,
            .entry_size = kTileBytesFp32,
            .num_entries = entries,
            .data_format_metadata = tt::DataFormat::Float32};
    };
    Group<DataflowBufferSpec> result = {
        fp32(EVEN_R, 2), fp32(EVEN_I, 2), fp32(ODD_R, 2), fp32(ODD_I, 2),
        fp32(TWIDDLE_R, 2), fp32(TWIDDLE_I, 2), fp32(OUT0_R, 2), fp32(OUT0_I, 2),
        fp32(OUT1_R, 2), fp32(OUT1_I, 2), fp32(TMP_R, 1), fp32(TMP_I, 1),
        fp32(TW_ODD_R, 1), fp32(TW_ODD_I, 1), fp32(STATE_R, 1), fp32(STATE_I, 1),
        fp32(SYNC, 1)};
    if (is_bf16) {
        for (const auto& name : {IN_R_BF16, IN_I_BF16, OUT_R_BF16, OUT_I_BF16}) {
            result.push_back(DataflowBufferSpec{
                .unique_id = name,
                .entry_size = kTileBytesBf16,
                .num_entries = 1,
                .data_format_metadata = tt::DataFormat::Float16_b});
        }
    }
    if (apply_post_twiddle) {
        result.push_back(fp32(POST_TW_R, 1));
        result.push_back(fp32(POST_TW_I, 1));
    }
    return result;
}

inline Group<DFBBinding> reader_bindings(bool is_bf16, bool apply_post_twiddle) {
    Group<DFBBinding> result;
    const auto add = [&result](const DFBSpecName& name, const char* accessor, DFBEndpointType endpoint) {
        result.push_back(DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = endpoint});
    };
    add(EVEN_R, "even_r", DFBEndpointType::PRODUCER);
    add(EVEN_I, "even_i", DFBEndpointType::PRODUCER);
    add(ODD_R, "odd_r", DFBEndpointType::PRODUCER);
    add(ODD_I, "odd_i", DFBEndpointType::PRODUCER);
    add(TWIDDLE_R, "twiddle_r", DFBEndpointType::PRODUCER);
    add(TWIDDLE_I, "twiddle_i", DFBEndpointType::PRODUCER);
    add(OUT0_R, "out0_r", DFBEndpointType::CONSUMER);
    add(OUT0_I, "out0_i", DFBEndpointType::CONSUMER);
    add(OUT1_R, "out1_r", DFBEndpointType::CONSUMER);
    add(OUT1_I, "out1_i", DFBEndpointType::CONSUMER);
    add(STATE_R, "state_r", DFBEndpointType::PRODUCER);
    add(STATE_I, "state_i", DFBEndpointType::PRODUCER);
    add(SYNC, "sync", DFBEndpointType::PRODUCER);
    if (is_bf16) {
        for (const auto& [name, accessor] :
             {std::pair{IN_R_BF16, "in_r_bf16"}, std::pair{IN_I_BF16, "in_i_bf16"}}) {
            add(name, accessor, DFBEndpointType::PRODUCER);
            add(name, accessor, DFBEndpointType::CONSUMER);
        }
    }
    if (apply_post_twiddle) {
        add(POST_TW_R, "post_twiddle_r", DFBEndpointType::PRODUCER);
        add(POST_TW_I, "post_twiddle_i", DFBEndpointType::PRODUCER);
    }
    return result;
}

inline Group<DFBBinding> writer_bindings(bool is_bf16, bool apply_post_twiddle) {
    Group<DFBBinding> result = {
        {.dfb_spec_name = STATE_R, .accessor_name = "state_r", .endpoint_type = DFBEndpointType::CONSUMER},
        {.dfb_spec_name = STATE_I, .accessor_name = "state_i", .endpoint_type = DFBEndpointType::CONSUMER},
        {.dfb_spec_name = SYNC, .accessor_name = "sync", .endpoint_type = DFBEndpointType::CONSUMER}};
    if (apply_post_twiddle) {
        result.push_back({.dfb_spec_name = POST_TW_R, .accessor_name = "post_twiddle_r", .endpoint_type = DFBEndpointType::CONSUMER});
        result.push_back({.dfb_spec_name = POST_TW_I, .accessor_name = "post_twiddle_i", .endpoint_type = DFBEndpointType::CONSUMER});
    }
    if (is_bf16) {
        for (const auto& [name, accessor] :
             {std::pair{OUT_R_BF16, "out_r_bf16"}, std::pair{OUT_I_BF16, "out_i_bf16"}}) {
            result.push_back({.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::PRODUCER});
            result.push_back({.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::CONSUMER});
        }
    }
    return result;
}

inline KernelSpec make_compute(uint32_t log2_sub_n) {
    ComputeGen1Config gen1{.fpu_math_fidelity = MathFidelity::HiFi4, .enable_32_bit_dest = true};
    for (const auto& name : {EVEN_R, EVEN_I, ODD_R, ODD_I, TWIDDLE_R, TWIDDLE_I, TMP_R, TMP_I, TW_ODD_R, TW_ODD_I}) {
        gen1.unpack_modes.emplace(name, UnpackMode::UnpackToDest);
    }
    ComputeHardwareConfig compute_hw = std::move(gen1);
    return KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/compute/batch_fft_compute.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = {
            {.dfb_spec_name = EVEN_R, .accessor_name = "even_r", .endpoint_type = DFBEndpointType::CONSUMER},
            {.dfb_spec_name = EVEN_I, .accessor_name = "even_i", .endpoint_type = DFBEndpointType::CONSUMER},
            {.dfb_spec_name = ODD_R, .accessor_name = "odd_r", .endpoint_type = DFBEndpointType::CONSUMER},
            {.dfb_spec_name = ODD_I, .accessor_name = "odd_i", .endpoint_type = DFBEndpointType::CONSUMER},
            {.dfb_spec_name = TWIDDLE_R, .accessor_name = "twiddle_r", .endpoint_type = DFBEndpointType::CONSUMER},
            {.dfb_spec_name = TWIDDLE_I, .accessor_name = "twiddle_i", .endpoint_type = DFBEndpointType::CONSUMER},
            {.dfb_spec_name = OUT0_R, .accessor_name = "out0_r", .endpoint_type = DFBEndpointType::PRODUCER},
            {.dfb_spec_name = OUT0_I, .accessor_name = "out0_i", .endpoint_type = DFBEndpointType::PRODUCER},
            {.dfb_spec_name = OUT1_R, .accessor_name = "out1_r", .endpoint_type = DFBEndpointType::PRODUCER},
            {.dfb_spec_name = OUT1_I, .accessor_name = "out1_i", .endpoint_type = DFBEndpointType::PRODUCER},
            {.dfb_spec_name = TMP_R, .accessor_name = "tmp_r", .endpoint_type = DFBEndpointType::PRODUCER},
            {.dfb_spec_name = TMP_R, .accessor_name = "tmp_r", .endpoint_type = DFBEndpointType::CONSUMER},
            {.dfb_spec_name = TMP_I, .accessor_name = "tmp_i", .endpoint_type = DFBEndpointType::PRODUCER},
            {.dfb_spec_name = TMP_I, .accessor_name = "tmp_i", .endpoint_type = DFBEndpointType::CONSUMER},
            {.dfb_spec_name = TW_ODD_R, .accessor_name = "tw_odd_r", .endpoint_type = DFBEndpointType::PRODUCER},
            {.dfb_spec_name = TW_ODD_R, .accessor_name = "tw_odd_r", .endpoint_type = DFBEndpointType::CONSUMER},
            {.dfb_spec_name = TW_ODD_I, .accessor_name = "tw_odd_i", .endpoint_type = DFBEndpointType::PRODUCER},
            {.dfb_spec_name = TW_ODD_I, .accessor_name = "tw_odd_i", .endpoint_type = DFBEndpointType::CONSUMER}},
        .compile_time_args = {{"log2_sub_n", log2_sub_n}},
        .runtime_arg_schema = {.runtime_arg_names = {"batch_per_core"}},
        .hw_config = std::move(compute_hw)};
}

}  // namespace ttnn::experimental::prim::stockham_spec
