// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// apply_twiddles_shared.hpp — the resource vocabulary shared by the three
// ops that run the same SFPU complex-multiply pipeline: apply_twiddles,
// apply_twiddles_xl, and complex_mul.
//
// All three bind apply_twiddles_writer.cpp and apply_twiddles_compute.cpp
// verbatim and differ only in their reader.  Those two kernels reference
// their dataflow buffers and arguments by name, so every factory has to
// declare the same names with the same roles — this header is the single
// definition of that contract, rather than three copies that can drift.
//
// Buffer layout (entry == one 32x32 tile):
//   a_r / a_i    input row tile        (fp32, 2 entries)
//   t_r / t_i    twiddle row tile      (fp32, 2 entries)
//   b_r / b_i    output row tile       (fp32, 2 entries)
//   tmp_r/tmp_i  SFPU cmul scratch     (fp32, 1 entry)
//   *_bf16       DRAM-boundary staging (bf16, 1 entry; bf16 configs only)

#pragma once

#include <cstdint>
#include <utility>

#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim::apply_tw_shared {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

inline constexpr uint32_t kTileElems = 32u * 32u;
inline constexpr uint32_t kTileBytesFp32 = kTileElems * 4u;  // 4096
inline constexpr uint32_t kTileBytesBf16 = kTileElems * 2u;  // 2048

// ── Shared resource names ────────────────────────────────────────────────
inline const DFBSpecName A_R{"a_r"};
inline const DFBSpecName A_I{"a_i"};
inline const DFBSpecName T_R{"t_r"};
inline const DFBSpecName T_I{"t_i"};
inline const DFBSpecName B_R{"b_r"};
inline const DFBSpecName B_I{"b_i"};
inline const DFBSpecName TMP_R{"tmp_r"};
inline const DFBSpecName TMP_I{"tmp_i"};
inline const DFBSpecName IN_R_BF16{"in_r_bf16"};
inline const DFBSpecName IN_I_BF16{"in_i_bf16"};
inline const DFBSpecName OUT_R_BF16{"out_r_bf16"};
inline const DFBSpecName OUT_I_BF16{"out_i_bf16"};

inline const KernelSpecName WRITER{"writer"};
inline const KernelSpecName COMPUTE{"compute"};

inline const TensorParamName OUT_R{"out_real"};
inline const TensorParamName OUT_I{"out_imag"};

// ── Dataflow buffers ─────────────────────────────────────────────────────
// The bf16 staging buffers are allocated only for bf16 configurations; the
// kernels gate their references behind the matching INPUT_BF16 /
// OUTPUT_BF16 define, so an fp32 program never names them.
inline Group<DataflowBufferSpec> make_dataflow_buffers(bool is_bf16) {
    auto fp32 = [](const DFBSpecName& id, uint32_t entries) {
        return DataflowBufferSpec{
            .unique_id = id,
            .entry_size = kTileBytesFp32,
            .num_entries = entries,
            .data_format_metadata = tt::DataFormat::Float32,
        };
    };

    Group<DataflowBufferSpec> dfbs = {
        fp32(A_R, 2),
        fp32(A_I, 2),
        fp32(T_R, 2),
        fp32(T_I, 2),
        fp32(B_R, 2),
        fp32(B_I, 2),
        fp32(TMP_R, 1),
        fp32(TMP_I, 1),
    };

    if (is_bf16) {
        for (const auto& id : {IN_R_BF16, IN_I_BF16, OUT_R_BF16, OUT_I_BF16}) {
            dfbs.push_back(DataflowBufferSpec{
                .unique_id = id,
                .entry_size = kTileBytesBf16,
                .num_entries = 1,
                .data_format_metadata = tt::DataFormat::Float16_b,
            });
        }
    }
    return dfbs;
}

// ── Reader bindings ──────────────────────────────────────────────────────
// Every reader fills the same four fp32 tiles and, for bf16 input, stages
// through the two input bf16 buffers.  It is the only kernel touching those
// staging buffers (fill then drain within one iteration), so it binds them
// as both endpoints.
inline Group<DFBBinding> reader_dfb_bindings(bool is_bf16) {
    Group<DFBBinding> bindings = {
        DFBBinding{.dfb_spec_name = A_R, .accessor_name = "a_r", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = A_I, .accessor_name = "a_i", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = T_R, .accessor_name = "t_r", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = T_I, .accessor_name = "t_i", .endpoint_type = DFBEndpointType::PRODUCER},
    };
    if (is_bf16) {
        bindings.push_back(
            DFBBinding{.dfb_spec_name = IN_R_BF16, .accessor_name = "in_r_bf16", .endpoint_type = DFBEndpointType::PRODUCER});
        bindings.push_back(
            DFBBinding{.dfb_spec_name = IN_R_BF16, .accessor_name = "in_r_bf16", .endpoint_type = DFBEndpointType::CONSUMER});
        bindings.push_back(
            DFBBinding{.dfb_spec_name = IN_I_BF16, .accessor_name = "in_i_bf16", .endpoint_type = DFBEndpointType::PRODUCER});
        bindings.push_back(
            DFBBinding{.dfb_spec_name = IN_I_BF16, .accessor_name = "in_i_bf16", .endpoint_type = DFBEndpointType::CONSUMER});
    }
    return bindings;
}

inline KernelSpec::CompilerOptions::Defines reader_defines(bool is_bf16) {
    KernelSpec::CompilerOptions::Defines defines;
    if (is_bf16) {
        defines["INPUT_BF16"] = "1";
    }
    return defines;
}

// ── Writer ───────────────────────────────────────────────────────────────
// `row_len` is the number of valid elements per row (N1 for apply_twiddles,
// P for the xl and complex_mul variants); the writer emits exactly that many
// so the tile's padding lanes never reach DRAM.
inline KernelSpec make_writer(tt::ARCH arch, uint32_t row_len, bool is_bf16) {
    Group<DFBBinding> bindings = {
        DFBBinding{.dfb_spec_name = B_R, .accessor_name = "b_r", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = B_I, .accessor_name = "b_i", .endpoint_type = DFBEndpointType::CONSUMER},
    };
    KernelSpec::CompilerOptions::Defines defines;
    if (is_bf16) {
        defines["OUTPUT_BF16"] = "1";
        bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT_R_BF16, .accessor_name = "out_r_bf16", .endpoint_type = DFBEndpointType::PRODUCER});
        bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT_R_BF16, .accessor_name = "out_r_bf16", .endpoint_type = DFBEndpointType::CONSUMER});
        bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT_I_BF16, .accessor_name = "out_i_bf16", .endpoint_type = DFBEndpointType::PRODUCER});
        bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT_I_BF16, .accessor_name = "out_i_bf16", .endpoint_type = DFBEndpointType::CONSUMER});
    }

    return KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/apply_twiddles_writer.cpp",
        .compiler_options = {.defines = defines},
        .dfb_bindings = std::move(bindings),
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = OUT_R, .accessor_name = "out_r"},
             TensorBinding{.tensor_parameter_name = OUT_I, .accessor_name = "out_i"}},
        .compile_time_args = {{"n1", row_len}},
        .runtime_arg_schema = {.runtime_arg_names = {"base_row", "num_rows"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };
}

// ── Compute ──────────────────────────────────────────────────────────────
// The scratch buffers are produced and consumed inside one iteration of the
// compute kernel itself, so it is both endpoints for them.  A 32-bit dest
// register with Float32 inputs requires the unpack mode to be stated
// outright; the legacy descriptor asked for unpack-to-dest on every fp32
// buffer, which for the consumed ones is UnpackToDest.
inline KernelSpec make_compute() {
    ComputeGen1Config gen1{
        .fpu_math_fidelity = MathFidelity::HiFi4,
        .enable_32_bit_dest = true,
    };
    for (const auto& id : {A_R, A_I, T_R, T_I, TMP_R, TMP_I}) {
        gen1.unpack_modes.emplace(id, UnpackMode::UnpackToDest);
    }
    ComputeHardwareConfig compute_hw = std::move(gen1);

    return KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/compute/apply_twiddles_compute.cpp",
        // Compute kernels default to O3 under the legacy descriptor but O2 here; keep O3.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = A_R, .accessor_name = "a_r", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = A_I, .accessor_name = "a_i", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = T_R, .accessor_name = "t_r", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = T_I, .accessor_name = "t_i", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = B_R, .accessor_name = "b_r", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = B_I, .accessor_name = "b_i", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = TMP_R, .accessor_name = "tmp_r", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = TMP_R, .accessor_name = "tmp_r", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = TMP_I, .accessor_name = "tmp_i", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = TMP_I, .accessor_name = "tmp_i", .endpoint_type = DFBEndpointType::CONSUMER}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = std::move(compute_hw),
    };
}

}  // namespace ttnn::experimental::prim::apply_tw_shared
