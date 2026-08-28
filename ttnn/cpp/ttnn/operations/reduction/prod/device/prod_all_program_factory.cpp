// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prod_all_device_operation.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <filesystem>

namespace ttnn::prim {

using namespace tt;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;
using ttnn::device_operation::ProgramArtifacts;

ProgramArtifacts ProdAllDeviceOperation::ProdAllProgramFactory::create_program_artifacts(
    const ProdAllParams& /*operation_attributes*/, const ProdAllInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    const auto arch = tensor_args.input.device()->arch();

    // Metal 2.0 named resource handles for the prod_all ProgramSpec.
    const m2::DFBSpecName INPUT_DFB{"input"};    // legacy c_0
    const m2::DFBSpecName OUTPUT_DFB{"output"};  // legacy c_3
    const m2::TensorParamName INPUT{"input"};
    const m2::TensorParamName OUTPUT{"output"};
    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};

    const m2::NodeCoord node{0, 0};

    const DataFormat in_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const DataFormat out_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t in_single_tile_size = tile_size(in_cb_data_format);
    const uint32_t out_single_tile_size = tile_size(out_cb_data_format);

    const uint32_t num_tiles = input.physical_volume() / input.tensor_spec().tile().get_tile_hw();
    TT_FATAL(num_tiles > 0, "Prod_all workload num_tiles must be > 0, got {}", num_tiles);

    // ------------------------------------------------------------------
    // Dataflow buffers (legacy c_0 / c_3). One DFB per legacy CB; two-entry FIFOs.
    // ------------------------------------------------------------------
    constexpr uint32_t num_input_tiles = 2;
    constexpr uint32_t num_output_tiles = 2;
    m2::DataflowBufferSpec input_dfb{
        .unique_id = INPUT_DFB,
        .entry_size = in_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = in_cb_data_format,
    };
    m2::DataflowBufferSpec output_dfb{
        .unique_id = OUTPUT_DFB,
        .entry_size = out_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = out_cb_data_format,
    };

    // ------------------------------------------------------------------
    // Compute hardware config (Style B — legacy set a Metal ComputeConfigDescriptor directly).
    // ------------------------------------------------------------------
    constexpr bool fp32_dest_acc_en = true;
    // On Wormhole B0, HiFi4 must not be combined with fp32_dest_acc_en due to a hardware bug
    // (see tenstorrent/tt-metal#38306); drop to HiFi3 only on that arch. Other architectures keep HiFi4.
    const bool needs_wh_fp32_workaround = fp32_dest_acc_en && arch == tt::ARCH::WORMHOLE_B0;
    const auto math_fidelity = needs_wh_fp32_workaround ? MathFidelity::HiFi3 : MathFidelity::HiFi4;

    m2::ComputeGen1Config compute_hw{
        .fpu_math_fidelity = math_fidelity,
        .sfpu_precision_mode = Precision::Approximate,  // legacy math_approx_mode = true
        .enable_32_bit_dest = fp32_dest_acc_en,
        .double_buffer_dest = true,  // legacy dst_full_sync_en = false -> !false
    };
    // Compute consumes INPUT_DFB; with enable_32_bit_dest an explicit unpack mode is required for a
    // Float32-formatted consumed DFB. Legacy set no unpack_to_dest_mode (default -> UnpackToSrc).
    if (in_cb_data_format == DataFormat::Float32) {
        compute_hw.unpack_modes.insert({INPUT_DFB, UnpackMode::UnpackToSrc});
    }

    // ------------------------------------------------------------------
    // Kernels.
    // ------------------------------------------------------------------
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                        "reader_unary_interleaved_start_id_metal2.cpp"},
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = INPUT_DFB, .accessor_name = "in", .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_metal2.cpp"},
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = OUTPUT_DFB, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/compute/prod_all.cpp"},
        // Match the legacy build: ComputeConfig defaults compute kernels to -O3 (Metal 2.0 defaults to -O2).
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = INPUT_DFB, .accessor_name = "in", .endpoint_type = m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{
                 .dfb_spec_name = OUTPUT_DFB, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .compile_time_args = {{"num_tiles", num_tiles}},
        .hw_config = compute_hw,
    };

    // ------------------------------------------------------------------
    // Tensor parameters.
    // ------------------------------------------------------------------
    m2::TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    m2::TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ------------------------------------------------------------------
    // Program spec.
    // ------------------------------------------------------------------
    m2::ProgramSpec spec{
        .name = "prod_all",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {input_dfb, output_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}},
    };

    // ------------------------------------------------------------------
    // Per-execution run args (single node). Reader reads the whole tensor; writer emits one tile.
    // ------------------------------------------------------------------
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        m2::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(node, {{"num_pages", num_tiles}, {"start_id", 0u}})},
        m2::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(node, {{"num_pages", 1u}, {"start_id", 0u}})},
        m2::KernelRunArgs{.kernel = COMPUTE},
    };
    run_args.tensor_args = {
        {INPUT, m2::TensorArgument{input}},
        {OUTPUT, m2::TensorArgument{output}},
    };

    return ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
