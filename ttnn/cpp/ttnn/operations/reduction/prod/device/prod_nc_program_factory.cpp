// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prod_nc_device_operation.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <filesystem>

namespace ttnn::prim {

using namespace tt;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;
using ttnn::device_operation::ProgramArtifacts;

ProgramArtifacts ProdNcDeviceOperation::ProdNcProgramFactory::create_program_artifacts(
    const ProdNcParams& operation_attributes, const ProdNcInputs& tensor_args, Tensor& /*tensor_return_value*/) {
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& output = tensor_args.output.mesh_tensor();
    const int64_t dim = operation_attributes.dim;

    TT_FATAL(dim == 0 || dim == 1, "Dimension ({}) must be either 0 or 1", dim);

    auto* device = tensor_args.input.device();
    const auto arch = device->arch();

    // Metal 2.0 named resource handles for the prod_nc ProgramSpec.
    const m2::DFBSpecName INPUT_DFB{"input"};    // legacy c_0
    const m2::DFBSpecName OUTPUT_DFB{"output"};  // legacy c_3
    const m2::TensorParamName INPUT{"input"};
    const m2::TensorParamName OUTPUT{"output"};
    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE_G1{"compute_g1"};
    const m2::KernelSpecName COMPUTE_G2{"compute_g2"};

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t single_tile_size = tile_size(cb_data_format);

    const auto& input_shape = input.padded_shape();
    const uint32_t tile_height = input.tensor_spec().tile().get_height();
    const uint32_t tile_width = input.tensor_spec().tile().get_width();
    const uint32_t tile_hw = input.tensor_spec().tile().get_tile_hw();

    [[maybe_unused]] const auto N = input_shape[0];
    const auto C = input_shape[1];
    const auto Ht = input_shape[2] / tile_height;
    const auto Wt = input_shape[3] / tile_width;
    TT_FATAL(Ht != 0 && Wt != 0, "Height and width in tiles must be non-zero (Ht={}, Wt={})", Ht, Wt);

    const uint32_t HtWt = Ht * Wt;
    const uint32_t CHtWt = C * Ht * Wt;
    const uint32_t num_reduce_input_tile = input_shape[dim];
    const uint32_t input_tile_offset = (dim == 0) ? CHtWt : HtWt;
    const uint32_t num_output_tiles = output.physical_volume() / tile_hw;

    log_debug(tt::LogOp, "N {} C {} Ht {} Wt {}", N, C, Ht, Wt);
    log_debug(
        tt::LogOp,
        "dim {} num_reduce_input_tile {} input_tile_offset {}, num_output_tiles {}",
        dim,
        num_reduce_input_tile,
        input_tile_offset,
        num_output_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;
    TT_FATAL(num_cores_y != 0, "Compute grid y-dimension must be non-zero");

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_output_tiles);

    validate_reduce_op_program_grid("Prod_nc", all_cores, grid, nullptr, true, {{&tensor_args.output, "output"}});

    ////////////////////////////////////////////////////////////////////////////
    //                         Dataflow buffers (legacy c_0 / c_3)
    ////////////////////////////////////////////////////////////////////////////
    constexpr uint32_t in0_t = 2;   // input
    constexpr uint32_t out0_t = 2;  // output
    m2::DataflowBufferSpec input_dfb{
        .unique_id = INPUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = in0_t,
        .data_format_metadata = cb_data_format,
    };
    m2::DataflowBufferSpec output_dfb{
        .unique_id = OUTPUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = out0_t,
        .data_format_metadata = cb_data_format,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Compute hardware config (Style B)
    ////////////////////////////////////////////////////////////////////////////
    // Enabling fp32 DEST accumulation for bf16 output forces the Wormhole HiFi3
    // workaround below, which adversely affects the accuracy of the reduction.
    const bool fp32_dest_acc_en = output.dtype() != DataType::BFLOAT16;
    // On Wormhole B0, HiFi4 must not be combined with fp32_dest_acc_en due to a hardware bug
    // (see tenstorrent/tt-metal#38306); drop to HiFi3 only on that arch. Other architectures keep HiFi4.
    const bool needs_wh_fp32_workaround = fp32_dest_acc_en && arch == tt::ARCH::WORMHOLE_B0;
    const auto math_fidelity = needs_wh_fp32_workaround ? MathFidelity::HiFi3 : MathFidelity::HiFi4;

    m2::ComputeGen1Config compute_hw{
        .fpu_math_fidelity = math_fidelity,
        // legacy math_approx_mode unset (default false) -> sfpu_precision_mode default (Precise)
        .enable_32_bit_dest = fp32_dest_acc_en,
        .double_buffer_dest = true,  // legacy dst_full_sync_en = false -> !false
    };
    // Compute consumes INPUT_DFB; with enable_32_bit_dest an explicit unpack mode is required for a
    // Float32-formatted consumed DFB. Legacy set no unpack_to_dest_mode (default -> UnpackToSrc).
    if (cb_data_format == DataFormat::Float32) {
        compute_hw.unpack_modes.insert({INPUT_DFB, UnpackMode::UnpackToSrc});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Kernels
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec reader{
        .unique_id = READER,
        .source =
            std::filesystem::path{"ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/dataflow/reader_prod_nc.cpp"},
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = INPUT_DFB, .accessor_name = "in", .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args = {{"dim", static_cast<uint32_t>(dim)}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_input_tiles", "num_output_tiles", "input_tile_offset", "start_id", "HtWt", "CHtWt"}},
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

    // One compute KernelSpec per legacy compute KernelDescriptor (per core group), preserving the
    // work-split multiplicity over disjoint node sets.
    auto make_compute = [&](const m2::KernelSpecName& unique_id) {
        return m2::KernelSpec{
            .unique_id = unique_id,
            .source =
                std::filesystem::path{"ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/compute/prod_nc.cpp"},
            // Match the legacy build: ComputeConfig defaults compute kernels to -O3 (Metal 2.0 defaults to -O2).
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {m2::DFBBinding{
                     .dfb_spec_name = INPUT_DFB, .accessor_name = "in", .endpoint_type = m2::DFBEndpointType::CONSUMER},
                 m2::DFBBinding{
                     .dfb_spec_name = OUTPUT_DFB,
                     .accessor_name = "out",
                     .endpoint_type = m2::DFBEndpointType::PRODUCER}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_input_tiles", "num_output_tiles"}},
            .hw_config = compute_hw,
        };
    };
    m2::KernelSpec compute_g1 = make_compute(COMPUTE_G1);
    const bool group_2_present = !core_group_2.ranges().empty();
    m2::KernelSpec compute_g2 = group_2_present ? make_compute(COMPUTE_G2) : m2::KernelSpec{};

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    m2::TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    m2::TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    ////////////////////////////////////////////////////////////////////////////
    //                      Per-node runtime args (node-first loop, transposed by the helper)
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelRunArgs reader_run{.kernel = READER};
    m2::KernelRunArgs writer_run{.kernel = WRITER};
    m2::KernelRunArgs compute_run_g1{.kernel = COMPUTE_G1};
    m2::KernelRunArgs compute_run_g2{.kernel = COMPUTE_G2};

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        m2::NodeCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        m2::AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"num_input_tiles", num_reduce_input_tile},
             {"num_output_tiles", num_tiles_per_core},
             {"input_tile_offset", input_tile_offset},
             {"start_id", tile_offset},
             {"HtWt", HtWt},
             {"CHtWt", CHtWt}});

        m2::AddRuntimeArgsForNode(
            writer_run.runtime_arg_values, core, {{"num_pages", num_tiles_per_core}, {"start_id", tile_offset}});

        if (core_group_1.contains(core)) {
            m2::AddRuntimeArgsForNode(
                compute_run_g1.runtime_arg_values,
                core,
                {{"num_input_tiles", num_reduce_input_tile}, {"num_output_tiles", num_tiles_per_core}});
        } else if (core_group_2.contains(core)) {
            TT_FATAL(group_2_present, "compute_g2 needs to be present");
            m2::AddRuntimeArgsForNode(
                compute_run_g2.runtime_arg_values,
                core,
                {{"num_input_tiles", num_reduce_input_tile}, {"num_output_tiles", num_tiles_per_core}});
        } else {
            TT_THROW("Core not in specified core ranges.");
        }
        tile_offset += num_tiles_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble spec + run args
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec{
        .name = "prod_nc",
        .kernels = {reader, writer, compute_g1},
        .dataflow_buffers = {input_dfb, output_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {m2::WorkUnitSpec{
            .name = "prod_nc_g1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1}},
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run, compute_run_g1};
    run_args.tensor_args = {
        {INPUT, m2::TensorArgument{input}},
        {OUTPUT, m2::TensorArgument{output}},
    };

    if (group_2_present) {
        spec.kernels.push_back(compute_g2);
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "prod_nc_g2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
        run_args.kernel_run_args.push_back(compute_run_g2);
    }

    return ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
