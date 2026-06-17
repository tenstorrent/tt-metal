// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_single_core_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts TilizeSingleCoreProgramFactory::create_program_artifacts(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    const auto& input_tensor = a.mesh_tensor();
    const auto& output_tensor = output.mesh_tensor();

    // Metal 2.0 named resource handles.
    const DFBSpecName SRC_DFB{"src"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL{"compute"};

    CoreRange default_core({0, 0}, {0, 0});
    CoreRange core = sub_core_grids.has_value() ? corerange_to_cores(sub_core_grids.value()).at(0) : default_core;
    CoreRangeSet core_ranges{core};
    const NodeCoord node = core.start_coord;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    uint32_t num_tiles = a.physical_volume() / TILE_HW;

    auto width = a.padded_shape()[-1];
    uint32_t stick_s = width;
    uint32_t num_sticks = a.physical_volume() / width;

    uint32_t num_tiles_in_row = stick_s / TILE_WIDTH;
    uint32_t num_tiles_per_block = 1;

    if (!operation_attributes.use_low_perf) {
        // Ensure we don't intrude into storage space
        uint32_t max_l1_size =
            (a.device()->l1_size_per_core() / 2) - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);
        uint32_t max_tiles = max_l1_size / (input_single_tile_size + output_single_tile_size);  // 2 CBs
        // Currently need the number of tiles in a row to be divisible by tiles in a block
        if (num_tiles_in_row <= max_tiles) {
            num_tiles_per_block = num_tiles_in_row;
        } else {
            for (uint32_t n_t = max_tiles; n_t > 0; n_t--) {
                if (num_tiles_in_row % n_t == 0) {
                    num_tiles_per_block = n_t;
                    break;
                }
            }
        }
    }

    uint32_t block_width_size = num_tiles_per_block * TILE_WIDTH * a.element_size();
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;

    const uint32_t num_input_tiles = num_tiles_per_block;
    const uint32_t num_output_tiles = num_tiles_per_block;

    // ------------------------------------------------------------------------
    // Dataflow buffers (formerly CB c_0 / c_16). One DFB per legacy CB.
    // ------------------------------------------------------------------------
    DataflowBufferSpec src_dfb_spec{
        .unique_id = SRC_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    // ------------------------------------------------------------------------
    // Tensor parameters (Case 1): legacy buffer-address RTA + TensorAccessorArgs
    // plumbing collapse to a binding.
    // ------------------------------------------------------------------------
    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_tensor.tensor_spec()};

    // ------------------------------------------------------------------------
    // Kernels.
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                        "reader_unary_stick_layout_split_rows_singlecore.cpp"},
        .dfb_bindings = {ProducerOf(SRC_DFB, "in")},
        .tensor_bindings = {{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks", "num_tiles_per_block", "block_width_size", "num_full_blocks_in_row", "start_stick_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_metal2.cpp"},
        .dfb_bindings = {ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    ComputeHardwareConfig::UnpackToDestModes unpack_to_dest_modes;
    if (fp32_llk_acc && input_cb_data_format == tt::DataFormat::Float32) {
        unpack_to_dest_modes.insert({SRC_DFB, UnpackToDestMode::UnpackToDestFp32});
    }

    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_compute_metal2.cpp"},
        .dfb_bindings = {ConsumerOf(SRC_DFB, "in"), ProducerOf(OUT_DFB, "out")},
        .compile_time_args =
            {{"per_core_block_cnt", num_tiles / num_tiles_per_block}, {"per_core_block_tile_cnt", num_tiles_per_block}},
        .hw_config =
            ComputeHardwareConfig{
                .fp32_dest_acc_en = fp32_llk_acc,
                .unpack_to_dest_mode = unpack_to_dest_modes,
            },
    };

    // ------------------------------------------------------------------------
    // Per-node runtime arg values (single core).
    // ------------------------------------------------------------------------
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    reader_run.runtime_arg_values.push_back(
        {node,
         {{"num_sticks", num_sticks},
          {"num_tiles_per_block", num_tiles_per_block},
          {"block_width_size", block_width_size},
          {"num_full_blocks_in_row", num_full_blocks_in_row},
          {"start_stick_id", 0u}}});

    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    writer_run.runtime_arg_values.push_back({node, {{"num_pages", num_tiles}, {"start_id", 0u}}});

    WorkUnitSpec wu{
        .name = "tilize_single_core",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = core_ranges,
    };

    ProgramSpec spec{
        .name = "tilize_single_core",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src_dfb_spec, out_dfb_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor)}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output_tensor)}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
