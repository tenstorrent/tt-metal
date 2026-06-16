// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_single_core_program_factory.hpp"

#include <filesystem>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

// Metal 2.0 program factory: builds the immutable ProgramSpec and its mutable ProgramRunArgs.
// Behavior-preserving port of the legacy ProgramDescriptor single-core tilize factory.
ttnn::device_operation::ProgramArtifacts TilizeSingleCoreProgramFactory::create_program_spec(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    CoreRange default_core({0, 0}, {0, 0});
    CoreRange core = sub_core_grids.has_value() ? corerange_to_cores(sub_core_grids.value()).at(0) : default_core;
    CoreRangeSet core_ranges{core};

    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

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

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "tilize_single_core";

    // "src0" (legacy CB c_0): input row-major sticks the reader fills and the compute tilizes.
    // "output" (legacy CB c_16): tilized output the compute produces and the writer drains.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"src0"},
            .entry_size = input_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = input_cb_data_format,
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"output"},
            .entry_size = output_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = output_cb_data_format,
        },
    };

    // Tilized reader. Reads the input tensor (binding) into the src0 DFB.
    // The legacy reader read positional RTA slots 0 (src addr -> TensorBinding), 1, 3, 4, 5, 8;
    // the unused legacy slots 2/6/7 (stick_size, num_leftover_tiles, leftover_width_in_row) are
    // dropped — the kernel never read them.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                        "reader_unary_stick_layout_split_rows_singlecore.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"src0"},
                    .accessor_name = "src0",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{
                    .tensor_parameter_name = m2::TensorParamName{"input"},
                    .accessor_name = "input",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_sticks", "num_tiles_per_block", "block_width_size", "num_full_blocks_in_row", "row_start_id"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    // Tilized writer (forked from eltwise/unary writer_unary_interleaved_start_id.cpp).
    // Drains the output DFB into the output tensor (binding).
    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_m2.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"output"},
                    .accessor_name = "output",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{
                    .tensor_parameter_name = m2::TensorParamName{"output"},
                    .accessor_name = "output",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    // Compute (forked from ttnn/cpp/ttnn/kernel/compute/tilize.cpp).
    // Consumes src0, produces output. per_core_block_cnt / per_core_block_tile_cnt stay CTAs.
    m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source =
            std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_m2.cpp"},
        .compiler_options = {},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"src0"},
                    .accessor_name = "src0",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"output"},
                    .accessor_name = "output",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args =
            {
                {"per_core_block_cnt", num_tiles / num_tiles_per_block},
                {"per_core_block_tile_cnt", num_tiles_per_block},
            },
        .hw_config =
            m2::ComputeHardwareConfig{
                .fp32_dest_acc_en = fp32_llk_acc,
            },
    };
    // Legacy set unpack_to_dest_mode[c_0]=UnpackToDestFp32 when fp32_llk_acc; preserve that
    // for the src0 DFB (the compute consumer of the fp32 input).
    if (fp32_llk_acc) {
        std::get<m2::ComputeHardwareConfig>(compute.hw_config).unpack_to_dest_mode = {
            {m2::DFBSpecName{"src0"}, UnpackToDestMode::UnpackToDestFp32}};
    }

    spec.kernels = {reader, writer, compute};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()},
    };
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "tilize_single_core",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute"}},
            .target_nodes = core_ranges,
        },
    };

    // ---- ProgramRunArgs (mutable) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    reader_run.runtime_arg_values.push_back(
        {core.start_coord,
         {{"num_sticks", num_sticks},
          {"num_tiles_per_block", num_tiles_per_block},
          {"block_width_size", block_width_size},
          {"num_full_blocks_in_row", num_full_blocks_in_row},
          {"row_start_id", 0u}}});

    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};
    writer_run.runtime_arg_values.push_back({core.start_coord, {{"num_pages", num_tiles}, {"start_id", 0u}}});

    run.kernel_run_args = {reader_run, writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"input"}, a.mesh_tensor()},
        {m2::TensorParamName{"output"}, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
