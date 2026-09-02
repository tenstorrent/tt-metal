// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_single_core_program_factory.hpp"
#include "ttnn/operations/data_movement/tilize/device/tilize_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TilizeSingleCoreProgramFactory::create_program_artifacts(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;
    const uint32_t tile_width = operation_attributes.tile.get_width();
    const uint32_t tile_height = operation_attributes.tile.get_height();
    const uint32_t tile_hw = operation_attributes.tile.get_tile_hw();

    CoreRange default_core({0, 0}, {0, 0});
    CoreRange core = sub_core_grids.has_value() ? corerange_to_cores(sub_core_grids.value()).at(0) : default_core;
    CoreRangeSet core_ranges{core};

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = operation_attributes.tile.get_tile_size(input_data_format);

    tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = operation_attributes.tile.get_tile_size(output_data_format);

    // UInt8 requires fp32 dest acc on Blackhole: hardware promotes 8-bit integers to 32-bit in
    // dest but keeps them as integers (not float), so the output DFB stays as UInt8 (not Float32).
    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B ||
                        a.dtype() == DataType::UINT8;

    uint32_t num_tiles = a.physical_volume() / tile_hw;

    auto width = a.padded_shape()[-1];
    uint32_t stick_s = width;
    uint32_t num_sticks = a.physical_volume() / width;

    uint32_t num_tiles_in_row = stick_s / tile_width;
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

    uint32_t block_width_size = num_tiles_per_block * tile_width * a.element_size();
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;

    const uint32_t num_input_tiles = num_tiles_per_block;
    const uint32_t num_output_tiles = num_tiles_per_block;

    // ---- Metal 2.0 spec resource names (function-local) ----
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName OUTPUT_DFB{"output"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    auto* device = a.device();

    // ---- Dataflow buffers (local staging) ----
    DataflowBufferSpec input_dfb{
        .unique_id = INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_data_format,
        .tile_format_metadata = operation_attributes.tile,
    };
    DataflowBufferSpec output_dfb{
        .unique_id = OUTPUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_data_format,
        .tile_format_metadata = operation_attributes.tile,
    };

    // ---- Tensor parameters ----
    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Kernels ----
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
            "reader_unary_stick_layout_split_rows_singlecore.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = INPUT_DFB,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = INPUT,
            .accessor_name = "src",
        }},
        .compile_time_args = {{"tile_height", tile_height}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks", "num_tiles_per_block", "block_width_size", "num_full_blocks_in_row", "start_stick_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUTPUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = OUTPUT,
            .accessor_name = "dst",
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    ComputeGen1Config compute_cfg;
    compute_cfg.enable_32_bit_dest = fp32_llk_acc;
    // UInt8 uses 32-bit dest as integer (not float): do not enable FP32 unpack-to-dest mode.
    if (fp32_llk_acc && a.dtype() != DataType::UINT8) {
        compute_cfg.unpack_modes.emplace(INPUT_DFB, UnpackMode::UnpackToDest);
    }
    // Gen2 (Quasar) hardware config. A KernelSpec's hw_config holds exactly one generation, so a
    // Gen1-only config cannot run on Quasar. Mirror the resolved Gen1 fields into a Gen2 config on
    // Quasar; WH/BH keep the Gen1 config untouched (this branch is not taken there).
    ComputeHardwareConfig compute_hw = compute_cfg;
    if (device->arch() == tt::ARCH::QUASAR) {
        ComputeGen2Config compute_cfg_gen2;
        compute_cfg_gen2.enable_32_bit_dest = compute_cfg.enable_32_bit_dest;
        // TODO(#52269): Quasar unpack_modes are copied from Gen1 and not yet optimized for Quasar.
        compute_cfg_gen2.unpack_modes = compute_cfg.unpack_modes;
        compute_hw = compute_cfg_gen2;
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_DFB,
                 .accessor_name = "in",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = OUTPUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .compile_time_args =
            {{"per_core_block_cnt", num_tiles / num_tiles_per_block}, {"per_core_block_tile_cnt", num_tiles_per_block}},
        .hw_config = std::move(compute_hw),
    };

    ProgramSpec spec{
        .name = "tilize_single_core",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {input_dfb, output_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "tilize",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = core_ranges,
        }},
    };

    // ---- Run args ----
    const NodeCoord node = core.start_coord;
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node,
                {{"num_sticks", num_sticks},
                 {"num_tiles_per_block", num_tiles_per_block},
                 {"block_width_size", block_width_size},
                 {"num_full_blocks_in_row", num_full_blocks_in_row},
                 {"start_stick_id", uint32_t{0}}}),
        },
        KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values =
                MakeRuntimeArgsForSingleNode(node, {{"num_pages", num_tiles}, {"start_id", uint32_t{0}}}),
        },
    };
    run_args.tensor_args = {{INPUT, a.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs TilizeSingleCoreProgramFactory::override_runtime_arguments(
    const TilizeParams& /*operation_attributes*/,
    const TilizeInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Every shape-derived arg is baked; only the input/output buffer addresses move on a cache hit.
    // On the custom concept the framework refreshes nothing on our behalf, so the two tensor bindings
    // are re-supplied here (this replaces the legacy patch_tilize_kernel_slot0 slot-0 re-point).
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    ProgramRunArgs params;
    params.tensor_args = {{INPUT, tensor_args.input_tensor.mesh_tensor()}, {OUTPUT, tensor_return_value.mesh_tensor()}};
    return params;
}

}  // namespace ttnn::prim
