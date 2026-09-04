// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_tiled_program_factory.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TransposeHCTiledProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    // Declared function-locally: this op's factories share one translation unit in the unity
    // build, so file-scope names would collide across them.
    const DFBSpecName IN0{"in0"};
    const DFBSpecName SCRATCH{"scratch"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    const auto& input_tensor = tensor_args.input;
    const auto& input = input_tensor.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    uint32_t sub_tile_line_bytes = 16 * input_tensor.element_size();
    uint32_t num_tensor_tiles = input_tensor.physical_volume() / TILE_HW;

    tt::DataFormat dfb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(dfb_data_format);

    log_debug(tt::LogOp, "transpose_hc_tiled");
    log_debug(tt::LogOp, "sub_tile_line_bytes: {}", sub_tile_line_bytes);
    log_debug(tt::LogOp, "dfb_data_format: {}", dfb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);

    IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t alignment = dst_buffer->alignment();
    bool misaligned = alignment > sub_tile_line_bytes;

    ProgramSpec spec{.name = "transpose_hc_tiled"};

    uint32_t num_input_tiles = 2;
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = dfb_data_format,
    });

    if (misaligned) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = SCRATCH,
            .entry_size = alignment,
            .num_entries = 1,
            .data_format_metadata = dfb_data_format,
        });
    }

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()});

    KernelSpec::CompilerOptions::Defines reader_defines;
    if (misaligned) {
        reader_defines.insert({"MISALIGNED", "1"});
    }

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "reader_unary_transpose_hc_interleaved_partitioned.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"WT",
                  "H",
                  "CT",
                  "HW_bytes",
                  "CHW_bytes",
                  "start_id",
                  "num_tiles",
                  "batch_addr",
                  "h",
                  "htWT",
                  "ct",
                  "ctoffs",
                  "wt"}},
        .hw_config = create_reader_datamovement_config(device->arch()),
    };
    reader.compile_time_args.insert({"subtile_line_bytes", sub_tile_line_bytes});
    reader.compile_time_args.insert({"float32_dtype", dfb_data_format == tt::DataFormat::Float32 ? 1u : 0u});
    if (misaligned) {
        reader.compile_time_args.insert({"alignment", alignment});
        // Only the reader touches the scratch buffer — it stages a NOC read there and copies
        // out by hand — so it is both the producer and the consumer of that buffer.
        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = SCRATCH,
            .accessor_name = "scratch",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = SCRATCH,
            .accessor_name = "scratch",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    // Shared Metal 2.0 writer from the eltwise/unary op; its binding vocabulary
    // (dfb::out, tensor::dst) and named argument set are fixed by that kernel.
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = create_writer_datamovement_config(device->arch()),
    };

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER},
        .target_nodes = all_cores,
    });

    auto input_shape = input_tensor.padded_shape();
    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1];
    uint32_t HW = H * W;
    uint32_t HW_bytes = HW * input_tensor.element_size();
    uint32_t CHW_bytes = C * HW * input_tensor.element_size();
    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ct = C / TILE_HEIGHT;
    uint32_t CtHWt = Ct * H * Wt;
    uint32_t CtWt = Ct * Wt;

    ProgramRunArgs run_args;
    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER};
    ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER};

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t num_tiles_read = 0;
    for (const auto& core : cores) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        uint32_t h = num_tiles_read / CtWt % H;
        uint32_t ct = num_tiles_read / Wt % Ct;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"WT", Wt},
             {"H", H},
             {"CT", Ct},
             {"HW_bytes", HW_bytes},
             {"CHW_bytes", CHW_bytes},
             {"start_id", num_tiles_read},
             {"num_tiles", num_tiles_per_core},
             {"batch_addr", num_tiles_read / CtHWt * CHW_bytes},
             {"h", h},
             {"htWT", h / TILE_HEIGHT * Wt},
             {"ct", ct},
             {"ctoffs", ct * TILE_HEIGHT * HW_bytes},
             {"wt", num_tiles_read % Wt}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_pages", num_tiles_per_core}, {"start_id", num_tiles_read}});

        num_tiles_read += num_tiles_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args.emplace(INPUT, input);
    run_args.tensor_args.emplace(OUTPUT, output);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
