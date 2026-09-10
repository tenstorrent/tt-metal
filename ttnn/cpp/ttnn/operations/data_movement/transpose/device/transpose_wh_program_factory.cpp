// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_program_factory.hpp"
#include "transpose_utils.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TransposeWHProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    // Declared function-locally: this op's factories share one translation unit in the unity
    // build, so file-scope names would collide across them.
    const DFBSpecName IN0{"in0"};
    const DFBSpecName OUT{"out"};
    const DFBSpecName TILIZE{"tilize"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    const auto& input_tensor = tensor_args.input;
    const auto& input = input_tensor.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    uint32_t num_tensor_tiles = input_tensor.physical_volume() / TILE_HW;
    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    uint32_t NC = input_tensor.logical_shape()[1] * input_tensor.logical_shape()[0];
    bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;
    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    tt::DataFormat src0_dfb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_dfb_data_format);
    tt::DataFormat dst_dfb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_dfb_data_format);

    IDevice* device = input_tensor.device();

    bool fp32_dest_acc_en = src0_dfb_data_format == tt::DataFormat::Float32 ||
                            src0_dfb_data_format == tt::DataFormat::Int32 ||
                            src0_dfb_data_format == tt::DataFormat::UInt32;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, row_major ? NC : num_tensor_tiles);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    ProgramSpec spec{.name = "transpose_wh"};

    uint32_t num_input_tiles = row_major ? wt * 2 : 2;
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0,
        .entry_size = src0_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = src0_dfb_data_format,
    });

    uint32_t num_output_tiles = row_major ? ht * 2 : 2;
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT,
        .entry_size = dst_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = dst_dfb_data_format,
    });

    if (row_major) {
        // Tilize intermediate: the compute kernel both fills it (tilize) and drains it
        // (transpose), so it is self-looped onto that one kernel.
        uint32_t num_im_tiles = ht * wt;
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = TILIZE,
            .entry_size = src0_single_tile_size,
            .num_entries = num_im_tiles,
            .data_format_metadata = src0_dfb_data_format,
        });
        // The legacy factory also allocated a second intermediate (c_25, "im2") on this path,
        // carrying a TODO to remove it. No kernel this factory binds ever referenced it — the
        // row-major compute kernel uses that index only under SHARDED, which only the sharded
        // row-major factory defines — so it is not re-created here.
    }

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()});

    // ---- reader ----
    KernelSpec reader{
        .unique_id = READER,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .hw_config = create_reader_datamovement_config(device->arch()),
    };
    if (row_major) {
        reader.source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "reader_unary_transpose_wh_interleaved_start_id_rm.cpp";
        reader.compile_time_args = {
            {"Ht", ht},
            {"H_per_tile", H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT},
            {"H_per_tile_last", H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT},
            {"Wt", wt},
            {"W_size_bytes", W * input_tensor.element_size()},
            {"l1_write_offset_bytes", wt * input_tensor.element_size() * TILE_WIDTH}};
        reader.runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_hw_blocks_per_core"}};
    } else {
        reader.source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "reader_unary_transpose_wh_interleaved_start_id.cpp";
        reader.runtime_arg_schema = {
            .runtime_arg_names = {"num_tiles", "start_id", "start_ht", "start_wt", "Ht", "Wt", "HtWt"}};
    }

    // ---- writer ----
    KernelSpec writer{
        .unique_id = WRITER,
        .hw_config = create_writer_datamovement_config(device->arch()),
    };
    if (row_major) {
        writer.source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "writer_unary_transpose_wh_interleaved_start_id_rm.cpp";
        writer.dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }};
        writer.tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}};
        writer.compile_time_args = {
            {"Ht", ht},
            {"Wt", wt},
            {"W_per_tile", W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH},
            {"W_per_tile_last", W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH},
            {"H_size_bytes", H * output_tensor.element_size()},
            {"l1_read_offset_bytes", ht * output_tensor.element_size() * TILE_HEIGHT}};
        writer.runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_hw_blocks_per_core"}};
    } else {
        // Shared Metal 2.0 writer from the eltwise/unary op; its binding vocabulary
        // (dfb::out, tensor::dst) and named argument set are fixed by that kernel.
        writer.source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp";
        writer.dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }};
        writer.tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}};
        writer.runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}};
    }

    // ---- compute ----
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (row_major && (input_tensor.dtype() == DataType::UINT32 || input_tensor.dtype() == DataType::INT32)) {
        compute_defines.insert({"DST_ACCUM_MODE", "1"});
    }

    // Legacy built a ComputeConfigDescriptor directly (no TTNN ComputeKernelConfig feeding it),
    // setting only fp32_dest_acc_en and unpack_to_dest_mode; every other field kept its Metal
    // default, which ComputeGen1Config reproduces.
    ComputeGen1Config compute_hw{.enable_32_bit_dest = fp32_dest_acc_en};
    if (src0_dfb_data_format == tt::DataFormat::Float32) {
        compute_hw.unpack_modes.insert({IN0, UnpackMode::UnpackToDest});
        if (row_major) {
            compute_hw.unpack_modes.insert({TILIZE, UnpackMode::UnpackToDest});
        }
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        // Legacy left opt_level unset on a ComputeConfigDescriptor, which resolves to O3;
        // a Metal 2.0 KernelSpec defaults to O2, so the level is restated explicitly here.
        .compiler_options = {.defines = std::move(compute_defines), .opt_level = KernelBuildOptLevel::O3},
        .hw_config = compute_hw,
    };
    if (row_major) {
        compute.source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_rm_metal2.cpp";
        compute.dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = IN0,
                .accessor_name = "in0",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = OUT,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            // Self-loop: this kernel tilizes into the intermediate and transposes back out of it.
            DFBBinding{
                .dfb_spec_name = TILIZE,
                .accessor_name = "tilize",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = TILIZE,
                .accessor_name = "tilize",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }};
        compute.compile_time_args = {{"Ht", ht}, {"Wt", wt}, {"HtWt", ht * wt}};
        compute.runtime_arg_schema = {.runtime_arg_names = {"num_hw_blocks_per_core"}};
    } else {
        // Shared Metal 2.0 compute fork (the legacy copy alongside it is still bound by the
        // qkv-heads ops); its binding vocabulary (dfb::cb_in / dfb::cb_out) is fixed by that kernel.
        compute.source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_metal2.cpp";
        compute.dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = IN0,
                .accessor_name = "cb_in",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = OUT,
                .accessor_name = "cb_out",
                .endpoint_type = DFBEndpointType::PRODUCER,
            }};
        compute.runtime_arg_schema = {.runtime_arg_names = {"NHtWt"}};
    }

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(std::move(compute));

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = all_cores,
    });

    // ---- run args ----
    ProgramRunArgs run_args;
    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER};
    ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER};
    ProgramRunArgs::KernelRunArgs compute_run_args{.kernel = COMPUTE};

    auto cores = corerange_to_cores(all_cores, std::nullopt);

    if (row_major) {
        auto rm_shape = input_tensor.logical_shape();
        uint32_t rm_W = rm_shape[3], rm_H = rm_shape[2];
        uint32_t num_sticks_read = 0, num_sticks_write = 0;
        for (const auto& core : cores) {
            uint32_t num_hw_blocks_per_core;
            if (core_group_1.contains(core)) {
                num_hw_blocks_per_core = num_tiles_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_hw_blocks_per_core = num_tiles_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"start_id", num_sticks_read}, {"num_hw_blocks_per_core", num_hw_blocks_per_core}});
            AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values, core, {{"num_hw_blocks_per_core", num_hw_blocks_per_core}});
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"start_id", num_sticks_write}, {"num_hw_blocks_per_core", num_hw_blocks_per_core}});
            num_sticks_read += num_hw_blocks_per_core * rm_H;
            num_sticks_write += num_hw_blocks_per_core * rm_W;
        }
    } else {
        auto tiled_shape = input_tensor.padded_shape();
        uint32_t tiled_W = tiled_shape[3], tiled_H = tiled_shape[2];
        uint32_t Wt_t = tiled_W / TILE_WIDTH;
        uint32_t Ht_t = tiled_H / TILE_HEIGHT;
        auto HtWt = Ht_t * Wt_t;
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
            uint32_t h = num_tiles_read % Ht_t;
            uint32_t w = num_tiles_read / Ht_t % Wt_t;
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"num_tiles", num_tiles_per_core},
                 {"start_id", tt::round_down(num_tiles_read, HtWt) + (h * Wt_t) + w},
                 {"start_ht", h},
                 {"start_wt", w},
                 {"Ht", Ht_t},
                 {"Wt", Wt_t},
                 {"HtWt", HtWt}});
            AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"NHtWt", num_tiles_per_core}});
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"num_pages", num_tiles_per_core}, {"start_id", num_tiles_read}});
            num_tiles_read += num_tiles_per_core;
        }
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args.emplace(INPUT, input);
    run_args.tensor_args.emplace(OUTPUT, output);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
