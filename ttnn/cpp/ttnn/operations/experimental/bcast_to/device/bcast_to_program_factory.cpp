// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include "bcast_to_device_operation.hpp"
#include "bcast_to_utils.hpp"

using namespace ttnn::operations::experimental::broadcast_to;

namespace ttnn::operations::experimental::broadcast_to {
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
const KernelSpecName COMPUTE{"compute"};
const DFBSpecName SRC{"src"};
const DFBSpecName DST{"dst"};
const TensorParamName INPUT{"input"};
const TensorParamName OUTPUT{"output"};

// Per-core runtime args shared by all three kernels; the writer also takes the output start tile.
const Group<std::string> RTA_NAMES{
    "start_n", "start_c", "start_t", "start_th", "start_tw", "num_tiles", "n_stride", "c_stride", "N", "C", "Ht", "Wt"};
const Group<std::string> WRITER_RTA_NAMES{
    "start_n",
    "start_c",
    "start_t",
    "start_th",
    "start_tw",
    "num_tiles",
    "n_stride",
    "c_stride",
    "N",
    "C",
    "Ht",
    "Wt",
    "start_tile_id",
};

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_shape_dims(const MeshTensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
}

void emplace_runtime_arguments(
    KernelRunArgs& reader_run_args,
    KernelRunArgs& writer_run_args,
    KernelRunArgs& compute_run_args,
    CoreCoord compute_with_storage_grid_size,
    const MeshTensor& input,
    const MeshTensor& output) {
    const auto [iN, iC, iHt, iWt] = extract_shape_dims(input);
    const auto [oN, oC, oHt, oWt] = extract_shape_dims(output);

    uint32_t num_output_tiles = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // Idle core: the kernels exit on num_tiles_per_core == 0, so every named arg stays
            // zero; the tensor bindings need no per-core value.
            for (const auto& name : RTA_NAMES) {
                reader_run_args.runtime_arg_values[name][core] = 0;
                compute_run_args.runtime_arg_values[name][core] = 0;
            }
            for (const auto& name : WRITER_RTA_NAMES) {
                writer_run_args.runtime_arg_values[name][core] = 0;
            }
            continue;
        }

        uint32_t oHtWt = oHt * oWt;
        uint32_t tiles_per_batch = oHtWt * oC;
        uint32_t start_n = start_tile_id / tiles_per_batch;
        uint32_t start_remaining = start_tile_id % tiles_per_batch;
        uint32_t start_c = start_remaining / oHtWt;
        uint32_t start_t = start_remaining % oHtWt;
        uint32_t start_th = start_t / oWt;
        uint32_t start_tw = start_t % oWt;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"start_n", start_n},
             {"start_c", start_c},
             {"start_t", start_t},
             {"start_th", start_th},
             {"start_tw", start_tw},
             {"num_tiles", num_tiles_per_core},
             {"n_stride", iHt * iWt * iC * (iN > 1)},
             {"c_stride", iHt * iWt * (iC > 1)},
             {"N", oN},
             {"C", oC},
             {"Ht", oHt},
             {"Wt", oWt}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"start_n", start_n},
             {"start_c", start_c},
             {"start_t", start_t},
             {"start_th", start_th},
             {"start_tw", start_tw},
             {"num_tiles", num_tiles_per_core},
             {"n_stride", iHt * iWt * iC * (iN > 1)},
             {"c_stride", iHt * iWt * (iC > 1)},
             {"N", oN},
             {"C", oC},
             {"Ht", oHt},
             {"Wt", oWt},
             {"start_tile_id", start_tile_id}});

        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            core,
            {{"start_n", start_n},
             {"start_c", start_c},
             {"start_t", start_t},
             {"start_th", start_th},
             {"start_tw", start_tw},
             {"num_tiles", num_tiles_per_core},
             {"n_stride", iHt * iWt * iC * (iN > 1)},
             {"c_stride", iHt * iWt * (iC > 1)},
             {"N", oN},
             {"C", oC},
             {"Ht", oHt},
             {"Wt", oWt}});

        start_tile_id += num_tiles_per_core;
    }
}
}  // namespace

ttnn::device_operation::ProgramArtifacts BcastToOperation::BcastToProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());

    uint32_t input_single_tile_size = tt::tile_size(input_data_format);

    // Device Setup
    const auto& device = input.device();

    // we parallelize the computation across the output tiles
    auto compute_with_storage_grid_size = device.compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRangeSet(CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1}));

    ProgramSpec spec;
    spec.name = "bcast_to";

    auto kernel_config = BcastToKernelConfig(operation_attributes.subtile_broadcast_type);
    // Under NONE the reader feeds the writer directly through `src` and the compute kernel is a
    // no-op, so the compute-produced `dst` buffer has no endpoints and is not declared.
    const bool uses_compute = operation_attributes.subtile_broadcast_type != SubtileBroadcastType::NONE;

    // How many tiles to store per input DFB (double buffer)
    constexpr uint32_t num_tiles_per_dfb = 2;
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SRC,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_dfb,
        .data_format_metadata = input_data_format,
    });
    if (uses_compute) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = DST,
            .entry_size = input_single_tile_size,
            .num_entries = num_tiles_per_dfb,
            .data_format_metadata = input_data_format,
        });
    }

    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT,
        .spec = input.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT,
        .spec = output.tensor_spec(),
    });

    // READER KERNEL
    KernelSpec reader{
        .unique_id = READER,
        .source = get_kernel_file_path(kernel_config.reader_kernel),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = SRC,
                    .accessor_name = "src",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT,
                    .accessor_name = "input",
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = RTA_NAMES},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    // WRITER KERNEL
    // The writer drains whichever buffer holds the finished tiles: `src` straight from the reader
    // under NONE, otherwise the compute kernel's `dst`. It names the binding `dst` either way.
    const DFBSpecName writer_dfb = (kernel_config.writer_kernel == KernelName::WriterNoBcast) ? SRC : DST;
    KernelSpec writer{
        .unique_id = WRITER,
        .source = get_kernel_file_path(kernel_config.writer_kernel),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = writer_dfb,
                    .accessor_name = "dst",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = OUTPUT,
                    .accessor_name = "output",
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = WRITER_RTA_NAMES},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // COMPUTE KERNEL
    // Enable fp32_dest_acc_en and unpack_to_dest_mode for 32-bit formats (Float32, Int32, UInt32)
    bool is_32bit_format = input_data_format == tt::DataFormat::Float32 || input_data_format == tt::DataFormat::Int32 ||
                           input_data_format == tt::DataFormat::UInt32;
    ComputeHardwareConfig compute_hw{
        .enable_32_bit_dest = is_32bit_format,
    };
    Group<DFBBinding> compute_dfb_bindings;
    if (uses_compute) {
        compute_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = SRC,
                .accessor_name = "src",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = DST,
                .accessor_name = "dst",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
        };
        if (is_32bit_format) {
            compute_hw.unpack_modes = {{SRC, UnpackMode::UnpackToDest}};
        }
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = get_kernel_file_path(kernel_config.compute_kernel),
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .runtime_arg_schema = {.runtime_arg_names = RTA_NAMES},
        .hw_config = compute_hw,
    };

    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    emplace_runtime_arguments(
        reader_run_args, writer_run_args, compute_run_args, compute_with_storage_grid_size, input, output);

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(std::move(compute));

    spec.work_units.push_back(WorkUnitSpec{
        .name = "bcast_to",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = all_device_cores,
    });

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args = {
        {INPUT, input},
        {OUTPUT, output},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}
}  // namespace ttnn::operations::experimental::broadcast_to
