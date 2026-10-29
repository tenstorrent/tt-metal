// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_device_operation.hpp"

#include <algorithm>
#include <numeric>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

constexpr const char* kReaderSingle =
    "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_single_core.cpp";
constexpr const char* kWriterSingle =
    "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_single_core.cpp";
constexpr const char* kReaderMulti =
    "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_multi_core.cpp";
constexpr const char* kWriterMulti =
    "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_multi_core.cpp";
constexpr const char* kReaderRmSingle =
    "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/"
    "gather_reader_rm_single_row_single_core.cpp";
constexpr const char* kWriterRmSingle =
    "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/"
    "gather_writer_rm_single_row_single_core.cpp";
constexpr const char* kReaderRmMulti =
    "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/"
    "gather_reader_rm_single_row_multi_core.cpp";
constexpr const char* kWriterRmMulti =
    "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/"
    "gather_writer_rm_single_row_multi_core.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts GatherDeviceOperation::SingleRowSingleCore::create_program_artifacts(
    const GatherParams& attributes, const GatherInputs& tensor_args, Tensor& output_tensor) {
    const tt::DataFormat input_tensor_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat input_index_tensor_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_index_tensor.dtype());
    const tt::DataFormat output_tensor_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_dfb_data_format);
    const uint32_t input_index_tensor_tile_size = tile_size(input_index_tensor_dfb_data_format);
    const uint32_t output_tensor_tile_size = tile_size(output_tensor_dfb_data_format);

    const auto& input_mesh_tensor = tensor_args.input_tensor.mesh_tensor();
    const auto& input_index_mesh_tensor = tensor_args.input_index_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();

    auto* input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto* input_index_tensor_buffer = tensor_args.input_index_tensor.buffer();
    auto* output_tensor_buffer = output_tensor.buffer();

    const bool input_tensor_is_dram = input_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool input_index_tensor_is_dram = input_index_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool output_tensor_is_dram = output_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();

    const auto input_index_shape = tensor_args.input_index_tensor.padded_shape();
    const auto input_shape = tensor_args.input_tensor.padded_shape();
    const uint32_t Ht = (input_index_shape[0] * input_index_shape[1] * input_index_shape[2]) / tile_height;
    const uint32_t Wt_input = input_shape[3] / tile_width;
    const uint32_t Wt_index = input_index_shape[3] / tile_width;

    // Calculate the number of cores available for computation
    auto* device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t max_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    // Create core grid
    CoreRangeSet core_grid =
        tt::tt_metal::num_cores_to_corerangeset(max_number_of_cores, compute_with_storage_grid_size, true);
    // Override core grid if sub_core_grids is provided in operation attributes
    if (attributes.sub_core_grids.has_value()) {
        core_grid = attributes.sub_core_grids.value();
    }

    const auto
        [total_number_of_cores,       // number of cores utilized
         core_range,                  // set of all cores used
         core_group_1,                // Primary core group
         core_group_2,                // Secondary core group
         num_tiles_per_core_group_1,  // Number of tiles each core in the primary group processes
         num_tiles_per_core_group_2   // Number of tiles each core in the secondary group processes
    ] = tt::tt_metal::split_work_to_cores(core_grid, Ht, true);
    const auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName INPUT_TENSOR_DFB{"input_tensor"};
    const DFBSpecName INPUT_INDEX_TENSOR_DFB{"input_index_tensor"};
    const DFBSpecName OUTPUT_TENSOR_DFB{"output_tensor"};
    const TensorParamName INPUT_TENSOR{"input_tensor"};
    const TensorParamName INPUT_INDEX_TENSOR{"input_index_tensor"};
    const TensorParamName OUTPUT_TENSOR{"output_tensor"};

    ProgramSpec spec;
    spec.name = "gather_single_row_single_core";

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_TENSOR_DFB,
        .entry_size = input_tensor_tile_size,
        .num_entries = Wt_input,
        .data_format_metadata = input_tensor_dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_INDEX_TENSOR_DFB,
        .entry_size = input_index_tensor_tile_size,
        .num_entries = 1,
        .data_format_metadata = input_index_tensor_dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUTPUT_TENSOR_DFB,
        .entry_size = output_tensor_tile_size,
        .num_entries = 1,
        .data_format_metadata = output_tensor_dfb_data_format,
    });

    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT_TENSOR,
        .spec = input_mesh_tensor.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT_INDEX_TENSOR,
        .spec = input_index_mesh_tensor.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_TENSOR,
        .spec = output_mesh_tensor.tensor_spec(),
    });

    // The reader is the only kernel that touches the index DFB: it fills it from memory and then
    // reads it, so it is bound as both producer and consumer.
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = kReaderSingle,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_TENSOR_DFB,
                    .accessor_name = "input_tensor",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT_INDEX_TENSOR_DFB,
                    .accessor_name = "input_index_tensor",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT_INDEX_TENSOR_DFB,
                    .accessor_name = "input_index_tensor",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_TENSOR_DFB,
                    .accessor_name = "output_tensor",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT_INDEX_TENSOR,
                    .accessor_name = "input_index_tensor",
                },
            },
        .compile_time_args =
            {
                {"input_index_tensor_is_dram", static_cast<uint32_t>(input_index_tensor_is_dram)},
                {"Ht", Ht},
                {"Wt_input", Wt_input},
                {"Wt_index", Wt_index},
                {"total_number_of_cores", total_number_of_cores},
                {"compute_with_storage_grid_size_x", static_cast<uint32_t>(compute_with_storage_grid_size.x)},
                {"compute_with_storage_grid_size_y", static_cast<uint32_t>(compute_with_storage_grid_size.y)},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"core_loop_count", "tile_width", "tile_height", "core_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = kWriterSingle,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_TENSOR_DFB,
                    .accessor_name = "input_tensor",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_TENSOR_DFB,
                    .accessor_name = "output_tensor",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT_TENSOR,
                    .accessor_name = "input_tensor",
                },
                TensorBinding{
                    .tensor_parameter_name = OUTPUT_TENSOR,
                    .accessor_name = "output_tensor",
                },
            },
        .compile_time_args =
            {
                {"input_tensor_is_dram", static_cast<uint32_t>(input_tensor_is_dram)},
                {"output_tensor_is_dram", static_cast<uint32_t>(output_tensor_is_dram)},
                {"Ht", Ht},
                {"Wt_input", Wt_input},
                {"Wt_index", Wt_index},
                {"total_number_of_cores", total_number_of_cores},
                {"compute_with_storage_grid_size_x", static_cast<uint32_t>(compute_with_storage_grid_size.x)},
                {"compute_with_storage_grid_size_y", static_cast<uint32_t>(compute_with_storage_grid_size.y)},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"core_loop_count", "core_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER},
        .target_nodes = core_range,
    });

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    uint32_t id = 0;  // Offset for the next core in the group
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values,
                    core,
                    {{"core_loop_count", work_per_core},
                     {"tile_width", tile_width},
                     {"tile_height", tile_height},
                     {"core_id", id}});
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values, core, {{"core_loop_count", work_per_core}, {"core_id", id}});
                id++;
            }
        }
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args = {
        {INPUT_TENSOR, input_mesh_tensor},
        {INPUT_INDEX_TENSOR, input_index_mesh_tensor},
        {OUTPUT_TENSOR, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

ttnn::device_operation::ProgramArtifacts GatherDeviceOperation::SingleRowMultiCore::create_program_artifacts(
    const GatherParams& attributes, const GatherInputs& tensor_args, Tensor& output_tensor) {
    const tt::DataFormat input_tensor_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat input_index_tensor_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_index_tensor.dtype());
    const tt::DataFormat output_tensor_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_dfb_data_format);
    const uint32_t input_index_tensor_tile_size = tile_size(input_index_tensor_dfb_data_format);
    const uint32_t output_tensor_tile_size = tile_size(output_tensor_dfb_data_format);

    const auto& input_mesh_tensor = tensor_args.input_tensor.mesh_tensor();
    const auto& input_index_mesh_tensor = tensor_args.input_index_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();

    auto* const input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto* const input_index_tensor_buffer = tensor_args.input_index_tensor.buffer();
    auto* const output_tensor_buffer = output_tensor.buffer();

    const bool input_tensor_is_dram = input_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool input_index_tensor_is_dram = input_index_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool output_tensor_is_dram = output_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();

    const auto input_index_shape = tensor_args.input_index_tensor.padded_shape();
    const auto input_shape = tensor_args.input_tensor.padded_shape();
    const uint32_t Ht = (input_index_shape[0] * input_index_shape[1] * input_index_shape[2]) / tile_height;
    const uint32_t Wt_input = input_shape[3] / tile_width;
    const uint32_t Wt_index = input_index_shape[3] / tile_width;

    // Calculate the number of cores available for computation
    auto* device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t max_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    // Create core grid
    CoreRangeSet core_grid =
        tt::tt_metal::num_cores_to_corerangeset(max_number_of_cores, compute_with_storage_grid_size, true);
    // Override core grid if sub_core_grids is provided in operation attributes
    if (attributes.sub_core_grids.has_value()) {
        core_grid = attributes.sub_core_grids.value();
    }

    const auto
        [total_number_of_cores,       // number of cores utilized
         core_range,                  // set of all cores used
         core_group_1,                // Primary core group
         core_group_2,                // Secondary core group
         num_tiles_per_core_group_1,  // Number of tiles each core in the primary group processes
         num_tiles_per_core_group_2   // Number of tiles each core in the secondary group processes
    ] = tt::tt_metal::split_work_to_cores(core_grid, Wt_index, true);
    const auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName INPUT_TENSOR_DFB{"input_tensor"};
    const DFBSpecName INPUT_INDEX_TENSOR_DFB{"input_index_tensor"};
    const DFBSpecName OUTPUT_TENSOR_DFB{"output_tensor"};
    const TensorParamName INPUT_TENSOR{"input_tensor"};
    const TensorParamName INPUT_INDEX_TENSOR{"input_index_tensor"};
    const TensorParamName OUTPUT_TENSOR{"output_tensor"};

    ProgramSpec spec;
    spec.name = "gather_single_row_multi_core";
    constexpr uint32_t buffer_scale_factor = 2;

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_TENSOR_DFB,
        .entry_size = input_tensor_tile_size,
        .num_entries = buffer_scale_factor,
        .data_format_metadata = input_tensor_dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_INDEX_TENSOR_DFB,
        .entry_size = input_index_tensor_tile_size,
        .num_entries = 1,
        .data_format_metadata = input_index_tensor_dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUTPUT_TENSOR_DFB,
        .entry_size = output_tensor_tile_size,
        .num_entries = 1,
        .data_format_metadata = output_tensor_dfb_data_format,
    });

    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT_TENSOR,
        .spec = input_mesh_tensor.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT_INDEX_TENSOR,
        .spec = input_index_mesh_tensor.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_TENSOR,
        .spec = output_mesh_tensor.tensor_spec(),
    });

    // The reader is the only kernel that touches the index DFB: it fills it from memory and then
    // reads it, so it is bound as both producer and consumer.
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = kReaderMulti,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_TENSOR_DFB,
                    .accessor_name = "input_tensor",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT_INDEX_TENSOR_DFB,
                    .accessor_name = "input_index_tensor",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT_INDEX_TENSOR_DFB,
                    .accessor_name = "input_index_tensor",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_TENSOR_DFB,
                    .accessor_name = "output_tensor",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT_INDEX_TENSOR,
                    .accessor_name = "input_index_tensor",
                },
            },
        .compile_time_args =
            {
                {"input_index_tensor_is_dram", static_cast<uint32_t>(input_index_tensor_is_dram)},
                {"Ht", Ht},
                {"Wt_input", Wt_input},
                {"Wt_index", Wt_index},
                {"total_number_of_cores", total_number_of_cores},
                {"compute_with_storage_grid_size_x", static_cast<uint32_t>(compute_with_storage_grid_size.x)},
                {"compute_with_storage_grid_size_y", static_cast<uint32_t>(compute_with_storage_grid_size.y)},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"core_loop_count", "tile_width", "tile_height", "core_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = kWriterMulti,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_TENSOR_DFB,
                    .accessor_name = "input_tensor",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_TENSOR_DFB,
                    .accessor_name = "output_tensor",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT_TENSOR,
                    .accessor_name = "input_tensor",
                },
                TensorBinding{
                    .tensor_parameter_name = OUTPUT_TENSOR,
                    .accessor_name = "output_tensor",
                },
            },
        .compile_time_args =
            {
                {"input_tensor_is_dram", static_cast<uint32_t>(input_tensor_is_dram)},
                {"output_tensor_is_dram", static_cast<uint32_t>(output_tensor_is_dram)},
                {"Ht", Ht},
                {"Wt_input", Wt_input},
                {"Wt_index", Wt_index},
                {"total_number_of_cores", total_number_of_cores},
                {"compute_with_storage_grid_size_x", static_cast<uint32_t>(compute_with_storage_grid_size.x)},
                {"compute_with_storage_grid_size_y", static_cast<uint32_t>(compute_with_storage_grid_size.y)},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"core_loop_count", "core_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER},
        .target_nodes = core_range,
    });

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    uint32_t id = 0;  // Offset for the next core in the group
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values,
                    core,
                    {{"core_loop_count", work_per_core},
                     {"tile_width", tile_width},
                     {"tile_height", tile_height},
                     {"core_id", id}});
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values, core, {{"core_loop_count", work_per_core}, {"core_id", id}});
                id++;
            }
        }
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args = {
        {INPUT_TENSOR, input_mesh_tensor},
        {INPUT_INDEX_TENSOR, input_index_mesh_tensor},
        {OUTPUT_TENSOR, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

// =========================================================================================
// ROW_MAJOR factories — row-distributed (single-core) and column-distributed (multi-core).
// =========================================================================================
ttnn::device_operation::ProgramArtifacts GatherDeviceOperation::RmSingleRowSingleCore::create_program_artifacts(
    const GatherParams& attributes, const GatherInputs& tensor_args, Tensor& output_tensor) {
    const tt::DataFormat input_tensor_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat input_index_tensor_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_index_tensor.dtype());
    const tt::DataFormat output_tensor_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    const auto& input_mesh_tensor = tensor_args.input_tensor.mesh_tensor();
    const auto& input_index_mesh_tensor = tensor_args.input_index_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();

    const uint32_t input_elem_size = tensor_args.input_tensor.element_size();
    const uint32_t input_index_elem_size = tensor_args.input_index_tensor.element_size();
    const uint32_t output_elem_size = output_tensor.element_size();

    const auto input_shape = tensor_args.input_tensor.padded_shape();
    const auto input_index_shape = tensor_args.input_index_tensor.padded_shape();

    const uint32_t W_input = input_shape[-1];
    const uint32_t W_index = input_index_shape[-1];
    // H = product of all leading dims. For 4D (pre-transform output) this is N*C*H*1.
    const uint32_t H = tensor_args.input_index_tensor.physical_volume() / W_index;

    const uint32_t input_stick_size_bytes = W_input * input_elem_size;
    const uint32_t input_index_stick_size_bytes = W_index * input_index_elem_size;
    const uint32_t output_stick_size_bytes = W_index * output_elem_size;

    auto* device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t max_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    CoreRangeSet core_grid =
        tt::tt_metal::num_cores_to_corerangeset(max_number_of_cores, compute_with_storage_grid_size, true);
    if (attributes.sub_core_grids.has_value()) {
        core_grid = attributes.sub_core_grids.value();
    }

    const auto [total_number_of_cores, core_range, core_group_1, core_group_2, rows_per_core_g1, rows_per_core_g2] =
        tt::tt_metal::split_work_to_cores(core_grid, H, true);
    const auto work_groups = {
        std::make_pair(core_group_1, rows_per_core_g1), std::make_pair(core_group_2, rows_per_core_g2)};

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName INPUT_TENSOR_DFB{"input_tensor"};
    const DFBSpecName INPUT_INDEX_TENSOR_DFB{"input_index_tensor"};
    const DFBSpecName OUTPUT_TENSOR_DFB{"output_tensor"};
    const TensorParamName INPUT_TENSOR{"input_tensor"};
    const TensorParamName INPUT_INDEX_TENSOR{"input_index_tensor"};
    const TensorParamName OUTPUT_TENSOR{"output_tensor"};

    ProgramSpec spec;
    spec.name = "gather_rm_single_row_single_core";

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_TENSOR_DFB,
        .entry_size = input_stick_size_bytes,
        .num_entries = 1,
        .data_format_metadata = input_tensor_dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_INDEX_TENSOR_DFB,
        .entry_size = input_index_stick_size_bytes,
        .num_entries = 1,
        .data_format_metadata = input_index_tensor_dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUTPUT_TENSOR_DFB,
        .entry_size = output_stick_size_bytes,
        .num_entries = 1,
        .data_format_metadata = output_tensor_dfb_data_format,
    });

    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT_TENSOR,
        .spec = input_mesh_tensor.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT_INDEX_TENSOR,
        .spec = input_index_mesh_tensor.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_TENSOR,
        .spec = output_mesh_tensor.tensor_spec(),
    });

    // The reader is the only kernel that touches the index DFB: it fills it from memory and then
    // reads it, so it is bound as both producer and consumer.
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = kReaderRmSingle,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_TENSOR_DFB,
                    .accessor_name = "input_tensor",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT_INDEX_TENSOR_DFB,
                    .accessor_name = "input_index_tensor",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT_INDEX_TENSOR_DFB,
                    .accessor_name = "input_index_tensor",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_TENSOR_DFB,
                    .accessor_name = "output_tensor",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT_INDEX_TENSOR,
                    .accessor_name = "input_index_tensor",
                },
            },
        .compile_time_args =
            {
                {"H", H},
                {"W_input", W_input},
                {"W_index", W_index},
                {"total_number_of_cores", total_number_of_cores},
                {"input_tensor_data_format_size", input_elem_size},
                {"input_index_tensor_data_format_size", input_index_elem_size},
                {"output_tensor_data_format_size", output_elem_size},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"core_loop_count", "core_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = kWriterRmSingle,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_TENSOR_DFB,
                    .accessor_name = "input_tensor",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_TENSOR_DFB,
                    .accessor_name = "output_tensor",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT_TENSOR,
                    .accessor_name = "input_tensor",
                },
                TensorBinding{
                    .tensor_parameter_name = OUTPUT_TENSOR,
                    .accessor_name = "output_tensor",
                },
            },
        .compile_time_args =
            {
                {"H", H},
                {"W_input", W_input},
                {"W_index", W_index},
                {"total_number_of_cores", total_number_of_cores},
                {"input_tensor_data_format_size", input_elem_size},
                {"output_tensor_data_format_size", output_elem_size},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"core_loop_count", "core_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER},
        .target_nodes = core_range,
    });

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    uint32_t id = 0;
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values, core, {{"core_loop_count", work_per_core}, {"core_id", id}});
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values, core, {{"core_loop_count", work_per_core}, {"core_id", id}});
                id++;
            }
        }
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args = {
        {INPUT_TENSOR, input_mesh_tensor},
        {INPUT_INDEX_TENSOR, input_index_mesh_tensor},
        {OUTPUT_TENSOR, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

// Column-distributed: each core owns [w_start, w_start+w_per_core) of every H row (wide W).
ttnn::device_operation::ProgramArtifacts GatherDeviceOperation::RmSingleRowMultiCore::create_program_artifacts(
    const GatherParams& attributes, const GatherInputs& tensor_args, Tensor& output_tensor) {
    const tt::DataFormat input_tensor_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat input_index_tensor_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_index_tensor.dtype());
    const tt::DataFormat output_tensor_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    const auto& input_mesh_tensor = tensor_args.input_tensor.mesh_tensor();
    const auto& input_index_mesh_tensor = tensor_args.input_index_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();

    auto* const input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto* const input_index_tensor_buffer = tensor_args.input_index_tensor.buffer();
    auto* const output_tensor_buffer = output_tensor.buffer();

    const uint32_t input_elem_size = tensor_args.input_tensor.element_size();
    const uint32_t input_index_elem_size = tensor_args.input_index_tensor.element_size();
    const uint32_t output_elem_size = output_tensor.element_size();

    const auto input_shape = tensor_args.input_tensor.padded_shape();
    const auto input_index_shape = tensor_args.input_index_tensor.padded_shape();

    const uint32_t W_input = input_shape[-1];
    const uint32_t W_index = input_index_shape[-1];
    const uint32_t H = tensor_args.input_index_tensor.physical_volume() / W_index;

    const uint32_t input_stick_size_bytes = W_input * input_elem_size;

    auto* device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t max_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    CoreRangeSet core_grid =
        tt::tt_metal::num_cores_to_corerangeset(max_number_of_cores, compute_with_storage_grid_size, true);
    if (attributes.sub_core_grids.has_value()) {
        core_grid = attributes.sub_core_grids.value();
    }

    // Split W_index across cores in units of w_unit columns, so that each core's byte offset into an
    // index row and into an output row (w_start * element size) is a multiple of the buffer alignment.
    // The DFB address on the core is aligned, and a NOC transfer needs the same low address bits on
    // both ends, so an unaligned row offset moves the wrong bytes. group_1 holds the larger slice when
    // the units don't divide evenly; the last core takes fewer columns when W_index is not a multiple
    // of w_unit.
    const uint32_t alignment = std::max(input_index_tensor_buffer->alignment(), output_tensor_buffer->alignment());
    const uint32_t w_unit = std::lcm(
        alignment / std::gcd(alignment, input_index_elem_size), alignment / std::gcd(alignment, output_elem_size));
    const auto [total_number_of_cores, core_range, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2] =
        tt::tt_metal::split_work_to_cores(core_grid, tt::div_up(W_index, w_unit), true);
    const uint32_t w_per_core_g1 = units_per_core_g1 * w_unit;
    const uint32_t w_per_core_g2 = units_per_core_g2 * w_unit;
    const auto work_groups = {std::make_pair(core_group_1, w_per_core_g1), std::make_pair(core_group_2, w_per_core_g2)};

    // DFB sticks sized for the LARGEST per-core slice (group 1); group 2 uses a shorter slice.
    const uint32_t max_w_per_core = std::max(w_per_core_g1, w_per_core_g2);
    const uint32_t max_input_index_slice_bytes = max_w_per_core * input_index_elem_size;
    const uint32_t max_output_slice_bytes = max_w_per_core * output_elem_size;

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName INPUT_TENSOR_DFB{"input_tensor"};
    const DFBSpecName INPUT_INDEX_TENSOR_DFB{"input_index_tensor"};
    const DFBSpecName OUTPUT_TENSOR_DFB{"output_tensor"};
    const TensorParamName INPUT_TENSOR{"input_tensor"};
    const TensorParamName INPUT_INDEX_TENSOR{"input_index_tensor"};
    const TensorParamName OUTPUT_TENSOR{"output_tensor"};

    ProgramSpec spec;
    spec.name = "gather_rm_single_row_multi_core";
    constexpr uint32_t buffer_scale_factor = 2;

    // Each input DFB page holds one input row. The page size is rounded up to the input buffer
    // alignment so that the second page starts at an aligned address, like the source row does.
    const uint32_t input_dfb_page_size = tt::align(input_stick_size_bytes, input_tensor_buffer->alignment());
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_TENSOR_DFB,
        .entry_size = input_dfb_page_size,
        .num_entries = buffer_scale_factor,
        .data_format_metadata = input_tensor_dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_INDEX_TENSOR_DFB,
        .entry_size = max_input_index_slice_bytes,
        .num_entries = 1,
        .data_format_metadata = input_index_tensor_dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUTPUT_TENSOR_DFB,
        .entry_size = max_output_slice_bytes,
        .num_entries = 1,
        .data_format_metadata = output_tensor_dfb_data_format,
    });

    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT_TENSOR,
        .spec = input_mesh_tensor.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT_INDEX_TENSOR,
        .spec = input_index_mesh_tensor.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_TENSOR,
        .spec = output_mesh_tensor.tensor_spec(),
    });

    // The reader is the only kernel that touches the index DFB: it fills it from memory and then
    // reads it, so it is bound as both producer and consumer.
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = kReaderRmMulti,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_TENSOR_DFB,
                    .accessor_name = "input_tensor",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT_INDEX_TENSOR_DFB,
                    .accessor_name = "input_index_tensor",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT_INDEX_TENSOR_DFB,
                    .accessor_name = "input_index_tensor",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_TENSOR_DFB,
                    .accessor_name = "output_tensor",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT_INDEX_TENSOR,
                    .accessor_name = "input_index_tensor",
                },
            },
        .compile_time_args =
            {
                {"H", H},
                {"W_input", W_input},
                {"W_index", W_index},
                {"input_tensor_data_format_size", input_elem_size},
                {"input_index_tensor_data_format_size", input_index_elem_size},
                {"output_tensor_data_format_size", output_elem_size},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"w_per_core", "w_start", "core_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = kWriterRmMulti,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_TENSOR_DFB,
                    .accessor_name = "input_tensor",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_TENSOR_DFB,
                    .accessor_name = "output_tensor",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT_TENSOR,
                    .accessor_name = "input_tensor",
                },
                TensorBinding{
                    .tensor_parameter_name = OUTPUT_TENSOR,
                    .accessor_name = "output_tensor",
                },
            },
        .compile_time_args =
            {
                {"H", H},
                {"W_input", W_input},
                {"W_index", W_index},
                {"input_tensor_data_format_size", input_elem_size},
                {"output_tensor_data_format_size", output_elem_size},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"w_per_core", "w_start", "core_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER},
        .target_nodes = core_range,
    });

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    uint32_t id = 0;
    uint32_t w_cursor = 0;  // Cumulative w_start across cores (iterating in group order matches split_work_to_cores).
    for (const auto& [group, w_per_group_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                const uint32_t w_start = w_cursor;
                const uint32_t w_per_core = std::min(w_per_group_core, W_index - w_cursor);
                AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values,
                    core,
                    {{"w_per_core", w_per_core}, {"w_start", w_start}, {"core_id", id}});
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"w_per_core", w_per_core}, {"w_start", w_start}, {"core_id", id}});
                w_cursor += w_per_core;
                id++;
            }
        }
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args = {
        {INPUT_TENSOR, input_mesh_tensor},
        {INPUT_INDEX_TENSOR, input_index_mesh_tensor},
        {OUTPUT_TENSOR, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
