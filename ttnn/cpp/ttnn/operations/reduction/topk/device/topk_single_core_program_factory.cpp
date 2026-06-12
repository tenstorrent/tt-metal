// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/device/topk_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include "tt-metalium/work_split.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <limits>
#include <string>
#include <vector>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TopKDeviceOperation::TopKSingleCoreProgramFactory::create_program_spec(
    const TopkParams& operation_attributes,
    const TopkInputs& tensor_args,
    std::tuple<Tensor, Tensor>& tensor_return_value) {
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName CB_IN0{"cb_in0"};
    const DFBSpecName CB_INDEX{"cb_index"};
    const DFBSpecName TRANSPOSED_VAL{"transposed_val"};
    const DFBSpecName TRANSPOSED_IND{"transposed_ind"};
    const DFBSpecName RESULT_PREP_VAL{"result_prep_val"};
    const DFBSpecName RESULT_PREP_IND{"result_prep_ind"};
    const DFBSpecName OUTPUT_VAL{"output_val"};
    const DFBSpecName OUTPUT_IND{"output_ind"};

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName INDICES_TENSOR{"indices"};
    const TensorParamName VALUE_TENSOR{"value"};
    const TensorParamName INDEX_TENSOR{"index"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL{"compute"};

    constexpr const char* READER_PATH =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_tensor.cpp";
    constexpr const char* WRITER_PATH =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_binary_interleaved.cpp";
    constexpr const char* COMPUTE_PATH = "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp";

    const auto& args = operation_attributes;
    auto& output_tensors = tensor_return_value;
    // Tensor references
    const auto& input_tensor = tensor_args.input;
    const auto& value_tensor = std::get<0>(output_tensors);
    const auto& index_tensor = std::get<1>(output_tensors);

    // Determine index output format based on dimension size constraints
    const ttnn::Shape input_shape = input_tensor.padded_shape();
    const bool uint16_output = (input_shape[args.dim] < std::numeric_limits<uint16_t>::max());

    // Data format conversions for circular buffer configurations
    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat output_val_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(value_tensor.dtype());
    const tt::DataFormat output_ind_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(index_tensor.dtype());

    // Use bf16 for compute intermediate buffers to avoid precision loss from bfp8/bfp4
    // shared-exponent grouping during sort (e.g. a single inf in a block makes all other
    // elements in that block encode to 0, corrupting the sort result).
    const tt::DataFormat compute_cb_data_format =
        (input_cb_data_format == tt::DataFormat::Bfp8_b || input_cb_data_format == tt::DataFormat::Bfp4_b)
            ? tt::DataFormat::Float16_b
            : input_cb_data_format;

    // Calculate tile sizes for memory allocation
    const uint32_t input_tile_size = tile_size(input_cb_data_format);
    const uint32_t value_tile_size = tile_size(output_val_cb_data_format);
    const uint32_t index_tile_size = tile_size(output_ind_cb_data_format);
    const uint32_t compute_tile_size = tile_size(compute_cb_data_format);

    // Tensor shape and dimension calculations
    const uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tile_height;
    const uint32_t Wt = input_shape[3] / tile_width;

    // Single core selection from the provided core grid
    const auto
        [total_number_of_cores,       // number of cores utilized
         core_range,                  // set of all cores used
         core_group_1,                // Primary core group
         core_group_2,                // Secondary core group
         num_tiles_per_core_group_1,  // Number of tiles each core in the primary group processes
         num_tiles_per_core_group_2   // Number of tiles each core in the secondary group processes
    ] = tt::tt_metal::split_work_to_cores(args.sub_core_grids, Ht, true);
    TT_FATAL(
        args.sub_core_grids.contains(core_range),
        "TopK single-core program core grid {} must be contained in sub_core_grids {}",
        core_range,
        args.sub_core_grids);
    const auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};
    const std::vector<CoreCoord>& cores = corerange_to_cores(core_range, total_number_of_cores, true);

    // Number of tiles needed to store K top elements
    const uint32_t Ktiles = tt::div_up(args.k, tile_width);

    // Pipeline Flow:
    // Input CB -> Reader Kernel -> Transposed CBs -> Compute Kernel -> Result Prep CBs -> Output CBs -> Writer Kernel
    const uint32_t num_cb_unit = 2;                         // Base unit for double buffering
    const uint32_t cb_in_units = num_cb_unit;               // 2 units total for input double buffering
    const uint32_t input_cb_tile_count = cb_in_units;       // Input stream buffer size
    const uint32_t transposed_cb_tile_count = 4;            // Transposed data staging
    const uint32_t result_prep_cb_tile_count = 2 * Ktiles;  // Intermediate TopK results (double-buffered)
    const uint32_t output_cb_tile_count = Ktiles;           // Final output buffer

    // ----------------------------------------------------------------------------
    // DataflowBufferSpecs (legacy CBs c_0..c_7, all normal/non-borrowed). The bf16
    // staging buffers (c_2..c_5) preserve sort precision for bfp8/bfp4 inputs.
    // ----------------------------------------------------------------------------
    DataflowBufferSpec cb_in0_spec{
        .unique_id = CB_IN0,
        .entry_size = input_tile_size,
        .num_entries = input_cb_tile_count,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec cb_index_spec{
        .unique_id = CB_INDEX,
        .entry_size = index_tile_size,
        .num_entries = input_cb_tile_count,
        .data_format_metadata = output_ind_cb_data_format,
    };
    DataflowBufferSpec transposed_val_spec{
        .unique_id = TRANSPOSED_VAL,
        .entry_size = compute_tile_size,
        .num_entries = transposed_cb_tile_count,
        .data_format_metadata = compute_cb_data_format,
    };
    DataflowBufferSpec transposed_ind_spec{
        .unique_id = TRANSPOSED_IND,
        .entry_size = index_tile_size,
        .num_entries = transposed_cb_tile_count,
        .data_format_metadata = output_ind_cb_data_format,
    };
    DataflowBufferSpec result_prep_val_spec{
        .unique_id = RESULT_PREP_VAL,
        .entry_size = compute_tile_size,
        .num_entries = result_prep_cb_tile_count,
        .data_format_metadata = compute_cb_data_format,
    };
    DataflowBufferSpec result_prep_ind_spec{
        .unique_id = RESULT_PREP_IND,
        .entry_size = index_tile_size,
        .num_entries = result_prep_cb_tile_count,
        .data_format_metadata = output_ind_cb_data_format,
    };
    DataflowBufferSpec output_val_spec{
        .unique_id = OUTPUT_VAL,
        .entry_size = value_tile_size,
        .num_entries = output_cb_tile_count,
        .data_format_metadata = output_val_cb_data_format,
    };
    DataflowBufferSpec output_ind_spec{
        .unique_id = OUTPUT_IND,
        .entry_size = index_tile_size,
        .num_entries = output_cb_tile_count,
        .data_format_metadata = output_ind_cb_data_format,
    };

    // ----------------------------------------------------------------------------
    // Tensor parameters. The optional precomputed-indices tensor is only bound on
    // the dead `#if not GENERATE_INDICES` reader path (GENERATE_INDICES is hardcoded
    // "1"); on the always-on path no indices TensorParameter is bound.
    // ----------------------------------------------------------------------------
    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_tensor.tensor_spec()};
    TensorParameter value_param{.unique_id = VALUE_TENSOR, .spec = value_tensor.tensor_spec()};
    TensorParameter index_param{.unique_id = INDEX_TENSOR, .spec = index_tensor.tensor_spec()};

    // ----------------------------------------------------------------------------
    // Reader: streams input tiles into cb_in0 (c_0) and generates index tiles into
    // cb_index (c_1, via generate_index_tile). GENERATE_INDICES is hardcoded "1"
    // (GH issue #36329), keeping the indices-tensor read path dead.
    // ----------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{READER_PATH},
        .compiler_options = {.defines = {{"GENERATE_INDICES", "1"}}},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = CB_INDEX, .accessor_name = "cb_index", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .compile_time_args =
            {{"Ht", Ht},
             {"Wt", Wt},
             {"total_number_of_cores", total_number_of_cores},
             {"uint16_output", static_cast<uint32_t>(uint16_output)}},
        .runtime_arg_schema = {.runtime_arg_names = {"id", "work_per_core"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // ----------------------------------------------------------------------------
    // Writer: consumes output_val (c_6) and output_ind (c_7), writes the value and
    // index output tensors page-by-page (Case 1 TensorAccessor).
    // ----------------------------------------------------------------------------
    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_PATH},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = OUTPUT_VAL,
                 .accessor_name = "output_val",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = OUTPUT_IND,
                 .accessor_name = "output_ind",
                 .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = VALUE_TENSOR, .accessor_name = "value"},
             TensorBinding{.tensor_parameter_name = INDEX_TENSOR, .accessor_name = "index"}},
        .compile_time_args = {{"Ht", Ht}, {"Kt", Ktiles}, {"total_number_of_cores", total_number_of_cores}},
        .runtime_arg_schema = {.runtime_arg_names = {"id", "work_per_core"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    // ----------------------------------------------------------------------------
    // Compute: consumes c_0/c_1, runs the insertion sort over the c_2..c_5 staging
    // self-loops (PRODUCER+CONSUMER on this same compute kernel), produces c_6/c_7.
    // The kernel performs runtime-dynamic CB selection over the staging buffers.
    // ----------------------------------------------------------------------------
    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source = std::filesystem::path{COMPUTE_PATH},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = CB_INDEX, .accessor_name = "cb_index", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = TRANSPOSED_VAL,
                 .accessor_name = "transposed_val",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = TRANSPOSED_VAL,
                 .accessor_name = "transposed_val",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = TRANSPOSED_IND,
                 .accessor_name = "transposed_ind",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = TRANSPOSED_IND,
                 .accessor_name = "transposed_ind",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = RESULT_PREP_VAL,
                 .accessor_name = "result_prep_val",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = RESULT_PREP_VAL,
                 .accessor_name = "result_prep_val",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = RESULT_PREP_IND,
                 .accessor_name = "result_prep_ind",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = RESULT_PREP_IND,
                 .accessor_name = "result_prep_ind",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = OUTPUT_VAL,
                 .accessor_name = "output_val",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = OUTPUT_IND,
                 .accessor_name = "output_ind",
                 .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args =
            {{"Ht", Ht}, {"Wt", Wt}, {"Ktiles", Ktiles}, {"largest", static_cast<uint32_t>(args.largest)}},
        .runtime_arg_schema = {.runtime_arg_names = {"work_per_core"}},
        .hw_config =
            ComputeHardwareConfig{
                .fp32_dest_acc_en = !uint16_output,
                .dst_full_sync_en = false,
            },
    };

    // ----------------------------------------------------------------------------
    // Per-core runtime args: id (core offset) and work_per_core (tiles for this core).
    // ----------------------------------------------------------------------------
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    KernelRunArgs compute_run{.kernel = COMPUTE_KERNEL};

    uint32_t id = 0;  // Offset for the next core in the group
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                const NodeCoord node = core;
                reader_run.runtime_arg_values.push_back({node, {{"id", id}, {"work_per_core", work_per_core}}});
                writer_run.runtime_arg_values.push_back({node, {{"id", id}, {"work_per_core", work_per_core}}});
                compute_run.runtime_arg_values.push_back({node, {{"work_per_core", work_per_core}}});
                id++;
            }
        }
    }

    WorkUnitSpec wu{
        .name = "topk_single_core",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = core_range,
    };

    ProgramSpec spec{
        .name = "topk_single_core",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers =
            {cb_in0_spec,
             cb_index_spec,
             transposed_val_spec,
             transposed_ind_spec,
             result_prep_val_spec,
             result_prep_ind_spec,
             output_val_spec,
             output_ind_spec},
        .tensor_parameters = {input_param, value_param, index_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run, compute_run};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor.mesh_tensor())}},
        {VALUE_TENSOR, TensorArgument{std::cref(value_tensor.mesh_tensor())}},
        {INDEX_TENSOR, TensorArgument{std::cref(index_tensor.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
