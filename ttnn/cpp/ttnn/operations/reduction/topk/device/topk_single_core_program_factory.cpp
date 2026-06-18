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

namespace {

// ---------------------------------------------------------------------------
// Named resource ids (replace the legacy magic CB indices and positional args).
// ---------------------------------------------------------------------------

// DataflowBuffers (one per legacy CB c_0..c_7).
const DFBSpecName INPUT_DFB{"input"};                      // c_0: input values
const DFBSpecName INDEX_DFB{"index"};                      // c_1: generated indices
const DFBSpecName TRANSPOSED_VAL_DFB{"transposed_val"};    // c_2: transposed value staging
const DFBSpecName TRANSPOSED_IND_DFB{"transposed_ind"};    // c_3: transposed index staging
const DFBSpecName RESULT_PREP_VAL_DFB{"result_prep_val"};  // c_4: result-prep values (double buffered)
const DFBSpecName RESULT_PREP_IND_DFB{"result_prep_ind"};  // c_5: result-prep indices (double buffered)
const DFBSpecName OUTPUT_VAL_DFB{"output_val"};            // c_6: output values
const DFBSpecName OUTPUT_IND_DFB{"output_ind"};            // c_7: output indices

// Tensor parameters.
const TensorParamName INPUT_TENSOR{"input"};
const TensorParamName INDICES_TENSOR{"indices"};
const TensorParamName VALUES_OUT_TENSOR{"values_out"};
const TensorParamName INDICES_OUT_TENSOR{"indices_out"};

// Kernels.
const KernelSpecName READER_KERNEL{"reader"};
const KernelSpecName WRITER_KERNEL{"writer"};
const KernelSpecName COMPUTE_KERNEL{"compute"};

}  // namespace

ttnn::device_operation::ProgramArtifacts TopKDeviceOperation::TopKSingleCoreProgramFactory::create_program_artifacts(
    const TopkParams& operation_attributes,
    const TopkInputs& tensor_args,
    std::tuple<Tensor, Tensor>& tensor_return_value) {
    const auto& args = operation_attributes;
    auto& output_tensors = tensor_return_value;
    // Tensor references
    const auto& input_tensor = tensor_args.input.mesh_tensor();
    const auto& value_tensor = std::get<0>(output_tensors).mesh_tensor();
    const auto& index_tensor = std::get<1>(output_tensors).mesh_tensor();

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

    // Number of tiles needed to store K top elements
    const uint32_t Ktiles = tt::div_up(args.k, tile_width);

    // GENERATE_INDICES is hardcoded to 1 today (GH issue #36329); the precomputed-indices
    // read path (the only consumer of the optional indices tensor) is compiled out. The
    // optional `indices` TensorParameter / binding is enumerated regardless, gated by the
    // same condition so the typed path stays correct when #36329 is fixed.
    constexpr bool generate_indices = true;
    const bool bind_precomputed_indices = tensor_args.indices.has_value() && !generate_indices;

    // Pipeline Flow:
    // Input CB -> Reader Kernel -> Transposed CBs -> Compute Kernel -> Result Prep CBs -> Output CBs -> Writer Kernel
    const uint32_t num_cb_unit = 2;                         // Base unit for double buffering
    const uint32_t cb_in_units = num_cb_unit;               // 2 units total for input double buffering
    const uint32_t input_cb_tile_count = cb_in_units;       // Input stream buffer size
    const uint32_t transposed_cb_tile_count = 4;            // Transposed data staging
    const uint32_t result_prep_cb_tile_count = 2 * Ktiles;  // Intermediate TopK results (double-buffered)
    const uint32_t output_cb_tile_count = Ktiles;           // Final output buffer

    // ----------------------------------------------------------------------
    // DataflowBuffer specs (one per legacy CB). Every DFB is bound to the
    // compute kernel, so each carries data_format_metadata.
    // ----------------------------------------------------------------------
    Group<DataflowBufferSpec> dfbs = {
        DataflowBufferSpec{
            .unique_id = INPUT_DFB,
            .entry_size = input_tile_size,
            .num_entries = input_cb_tile_count,
            .data_format_metadata = input_cb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = INDEX_DFB,
            .entry_size = index_tile_size,
            .num_entries = input_cb_tile_count,
            .data_format_metadata = output_ind_cb_data_format,
        },
        // Uses bf16 when input is bfp8/bfp4 so the insertion sort operates at higher
        // precision and avoids shared-exponent corruption of tiles adjacent to inf values.
        DataflowBufferSpec{
            .unique_id = TRANSPOSED_VAL_DFB,
            .entry_size = compute_tile_size,
            .num_entries = transposed_cb_tile_count,
            .data_format_metadata = compute_cb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = TRANSPOSED_IND_DFB,
            .entry_size = index_tile_size,
            .num_entries = transposed_cb_tile_count,
            .data_format_metadata = output_ind_cb_data_format,
        },
        // Uses bf16 when input is bfp8/bfp4 (same rationale as transposed_val).
        DataflowBufferSpec{
            .unique_id = RESULT_PREP_VAL_DFB,
            .entry_size = compute_tile_size,
            .num_entries = result_prep_cb_tile_count,
            .data_format_metadata = compute_cb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = RESULT_PREP_IND_DFB,
            .entry_size = index_tile_size,
            .num_entries = result_prep_cb_tile_count,
            .data_format_metadata = output_ind_cb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = OUTPUT_VAL_DFB,
            .entry_size = value_tile_size,
            .num_entries = output_cb_tile_count,
            .data_format_metadata = output_val_cb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = OUTPUT_IND_DFB,
            .entry_size = index_tile_size,
            .num_entries = output_cb_tile_count,
            .data_format_metadata = output_ind_cb_data_format,
        },
    };

    // ----------------------------------------------------------------------
    // Tensor parameters (one per distinct originating tensor).
    // ----------------------------------------------------------------------
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = INPUT_TENSOR, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = VALUES_OUT_TENSOR, .spec = value_tensor.tensor_spec()},
        TensorParameter{.unique_id = INDICES_OUT_TENSOR, .spec = index_tensor.tensor_spec()},
    };
    if (bind_precomputed_indices) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = INDICES_TENSOR, .spec = tensor_args.indices->mesh_tensor().tensor_spec()});
    }

    // ----------------------------------------------------------------------
    // Reader kernel.
    // ----------------------------------------------------------------------
    KernelSpec reader{
        .unique_id = READER_KERNEL,
        .source = "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_tensor.cpp",
        .compiler_options =
            {
                .defines = generate_indices ? KernelSpec::CompilerOptions::Defines{{"GENERATE_INDICES", "1"}}
                                            : KernelSpec::CompilerOptions::Defines{},
            },
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = INDEX_DFB, .accessor_name = "index", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "inout"},
            },
        .compile_time_args =
            {
                {"Ht", Ht},
                {"Wt", Wt},
                {"total_number_of_cores", total_number_of_cores},
                {"uint16_output", static_cast<uint32_t>(uint16_output)},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"id", "work_per_core"},
            },
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };
    if (bind_precomputed_indices) {
        reader.tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = INDICES_TENSOR, .accessor_name = "indices"});
    }

    // ----------------------------------------------------------------------
    // Writer kernel.
    // ----------------------------------------------------------------------
    KernelSpec writer{
        .unique_id = WRITER_KERNEL,
        .source = "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_binary_interleaved.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUTPUT_VAL_DFB,
                    .accessor_name = "values",
                    .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = OUTPUT_IND_DFB,
                    .accessor_name = "indices",
                    .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = VALUES_OUT_TENSOR, .accessor_name = "values"},
                TensorBinding{.tensor_parameter_name = INDICES_OUT_TENSOR, .accessor_name = "out_indices"},
            },
        .compile_time_args =
            {
                {"Ht", Ht},
                {"Kt", Ktiles},
                {"total_number_of_cores", total_number_of_cores},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"id", "work_per_core"},
            },
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    // ----------------------------------------------------------------------
    // Compute kernel.
    // c_2..c_5 are self-loop (PRODUCER + CONSUMER): the compute kernel both
    // reserves/pushes and waits/pops these staging / result-prep buffers itself.
    // ----------------------------------------------------------------------
    KernelSpec compute{
        .unique_id = COMPUTE_KERNEL,
        .source = "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = INDEX_DFB, .accessor_name = "index", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = TRANSPOSED_VAL_DFB,
                    .accessor_name = "transposed_val",
                    .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = TRANSPOSED_VAL_DFB,
                    .accessor_name = "transposed_val",
                    .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = TRANSPOSED_IND_DFB,
                    .accessor_name = "transposed_ind",
                    .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = TRANSPOSED_IND_DFB,
                    .accessor_name = "transposed_ind",
                    .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = RESULT_PREP_VAL_DFB,
                    .accessor_name = "result_prep_val",
                    .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = RESULT_PREP_VAL_DFB,
                    .accessor_name = "result_prep_val",
                    .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = RESULT_PREP_IND_DFB,
                    .accessor_name = "result_prep_ind",
                    .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = RESULT_PREP_IND_DFB,
                    .accessor_name = "result_prep_ind",
                    .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = OUTPUT_VAL_DFB,
                    .accessor_name = "output_val",
                    .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = OUTPUT_IND_DFB,
                    .accessor_name = "output_ind",
                    .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .compile_time_args =
            {
                {"Ht", Ht},
                {"Wt", Wt},
                {"output_tiles", Ktiles},
                {"largest", static_cast<uint32_t>(args.largest)},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"work_per_core"},
            },
        .hw_config =
            ComputeHardwareConfig{
                .fp32_dest_acc_en = !uint16_output,
                .dst_full_sync_en = false,
            },
    };

    // ----------------------------------------------------------------------
    // Assemble the spec.
    // ----------------------------------------------------------------------
    ProgramSpec spec{
        .name = "topk_single_core",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{
                    .name = "topk",
                    .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
                    .target_nodes = core_range,
                },
            },
    };

    // ----------------------------------------------------------------------
    // Run args: per-core (id, work_per_core) for reader/writer; (work_per_core)
    // for compute. id is the sequential per-core offset, matching the legacy loop.
    // ----------------------------------------------------------------------
    Group<KernelRunArgs::NodeRuntimeArgs> reader_node_args;
    Group<KernelRunArgs::NodeRuntimeArgs> writer_node_args;
    Group<KernelRunArgs::NodeRuntimeArgs> compute_node_args;

    uint32_t id = 0;  // Offset for the next core in the group
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                reader_node_args.push_back({.node = core, .args = {{"id", id}, {"work_per_core", work_per_core}}});
                writer_node_args.push_back({.node = core, .args = {{"id", id}, {"work_per_core", work_per_core}}});
                compute_node_args.push_back({.node = core, .args = {{"work_per_core", work_per_core}}});
                id++;
            }
        }
    }

    ProgramRunArgs run_args{
        .kernel_run_args =
            {
                KernelRunArgs{.kernel = READER_KERNEL, .runtime_arg_values = std::move(reader_node_args)},
                KernelRunArgs{.kernel = WRITER_KERNEL, .runtime_arg_values = std::move(writer_node_args)},
                KernelRunArgs{.kernel = COMPUTE_KERNEL, .runtime_arg_values = std::move(compute_node_args)},
            },
        .tensor_args =
            {
                {INPUT_TENSOR, input_tensor},
                {VALUES_OUT_TENSOR, value_tensor},
                {INDICES_OUT_TENSOR, index_tensor},
            },
    };
    if (bind_precomputed_indices) {
        run_args.tensor_args.emplace(INDICES_TENSOR, tensor_args.indices->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
