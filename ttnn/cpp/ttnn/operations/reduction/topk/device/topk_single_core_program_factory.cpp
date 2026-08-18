// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/device/topk_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include "tt-metalium/work_split.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <string>
#include <utility>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

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

    // Determine index output width from the actual index tensor dtype. UINT16 uses 16-bit indices;
    // UINT32/INT32 use 32-bit indices (both share the same 4-byte tile layout). Deriving this from the
    // tensor dtype (rather than the dimension size) keeps index generation and the CB data format in
    // sync for preallocated 32-bit outputs, regardless of the dimension size.
    const ttnn::Shape input_shape = input_tensor.padded_shape();
    const bool uint16_output = (index_tensor.dtype() == DataType::UINT16);

    // Data format conversions for dataflow buffer configurations
    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat output_val_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(value_tensor.dtype());
    // The on-device sort datapath handles 32-bit indices as UInt32. INT32 shares the same 4-byte
    // little-endian layout for the non-negative positions TopK produces, so run the compute in UInt32
    // and let the writer copy the raw tile bytes into the (INT32-typed) output buffer unchanged.
    tt::DataFormat output_ind_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(index_tensor.dtype());
    if (output_ind_cb_data_format == tt::DataFormat::Int32) {
        output_ind_cb_data_format = tt::DataFormat::UInt32;
    }

    // Use bf16 for compute intermediate buffers to avoid precision loss from bfp8/bfp4
    // shared-exponent grouping during sort (e.g. a single inf in a block makes all other
    // elements in that block encode to 0, corrupting the sort result).
    // fp32 is kept as-is: with a 32-bit dest register and unpack-to-dest the value buffers stay fp32
    // and the sort's default SFPLOAD mode resolves to FP32 under fp32 dest-acc, so no downcast is
    // needed.
    const bool is_fp32_input = input_cb_data_format == tt::DataFormat::Float32;
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
    // Input DFB -> Reader Kernel -> Transposed DFBs -> Compute Kernel -> Result Prep DFBs -> Output DFBs -> Writer
    // Kernel
    const uint32_t num_cb_unit = 2;                         // Base unit for double buffering
    const uint32_t cb_in_units = num_cb_unit;               // 2 units total for input double buffering
    const uint32_t input_cb_tile_count = cb_in_units;       // Input stream buffer size
    const uint32_t transposed_cb_tile_count = 4;            // Transposed data staging
    const uint32_t result_prep_cb_tile_count = 2 * Ktiles;  // Intermediate TopK results (double-buffered)
    const uint32_t output_cb_tile_count = Ktiles;           // Final output buffer

    // Program-scope resource names. These identify resources within the ProgramSpec and never reach
    // device code: the kernel-side token comes from each binding's accessor_name, which is local to
    // the kernel declaring it. Renaming a name here therefore does not rename any `dfb::` or
    // `tensor::` token. One buffer can be `dfb::values` in the writer and `dfb::output_val` in the
    // compute kernel, and each kernel's `tensor::indices` resolves to whichever tensor parameter
    // that kernel binds.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName INDEX_DFB{"index"};
    const DFBSpecName TRANSPOSED_VAL_DFB{"transposed_val"};
    const DFBSpecName TRANSPOSED_IND_DFB{"transposed_ind"};
    const DFBSpecName RESULT_PREP_VAL_DFB{"result_prep_val"};
    const DFBSpecName RESULT_PREP_IND_DFB{"result_prep_ind"};
    const DFBSpecName OUTPUT_VAL_DFB{"output_val"};
    const DFBSpecName OUTPUT_IND_DFB{"output_ind"};

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName INPUT_INDICES_TENSOR{"input_indices"};
    const TensorParamName VALUES_TENSOR{"values"};
    const TensorParamName INDEX_TENSOR{"index"};

    ProgramSpec spec;
    spec.name = "topk_single_core";

    // Dataflow Buffer Creations:
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_DFB,
        .entry_size = input_tile_size,
        .num_entries = input_cb_tile_count,
        .data_format_metadata = input_cb_data_format,
    });

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INDEX_DFB,
        .entry_size = index_tile_size,
        .num_entries = input_cb_tile_count,
        .data_format_metadata = output_ind_cb_data_format,
    });

    // Uses bf16 when input is bfp8/bfp4 so that the insertion sort operates at higher
    // precision and avoids shared-exponent corruption of tiles adjacent to inf values.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = TRANSPOSED_VAL_DFB,
        .entry_size = compute_tile_size,
        .num_entries = transposed_cb_tile_count,
        .data_format_metadata = compute_cb_data_format,
    });

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = TRANSPOSED_IND_DFB,
        .entry_size = index_tile_size,
        .num_entries = transposed_cb_tile_count,
        .data_format_metadata = output_ind_cb_data_format,
    });

    // Uses bf16 when input is bfp8/bfp4 (same rationale as TRANSPOSED_VAL_DFB).
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = RESULT_PREP_VAL_DFB,
        .entry_size = compute_tile_size,
        .num_entries = result_prep_cb_tile_count,
        .data_format_metadata = compute_cb_data_format,
    });

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = RESULT_PREP_IND_DFB,
        .entry_size = index_tile_size,
        .num_entries = result_prep_cb_tile_count,
        .data_format_metadata = output_ind_cb_data_format,
    });

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUTPUT_VAL_DFB,
        .entry_size = value_tile_size,
        .num_entries = output_cb_tile_count,
        .data_format_metadata = output_val_cb_data_format,
    });

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUTPUT_IND_DFB,
        .entry_size = index_tile_size,
        .num_entries = output_cb_tile_count,
        .data_format_metadata = output_ind_cb_data_format,
    });

    // Tensor Parameter Declarations:
    // The optional input indices tensor is declared only when the caller supplies one, but no build
    // reads it today: the reader's sole use sits behind `#if not GENERATE_INDICES` and that define is
    // pinned on (GH issue #36329), so the host condition and the kernel condition do not track each
    // other. The parameter, its binding and its run arg are provisioned for the fix rather than
    // consumed, and the declared spec is still re-checked against the supplied tensor every
    // dispatch.
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_TENSOR, .spec = input_tensor.tensor_spec()});
    if (tensor_args.indices.has_value()) {
        spec.tensor_parameters.push_back(TensorParameter{
            .unique_id = INPUT_INDICES_TENSOR, .spec = tensor_args.indices->mesh_tensor().tensor_spec()});
    }
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = VALUES_TENSOR, .spec = value_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INDEX_TENSOR, .spec = index_tensor.tensor_spec()});

    // Kernel Creations:
    Group<TensorBinding> reader_tensor_bindings = {
        TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}};
    if (tensor_args.indices.has_value()) {
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = INPUT_INDICES_TENSOR, .accessor_name = "indices"});
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_tensor.cpp",
        .compiler_options =
            {
                .defines = {{"GENERATE_INDICES", "1"}},  // tensor_args.indices.has_value() ? "0" : "1" - GH issue:
                                                         // #36329
            },
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_DFB,  // Input values
                    .accessor_name = "input",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = INDEX_DFB,  // Generated indices
                    .accessor_name = "index",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings = reader_tensor_bindings,
        .compile_time_args =
            {
                {"Ht", Ht},                                        // Height in tiles
                {"Wt", Wt},                                        // Width in tiles
                {"total_number_of_cores", total_number_of_cores},  // Total number of cores
                // Index width must match the index tensor dtype: fp32 requires 32-bit iota.
                // 16-bit iota packs two indices per word, producing incorrect INT32 reads.
                {"uint16_output",
                 static_cast<uint32_t>(output_ind_cb_data_format == tt::DataFormat::UInt16)},  // Index format flag
            },
        .runtime_arg_schema = {.runtime_arg_names = {"id", "work_per_core"}},
        .hw_config = ttnn::create_reader_datamovement_config(input_tensor.device().arch()),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_binary_interleaved.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUTPUT_VAL_DFB,  // Output values
                    .accessor_name = "values",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_IND_DFB,  // Output indices
                    .accessor_name = "indices",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = VALUES_TENSOR, .accessor_name = "values"},
                TensorBinding{.tensor_parameter_name = INDEX_TENSOR, .accessor_name = "indices"},
            },
        .compile_time_args =
            {
                {"Ht", Ht},                                        // Height in tiles
                {"Kt", Ktiles},                                    // K value in tiles
                {"total_number_of_cores", total_number_of_cores},  // Total number of cores
            },
        .runtime_arg_schema = {.runtime_arg_names = {"id", "work_per_core"}},
        .hw_config = ttnn::create_writer_datamovement_config(input_tensor.device().arch()),
    };

    // fp32 input: unpack the value-holding buffers straight to a 32-bit dest register so the sort's
    // default SFPLOAD mode resolves to FP32 and compares full-precision values. These are keyed by
    // buffer name, and an omitted entry means unpack-to-source, which is what every other buffer
    // wants. They are also the only 32-bit-float buffers the compute kernel consumes, and an
    // explicit choice is required for those whenever the dest register is 32-bit.
    ComputeUnpackModes compute_unpack_modes;
    if (is_fp32_input) {
        compute_unpack_modes = {
            {INPUT_DFB, UnpackMode::UnpackToDest},
            {TRANSPOSED_VAL_DFB, UnpackMode::UnpackToDest},
            {RESULT_PREP_VAL_DFB, UnpackMode::UnpackToDest},
        };
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp",
        // A compute kernel's legacy default optimization level is O3, while the Metal 2.0
        // default is O2 for every kernel kind, so it has to be stated to keep the level.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        // The four workspace buffers are touched by this kernel alone, which both fills and
        // drains each of them, so each is bound at both endpoints under one accessor name.
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_DFB,  // Input values
                    .accessor_name = "input",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = INDEX_DFB,  // Input indices
                    .accessor_name = "index",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = TRANSPOSED_VAL_DFB,  // Transposed values
                    .accessor_name = "transposed_val",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = TRANSPOSED_VAL_DFB,
                    .accessor_name = "transposed_val",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = TRANSPOSED_IND_DFB,  // Transposed indices
                    .accessor_name = "transposed_ind",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = TRANSPOSED_IND_DFB,
                    .accessor_name = "transposed_ind",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = RESULT_PREP_VAL_DFB,  // Result prep values
                    .accessor_name = "result_prep_val",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = RESULT_PREP_VAL_DFB,
                    .accessor_name = "result_prep_val",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = RESULT_PREP_IND_DFB,  // Result prep indices
                    .accessor_name = "result_prep_ind",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = RESULT_PREP_IND_DFB,
                    .accessor_name = "result_prep_ind",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_VAL_DFB,  // Output values
                    .accessor_name = "output_val",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_IND_DFB,  // Output indices
                    .accessor_name = "output_ind",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args =
            {
                {"Ht", Ht},                                           // Height in tiles
                {"Wt", Wt},                                           // Width in tiles
                {"output_tiles", Ktiles},                             // K value in tiles
                {"largest", static_cast<uint32_t>(args.largest)},     // Sort order: largest (true) or smallest (false)
                {"stable_sort", static_cast<uint32_t>(args.stable)},  // Stable sort: ties keep the lowest index
            },
        .runtime_arg_schema = {.runtime_arg_names = {"work_per_core"}},
        // A 32-bit dest register is needed in two independent cases: a UInt32 index output (wide
        // rows), so an index survives the sort intact, and an fp32 input, so values keep full
        // precision through it. double_buffer_dest is the inverse of the legacy dst_full_sync_en flag.
        .hw_config =
            ComputeGen1Config{
                .enable_32_bit_dest = !uint16_output || is_fp32_input,
                .double_buffer_dest = true,
                .unpack_modes = std::move(compute_unpack_modes),
            },
    };

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(std::move(compute));

    spec.work_units.push_back(
        WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = core_range});

    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    uint32_t id = 0;  // Offset for the next core in the group
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values, core, {{"id", id}, {"work_per_core", work_per_core}});
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values, core, {{"id", id}, {"work_per_core", work_per_core}});
                AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"work_per_core", work_per_core}});
                id++;
            }
        }
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));

    run_args.tensor_args.emplace(INPUT_TENSOR, input_tensor);
    if (tensor_args.indices.has_value()) {
        run_args.tensor_args.emplace(INPUT_INDICES_TENSOR, tensor_args.indices->mesh_tensor());
    }
    run_args.tensor_args.emplace(VALUES_TENSOR, value_tensor);
    run_args.tensor_args.emplace(INDEX_TENSOR, index_tensor);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
