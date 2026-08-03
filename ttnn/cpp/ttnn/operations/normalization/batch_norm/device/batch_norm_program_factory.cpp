// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_device_operation.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <bit>
#include <cmath>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::normalization;

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_shape_dims(const ttnn::Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
}

void populate_runtime_arguments(
    tt::tt_metal::experimental::KernelRunArgs& reader_run_args,
    tt::tt_metal::experimental::KernelRunArgs& writer_run_args,
    tt::tt_metal::experimental::KernelRunArgs& compute_run_args,
    tt::tt_metal::CoreCoord compute_with_storage_grid_size,
    bool any_float32,
    const BatchNormOperation::operation_attributes_t& operation_attributes,
    const BatchNormOperation::tensor_args_t& tensor_args,
    BatchNormOperation::tensor_return_value_t& c) {
    using tt::tt_metal::experimental::AddRuntimeArgsForNode;

    const auto& [input_tensor, batch_mean_tensor, batch_var_tensor, weight_tensor, bias_tensor, _] = tensor_args;
    const auto eps = operation_attributes.eps;

    const auto [aN, aC, aHt, aWt] = extract_shape_dims(input_tensor);
    const auto [bN, bC, bHt, bWt] = extract_shape_dims(batch_mean_tensor);
    const auto [cN, cC, cHt, cWt] = extract_shape_dims(c);

    uint32_t num_output_tiles = c.physical_volume() / c.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto
        [_unused_num_cores,
         _unused_all_cores,
         core_group_1,
         core_group_2,
         num_tiles_per_core_group_1,
         num_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);

    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // Cores outside both work groups still run the kernels, so every named runtime argument
            // must be set on them; the compute kernel early-returns on num_tiles == 0.
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"eps", 0},
                 {"start_tile_id", 0},
                 {"num_tiles", 0},
                 {"HtWt", 0},
                 {"n_stride", 0},
                 {"c_stride", 0},
                 {"N", 0},
                 {"C", 0}});
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"start_tile_id", 0},
                 {"num_tiles", 0},
                 {"HtWt", 0},
                 {"n_stride", 0},
                 {"c_stride", 0},
                 {"N", 0},
                 {"C", 0}});
            AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values, core, {{"num_tiles", 0}, {"tile_freq", 0}, {"tile_start", 0}});
            continue;
        }

        uint32_t cHtWt = cHt * cWt;
        const auto scalar = eps;
        const auto packed_scalar_eps =
            any_float32 ? std::bit_cast<uint32_t>(scalar) : pack_two_bfloat16_into_uint32({scalar, scalar});

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"eps", packed_scalar_eps},
             {"start_tile_id", start_tile_id},
             {"num_tiles", num_tiles_per_core},
             {"HtWt", cHtWt},
             {"n_stride", aHt * aWt * aC * static_cast<uint32_t>(aN > 1)},
             {"c_stride", aHt * aWt * static_cast<uint32_t>(aC > 1)},
             {"N", cN},
             {"C", cC}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"start_tile_id", start_tile_id},
             {"num_tiles", num_tiles_per_core},
             {"HtWt", cHtWt},
             {"n_stride", bHt * bWt * bC * static_cast<uint32_t>(bN > 1)},
             {"c_stride", bHt * bWt * static_cast<uint32_t>(bC > 1)},
             {"N", cN},
             {"C", cC}});

        auto counter = start_tile_id % cHtWt;
        auto freq = cHtWt;

        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            core,
            {{"num_tiles", num_tiles_per_core}, {"tile_freq", freq}, {"tile_start", counter}});

        start_tile_id += num_tiles_per_core;
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::normalization {
ttnn::device_operation::ProgramArtifacts BatchNormOperation::BatchNormFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& [input_tensor, batch_mean_tensor, batch_var_tensor, weight_tensor, bias_tensor, _] = tensor_args;

    auto* device = input_tensor.device();

    const bool weight_has_value = weight_tensor.has_value();
    const bool bias_has_value = bias_tensor.has_value();

    auto a_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    auto b_data_format = datatype_to_dataformat_converter(batch_mean_tensor.dtype());
    auto c_data_format = datatype_to_dataformat_converter(output.dtype());
    auto d_data_format = datatype_to_dataformat_converter(batch_var_tensor.dtype());
    auto e_data_format =
        weight_has_value ? datatype_to_dataformat_converter(weight_tensor->dtype()) : DataFormat::Float16_b;
    auto f_data_format =
        bias_has_value ? datatype_to_dataformat_converter(bias_tensor->dtype()) : DataFormat::Float16_b;

    const bool any_float32 =
        (a_data_format == DataFormat::Float32 || b_data_format == DataFormat::Float32 ||
         c_data_format == DataFormat::Float32 || d_data_format == DataFormat::Float32 ||
         e_data_format == DataFormat::Float32 || f_data_format == DataFormat::Float32);
    auto interm_data_format = any_float32 ? DataFormat::Float32 : a_data_format;

    uint32_t a_single_tile_size = tt::tile_size(a_data_format);
    uint32_t b_single_tile_size = tt::tile_size(b_data_format);
    uint32_t c_single_tile_size = tt::tile_size(c_data_format);
    uint32_t d_single_tile_size = tt::tile_size(d_data_format);
    uint32_t e_single_tile_size = tt::tile_size(e_data_format);
    uint32_t f_single_tile_size = tt::tile_size(f_data_format);
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);

    // If accumulation occurs in float32 but output dtype is different, the compute kernel must typecast from
    // float32 to output dtype
    const bool needs_output_typecast =
        (interm_data_format == DataFormat::Float32 && c_data_format != DataFormat::Float32);

    // we parallelize the computation across the output tiles
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRangeSet(CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1}));

    // Number of tiles to store per input DFB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    uint32_t b_num_tiles_per_cb = num_tiles_per_cb;

    // ---- Program-scope resource names (drive the generated dfb:: / tensor:: tokens) ----
    // Declared function-local: this factory and running_statistics_program_factory.cpp land in the
    // same unity-build translation unit, so no anonymous-namespace constants are introduced.
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName BATCH_MEAN_DFB{"batch_mean"};
    const DFBSpecName OUTPUT_0_DFB{"output_0"};
    const DFBSpecName BATCH_VAR_DFB{"batch_var"};
    const DFBSpecName EPS_DFB{"eps"};
    const DFBSpecName WEIGHT_DFB{"weight"};
    const DFBSpecName BIAS_DFB{"bias"};
    const DFBSpecName DEN_DFB{"den"};
    const DFBSpecName TEMP_1_DFB{"temp_1"};
    const DFBSpecName WRITER_OUTPUT_DFB{"writer_output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName BATCH_MEAN_TENSOR{"batch_mean"};
    const TensorParamName BATCH_VAR_TENSOR{"batch_var"};
    const TensorParamName WEIGHT_TENSOR{"weight"};
    const TensorParamName BIAS_TENSOR{"bias"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    ProgramSpec spec;
    spec.name = "batch_norm";

    // ---- Dataflow buffers ----
    // Input buffers
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_DFB,
        .entry_size = a_single_tile_size,
        .num_entries = num_tiles_per_cb,
        .data_format_metadata = a_data_format,
    });  // input
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = BATCH_MEAN_DFB,
        .entry_size = b_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = b_data_format,
    });  // batch_mean
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUTPUT_0_DFB,
        .entry_size = needs_output_typecast ? interm_single_tile_size : c_single_tile_size,
        .num_entries = num_tiles_per_cb,
        .data_format_metadata = needs_output_typecast ? interm_data_format : c_data_format,
    });  // compute output (staging when typecast)

    // The writer drains the writer-facing DFB when the compute kernel has to typecast, and the
    // compute-output DFB itself otherwise: with no typecast the legacy factory pointed both CB
    // indices at the same buffer, so the writer's one accessor name resolves here rather than
    // through a kernel-side alias.
    const DFBSpecName writer_output_dfb = needs_output_typecast ? WRITER_OUTPUT_DFB : OUTPUT_0_DFB;
    if (needs_output_typecast) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = WRITER_OUTPUT_DFB,
            .entry_size = c_single_tile_size,
            .num_entries = num_tiles_per_cb,
            .data_format_metadata = c_data_format,
        });  // writer-facing output (BF16)
    }
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = BATCH_VAR_DFB,
        .entry_size = d_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = d_data_format,
    });  // batch_var
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = EPS_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = interm_data_format,
    });  // eps
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WEIGHT_DFB,
        .entry_size = e_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = e_data_format,
    });  // weight
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = BIAS_DFB,
        .entry_size = f_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = f_data_format,
    });  // bias

    // Temporary buffers to store intermediate results
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = DEN_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = num_tiles_per_cb,
        .data_format_metadata = interm_data_format,
    });  // to store 1/(sqrt(batch_var + eps))
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = TEMP_1_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = num_tiles_per_cb,
        .data_format_metadata = interm_data_format,
    });

    // ---- Tensor parameters (replace the buffer-address RTAs and the TensorAccessorArgs plumbing) ----
    // weight / bias are declared only when present: there is no tensor to supply as a TensorArgument
    // otherwise, so their kernel-side accessors are #ifdef-gated instead.
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_TENSOR, .spec = input_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = BATCH_MEAN_TENSOR, .spec = batch_mean_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = BATCH_VAR_TENSOR, .spec = batch_var_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()});
    if (weight_has_value) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = WEIGHT_TENSOR, .spec = weight_tensor->tensor_spec()});
    }
    if (bias_has_value) {
        spec.tensor_parameters.push_back(TensorParameter{.unique_id = BIAS_TENSOR, .spec = bias_tensor->tensor_spec()});
    }

    // ---- READER KERNEL ----
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_DFB,
                    .accessor_name = "src",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = EPS_DFB,
                    .accessor_name = "eps",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}},
        .compile_time_args = {{"fill_eps_fp32", static_cast<uint32_t>(any_float32)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"eps", "start_tile_id", "num_tiles", "HtWt", "n_stride", "c_stride", "N", "C"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    });

    // ---- WRITER KERNEL ----
    // The writer is a producer on batch_mean / batch_var / weight / bias (it reads tensor memory
    // into them) as well as the consumer of the compute output.
    KernelSpec::CompilerOptions::Defines writer_defines;
    Group<TensorBinding> writer_tensor_bindings = {
        TensorBinding{.tensor_parameter_name = BATCH_MEAN_TENSOR, .accessor_name = "src"},
        TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"},
        TensorBinding{.tensor_parameter_name = BATCH_VAR_TENSOR, .accessor_name = "batch_var"},
    };
    if (weight_has_value) {
        writer_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = WEIGHT_TENSOR, .accessor_name = "weight"});
        writer_defines.emplace("WEIGHT_HAS_VALUE", "1");
    }
    if (bias_has_value) {
        writer_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = BIAS_TENSOR, .accessor_name = "bias"});
        writer_defines.emplace("BIAS_HAS_VALUE", "1");
    }

    auto param_data_format =
        weight_has_value ? e_data_format : (bias_has_value ? f_data_format : DataFormat::Float16_b);

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp",
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = BATCH_MEAN_DFB,
                    .accessor_name = "src",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = writer_output_dfb,
                    .accessor_name = "dst",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = BATCH_VAR_DFB,
                    .accessor_name = "batch_var",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // Bound unconditionally even when the optional tensor is absent: the legacy host
                // allocated these buffers in every configuration and the kernel constructs their
                // DataflowBuffer objects outside the conditional.
                DFBBinding{
                    .dfb_spec_name = WEIGHT_DFB,
                    .accessor_name = "weight",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = BIAS_DFB,
                    .accessor_name = "bias",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings = std::move(writer_tensor_bindings),
        .compile_time_args =
            {
                {"batch_stat_is_fp32", static_cast<uint32_t>(b_data_format == DataFormat::Float32)},
                {"param_is_fp32", static_cast<uint32_t>(param_data_format == DataFormat::Float32)},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"start_tile_id", "num_tiles", "HtWt", "n_stride", "c_stride", "N", "C"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    });

    // ---- COMPUTE KERNEL ----
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    // den holds 1/sqrt(batch_var + eps), packed and then re-read by this same kernel; temp_1 is
    // reached only through the kernel's runtime dfb_affine_or_out / dfb_scaled_output aliases and is
    // likewise both packed and re-read here. Each is bound as PRODUCER and CONSUMER (a self-loop).
    Group<DFBBinding> compute_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = INPUT_DFB,
            .accessor_name = "input",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = BATCH_MEAN_DFB,
            .accessor_name = "batch_mean",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = OUTPUT_0_DFB,
            .accessor_name = "output_0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = BATCH_VAR_DFB,
            .accessor_name = "batch_var",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = EPS_DFB,
            .accessor_name = "eps",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = WEIGHT_DFB,
            .accessor_name = "weight",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = BIAS_DFB,
            .accessor_name = "bias",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = DEN_DFB,
            .accessor_name = "den",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = DEN_DFB,
            .accessor_name = "den",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = TEMP_1_DFB,
            .accessor_name = "temp_1",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = TEMP_1_DFB,
            .accessor_name = "temp_1",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
    };

    // On the typecast path this kernel re-reads its own FP32 staging buffer to typecast it into the
    // writer-facing DFB, so output_0 becomes a compute self-loop and the writer-facing DFB appears.
    // The define lets the kernel name the writer-facing token only where it is bound.
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (needs_output_typecast) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = OUTPUT_0_DFB,
            .accessor_name = "output_0",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WRITER_OUTPUT_DFB,
            .accessor_name = "output_final",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_defines.emplace("NEEDS_OUTPUT_TYPECAST", "1");
    }

    auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config);
    if (auto* compute_gen1 = std::get_if<ComputeGen1Config>(&compute_hw); compute_gen1 && fp32_dest_acc_en) {
        // Legacy set unpack_to_dest_mode[cb] = UnpackToDestFp32 on these eight CBs when fp32
        // accumulation is on, plus the compute-output CB only on the typecast path; reindexed onto
        // DFB names and translated to the Metal 2.0 spelling. Leaving output_0's entry ungated would
        // be accepted by the validator but would silently flip its unpack mode in the other config.
        ComputeUnpackModes unpack_modes = {
            {INPUT_DFB, UnpackMode::UnpackToDest},
            {BATCH_MEAN_DFB, UnpackMode::UnpackToDest},
            {BATCH_VAR_DFB, UnpackMode::UnpackToDest},
            {EPS_DFB, UnpackMode::UnpackToDest},
            {DEN_DFB, UnpackMode::UnpackToDest},
            {WEIGHT_DFB, UnpackMode::UnpackToDest},
            {TEMP_1_DFB, UnpackMode::UnpackToDest},
            {BIAS_DFB, UnpackMode::UnpackToDest},
        };
        if (needs_output_typecast) {
            unpack_modes.insert({OUTPUT_0_DFB, UnpackMode::UnpackToDest});
        }
        compute_gen1->unpack_modes = std::move(unpack_modes);
    }

    // Both compute sources bind this one KernelSpec, so the named compile-time argument set is the
    // superset the SFPU source reads; the plain source ignores the four it does not read.
    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = fmt::format(
            "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_{}.cpp",
            (fp32_dest_acc_en || any_float32) ? "sfpu_kernel" : "kernel"),
        // O3 is the legacy ComputeConfig default; Metal 2.0's CompilerOptions defaults to O2, so the
        // level has to be stated explicitly to keep the compute kernel where it was.
        .compiler_options = {.defines = std::move(compute_defines), .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args =
            {
                {"weight_has_value", static_cast<uint32_t>(weight_has_value)},
                {"bias_has_value", static_cast<uint32_t>(bias_has_value)},
                {"needs_output_typecast", static_cast<uint32_t>(needs_output_typecast)},
                {"tc_in_fmt", static_cast<uint32_t>(DataFormat::Float32)},
                {"tc_out_fmt",
                 needs_output_typecast ? static_cast<uint32_t>(c_data_format)
                                       : static_cast<uint32_t>(DataFormat::Float32)},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "tile_freq", "tile_start"}},
        .hw_config = compute_hw,
    });

    // ---- Work unit (placement) ----
    // All three legacy KernelDescriptors shared one core_ranges, so one work unit reproduces
    // placement exactly and satisfies the local-DFB identical-work-unit-membership invariant.
    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = all_device_cores,
    });

    // ---- Runtime arguments per core ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    CMAKE_UNIQUE_NAMESPACE::populate_runtime_arguments(
        reader_run_args,
        writer_run_args,
        compute_run_args,
        compute_with_storage_grid_size,
        any_float32,
        operation_attributes,
        tensor_args,
        output);

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));

    run_args.tensor_args.emplace(INPUT_TENSOR, TensorArgument{input_tensor.mesh_tensor()});
    run_args.tensor_args.emplace(BATCH_MEAN_TENSOR, TensorArgument{batch_mean_tensor.mesh_tensor()});
    run_args.tensor_args.emplace(BATCH_VAR_TENSOR, TensorArgument{batch_var_tensor.mesh_tensor()});
    run_args.tensor_args.emplace(OUTPUT_TENSOR, TensorArgument{output.mesh_tensor()});
    if (weight_has_value) {
        run_args.tensor_args.emplace(WEIGHT_TENSOR, TensorArgument{weight_tensor->mesh_tensor()});
    }
    if (bias_has_value) {
        run_args.tensor_args.emplace(BIAS_TENSOR, TensorArgument{bias_tensor->mesh_tensor()});
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::normalization
