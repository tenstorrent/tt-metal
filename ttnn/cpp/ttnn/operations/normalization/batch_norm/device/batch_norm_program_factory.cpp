// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_device_operation.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <bit>
#include <cmath>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::normalization;
using namespace tt::tt_metal::experimental;

// Kernel identities within the ProgramSpec. Names are file-local: CMake gives every source in a
// unity-build target its own CMAKE_UNIQUE_NAMESPACE, so the sibling running-statistics factory can
// use the same spellings without colliding.
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
const KernelSpecName COMPUTE{"compute"};

// Dataflow buffers, in the order the legacy factory allocated their circular buffers.
const DFBSpecName INPUT_DFB{"input"};            // input tiles, DRAM -> compute
const DFBSpecName BATCH_MEAN_DFB{"batch_mean"};  // per-channel batch mean, broadcast against input
const DFBSpecName OUT_DFB{"out"};                // compute result; FP32 staging when typecasting
const DFBSpecName BATCH_VAR_DFB{"batch_var"};    // per-channel batch variance
const DFBSpecName EPS_DFB{"eps"};                // one tile filled with eps, held for the whole kernel
const DFBSpecName WEIGHT_DFB{"weight"};          // optional affine scale
const DFBSpecName BIAS_DFB{"bias"};              // optional affine shift
const DFBSpecName DEN_DFB{"den"};                // 1/(sqrt(batch_var + eps))
const DFBSpecName TEMP_1_DFB{"temp_1"};          // (input - batch_mean)/(sqrt(batch_var + eps))
const DFBSpecName WRITER_OUT_DFB{"writer_out"};  // writer-facing output, only when typecasting

const TensorParamName INPUT{"input"};
const TensorParamName BATCH_MEAN{"batch_mean"};
const TensorParamName BATCH_VAR{"batch_var"};
const TensorParamName WEIGHT{"weight"};
const TensorParamName BIAS{"bias"};
const TensorParamName OUTPUT{"output"};

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_shape_dims(const tt::tt_metal::MeshTensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
}

DataflowBufferSpec make_dfb(
    const DFBSpecName& unique_id, tt::DataFormat data_format, uint32_t entry_size, uint32_t num_entries) {
    return DataflowBufferSpec{
        .unique_id = unique_id,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = data_format,
    };
}

void populate_runtime_arguments(
    KernelRunArgs& reader_run_args,
    KernelRunArgs& writer_run_args,
    KernelRunArgs& compute_run_args,
    NodeCoord compute_with_storage_grid_size,
    bool any_float32,
    const BatchNormOperation::operation_attributes_t& operation_attributes,
    const tt::tt_metal::MeshTensor& input_tensor,
    const tt::tt_metal::MeshTensor& batch_mean_tensor,
    const tt::tt_metal::MeshTensor& c) {
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

    auto cores = grid_to_nodes(num_cores_total, num_cores_x, num_cores_y, row_major);

    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // The kernels are placed on every device core, so the nodes outside both work groups
            // still need a value for every named argument. They get zeros, and the kernels return
            // early on num_tiles == 0 -- same as the all-zero argument vector legacy handed them.
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"eps", 0u},
                 {"start_tile_id", 0u},
                 {"num_tiles", 0u},
                 {"HtWt", 0u},
                 {"n_stride", 0u},
                 {"c_stride", 0u},
                 {"N", 0u},
                 {"C", 0u}});
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"start_tile_id", 0u},
                 {"num_tiles", 0u},
                 {"HtWt", 0u},
                 {"n_stride", 0u},
                 {"c_stride", 0u},
                 {"N", 0u},
                 {"C", 0u}});
            AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values, core, {{"num_tiles", 0u}, {"tile_freq", 0u}, {"tile_start", 0u}});
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

        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            core,
            {{"num_tiles", num_tiles_per_core}, {"tile_freq", cHtWt}, {"tile_start", start_tile_id % cHtWt}});

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
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& input_tensor = tensor_args.input.mesh_tensor();
    const auto& batch_mean_tensor = tensor_args.batch_mean.mesh_tensor();
    const auto& batch_var_tensor = tensor_args.batch_var.mesh_tensor();
    const auto& output_tensor = output.mesh_tensor();
    const auto& weight_tensor = tensor_args.weight;
    const auto& bias_tensor = tensor_args.bias;

    IDevice* device = &input_tensor.mutable_device();

    const bool weight_has_value = weight_tensor.has_value();
    const bool bias_has_value = bias_tensor.has_value();

    auto a_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    auto b_data_format = datatype_to_dataformat_converter(batch_mean_tensor.dtype());
    auto c_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
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
    auto all_device_cores = NodeRangeSet(NodeRange({0, 0}, {num_cores_x - 1, num_cores_y - 1}));

    // Number of tiles to store per input DFB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    uint32_t b_num_tiles_per_cb = num_tiles_per_cb;

    Group<DataflowBufferSpec> dataflow_buffers{
        make_dfb(INPUT_DFB, a_data_format, a_single_tile_size, num_tiles_per_cb),
        make_dfb(BATCH_MEAN_DFB, b_data_format, b_single_tile_size, b_num_tiles_per_cb),
        // The compute kernel packs here. When the accumulation format is wider than the output
        // dtype this is FP32 staging that the typecast stage reads back; otherwise it is the
        // buffer the writer drains.
        make_dfb(
            OUT_DFB,
            needs_output_typecast ? interm_data_format : c_data_format,
            needs_output_typecast ? interm_single_tile_size : c_single_tile_size,
            num_tiles_per_cb),
        make_dfb(BATCH_VAR_DFB, d_data_format, d_single_tile_size, b_num_tiles_per_cb),
        make_dfb(EPS_DFB, interm_data_format, interm_single_tile_size, b_num_tiles_per_cb),
        make_dfb(WEIGHT_DFB, e_data_format, e_single_tile_size, b_num_tiles_per_cb),
        make_dfb(BIAS_DFB, f_data_format, f_single_tile_size, b_num_tiles_per_cb),
        // Intermediates, produced and consumed entirely inside the compute kernel.
        make_dfb(DEN_DFB, interm_data_format, interm_single_tile_size, num_tiles_per_cb),
        make_dfb(TEMP_1_DFB, interm_data_format, interm_single_tile_size, num_tiles_per_cb),
    };
    if (needs_output_typecast) {
        // Writer-facing output at the output dtype, fed by the typecast stage.
        dataflow_buffers.push_back(make_dfb(WRITER_OUT_DFB, c_data_format, c_single_tile_size, num_tiles_per_cb));
    }
    // The DFB the writer drains is whichever buffer the compute kernel finally packs into. One
    // binding under one accessor name covers both paths, so the writer needs no preprocessor gate.
    const DFBSpecName& writer_output_dfb = needs_output_typecast ? WRITER_OUT_DFB : OUT_DFB;

    Group<TensorParameter> tensor_parameters{
        TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = BATCH_MEAN, .spec = batch_mean_tensor.tensor_spec()},
        TensorParameter{.unique_id = BATCH_VAR, .spec = batch_var_tensor.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()},
    };

    // READER KERNEL
    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm.cpp",
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_DFB,
                 .accessor_name = "src",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = EPS_DFB,
                 .accessor_name = "eps",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args = {{"fill_eps_fp32", static_cast<uint32_t>(any_float32)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"eps", "start_tile_id", "num_tiles", "HtWt", "n_stride", "c_stride", "N", "C"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // WRITER KERNEL
    // It is a reader-writer: it pulls batch_mean / batch_var / weight / bias from DRAM on the
    // compute kernel's behalf, so it PRODUCES those four DFBs and only CONSUMES the output one.
    Group<DFBBinding> writer_dfb_bindings{
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
        // weight and bias are bound whether or not their tensor is present: the kernel reads their
        // entry size outside the has-value guards, and legacy allocated the buffers unconditionally.
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
    };

    Group<TensorBinding> writer_tensor_bindings{
        TensorBinding{.tensor_parameter_name = BATCH_MEAN, .accessor_name = "batch_mean"},
        TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"},
        TensorBinding{.tensor_parameter_name = BATCH_VAR, .accessor_name = "batch_var"},
    };
    // An absent optional tensor has no binding and therefore no tensor:: token, so the kernel's
    // accessor construction has to disappear at the preprocessor stage rather than under an
    // if constexpr -- which would still look the name up in its discarded branch.
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (weight_has_value) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = WEIGHT, .spec = weight_tensor->mesh_tensor().tensor_spec()});
        writer_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = WEIGHT, .accessor_name = "weight"});
        writer_defines["WEIGHT_HAS_VALUE"] = "1";
    }
    if (bias_has_value) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = BIAS, .spec = bias_tensor->mesh_tensor().tensor_spec()});
        writer_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = BIAS, .accessor_name = "bias"});
        writer_defines["BIAS_HAS_VALUE"] = "1";
    }

    KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp",
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings = std::move(writer_dfb_bindings),
        .tensor_bindings = std::move(writer_tensor_bindings),
        .compile_time_args =
            {{"batch_stat_is_fp32", static_cast<uint32_t>(b_data_format == DataFormat::Float32)},
             {"param_is_fp32",
              static_cast<uint32_t>(
                  (weight_has_value ? e_data_format : (bias_has_value ? f_data_format : DataFormat::Float16_b)) ==
                  DataFormat::Float32)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"start_tile_id", "num_tiles", "HtWt", "n_stride", "c_stride", "N", "C"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // COMPUTE KERNEL
    // fp32_dest_acc_en selects the compute source and gates the unpack_modes list below, so it is
    // read directly. to_compute_hardware_config carries the four knobs it covers -- math_fidelity,
    // math_approx_mode, fp32_dest_acc_en and dst_full_sync_en -- into hw_config; packer_l1_acc and
    // throttle_level are deliberately not translated. ComputeGen1Config has no packer_l1_acc field,
    // so the value this op resolves for it stays unapplied, as it also was under the descriptor API.
    const bool fp32_dest_acc_en = ttnn::get_fp32_dest_acc_en(operation_attributes.compute_kernel_config);
    const bool use_sfpu_kernel = fp32_dest_acc_en || any_float32;

    Group<DFBBinding> compute_dfb_bindings{
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
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
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
        // den and temp_1 never leave the compute kernel: it packs a partial result and reads it
        // straight back, so it holds both ends of each FIFO.
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
        // weight and bias are selected by a runtime if inside the compute kernel, so they are bound
        // whether or not their tensor is present.
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
    };

    KernelSpec::CompileTimeArgs compute_compile_time_args{
        {"weight_has_value", static_cast<uint32_t>(weight_has_value)},
        {"bias_has_value", static_cast<uint32_t>(bias_has_value)},
    };
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (use_sfpu_kernel) {
        compute_compile_time_args["tc_in_fmt"] = static_cast<uint32_t>(DataFormat::Float32);
        compute_compile_time_args["tc_out_fmt"] =
            needs_output_typecast ? static_cast<uint32_t>(c_data_format) : static_cast<uint32_t>(DataFormat::Float32);
    }
    if (needs_output_typecast) {
        // needs_output_typecast implies interm == Float32 implies any_float32, so this only ever
        // reaches the SFPU source -- the only one carrying the typecast stage.
        compute_defines["NEEDS_OUTPUT_TYPECAST"] = "1";
        // The typecast stage reads its own FP32 staging output back before packing the narrower
        // result into the writer-facing buffer.
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WRITER_OUT_DFB,
            .accessor_name = "writer_out",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }

    auto compute_hw_config =
        ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config);
    if (fp32_dest_acc_en) {
        // Re-key of the legacy unpack_to_dest_mode vector, which was indexed by CB id. Every DFB the
        // compute kernel consumes is listed; the writer-facing output is producer-only, so it gets no
        // entry. An omitted DFB keeps the UnpackToSrc default.
        auto& unpack_modes = std::get<ComputeGen1Config>(compute_hw_config).unpack_modes;
        for (const auto& dfb_name :
             {INPUT_DFB, BATCH_MEAN_DFB, BATCH_VAR_DFB, EPS_DFB, DEN_DFB, WEIGHT_DFB, TEMP_1_DFB, BIAS_DFB}) {
            unpack_modes[dfb_name] = UnpackMode::UnpackToDest;
        }
        if (needs_output_typecast) {
            unpack_modes[OUT_DFB] = UnpackMode::UnpackToDest;
        }
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = fmt::format(
            "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_{}.cpp",
            use_sfpu_kernel ? "sfpu_kernel" : "kernel"),
        // O3 is the level the legacy ComputeConfigDescriptor defaulted a compute kernel to; Metal
        // 2.0's CompilerOptions defaults to O2 for every kernel kind, so it is stated here.
        .compiler_options = {.defines = std::move(compute_defines), .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args = std::move(compute_compile_time_args),
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "tile_freq", "tile_start"}},
        .hw_config = std::move(compute_hw_config),
    };

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
        input_tensor,
        batch_mean_tensor,
        output_tensor);

    ProgramSpec spec{
        .name = "batch_norm",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        // Legacy placed every kernel on every device core and padded the idle ones with zero
        // arguments rather than narrowing the grid; that placement is preserved here.
        .work_units = {WorkUnitSpec{
            .name = "batch_norm",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = all_device_cores,
        }},
    };

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)},
        .tensor_args =
            {{INPUT, input_tensor},
             {BATCH_MEAN, batch_mean_tensor},
             {BATCH_VAR, batch_var_tensor},
             {OUTPUT, output_tensor}},
    };
    if (weight_has_value) {
        run_args.tensor_args.emplace(WEIGHT, weight_tensor->mesh_tensor());
    }
    if (bias_has_value) {
        run_args.tensor_args.emplace(BIAS, bias_tensor->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::normalization
