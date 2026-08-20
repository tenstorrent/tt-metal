// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "running_statistics_device_operation.hpp"

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
// unity-build target its own CMAKE_UNIQUE_NAMESPACE, so the sibling batch-norm factory can use the
// same spellings without colliding.
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
const KernelSpecName COMPUTE{"compute"};

// Dataflow buffers, in the order the legacy factory allocated their circular buffers.
const DFBSpecName BATCH_MEAN_DFB{"batch_mean"};
const DFBSpecName BATCH_VAR_DFB{"batch_var"};
const DFBSpecName OUTPUT_DFB{"output"};
const DFBSpecName OLD_RUNNING_MEAN_DFB{"old_running_mean"};
const DFBSpecName OLD_RUNNING_VAR_DFB{"old_running_var"};
const DFBSpecName MOMENTUM_DFB{"momentum"};
const DFBSpecName ONE_DFB{"one"};                                  // one tile filled with 1.0
const DFBSpecName UPDATED_MEAN_DFB{"updated_mean"};                // FP32 staging when typecasting
const DFBSpecName UPDATED_VAR_DFB{"updated_var"};                  // FP32 staging when typecasting
const DFBSpecName WRITER_UPDATED_MEAN_DFB{"writer_updated_mean"};  // only when typecasting the mean
const DFBSpecName WRITER_UPDATED_VAR_DFB{"writer_updated_var"};    // only when typecasting the var

const TensorParamName BATCH_MEAN{"batch_mean"};
const TensorParamName BATCH_VAR{"batch_var"};
const TensorParamName RUNNING_MEAN{"running_mean"};
const TensorParamName RUNNING_VAR{"running_var"};
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
    const RunningStatistics::operation_attributes_t& operation_attributes,
    const tt::tt_metal::MeshTensor& batch_mean_tensor,
    const tt::tt_metal::MeshTensor& batch_var_tensor,
    const tt::tt_metal::MeshTensor& c) {
    const auto momentum = operation_attributes.momentum;

    const auto [aN, aC, aHt, aWt] = extract_shape_dims(batch_mean_tensor);
    const auto [bN, bC, bHt, bWt] = extract_shape_dims(batch_var_tensor);
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
            // still need a value for every named argument. They get zeros, and the kernels do no
            // work at num_tiles == 0 -- same as the all-zero argument vector legacy handed them.
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"momentum", 0u},
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
            AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"num_tiles", 0u}});
            continue;
        }

        uint32_t cHtWt = cHt * cWt;
        const auto scalar = momentum;
        const auto packed_scalar_momentum =
            any_float32 ? std::bit_cast<uint32_t>(scalar) : pack_two_bfloat16_into_uint32({scalar, scalar});

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"momentum", packed_scalar_momentum},
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

        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"num_tiles", num_tiles_per_core}});

        start_tile_id += num_tiles_per_core;
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::normalization {
ttnn::device_operation::ProgramArtifacts RunningStatistics::RunningStatisticsProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& batch_mean_tensor = tensor_args.batch_mean.mesh_tensor();
    const auto& batch_var_tensor = tensor_args.batch_var.mesh_tensor();
    const auto& output_tensor = output.mesh_tensor();
    const auto& running_mean_tensor = tensor_args.running_mean;
    const auto& running_var_tensor = tensor_args.running_var;

    IDevice* device = &batch_mean_tensor.mutable_device();

    const bool running_mean_has_value = running_mean_tensor.has_value();
    const bool running_var_has_value = running_var_tensor.has_value();

    auto a_data_format = datatype_to_dataformat_converter(batch_mean_tensor.dtype());
    auto b_data_format = datatype_to_dataformat_converter(batch_var_tensor.dtype());
    auto c_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    auto d_data_format =
        running_mean_has_value ? datatype_to_dataformat_converter(running_mean_tensor->dtype()) : DataFormat::Float16_b;
    auto e_data_format =
        running_var_has_value ? datatype_to_dataformat_converter(running_var_tensor->dtype()) : DataFormat::Float16_b;

    const bool any_float32 =
        (a_data_format == DataFormat::Float32 || b_data_format == DataFormat::Float32 ||
         c_data_format == DataFormat::Float32 || d_data_format == DataFormat::Float32 ||
         e_data_format == DataFormat::Float32);
    auto interm_data_format = any_float32 ? DataFormat::Float32 : a_data_format;

    uint32_t a_single_tile_size = tt::tile_size(a_data_format);
    uint32_t b_single_tile_size = tt::tile_size(b_data_format);
    uint32_t c_single_tile_size = tt::tile_size(c_data_format);
    uint32_t d_single_tile_size = tt::tile_size(d_data_format);
    uint32_t e_single_tile_size = tt::tile_size(e_data_format);
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);

    auto running_stat_data_format =
        running_mean_has_value ? d_data_format : (running_var_has_value ? e_data_format : DataFormat::Float16_b);
    const bool stat_format_needs_typecast =
        (interm_data_format == DataFormat::Float32 && running_stat_data_format != DataFormat::Float32);
    const bool needs_mean_typecast = running_mean_has_value && stat_format_needs_typecast;
    const bool needs_var_typecast = running_var_has_value && stat_format_needs_typecast;

    // we parallelize the computation across the output tiles
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = NodeRangeSet(NodeRange({0, 0}, {num_cores_x - 1, num_cores_y - 1}));

    // Number of tiles to store per input DFB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    uint32_t b_num_tiles_per_cb = num_tiles_per_cb;

    Group<DataflowBufferSpec> dataflow_buffers{
        make_dfb(BATCH_MEAN_DFB, a_data_format, a_single_tile_size, num_tiles_per_cb),
        make_dfb(BATCH_VAR_DFB, b_data_format, b_single_tile_size, b_num_tiles_per_cb),
        make_dfb(OUTPUT_DFB, c_data_format, c_single_tile_size, num_tiles_per_cb),
        make_dfb(OLD_RUNNING_MEAN_DFB, d_data_format, d_single_tile_size, b_num_tiles_per_cb),
        make_dfb(OLD_RUNNING_VAR_DFB, e_data_format, e_single_tile_size, b_num_tiles_per_cb),
        make_dfb(MOMENTUM_DFB, interm_data_format, interm_single_tile_size, b_num_tiles_per_cb),
        make_dfb(ONE_DFB, interm_data_format, interm_single_tile_size, b_num_tiles_per_cb),
        // The compute kernel packs the updated stats here. When the accumulation format is wider
        // than the stat dtype these are FP32 staging that the typecast stage reads back; otherwise
        // they are the buffers the writer drains.
        make_dfb(
            UPDATED_MEAN_DFB,
            needs_mean_typecast ? interm_data_format : d_data_format,
            needs_mean_typecast ? interm_single_tile_size : d_single_tile_size,
            b_num_tiles_per_cb),
        make_dfb(
            UPDATED_VAR_DFB,
            needs_var_typecast ? interm_data_format : e_data_format,
            needs_var_typecast ? interm_single_tile_size : e_single_tile_size,
            b_num_tiles_per_cb),
    };
    if (needs_mean_typecast) {
        dataflow_buffers.push_back(
            make_dfb(WRITER_UPDATED_MEAN_DFB, d_data_format, d_single_tile_size, b_num_tiles_per_cb));
    }
    if (needs_var_typecast) {
        dataflow_buffers.push_back(
            make_dfb(WRITER_UPDATED_VAR_DFB, e_data_format, e_single_tile_size, b_num_tiles_per_cb));
    }
    // The DFBs the writer drains are whichever buffers the compute kernel finally packs into. One
    // binding under one accessor name covers both paths, so the writer needs no preprocessor gate.
    // The two stats are keyed independently: one may typecast while the other does not.
    const DFBSpecName& writer_updated_mean_dfb = needs_mean_typecast ? WRITER_UPDATED_MEAN_DFB : UPDATED_MEAN_DFB;
    const DFBSpecName& writer_updated_var_dfb = needs_var_typecast ? WRITER_UPDATED_VAR_DFB : UPDATED_VAR_DFB;

    Group<TensorParameter> tensor_parameters{
        TensorParameter{.unique_id = BATCH_MEAN, .spec = batch_mean_tensor.tensor_spec()},
        TensorParameter{.unique_id = BATCH_VAR, .spec = batch_var_tensor.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()},
    };

    // READER KERNEL
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_running_statistics.cpp",
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = BATCH_MEAN_DFB,
                 .accessor_name = "src",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = MOMENTUM_DFB,
                 .accessor_name = "momentum",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = ONE_DFB,
                 .accessor_name = "one",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = BATCH_MEAN, .accessor_name = "batch_mean"}},
        .compile_time_args = {{"fill_momentum_fp32", static_cast<uint32_t>(any_float32)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"momentum", "start_tile_id", "num_tiles", "HtWt", "n_stride", "c_stride", "N", "C"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // WRITER KERNEL
    // It is a reader-writer: it pulls batch_var and the old running stats from DRAM on the compute
    // kernel's behalf (so it PRODUCES those three DFBs), and writes the updated stats back to the
    // same pages of the same tensors it read them from.
    Group<DFBBinding> writer_dfb_bindings{
        DFBBinding{
            .dfb_spec_name = BATCH_VAR_DFB,
            .accessor_name = "src",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = OUTPUT_DFB,
            .accessor_name = "dst",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        // The old-stat DFBs are bound whether or not their tensor is present: the kernel reads their
        // entry size outside the has-value guards, and legacy allocated the buffers unconditionally.
        DFBBinding{
            .dfb_spec_name = OLD_RUNNING_MEAN_DFB,
            .accessor_name = "old_mean",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = OLD_RUNNING_VAR_DFB,
            .accessor_name = "old_var",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = writer_updated_mean_dfb,
            .accessor_name = "new_mean",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = writer_updated_var_dfb,
            .accessor_name = "new_var",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
    };

    Group<TensorBinding> writer_tensor_bindings{
        TensorBinding{.tensor_parameter_name = BATCH_VAR, .accessor_name = "batch_var"},
        TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"},
    };
    // An absent optional tensor has no binding and therefore no tensor:: token, so the kernel's
    // accessor construction has to disappear at the preprocessor stage rather than under an
    // if constexpr -- which would still look the name up in its discarded branch.
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (running_mean_has_value) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = RUNNING_MEAN, .spec = running_mean_tensor->mesh_tensor().tensor_spec()});
        writer_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = RUNNING_MEAN, .accessor_name = "running_mean"});
        writer_defines["OLD_RUNNING_MEAN_HAS_VALUE"] = "1";
    }
    if (running_var_has_value) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = RUNNING_VAR, .spec = running_var_tensor->mesh_tensor().tensor_spec()});
        writer_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = RUNNING_VAR, .accessor_name = "running_var"});
        writer_defines["OLD_RUNNING_VAR_HAS_VALUE"] = "1";
    }

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_running_statistics.cpp",
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings = std::move(writer_dfb_bindings),
        .tensor_bindings = std::move(writer_tensor_bindings),
        .compile_time_args =
            {{"old_stat_is_fp32", static_cast<uint32_t>(running_stat_data_format == DataFormat::Float32)}},
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
            .dfb_spec_name = BATCH_MEAN_DFB,
            .accessor_name = "batch_mean",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = BATCH_VAR_DFB,
            .accessor_name = "batch_var",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = OUTPUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        // The old-stat DFBs are gated by an if constexpr inside the kernel but bound unconditionally,
        // matching the legacy unconditional CB allocation and the kernel's unconditional handles.
        DFBBinding{
            .dfb_spec_name = OLD_RUNNING_MEAN_DFB,
            .accessor_name = "old_running_mean",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = OLD_RUNNING_VAR_DFB,
            .accessor_name = "old_running_var",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = MOMENTUM_DFB,
            .accessor_name = "momentum",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = ONE_DFB,
            .accessor_name = "one",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = UPDATED_MEAN_DFB,
            .accessor_name = "updated_mean",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = UPDATED_VAR_DFB,
            .accessor_name = "updated_var",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };

    KernelSpec::CompileTimeArgs compute_compile_time_args{
        {"old_running_mean_has_value", static_cast<uint32_t>(running_mean_has_value)},
        {"old_running_var_has_value", static_cast<uint32_t>(running_var_has_value)},
    };
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (use_sfpu_kernel) {
        compute_compile_time_args["tc_in_fmt"] = static_cast<uint32_t>(DataFormat::Float32);
        compute_compile_time_args["tc_out_fmt"] = stat_format_needs_typecast
                                                      ? static_cast<uint32_t>(running_stat_data_format)
                                                      : static_cast<uint32_t>(DataFormat::Float32);
    }
    // needs_*_typecast implies interm == Float32 implies any_float32, so these only ever reach the
    // SFPU source -- the only one carrying the typecast stage. The host computes the two flags so a
    // single define gates a single alias, rather than the kernel re-deriving them from two CTAs.
    if (needs_mean_typecast) {
        compute_defines["NEEDS_MEAN_TYPECAST"] = "1";
        // The typecast stage reads its own FP32 staging output back before packing the narrower
        // result into the writer-facing buffer.
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UPDATED_MEAN_DFB,
            .accessor_name = "updated_mean",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WRITER_UPDATED_MEAN_DFB,
            .accessor_name = "writer_updated_mean",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }
    if (needs_var_typecast) {
        compute_defines["NEEDS_VAR_TYPECAST"] = "1";
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UPDATED_VAR_DFB,
            .accessor_name = "updated_var",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WRITER_UPDATED_VAR_DFB,
            .accessor_name = "writer_updated_var",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }

    auto compute_hw_config =
        ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config);
    if (fp32_dest_acc_en) {
        // Re-key of the legacy unpack_to_dest_mode vector, which was indexed by CB id. The
        // writer-facing stat buffers are producer-only for this kernel, so they get no entry. An
        // omitted DFB keeps the UnpackToSrc default.
        auto& unpack_modes = std::get<ComputeGen1Config>(compute_hw_config).unpack_modes;
        for (const auto& dfb_name :
             {BATCH_MEAN_DFB,
              BATCH_VAR_DFB,
              OUTPUT_DFB,
              OLD_RUNNING_MEAN_DFB,
              OLD_RUNNING_VAR_DFB,
              UPDATED_MEAN_DFB,
              UPDATED_VAR_DFB,
              MOMENTUM_DFB,
              ONE_DFB}) {
            unpack_modes[dfb_name] = UnpackMode::UnpackToDest;
        }
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = fmt::format(
            "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/running_statistics_{}.cpp",
            use_sfpu_kernel ? "sfpu_kernel" : "kernel"),
        // O3 is the level the legacy ComputeConfigDescriptor defaulted a compute kernel to; Metal
        // 2.0's CompilerOptions defaults to O2 for every kernel kind, so it is stated here.
        .compiler_options = {.defines = std::move(compute_defines), .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args = std::move(compute_compile_time_args),
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
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
        batch_mean_tensor,
        batch_var_tensor,
        output_tensor);

    ProgramSpec spec{
        .name = "running_statistics",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        // Legacy placed every kernel on every device core and padded the idle ones with zero
        // arguments rather than narrowing the grid; that placement is preserved here.
        .work_units = {WorkUnitSpec{
            .name = "running_statistics",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = all_device_cores,
        }},
    };

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)},
        .tensor_args = {{BATCH_MEAN, batch_mean_tensor}, {BATCH_VAR, batch_var_tensor}, {OUTPUT, output_tensor}},
    };
    if (running_mean_has_value) {
        run_args.tensor_args.emplace(RUNNING_MEAN, running_mean_tensor->mesh_tensor());
    }
    if (running_var_has_value) {
        run_args.tensor_args.emplace(RUNNING_VAR, running_var_tensor->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::normalization
