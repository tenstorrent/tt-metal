// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/moreh/moreh_norm/device/moreh_norm_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_norm {

ttnn::device_operation::ProgramArtifacts MorehNormOperation::ProgramFactoryWOther::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& input = tensor_args.input.mesh_tensor();
    const auto& out = output.mesh_tensor();
    const auto p = operation_attributes.p;

    ////////////////////////////////////////////////////////////////////////////
    //                      Resource names
    ////////////////////////////////////////////////////////////////////////////
    // Declared function-local: ttnn_op_moreh is a unity build, so anonymous-namespace constants of
    // the same name in the three sibling factory .cpp files would collide in the merged TU.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};

    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName ONE_DFB{"one"};
    const DFBSpecName MASK_W_DFB{"mask_w"};
    const DFBSpecName OUTPUT_DFB{"output"};
    const DFBSpecName VAL_DFB{"val"};        // f(x)
    const DFBSpecName CAL_DFB{"cal"};        // calculate f(x) over dimension
    const DFBSpecName REDUCE_DFB{"reduce"};  // reduce f(x)

    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto& device = input.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.padded_shape();
    const auto input_rank = input_shape.rank();
    auto logical_shape = input.logical_shape();
    if (logical_shape.rank() < 2) {
        logical_shape = logical_shape.to_rank(2);
    }

    const auto H = input_shape[-2];
    const auto W = input_shape[-1];

    const auto Ht = H / tt::constants::TILE_HEIGHT;
    const auto Wt = W / tt::constants::TILE_WIDTH;

    const auto num_units = input.physical_volume() / H / W * Ht;

    const auto origin_w = logical_shape[input_rank - 1];

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device.compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    auto arch = device.arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, operation_attributes.compute_kernel_config);

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_units_per_core_group_1,
         num_units_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_units);

    ////////////////////////////////////////////////////////////////////////////
    //                       DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;

    const uint32_t in0_t{1};  // input
    const uint32_t in1_t{1};  // one
    const uint32_t in2_t{1};  // mask_w

    const uint32_t out0_t{1};  // output

    const uint32_t im0_t{1};  // f(x)
    const uint32_t im1_t{1};  // calculate f(x) over dimension
    const uint32_t im2_t{1};  // reduce f(x)

    // No node_ranges: a DFB's placement is derived from the WorkUnitSpec membership of the kernels
    // that bind it. The legacy CBs carried `.core_ranges = all_cores`, which the reader/writer
    // membership in both work units reproduces.
    DataflowBufferSpec dfb_input{
        .unique_id = INPUT_DFB,
        .entry_size = tile_size(cb_data_format),
        .num_entries = in0_t,
        .data_format_metadata = cb_data_format,
    };
    DataflowBufferSpec dfb_one{
        .unique_id = ONE_DFB,
        .entry_size = tile_size(cb_data_format),
        .num_entries = in1_t,
        .data_format_metadata = cb_data_format,
    };
    DataflowBufferSpec dfb_mask_w{
        .unique_id = MASK_W_DFB,
        .entry_size = tile_size(cb_data_format),
        .num_entries = in2_t,
        .data_format_metadata = cb_data_format,
    };
    DataflowBufferSpec dfb_output{
        .unique_id = OUTPUT_DFB,
        .entry_size = tile_size(cb_data_format),
        .num_entries = out0_t,
        .data_format_metadata = cb_data_format,
    };
    DataflowBufferSpec dfb_val{
        .unique_id = VAL_DFB,
        .entry_size = tile_size(intermed_data_format),
        .num_entries = im0_t,
        .data_format_metadata = intermed_data_format,
    };
    DataflowBufferSpec dfb_cal{
        .unique_id = CAL_DFB,
        .entry_size = tile_size(intermed_data_format),
        .num_entries = im1_t,
        .data_format_metadata = intermed_data_format,
    };
    DataflowBufferSpec dfb_reduce{
        .unique_id = REDUCE_DFB,
        .entry_size = tile_size(intermed_data_format),
        .num_entries = im2_t,
        .data_format_metadata = intermed_data_format,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/"
        "reader_moreh_norm_w.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/"
        "writer_moreh_norm_w.cpp";

    KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_file,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_DFB,
                    .accessor_name = "input",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = ONE_DFB,
                    .accessor_name = "one",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = MASK_W_DFB,
                    .accessor_name = "mask_w",
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
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"input_is_dram", "num_rows_per_core", "Wt", "tile_offset", "origin_w"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = writer_kernel_file,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUTPUT_DFB,
                    .accessor_name = "output",
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
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"output_is_dram", "num_rows_per_core", "Wt", "tile_offset"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelSpec::CompilerOptions::Defines compute_defines;

    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";
    if (p == 0.0f) {
        compute_defines["REDUCE_OP"] = "PoolType::SUM";
    } else {
        compute_defines["REDUCE_OP"] = "PoolType::MAX";
    }

    const KernelSpec::CompileTimeArgs compute_compile_time_args{
        {"is_zero", static_cast<uint32_t>(p == 0.0f)},
        {"minus_inf", static_cast<uint32_t>(p == -std::numeric_limits<float>::infinity())},
    };

    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/"
        "moreh_norm_w_kernel.cpp";

    auto compute_hw_config = ttnn::to_compute_hardware_config(arch, operation_attributes.compute_kernel_config);
    // The legacy config set UnpackToDestMode::Default for every CB index; Default is UnpackToSrc.
    // An explicit entry is *required* for any Float32 DFB the compute kernel consumes while
    // enable_32_bit_dest (= fp32_dest_acc_en) is set. That is reachable two independent ways here:
    // fp32_dest_acc_en makes intermed_data_format Float32, and cb_data_format is Float32 whenever the
    // input dtype is float32. Naming every consumed DFB covers both without a dtype-dependent branch,
    // and reproduces the legacy all-Default vector byte for byte. `output` is producer-only on
    // compute, so it takes no entry.
    std::get<ComputeGen1Config>(compute_hw_config).unpack_modes = {
        {INPUT_DFB, UnpackMode::UnpackToSrc},
        {ONE_DFB, UnpackMode::UnpackToSrc},
        {MASK_W_DFB, UnpackMode::UnpackToSrc},
        {VAL_DFB, UnpackMode::UnpackToSrc},
        {CAL_DFB, UnpackMode::UnpackToSrc},
        {REDUCE_DFB, UnpackMode::UnpackToSrc},
    };

    // One KernelSpec per legacy compute KernelDescriptor. The two are identical apart from their
    // unique_id — the per-group work count travels as a runtime arg, as it did in legacy — but they
    // must stay separate specs so each can sit in its own WorkUnitSpec and so land on its own core
    // group. `val`, `cal` and `reduced` are compute-private accumulators: the compute kernel is their
    // only toucher on any node, so each is self-looped (bound PRODUCER *and* CONSUMER) under a single
    // accessor name, giving the kernel one DataflowBuffer object that drives both directions.
    auto make_compute = [&](KernelSpecName unique_id) {
        return KernelSpec{
            .unique_id = std::move(unique_id),
            .source = compute_kernel_file,
            // opt_level is explicit because the default differs by API: legacy ComputeConfig
            // defaults to O3, while Metal 2.0's type-agnostic CompilerOptions defaults to O2 for
            // compute and data movement alike. The legacy compute descriptors set no opt_level, so
            // they resolved to O3; leaving this unset would silently drop a level.
            .compiler_options =
                {
                    .defines = compute_defines,
                    .opt_level = tt::tt_metal::KernelBuildOptLevel::O3,
                },
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = INPUT_DFB,
                        .accessor_name = "x",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = ONE_DFB,
                        .accessor_name = "one",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = MASK_W_DFB,
                        .accessor_name = "mask_w",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = OUTPUT_DFB,
                        .accessor_name = "y",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = VAL_DFB,
                        .accessor_name = "val",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = VAL_DFB,
                        .accessor_name = "val",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = CAL_DFB,
                        .accessor_name = "cal",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = CAL_DFB,
                        .accessor_name = "cal",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = REDUCE_DFB,
                        .accessor_name = "reduce",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = REDUCE_DFB,
                        .accessor_name = "reduce",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                },
            .compile_time_args = compute_compile_time_args,
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"num_rows_per_core", "Wt", "origin_w"},
                },
            .hw_config = compute_hw_config,
        };
    };

    const bool has_core_group_2 = !core_group_2.ranges().empty();

    Group<KernelSpec> kernels = {std::move(reader), std::move(writer), make_compute(COMPUTE_G1)};

    // reader and writer belong to BOTH work units, so their derived node set is
    // core_group_1 | core_group_2 == all_cores (the legacy core_ranges), while each compute spec stays
    // on its own group. This co-membership is also what the local-DFB invariant needs: on every node,
    // each DFB sees exactly one producer instance and one consumer instance.
    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{
            .name = "wu_g1",
            .kernels = {READER, WRITER, COMPUTE_G1},
            .target_nodes = core_group_1,
        },
    };
    if (has_core_group_2) {
        kernels.push_back(make_compute(COMPUTE_G2));
        work_units.push_back(WorkUnitSpec{
            .name = "wu_g2",
            .kernels = {READER, WRITER, COMPUTE_G2},
            .target_nodes = core_group_2,
        });
    }

    ProgramSpec spec{
        .name = "moreh_norm_w_other",
        .kernels = std::move(kernels),
        .dataflow_buffers =
            {std::move(dfb_input),
             std::move(dfb_one),
             std::move(dfb_mask_w),
             std::move(dfb_output),
             std::move(dfb_val),
             std::move(dfb_cal),
             std::move(dfb_reduce)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = out.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    // The legacy per-core loop is kept as-is; AddRuntimeArgsForNode transposes each node's values
    // into ProgramRunArgs' name-first (name -> node -> value) table.
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_g1_run_args{.kernel = COMPUTE_G1};
    KernelRunArgs compute_g2_run_args{.kernel = COMPUTE_G2};

    const auto input_is_dram = static_cast<uint32_t>(is_dram(tensor_args.input));
    const auto output_is_dram = static_cast<uint32_t>(is_dram(output));

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_units_per_core;
        if (core_group_1.contains(core)) {
            num_units_per_core = num_units_per_core_group_1;
            AddRuntimeArgsForNode(
                compute_g1_run_args.runtime_arg_values,
                core,
                {
                    {"num_rows_per_core", num_units_per_core},
                    {"Wt", static_cast<uint32_t>(Wt)},
                    {"origin_w", static_cast<uint32_t>(origin_w)},
                });
        } else if (core_group_2.contains(core)) {
            num_units_per_core = num_units_per_core_group_2;
            AddRuntimeArgsForNode(
                compute_g2_run_args.runtime_arg_values,
                core,
                {
                    {"num_rows_per_core", num_units_per_core},
                    {"Wt", static_cast<uint32_t>(Wt)},
                    {"origin_w", static_cast<uint32_t>(origin_w)},
                });
        } else {
            TT_THROW("Core not in specified core ranges.");
        }
        // reader
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"input_is_dram", input_is_dram},
             {"num_rows_per_core", num_units_per_core},
             {"Wt", static_cast<uint32_t>(Wt)},
             {"tile_offset", tile_offset},
             {"origin_w", static_cast<uint32_t>(origin_w)}});

        // writer
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"output_is_dram", output_is_dram},
             {"num_rows_per_core", num_units_per_core},
             {"Wt", static_cast<uint32_t>(Wt)},
             {"tile_offset", tile_offset}});

        tile_offset += num_units_per_core * Wt;
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_g1_run_args)};
    if (has_core_group_2) {
        run_args.kernel_run_args.push_back(std::move(compute_g2_run_args));
    }
    run_args.tensor_args = {
        {INPUT, input},
        {OUTPUT, out},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}
}  // namespace ttnn::operations::moreh::moreh_norm
