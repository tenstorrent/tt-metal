// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "moreh_mean_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

namespace ttnn::operations::moreh::moreh_mean {

ttnn::device_operation::ProgramArtifacts MorehMeanOperation::MorehMeanNCFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& input = tensor_args.input;
    auto dim = operation_attributes.dim;

    auto compute_kernel_config =
        init_device_compute_kernel_config(input.device()->arch(), operation_attributes.compute_kernel_config);

    auto* device = input.device();

    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    const auto cb_data_format = datatype_to_dataformat_converter(output.dtype());

    const auto& input_shape = input.padded_shape();
    const auto Ht = input_shape[-2] / constants::TILE_HEIGHT;
    const auto Wt = input_shape[-1] / constants::TILE_WIDTH;
    const auto HtWt = Ht * Wt;
    const auto num_reduce_input_tile = input_shape[dim];

    const auto rank = input_shape.rank();
    auto input_tile_stride = HtWt;
    for (int i = dim + 1; i < rank - 2; i++) {
        input_tile_stride *= input_shape[i];
    }

    uint32_t inner_size = 1;
    for (int i = dim + 1; i < rank - 2; i++) {
        inner_size *= input_shape[i];
    }

    const auto units_to_divide = output.physical_volume() / constants::TILE_HW;

    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores_wt_core_range(core_range, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    // ---- Program-scope resource names (drive the generated dfb:: / tensor:: tokens) ----
    // Declared function-local: the three moreh_mean factory .cpp files land in the same
    // unity-build translation unit, so no anonymous-namespace constants are introduced.
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName SCALAR_DFB{"scalar"};
    const DFBSpecName INTERMED0_DFB{"intermed0"};
    const DFBSpecName OUT_DFB{"out"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    ProgramSpec spec;
    spec.name = "moreh_mean_nc";

    // ---- Dataflow buffers ----
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_DFB,
        .entry_size = tile_size(cb_data_format),
        .num_entries = 2,
        .data_format_metadata = cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN1_DFB,
        .entry_size = tile_size(cb_data_format),
        .num_entries = 1,
        .data_format_metadata = cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SCALAR_DFB,
        .entry_size = tile_size(cb_data_format),
        .num_entries = 1,
        .data_format_metadata = cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INTERMED0_DFB,
        .entry_size = tile_size(cb_data_format),
        .num_entries = 1,
        .data_format_metadata = cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = tile_size(cb_data_format),
        .num_entries = 2,
        .data_format_metadata = cb_data_format,
    });

    // ---- Tensor parameters (replace the buffer-address RTA + TensorAccessorArgs plumbing) ----
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_TENSOR, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()});

    // ---- Reader kernel ----
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/reader_moreh_mean_nc.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_DFB,
                    .accessor_name = "input",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = IN1_DFB,
                    .accessor_name = "in1",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SCALAR_DFB,
                    .accessor_name = "scalar",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_input_tiles", "num_output_tiles", "input_tile_stride", "start_id", "HtWt", "inner_size"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });

    // ---- Writer kernel ----
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/writer_moreh_mean_nc.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    });

    // ---- Compute kernels (two groups) ----
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }
    // No unpack_modes entry: this factory never widens the intermediate DFB to Float32, so no DFB the
    // compute kernel consumes carries a 32-bit format. Legacy left unpack_to_dest_mode all-Default,
    // which is exactly an empty table.
    auto compute_hw = ttnn::to_compute_hardware_config(compute_kernel_config);

    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t units_per_core) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/moreh_mean_nc.cpp",
            // O3 is legacy ComputeConfig's default; Metal 2.0's CompilerOptions defaults to O2, so
            // the level has to be stated explicitly to keep the compute kernel where it was.
            .compiler_options = {.defines = compute_defines, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = INPUT_DFB,
                        .accessor_name = "input",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = IN1_DFB,
                        .accessor_name = "in1",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = SCALAR_DFB,
                        .accessor_name = "scalar",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    // intermed0 carries the running sum across the reduced dim: packed by this kernel
                    // each iteration and read back on the next one.
                    DFBBinding{
                        .dfb_spec_name = INTERMED0_DFB,
                        .accessor_name = "intermed0",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = INTERMED0_DFB,
                        .accessor_name = "intermed0",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                },
            // Reproduces the legacy per-group CTA. The compute kernel does not read it (it takes both
            // its loop bounds from RTAs); it is kept so the two per-group KernelSpecs stay the
            // faithful 1:1 image of the two legacy KernelDescriptors.
            .compile_time_args = {{"units_per_core", units_per_core}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_input_tiles", "num_output_tiles"}},
            .hw_config = compute_hw,
        };
    };

    spec.kernels.push_back(make_compute(COMPUTE_G1, units_per_core_group_1));
    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        spec.kernels.push_back(make_compute(COMPUTE_G2, units_per_core_group_2));
    }

    // ---- Work units (placement) ----
    // Reader and writer belong to both work units, so their derived node set is the union of the two
    // core groups (the legacy `all_cores`), while each core group hosts its own compute instance.
    spec.work_units.push_back(
        WorkUnitSpec{.name = "wu_g1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1});
    if (has_core_group_2) {
        spec.work_units.push_back(
            WorkUnitSpec{.name = "wu_g2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    // ---- Runtime args per core ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_g1_run_args{.kernel = COMPUTE_G1};
    KernelRunArgs compute_g2_run_args{.kernel = COMPUTE_G2};

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / core_h, i % core_h};

        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
            AddRuntimeArgsForNode(
                compute_g1_run_args.runtime_arg_values,
                core,
                {{"num_input_tiles", static_cast<uint32_t>(num_reduce_input_tile)},
                 {"num_output_tiles", units_per_core}});
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
            AddRuntimeArgsForNode(
                compute_g2_run_args.runtime_arg_values,
                core,
                {{"num_input_tiles", static_cast<uint32_t>(num_reduce_input_tile)},
                 {"num_output_tiles", units_per_core}});
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_input_tiles", static_cast<uint32_t>(num_reduce_input_tile)},
             {"num_output_tiles", units_per_core},
             {"input_tile_stride", static_cast<uint32_t>(input_tile_stride)},
             {"start_id", tile_offset},
             {"HtWt", static_cast<uint32_t>(HtWt)},
             {"inner_size", inner_size}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"num_tiles", units_per_core}, {"start_id", tile_offset}});

        tile_offset += units_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_g1_run_args));
    if (has_core_group_2) {
        run_args.kernel_run_args.push_back(std::move(compute_g2_run_args));
    }

    run_args.tensor_args.emplace(INPUT_TENSOR, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(OUTPUT_TENSOR, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_mean
