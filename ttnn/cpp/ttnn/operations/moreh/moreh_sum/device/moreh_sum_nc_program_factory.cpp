// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "moreh_sum_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_sum {

ttnn::device_operation::ProgramArtifacts MorehSumOperation::MorehSumNCFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& input = tensor_args.input;
    auto dim = operation_attributes.dim;

    const DeviceComputeKernelConfig& compute_kernel_config = operation_attributes.compute_kernel_config;

    IDevice* device = input.device();

    const auto dfb_data_format = datatype_to_dataformat_converter(output.dtype());

    const auto& input_shape = input.padded_shape();
    const auto [Wt, Ht, inner_tile_size, reduce_tile_size] =
        extract_and_scale_spatial_dims(input_shape, static_cast<uint32_t>(dim));
    const auto num_reduce_input_tile = input_shape[dim];
    const auto num_output_tiles = output.physical_volume() / constants::TILE_HW;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);

    log_debug(
        tt::LogOp, "reduce_tile_size {} inner_tile_size {} Ht {} Wt {}", reduce_tile_size, inner_tile_size, Ht, Wt);
    log_debug(
        tt::LogOp, "dim {} num_reduce_input_tile {} num_output_tiles {}", dim, num_reduce_input_tile, num_output_tiles);
    log_debug(
        tt::LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const uint32_t in0_t = 2;   // input
    const uint32_t in1_t = 1;   // zero
    const uint32_t out0_t = 2;  // output
    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = split_work_to_cores(grid, num_output_tiles);

    uint32_t dfb_tile_size = tile_size(dfb_data_format);

    // ---- Program-scope resource names (drive the generated dfb:: / tensor:: tokens) ----
    // Declared function-local: the six moreh_sum factory .cpp files land in the same
    // unity-build translation unit, so no anonymous-namespace constants are introduced.
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName ZERO_DFB{"zero"};
    const DFBSpecName OUT_DFB{"out"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    ProgramSpec spec;
    spec.name = "moreh_sum_nc";

    // ---- Dataflow buffers ----
    // Legacy also allocated a one-tile CBIndex::c_24 "accumulated sum" intermediate that no kernel in
    // this factory ever referenced: this compute path accumulates in DST (add_tiles with
    // acc_to_dest = true) and never needs an L1 intermediate. It is dropped here — a dead buffer has
    // no behavior, and a DFB with no endpoint binding cannot be expressed at all.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_DFB,
        .entry_size = dfb_tile_size,
        .num_entries = in0_t,
        .data_format_metadata = dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = ZERO_DFB,
        .entry_size = dfb_tile_size,
        .num_entries = in1_t,
        .data_format_metadata = dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = dfb_tile_size,
        .num_entries = out0_t,
        .data_format_metadata = dfb_data_format,
    });

    // ---- Tensor parameters (replace the Buffer* RTA + TensorAccessorArgs plumbing) ----
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_TENSOR, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()});

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // The reader source is shared with MorehSumNCIntFactory, which allocates no zero buffer: the
    // zero-tile block there is preprocessed away, so the binding is emitted only alongside USE_FPU.
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_nc_impl_kernels/reader_moreh_sum_nc.cpp",
        .compiler_options = {.defines = {{"USE_FPU", "1"}}},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INPUT_DFB,
                    .accessor_name = "input",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = ZERO_DFB,
                    .accessor_name = "zero",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_input_tiles", "num_output_tiles", "start_id", "dim", "reduce_tile_size", "inner_tile_size"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_nc_impl_kernels/writer_moreh_sum_nc.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    // No unpack_modes entry: legacy left unpack_to_dest_mode all-Default, which is exactly an empty
    // table. With the dead Float32-capable intermediate gone, every DFB this kernel consumes carries
    // the output data format, so Metal 2.0's explicit-entry requirement does not fire either.
    auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config);

    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t units_per_core) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_nc_impl_kernels/moreh_sum_nc.cpp",
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
                    // The zero tile the reader stages is the second operand of every accumulating
                    // add; it is waited on once and never popped, so it is read many times.
                    DFBBinding{
                        .dfb_spec_name = ZERO_DFB,
                        .accessor_name = "zero",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                },
            .compile_time_args =
                {
                    {"num_output_tiles", units_per_core},
                    {"num_input_tiles", static_cast<uint32_t>(num_reduce_input_tile)},
                },
            .hw_config = compute_hw,
        };
    };

    spec.kernels.push_back(make_compute(COMPUTE_G1, num_cols_per_core_group_1));
    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        spec.kernels.push_back(make_compute(COMPUTE_G2, num_cols_per_core_group_2));
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

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_input_tiles", static_cast<uint32_t>(num_reduce_input_tile)},
             {"num_output_tiles", num_tiles_per_core},
             {"start_id", tile_offset},
             {"dim", static_cast<uint32_t>(dim)},
             {"reduce_tile_size", reduce_tile_size},
             {"inner_tile_size", inner_tile_size}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"num_tiles", num_tiles_per_core}, {"start_id", tile_offset}});

        tile_offset += num_tiles_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(INPUT_TENSOR, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(OUTPUT_TENSOR, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_sum
