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

ttnn::device_operation::ProgramArtifacts MorehSumOperation::MorehSumHIntFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& input = tensor_args.input;
    const DeviceComputeKernelConfig& compute_kernel_config = operation_attributes.compute_kernel_config;

    IDevice* device{input.device()};

    const auto dfb_data_format{datatype_to_dataformat_converter(output.dtype())};
    const auto& shape{input.padded_shape()};

    const auto [W, H, other_dims_product] = extract_spatial_dims(shape);
    uint32_t Wt{W / constants::TILE_WIDTH};
    uint32_t Ht{H / constants::TILE_HEIGHT};
    uint32_t HtWt{Ht * Wt};
    [[maybe_unused]] uint32_t num_tiles = input.physical_volume() / constants::TILE_HW;
    auto num_cols{other_dims_product * Wt};

    // check mask for h-dim
    const auto& input_shape_without_padding{input.logical_shape()};
    const auto origin_H{input_shape_without_padding[-2]};
    const bool do_mask_h{(origin_H % constants::TILE_HEIGHT) != 0};
    const auto mask_h{do_mask_h ? origin_H % constants::TILE_HEIGHT : constants::TILE_HEIGHT};

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);
    log_debug(
        tt::LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);

    if (!fp32_dest_acc_en) {
        log_warning(tt::LogOp, "fp32_dest_acc_en should be set for integer sum");
        fp32_dest_acc_en = true;
    }
    log_debug(tt::LogOp, "do_mask_h {} mask_h {}", do_mask_h, mask_h);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid{device->compute_with_storage_grid_size()};
    const auto num_cores_y{grid.y};

    const uint32_t in0_t{2};        // input
    const uint32_t in1_t{1};        // mask
    const uint32_t intermed0_t{1};  // accumulated sum
    const uint32_t out0_t{2};       // output
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            split_work_to_cores(grid, num_cols);

    log_debug(
        tt::LogOp,
        "num_tiles {}, num_cols {}, num_cols_per_core_group_1 {}, num_cols_per_core_group_2 {}",
        num_tiles,
        num_cols,
        num_cols_per_core_group_1,
        num_cols_per_core_group_2);

    uint32_t dfb_tile_size = tile_size(dfb_data_format);

    // ---- Program-scope resource names (drive the generated dfb:: / tensor:: tokens) ----
    // Declared function-local: the six moreh_sum factory .cpp files land in the same
    // unity-build translation unit, so no anonymous-namespace constants are introduced.
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName MASK_H_DFB{"mask_h"};
    const DFBSpecName INTERMED0_DFB{"intermed0"};
    const DFBSpecName OUT_DFB{"out"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    ProgramSpec spec;
    spec.name = "moreh_int_sum_h";

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_DFB,
        .entry_size = dfb_tile_size,
        .num_entries = in0_t,
        .data_format_metadata = dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MASK_H_DFB,
        .entry_size = dfb_tile_size,
        .num_entries = in1_t,
        .data_format_metadata = dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INTERMED0_DFB,
        .entry_size = dfb_tile_size,
        .num_entries = intermed0_t,
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
    // The mask DFB is produced only when masking is active; the reader's DO_MASK_H define already
    // gates that production, and gates the dfb::mask_h reference along with it.
    Group<DFBBinding> reader_dfb_bindings = {DFBBinding{
        .dfb_spec_name = INPUT_DFB,
        .accessor_name = "input",
        .endpoint_type = DFBEndpointType::PRODUCER,
    }};
    KernelSpec::CompilerOptions::Defines reader_defines;
    if (do_mask_h) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = MASK_H_DFB,
            .accessor_name = "mask_h",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_defines.emplace("DO_MASK_H", "1");
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/reader_moreh_int_sum_h.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}},
        .compile_time_args =
            {
                {"Ht", Ht},
                {"Wt", Wt},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"col_start_tile_id", "curr_col_in_batch", "num_cols", "mask_h"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/writer_moreh_int_sum_h.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    auto compute_hw = ttnn::to_compute_hardware_config(compute_kernel_config);
    // This factory overrides the caller's fp32_dest_acc_en above (integer sum requires a 32-bit
    // dest); carry the *forced* value rather than the one the attributes came in with.
    // Every DFB here carries the Int32 output format and the dest register is 32-bit, so the
    // Src-vs-Dest choice is real for each one the compute kernel consumes. Issue #49936 extends
    // the choice to Int32/UInt32, where an unspecified consumer becomes a hard error.
    compute_hw.enable_32_bit_dest = fp32_dest_acc_en;
    compute_hw.unpack_modes = ComputeUnpackModes{
        {INPUT_DFB, UnpackMode::UnpackToSrc},
        {MASK_H_DFB, UnpackMode::UnpackToSrc},
        {INTERMED0_DFB, UnpackMode::UnpackToSrc},
    };

    // The compute kernel binds the mask DFB in every configuration: it constructs the buffer object
    // unconditionally and gates only its FIFO calls on do_mask_h. When masking is off the reader does
    // not produce into it, leaving compute the single toucher — bound as both PRODUCER and CONSUMER
    // (self-loop) so the DFB still presents one endpoint of each kind per node.
    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t units_per_core) {
        Group<DFBBinding> dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = INPUT_DFB,
                .accessor_name = "input",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = MASK_H_DFB,
                .accessor_name = "mask_h",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            // intermed0 carries the running sum down the column: packed by this kernel each
            // iteration and read back on the next one.
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
        };
        if (!do_mask_h) {
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = MASK_H_DFB,
                .accessor_name = "mask_h",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
        }
        return KernelSpec{
            .unique_id = unique_id,
            .source = "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/moreh_int_sum_h.cpp",
            // O3 is legacy ComputeConfig's default; Metal 2.0's CompilerOptions defaults to O2, so
            // the level has to be stated explicitly to keep the compute kernel where it was.
            .compiler_options = {.defines = compute_defines, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
            .dfb_bindings = std::move(dfb_bindings),
            .compile_time_args =
                {
                    {"num_cols", units_per_core},
                    {"Ht", Ht},
                    {"origin_H", origin_H},
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

    for (uint32_t i = 0, num_cols_read = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_cols_per_core{0};
        if (core_group_1.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"col_start_tile_id", (num_cols_read / Wt * HtWt) + (num_cols_read % Wt)},
             {"curr_col_in_batch", num_cols_read % Wt},
             {"num_cols", num_cols_per_core},
             {"mask_h", mask_h}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"num_tiles", num_cols_per_core},  // number of tiles to write
                {"start_id", num_cols_read}        // output tile start index
            });

        num_cols_read += num_cols_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(INPUT_TENSOR, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(OUTPUT_TENSOR, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_sum
