// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "moreh_sum_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

namespace ttnn::operations::moreh::moreh_sum {

ttnn::device_operation::ProgramArtifacts MorehSumOperation::MorehSumHFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& input = tensor_args.input;
    const DeviceComputeKernelConfig& compute_kernel_config = operation_attributes.compute_kernel_config;

    ReduceOpMath reduce_op = ReduceOpMath::SUM;
    ReduceOpDim reduce_dim = ReduceOpDim::H;
    const auto& shape = input.padded_shape();
    const auto [W, H, other_dims_product] = extract_spatial_dims(shape);

    uint32_t Wt = W / constants::TILE_WIDTH;
    uint32_t Ht = H / constants::TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    // check mask for h-dim
    const auto& input_shape_without_padding = input.logical_shape();
    const auto origin_H = input_shape_without_padding[-2];
    const bool do_mask_h = (origin_H % constants::TILE_HEIGHT) != 0;
    const auto mask_h = do_mask_h ? origin_H % constants::TILE_HEIGHT : constants::TILE_HEIGHT;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);
    log_debug(
        tt::LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);

    DataFormat src0_dfb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t src0_single_tile_size = tile_size(src0_dfb_data_format);
    DataFormat scaler_dfb_data_format = DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tile_size(scaler_dfb_data_format);
    DataFormat mask_h_dfb_data_format = DataFormat::Float16_b;
    uint32_t mask_h_single_tile_size = tile_size(mask_h_dfb_data_format);
    DataFormat intermed_dfb_data_format = (fp32_dest_acc_en) ? DataFormat::Float32 : DataFormat::Float16_b;
    DataFormat intermed1_dfb_data_format = DataFormat::Float16_b;
    uint32_t intermed_single_tile_size = tile_size(intermed_dfb_data_format);
    DataFormat dst_dfb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tile_size(dst_dfb_data_format);

    IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_cols = other_dims_product * Wt;

    const CoreRange all_core_range(
        {0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
        split_work_to_cores_wt_core_range(all_core_range, num_cols);

    // ---- Program-scope resource names (drive the generated dfb:: / tensor:: tokens) ----
    // Declared function-local: the six moreh_sum factory .cpp files land in the same
    // unity-build translation unit, so no anonymous-namespace constants are introduced.
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName SCALER_DFB{"scaler"};
    const DFBSpecName MASK_H_DFB{"mask_h"};
    const DFBSpecName ACCUM_DST_DFB{"accum_dst"};
    const DFBSpecName MASKED_INPUT_DFB{"masked_input"};
    const DFBSpecName OUT_DFB{"out"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    ProgramSpec spec;
    spec.name = "moreh_sum_h";

    // ---- Dataflow buffers ----
    constexpr uint32_t num_input_tiles = 2;
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_DFB,
        .entry_size = src0_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = src0_dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SCALER_DFB,
        .entry_size = scaler_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = scaler_dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MASK_H_DFB,
        .entry_size = mask_h_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = mask_h_dfb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = ACCUM_DST_DFB,
        .entry_size = intermed_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = intermed_dfb_data_format,
    });
    uint32_t intermed1_single_tile_size = tile_size(intermed1_dfb_data_format);
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MASKED_INPUT_DFB,
        .entry_size = intermed1_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = intermed1_dfb_data_format,
    });
    constexpr uint32_t num_output_tiles = 2;
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = dst_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = dst_dfb_data_format,
    });

    // ---- Tensor parameters (replace the Buffer* RTA + TensorAccessorArgs plumbing) ----
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_TENSOR, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()});

    // ---- Reader kernel ----
    // The mask DFB is produced only when masking is active; the reader's DO_MASK_H define already
    // gates that production, and gates the dfb::mask_h reference along with it.
    Group<DFBBinding> reader_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = INPUT_DFB,
            .accessor_name = "input",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = SCALER_DFB,
            .accessor_name = "scaler",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    KernelSpec::CompilerOptions::Defines reader_defines = {{"REDUCE_SCALER", "1"}};
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
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/reader_moreh_sum_h.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}},
        .compile_time_args =
            {
                {"Ht", Ht},
                {"Wt", Wt},
                // Read by the kernel but unused there (the column loop strides by Wt); kept as-is,
                // since dropping it would be a change to the kernel's argument list beyond the port.
                {"HtWt", HtWt},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"col_start_tile_id", "curr_col_in_batch", "num_cols", "mask_h"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    });

    // ---- Writer kernel ----
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/writer_moreh_sum_h.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    });

    // ---- Compute kernels (two groups) ----
    auto reduce_defines_map = reduce_op_utils::get_defines(reduce_op, reduce_dim);
    if (fp32_dest_acc_en) {
        reduce_defines_map["FP32_DEST_ACC_EN"] = "1";
    }
    KernelSpec::CompilerOptions::Defines reduce_defines(reduce_defines_map);

    auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config);
    if (auto* compute_gen1 = std::get_if<ComputeGen1Config>(&compute_hw); compute_gen1 && fp32_dest_acc_en) {
        // Legacy set unpack_to_dest_mode[CBIndex::c_24] = UnpackToDestFp32 when fp32 accumulation is
        // on; reindexed onto the DFB name and translated to the Metal 2.0 spelling. Metal 2.0 also
        // *requires* an explicit entry here (accum_dst is Float32 and the kernel consumes it with a
        // 32-bit dest register).
        compute_gen1->unpack_modes = ComputeUnpackModes{{ACCUM_DST_DFB, UnpackMode::UnpackToDest}};
    }

    // The compute kernel binds the mask DFB in every configuration: it constructs the buffer object
    // unconditionally and gates only some of its FIFO calls on do_mask_h. When masking is off the
    // reader does not produce into it, leaving compute the single toucher — bound as both PRODUCER
    // and CONSUMER (self-loop) so the DFB still presents one endpoint of each kind per node.
    // masked_input is bound PRODUCER+CONSUMER in *every* configuration so its DFB token exists while
    // the kernel's discarded do_mask_h branch is still type-checked.
    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t units_per_core) {
        Group<DFBBinding> dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = INPUT_DFB,
                .accessor_name = "input",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = SCALER_DFB,
                .accessor_name = "scaler",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = MASK_H_DFB,
                .accessor_name = "mask_h",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            // accum_dst holds the running reduction result: produced by the reduce output and read
            // back by the next iteration's accumulation.
            DFBBinding{
                .dfb_spec_name = ACCUM_DST_DFB,
                .accessor_name = "accum_dst",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = ACCUM_DST_DFB,
                .accessor_name = "accum_dst",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            // masked_input is packed by this kernel and immediately re-read as the reduce input.
            DFBBinding{
                .dfb_spec_name = MASKED_INPUT_DFB,
                .accessor_name = "masked_input",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = MASKED_INPUT_DFB,
                .accessor_name = "masked_input",
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
            .source = "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/moreh_sum_h.cpp",
            // O3 is legacy ComputeConfig's default; Metal 2.0's CompilerOptions defaults to O2, so
            // the level has to be stated explicitly to keep the compute kernel where it was.
            .compiler_options = {.defines = reduce_defines, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
            .dfb_bindings = std::move(dfb_bindings),
            .compile_time_args =
                {
                    {"Ht", Ht},
                    // The kernel unpacks this into a local named `Wt`, but the value is the
                    // per-core work-split count, not the tensor's tile width.
                    {"units_per_core", units_per_core},
                    {"NC", 1},
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

    // ---- Runtime args per core ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    for (uint32_t i = 0, num_cols_read = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_cols_per_core = 0;
        if (core_group_1.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
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
