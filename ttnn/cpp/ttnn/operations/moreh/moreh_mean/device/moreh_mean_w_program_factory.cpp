// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include "moreh_mean_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

namespace ttnn::operations::moreh::moreh_mean {

ttnn::device_operation::ProgramArtifacts MorehMeanOperation::MorehMeanWFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& input = tensor_args.input;
    auto compute_kernel_config =
        init_device_compute_kernel_config(input.device()->arch(), operation_attributes.compute_kernel_config);
    const auto& shape = input.padded_shape();

    auto* device = input.device();

    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    uint32_t W = shape[-1], H = shape[-2];
    uint32_t Wt = W / constants::TILE_WIDTH;
    uint32_t Ht = H / constants::TILE_HEIGHT;

    // check mask for w-dim
    const auto& input_shape_without_padding = input.logical_shape();
    const auto origin_W = input_shape_without_padding[-1];
    const bool do_mask_w = (origin_W % constants::TILE_WIDTH) != 0;
    const auto mask_w = do_mask_w ? origin_W % constants::TILE_WIDTH : constants::TILE_WIDTH;

    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto units_to_divide = input.physical_volume() / W / H * Ht;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores_wt_core_range(core_range, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    // ---- Program-scope resource names (drive the generated dfb:: / tensor:: tokens) ----
    // Declared function-local: the three moreh_mean factory .cpp files land in the same
    // unity-build translation unit, so no anonymous-namespace constants are introduced.
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName SCALER_DFB{"scaler"};
    const DFBSpecName MASK_W_DFB{"mask_w"};
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
    spec.name = "moreh_mean_w";

    // ---- Dataflow buffers ----
    constexpr uint32_t num_input_tiles = 2;
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_DFB,
        .entry_size = tile_size(data_format),
        .num_entries = num_input_tiles,
        .data_format_metadata = data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SCALER_DFB,
        .entry_size = tile_size(data_format),
        .num_entries = 1,
        .data_format_metadata = data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MASK_W_DFB,
        .entry_size = tile_size(data_format),
        .num_entries = 1,
        .data_format_metadata = data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = ACCUM_DST_DFB,
        .entry_size = tile_size(fp32_dest_acc_en_data_format),
        .num_entries = 1,
        .data_format_metadata = fp32_dest_acc_en_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MASKED_INPUT_DFB,
        .entry_size = tile_size(data_format),
        .num_entries = 1,
        .data_format_metadata = data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = tile_size(data_format),
        .num_entries = 1,
        .data_format_metadata = data_format,
    });

    // ---- Tensor parameters (replace the buffer-address RTA + TensorAccessorArgs plumbing) ----
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_TENSOR, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()});

    // ---- Reader kernel ----
    float scaler = 1.0f / origin_W;
    bfloat16 bfloat_scaler_value(scaler);
    auto packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});

    // The mask DFB is produced only when masking is active; the matching DO_MASK_W define already
    // gates the kernel-side production, and gates the dfb::mask_w reference with it.
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
    KernelSpec::CompilerOptions::Defines reader_defines;
    if (do_mask_w) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = MASK_W_DFB,
            .accessor_name = "mask_w",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_defines.emplace("DO_MASK_W", "1");
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/reader_moreh_mean_w.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}},
        .compile_time_args = {{"scaler", packed_scaler_value}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id", "mask_w"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });

    // ---- Writer kernel ----
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/writer_moreh_mean_unary_interleaved_start_id.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    });

    // ---- Compute kernels (two groups) ----
    auto reduce_op = ReduceOpMath::AVG;
    auto reduce_dim = ReduceOpDim::W;
    auto compute_defines_map = reduce_op_utils::get_defines(reduce_op, reduce_dim);
    if (fp32_dest_acc_en) {
        compute_defines_map["FP32_DEST_ACC_EN"] = "1";
    }
    KernelSpec::CompilerOptions::Defines compute_defines(compute_defines_map);

    auto compute_hw = ttnn::to_compute_hardware_config(compute_kernel_config);
    // Legacy left unpack_to_dest_mode entirely Default here (unlike the H factory). Metal 2.0
    // requires the choice to be explicit once a consumed DFB is Float32 with a 32-bit dest
    // register, so spell out the legacy Default: UnpackToSrc. Value-preserving.
    if (fp32_dest_acc_en) {
        compute_hw.unpack_modes = ComputeUnpackModes{{ACCUM_DST_DFB, UnpackMode::UnpackToSrc}};
    }

    // The compute kernel binds the mask DFB in every configuration: it constructs the buffer object
    // unconditionally and only its FIFO calls are dead when masking is off. In that configuration the
    // reader does not produce into it, leaving compute the single toucher — bound as both PRODUCER
    // and CONSUMER (self-loop) so the DFB still presents one endpoint of each kind per node.
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
                .dfb_spec_name = MASK_W_DFB,
                .accessor_name = "mask_w",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            // accum_dst holds the partial row product across the Wt-1 leading tiles: packed by this
            // kernel, then read back to seed the final matmul.
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
            // masked_input is packed by this kernel and then re-read through the runtime-selected
            // cb_input handle.
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
        if (!do_mask_w) {
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = MASK_W_DFB,
                .accessor_name = "mask_w",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
        }
        return KernelSpec{
            .unique_id = unique_id,
            .source = "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/moreh_mean_w.cpp",
            // O3 is legacy ComputeConfig's default; Metal 2.0's CompilerOptions defaults to O2, so
            // the level has to be stated explicitly to keep the compute kernel where it was.
            .compiler_options = {.defines = compute_defines, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
            .dfb_bindings = std::move(dfb_bindings),
            .compile_time_args =
                {
                    // The kernel unpacks this into a local named `Ht`, but the value is the per-core
                    // work-split count, not the tensor's tile height.
                    {"units_per_core", units_per_core},
                    {"Wt", Wt},
                    {"NC", 1},
                    {"origin_W", origin_W},
                },
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
    uint32_t out_dim_divider = Wt;
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core = 0;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        uint32_t num_tensor_tiles_per_core = units_per_core * Wt;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_tiles", num_tensor_tiles_per_core}, {"start_id", tile_offset}, {"mask_w", mask_w}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_tiles", num_tensor_tiles_per_core / out_dim_divider}, {"start_id", tile_offset / out_dim_divider}});

        tile_offset += num_tensor_tiles_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(INPUT_TENSOR, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(OUTPUT_TENSOR, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_mean
