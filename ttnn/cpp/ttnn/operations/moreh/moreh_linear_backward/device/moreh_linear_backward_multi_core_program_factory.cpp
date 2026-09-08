// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>

#include "ttnn/cpp/ttnn/kernel_lib/host/reduce_host.hpp"
#include <vector>

#include "moreh_linear_backward_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_linear_backward {

ttnn::device_operation::ProgramArtifacts
MorehBiasAddBackwardOperation::MultiCoreProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& bias_grad) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& output_grad = tensor_args.output_grad;
    const auto& output_grad_mesh = output_grad.mesh_tensor();
    const auto& bias_grad_mesh = bias_grad.mesh_tensor();

    const auto& output_grad_shape_wo_padding = output_grad.logical_shape();

    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    const uint32_t logical_height = output_grad_shape_wo_padding[-2];
    const bool has_partial_height = logical_height % constants::TILE_HEIGHT != 0;

    const auto& output_grad_shape = output_grad.padded_shape();
    uint32_t batch_num = output_grad.physical_volume() / output_grad_shape[-2] / output_grad_shape[-1];
    uint32_t Ht = output_grad_shape[-2] / constants::TILE_HEIGHT;
    uint32_t Wt = output_grad_shape[-1] / constants::TILE_WIDTH;
    uint32_t num_tiles = batch_num * Ht;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    IDevice* device = output_grad.device();
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;
    auto arch = device->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);
    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = split_work_to_cores(grid, Wt);

    ////////////////////////////////////////////////////////////////////////////
    //         Program-scope resource names (drive the generated dfb:: / tensor:: tokens)
    ////////////////////////////////////////////////////////////////////////////
    // Declared function-local: this factory and the single-core one land in the same unity-build
    // translation unit, so no anonymous-namespace constants are introduced.
    // `out` / `dst` / `num_tiles` / `start_id` are the writer kernel's own vocabulary, and
    // writer_moreh_bias_backward.cpp is bound by both factories — the two specs must agree on them.
    const DFBSpecName IN0_DFB{"in0"};
    const DFBSpecName SCALER_DFB{"scaler"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName INTERMED1_DFB{"intermed1"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};
    const TensorParamName OUTPUT_GRAD_TENSOR{"output_grad"};
    const TensorParamName BIAS_GRAD_TENSOR{"bias_grad"};

    ProgramSpec spec;
    spec.name = "moreh_bias_add_backward_multi_core";

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 2;

    const uint32_t out0_t = 1;
    const uint32_t im1_t = 1;
    auto dfb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : dfb_data_format;

    namespace reduce_host = ttnn::kernel_lib::host;
    // Full-height batches form one contiguous column. A partial height must
    // be handled at each batch boundary, using explicit accumulating calls.
    const uint32_t reduce_repetitions = has_partial_height ? batch_num : 1;
    const uint32_t call_height = has_partial_height ? logical_height : batch_num * logical_height;
    const TensorLayout input_layout(output_grad.dtype(), PageConfig(Layout::TILE), MemoryConfig{});
    const TensorLayout output_layout(bias_grad.dtype(), PageConfig(Layout::TILE), MemoryConfig{});
    const reduce_host::ReduceCallConfig reduction{
        TensorSpec(Shape{call_height, 32}, input_layout),
        TensorSpec(Shape{1, 32}, output_layout),
        ReduceOpMath::SUM,
        ReduceOpDim::H,
        1.0F,
        ReduceFp32Mode::Fast,
        in0_t * tile_size(dfb_data_format)};
    std::vector<reduce_host::ReduceCbConfig> reductions(std::min(reduce_repetitions, 3U), {0, reduction});
    auto reduce_sequence = reduce_host::make_reduce_sequence_plan(
        reductions,
        {.auxiliary_cb_id = 1, .accumulator_cb_id = 3, .output_cb_id = 2},
        {.arch = arch,
         .fp32_dest_acc_en = fp32_dest_acc_en,
         .dst_full_sync_en = dst_full_sync_en,
         .available_l1_bytes = 16 * tile_size(dfb_data_format)});
    reduce_sequence.calls.back().accumulation_index = reduce_repetitions - 1;
    const auto* auxiliary = reduce_sequence.calls.front().plan.find_cb(reduce_host::ReduceCbRole::Auxiliary);
    const uint32_t auxiliary_tiles = reduce_sequence.auxiliary.tiles.size();

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0_DFB,
        .entry_size = tile_size(dfb_data_format),
        .num_entries = in0_t,
        .data_format_metadata = dfb_data_format,
    });  // output_grad
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SCALER_DFB,
        .entry_size = auxiliary->page_size,
        .num_entries = auxiliary_tiles,
        .data_format_metadata = auxiliary->data_format,
    });  // scaler
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = tile_size(dfb_data_format),
        .num_entries = out0_t,
        .data_format_metadata = dfb_data_format,
    });  // bias_grad
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INTERMED1_DFB,
        .entry_size = tile_size(fp32_dest_acc_en_data_format),
        .num_entries = im1_t,
        .data_format_metadata = fp32_dest_acc_en_data_format,
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    // These replace the buffer-address runtime args and the host-side tensor-accessor argument plumbing.
    // tensor_args.bias is deliberately absent: it is read on the host only, for the output spec.
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = OUTPUT_GRAD_TENSOR, .spec = output_grad_mesh.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = BIAS_GRAD_TENSOR, .spec = bias_grad_mesh.tensor_spec()});

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/reader_moreh_bias_backward_h.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN0_DFB,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SCALER_DFB,
                    .accessor_name = "scaler",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_GRAD_TENSOR, .accessor_name = "src0"}},
        // `batch_num` names the kernel-side local it lands in; the value is batch_num * Ht, the
        // number of tiles in one full column.
        .runtime_arg_schema = {.runtime_arg_names = {"batch_num", "Wt", "Wt_per_core", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
        .advanced_options = {.compile_time_varargs = reduce_sequence.get_auxiliary_compile_time_args()},
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/writer_moreh_bias_backward.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = BIAS_GRAD_TENSOR, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Style A: the op resolves a TTNN DeviceComputeKernelConfig, so the TTNN helper carries its
    // values across (including the math_approx_mode bool -> Precision mapping and the
    // dst_full_sync_en -> double_buffer_dest inversion).
    auto compute_hw = ttnn::to_compute_hardware_config(arch, compute_kernel_config);

    // Legacy carried an unpack-to-dest-mode vector indexed by buffer index, every entry left at its
    // default, except the intermed1 buffer (index 25) which it set to the fp32 unpack-to-dest mode
    // under fp32 accumulation. Metal 2.0 keys the same information by DFB name and additionally
    // requires an explicit entry wherever a compute kernel consumes a Float32 DFB with a 32-bit Dest
    // register, so the legacy default has to be stated for the other DFBs this kernel consumes:
    // intermed1 is Float32 whenever fp32_dest_acc_en is set, and the rest are Float32 whenever
    // output_grad is. The legacy default is UnpackToSrc, which is legal for any format, so
    // transcribing the whole legacy row reproduces the legacy unpack vector byte-for-byte in every
    // configuration.
    ComputeUnpackModes dfb_unpack_modes = {
        {IN0_DFB, UnpackMode::UnpackToSrc},
        {SCALER_DFB, UnpackMode::UnpackToSrc},
        {INTERMED1_DFB, UnpackMode::UnpackToSrc},
    };
    if (fp32_dest_acc_en) {
        dfb_unpack_modes[INTERMED1_DFB] = UnpackMode::UnpackToDest;  // legacy: c_25 = UnpackToDestFp32
    }
    // Assign through the generation-neutral accessor rather than std::get<ComputeGen1Config>: the
    // helper above returns whichever alternative matches `arch`, so naming Gen1 here would throw
    // std::bad_variant_access on Quasar. (The local is named dfb_unpack_modes so it does not shadow
    // the accessor.)
    // TODO(#52269): Quasar unpack_modes are copied from Gen1 and not yet optimized for Quasar.
    unpack_modes(compute_hw) = std::move(dfb_unpack_modes);

    // Two KernelSpecs of the same source over the two disjoint core groups, differing only in the
    // per-group compile-time count — the Metal 2.0 form of the legacy two-kernel-descriptor work
    // split. Each node hosts exactly one instance, so every shared-DFB binding below is an ordinary
    // single-role binding.
    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t units_per_core) {
        return KernelSpec{
            .unique_id = unique_id,
            .source =
                "ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/"
                "moreh_bias_backward_multi_core_h.cpp",
            // O3 is the legacy ComputeConfigDescriptor default; Metal 2.0's CompilerOptions defaults
            // to O2, so the level has to be stated explicitly on each spec to keep the compute
            // kernels where they were.
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN0_DFB,
                        .accessor_name = "in0",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = SCALER_DFB,
                        .accessor_name = "scaler",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },

                    // intermed1 holds the running reduction result: written as the reduce output and
                    // read back by the next iteration's accumulation. Also a self-loop.
                    DFBBinding{
                        .dfb_spec_name = INTERMED1_DFB,
                        .accessor_name = "intermed1",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = INTERMED1_DFB,
                        .accessor_name = "intermed1",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                },
            .compile_time_args =
                {{"units_per_core", units_per_core},
                 {"reduce_repetitions", reduce_repetitions},
                 {"reduce_auxiliary_tiles", auxiliary_tiles}},
            .hw_config = compute_hw,
            .advanced_options = {.compile_time_varargs = reduce_sequence.get_compile_time_args()},
        };
    };

    spec.kernels.push_back(make_compute(COMPUTE_G1, num_cols_per_core_group_1));
    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        spec.kernels.push_back(make_compute(COMPUTE_G2, num_cols_per_core_group_2));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Work units (placement)
    ////////////////////////////////////////////////////////////////////////////
    // Reader and writer belong to both work units, so their derived node set is the union of the two
    // core groups — the legacy `all_cores`. Each core group hosts its own compute instance.
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

        uint32_t num_cols_per_core = 0;
        if (core_group_1.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"batch_num", num_tiles}, {"Wt", Wt}, {"Wt_per_core", num_cols_per_core}, {"start_id", tile_offset}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"num_tiles", num_cols_per_core}, {"start_id", tile_offset}});

        tile_offset += num_cols_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(OUTPUT_GRAD_TENSOR, TensorArgument{output_grad_mesh});
    run_args.tensor_args.emplace(BIAS_GRAD_TENSOR, TensorArgument{bias_grad_mesh});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_linear_backward
