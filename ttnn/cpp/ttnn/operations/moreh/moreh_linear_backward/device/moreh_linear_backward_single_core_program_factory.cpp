// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "moreh_linear_backward_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_linear_backward {

ttnn::device_operation::ProgramArtifacts
MorehBiasAddBackwardOperation::SingleCoreProgramFactory::create_program_artifacts(
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

    const bool do_mask_h = (output_grad_shape_wo_padding[-2] % constants::TILE_HEIGHT) != 0;
    const uint32_t mask_h =
        do_mask_h ? output_grad_shape_wo_padding[-2] % constants::TILE_HEIGHT : constants::TILE_HEIGHT;
    const bool do_mask_w = (output_grad_shape_wo_padding[-1] % constants::TILE_WIDTH) != 0;
    const uint32_t mask_w =
        do_mask_w ? output_grad_shape_wo_padding[-1] % constants::TILE_WIDTH : constants::TILE_WIDTH;

    const auto& output_grad_shape = output_grad.padded_shape();
    uint32_t batch_num = output_grad.physical_volume() / output_grad_shape[-2] / output_grad_shape[-1];
    uint32_t Ht = output_grad_shape[-2] / constants::TILE_HEIGHT;
    uint32_t Wt = output_grad_shape[-1] / constants::TILE_WIDTH;
    uint32_t num_tiles = output_grad.physical_volume() / constants::TILE_HW;

    const uint32_t in0_t = 2;
    const uint32_t in1_t = 1;
    const uint32_t in2_t = (do_mask_h || do_mask_w) ? 2 : 0;  // mask_h_w

    const uint32_t out0_t = 1;
    const uint32_t im0_t = 1;
    const uint32_t im1_t = 1;

    // The mask buffer is allocated only when a mask is actually applied, so its DFB, both of its
    // endpoint bindings, and the kernel-side references to it are all gated on this one condition.
    const bool do_mask_h_w = in2_t > 0;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    const NodeCoord node = {0, 0};

    IDevice* device = output_grad.device();
    auto arch = device->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //         Program-scope resource names (drive the generated dfb:: / tensor:: tokens)
    ////////////////////////////////////////////////////////////////////////////
    // Declared function-local: this factory and the multi-core one land in the same unity-build
    // translation unit, so no anonymous-namespace constants are introduced.
    // `out` / `dst` / `num_tiles` / `start_id` are the writer kernel's own vocabulary, and
    // writer_moreh_bias_backward.cpp is bound by both factories — the two specs must agree on them.
    const DFBSpecName IN0_DFB{"in0"};
    const DFBSpecName SCALER_DFB{"scaler"};
    const DFBSpecName MASK_H_W_DFB{"mask_h_w"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName INTERMED0_DFB{"intermed0"};
    const DFBSpecName INTERMED1_DFB{"intermed1"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const TensorParamName OUTPUT_GRAD_TENSOR{"output_grad"};
    const TensorParamName BIAS_GRAD_TENSOR{"bias_grad"};

    ProgramSpec spec;
    spec.name = "moreh_bias_add_backward_single_core";

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    auto dfb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : dfb_data_format;

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0_DFB,
        .entry_size = tile_size(dfb_data_format),
        .num_entries = in0_t,
        .data_format_metadata = dfb_data_format,
    });  // output_grad
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SCALER_DFB,
        .entry_size = tile_size(dfb_data_format),
        .num_entries = in1_t,
        .data_format_metadata = dfb_data_format,
    });  // scaler
    if (do_mask_h_w) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = MASK_H_W_DFB,
            .entry_size = tile_size(dfb_data_format),
            .num_entries = in2_t,
            .data_format_metadata = dfb_data_format,
        });  // mask_h_w
    }
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = tile_size(dfb_data_format),
        .num_entries = out0_t,
        .data_format_metadata = dfb_data_format,
    });  // bias_grad
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INTERMED0_DFB,
        .entry_size = tile_size(dfb_data_format),
        .num_entries = im0_t,
        .data_format_metadata = dfb_data_format,
    });
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
    Group<DFBBinding> reader_dfb_bindings = {
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
    };
    KernelSpec::CompilerOptions::Defines reader_defines;
    if (do_mask_h_w) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = MASK_H_W_DFB,
            .accessor_name = "mask_h_w",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_defines.emplace("DO_MASK_H_W", "1");
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/reader_moreh_bias_backward_hw.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_GRAD_TENSOR, .accessor_name = "src"}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"num_tiles", "start_id", "mask_h", "mask_w", "do_mask_h", "do_mask_w"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
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
        .hw_config = ttnn::create_writer_datamovement_config(),
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelSpec::CompilerOptions::Defines compute_defines = {
        {"REDUCE_OP", "PoolType::SUM"},
        {"REDUCE_DIM", "ReduceDim::REDUCE_SCALAR"},
    };
    if (fp32_dest_acc_en) {
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    Group<DFBBinding> compute_dfb_bindings = {
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
        // intermed0 stages the masked input tile: this kernel packs it and immediately re-reads it
        // as the reduce input, so it binds both endpoints (self-loop).
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
        // intermed1 holds the running reduction result: written as the reduce output and read back
        // by the next iteration's accumulation. Also a self-loop.
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
    };
    if (do_mask_h_w) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = MASK_H_W_DFB,
            .accessor_name = "mask_h_w",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_defines.emplace("DO_MASK_H_W", "1");
    }

    // Style A: the op resolves a TTNN DeviceComputeKernelConfig, so the TTNN helper carries its
    // values across (including the math_approx_mode bool -> Precision mapping and the
    // dst_full_sync_en -> double_buffer_dest inversion).
    auto compute_hw = ttnn::to_compute_hardware_config(compute_kernel_config);

    // Legacy carried an unpack-to-dest-mode vector indexed by buffer index and left every entry at
    // its default in this factory. Metal 2.0 keys the same information by DFB name and requires an
    // explicit entry wherever a compute kernel consumes a Float32 DFB with a 32-bit Dest register,
    // so the legacy default has to be stated for the DFBs this kernel consumes: intermed1 is Float32
    // whenever fp32_dest_acc_en is set, and the rest are Float32 whenever output_grad is. The legacy
    // default is UnpackToSrc, which is legal for any format, so transcribing the whole legacy row
    // reproduces the legacy unpack vector byte-for-byte in every configuration.
    //
    // Note the divergence from the multi-core factory, which sets intermed1 to UnpackToDest under
    // the same fp32_dest_acc_en while this factory leaves it at UnpackToSrc. That looks unintended
    // rather than deliberate: intermed1 is the running reduction accumulator and is read back on
    // every iteration, so unpacking it to SrcA/SrcB narrows a 32-bit partial to the source
    // registers' 19 bits — the precision fp32_dest_acc_en was asked for. Reproduced as-is anyway,
    // because a port makes no functional change; correcting it is the op owner's call.
    ComputeUnpackModes dfb_unpack_modes = {
        {IN0_DFB, UnpackMode::UnpackToSrc},
        {SCALER_DFB, UnpackMode::UnpackToSrc},
        {INTERMED0_DFB, UnpackMode::UnpackToSrc},
        {INTERMED1_DFB, UnpackMode::UnpackToSrc},
    };
    if (do_mask_h_w) {
        // An entry naming a DFB the kernel does not bind is rejected, so this one shares the
        // binding's condition.
        dfb_unpack_modes.emplace(MASK_H_W_DFB, UnpackMode::UnpackToSrc);
    }
    // TODO(#52269): Quasar unpack_modes are copied from Gen1 and not yet optimized for Quasar.
    compute_hw.unpack_modes = std::move(dfb_unpack_modes);

    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/"
                  "moreh_bias_backward_single_core_hw.cpp",
        // O3 is the legacy ComputeConfigDescriptor default; Metal 2.0's CompilerOptions defaults to
        // O2, so the level has to be stated explicitly to keep the compute kernel where it was.
        .compiler_options = {.defines = std::move(compute_defines), .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .runtime_arg_schema = {.runtime_arg_names = {"batch_num", "Ht", "Wt", "do_mask_h", "do_mask_w"}},
        .hw_config = compute_hw,
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      Work units (placement)
    ////////////////////////////////////////////////////////////////////////////
    spec.work_units.push_back(WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node});

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        {.kernel = READER,
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(
             node,
             {{"num_tiles", num_tiles},
              {"start_id", 0u},
              {"mask_h", mask_h},
              {"mask_w", mask_w},
              {"do_mask_h", static_cast<uint32_t>(do_mask_h)},
              {"do_mask_w", static_cast<uint32_t>(do_mask_w)}})},
        {.kernel = WRITER,
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"num_tiles", 1u}, {"start_id", 0u}})},
        {.kernel = COMPUTE,
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(
             node,
             {{"batch_num", batch_num},
              {"Ht", Ht},
              {"Wt", Wt},
              {"do_mask_h", static_cast<uint32_t>(do_mask_h)},
              {"do_mask_w", static_cast<uint32_t>(do_mask_w)}})},
    };

    run_args.tensor_args.emplace(OUTPUT_GRAD_TENSOR, TensorArgument{output_grad_mesh});
    run_args.tensor_args.emplace(BIAS_GRAD_TENSOR, TensorArgument{bias_grad_mesh});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_linear_backward
