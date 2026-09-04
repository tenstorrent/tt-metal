// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gelu_backward_program_factory.hpp"
#include "gelu_backward_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt_stl/assert.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;

ttnn::device_operation::ProgramArtifacts GeluBackwardProgramFactory::create_program_artifacts(
    const GeluBackwardParams& args, const GeluBackwardInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    // Work against the Metalium device-tensor type throughout; the TTNN wrappers only carry us in.
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& grad_output = tensor_args.grad_output.mesh_tensor();
    const auto& grad_in = output.mesh_tensor();

    // Program-scope resource names. Declared function-local (not at namespace scope) so that the
    // unity build, which concatenates every op's factory into one translation unit, cannot collide
    // these very generic identifiers across ops.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // One DFB per legacy circular buffer: the two operand streams the reader fills for compute, and
    // the result stream compute packs for the writer to drain.
    const DFBSpecName GRAD_OUTPUT_DFB{"grad_output"};
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName GRAD_IN_DFB{"grad_in"};

    const TensorParamName GRAD_OUTPUT{"grad_output"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    DataFormat src0_cb_data_format = datatype_to_dataformat_converter(grad_output.dtype());
    uint32_t src0_single_tile_size = tile_size(src0_cb_data_format);
    DataFormat src1_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t src1_single_tile_size = tile_size(src1_cb_data_format);
    DataFormat dst_cb_data_format = datatype_to_dataformat_converter(grad_in.dtype());
    uint32_t dst_single_tile_size = tile_size(dst_cb_data_format);

    uint32_t num_tiles = input.physical_volume() / TILE_HW;

    const auto& device = input.device();
    auto compute_with_storage_grid_size = device.compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    constexpr uint32_t num_input_tiles = 2;
    constexpr uint32_t num_output_tiles = 2;

    // --- Dataflow Buffers ---
    // The legacy CBs left CBFormatDescriptor::tile unset, so tile_format_metadata stays unset too
    // (standard 32x32 tiles).
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = GRAD_OUTPUT_DFB,
            .entry_size = src0_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = src0_cb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = INPUT_DFB,
            .entry_size = src1_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = src1_cb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = GRAD_IN_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = dst_cb_data_format,
        },
    };

    // --- Reader Kernel ---
    KernelSpec reader_spec{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/"
            "reader_binary_interleaved_start_id_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = GRAD_OUTPUT_DFB,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT_DFB,
                    .accessor_name = "in1",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = GRAD_OUTPUT, .accessor_name = "src0"},
                TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src1"},
            },
        // This op is interleaved-only, so the shared reader's block/width-sharded path is never
        // taken; the flag stays 0 exactly as the legacy factory passed it.
        .compile_time_args = {{"block_or_width_sharded", 0u}},
        // block_height / block_width / num_cores_y feed only that untaken path, but the shared
        // kernel reads them unconditionally, so all five arguments are supplied.
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_tiles", "start_id", "block_height", "block_width", "num_cores_y"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device.arch()),
    };

    // --- Writer Kernel ---
    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = GRAD_IN_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device.arch()),
    };

    // --- Compute Kernel ---
    // The legacy factory set a Metal ComputeConfigDescriptor directly rather than resolving a TTNN
    // ComputeKernelConfig, so build the Gen1 config by hand and leave every field the op did not set
    // at its default: ComputeGen1Config's defaults coincide with the legacy descriptor's
    // (math_approx_mode=false -> sfpu_precision_mode=Precise, dst_full_sync_en=false ->
    // double_buffer_dest=true, bfp8_pack_precise=false -> bfp_pack_precision_mode=Approximate).
    // The migrated chain computes through DEST, so Float32 inputs also require a 32-bit DEST even
    // when the output format is narrower.
    bool fp32_dest_acc_en = (src0_cb_data_format == DataFormat::Float32) ||
                            (src1_cb_data_format == DataFormat::Float32) ||
                            (dst_cb_data_format == DataFormat::Float32) || (dst_cb_data_format == DataFormat::Int32) ||
                            (dst_cb_data_format == DataFormat::UInt32);

    // Legacy asked for UnpackToDestFp32 on both operand CBs unconditionally, but the legacy lowering
    // honours that request only for a Float32-formatted buffer — get_unpack_dst_formats() consults
    // the entry inside `if (src_format == DataFormat::Float32 && ...)` and otherwise leaves the
    // buffer on the SrcA/B path. Reproduce that exactly by emitting UnpackToDest only where the
    // legacy request was live; an omitted entry means UnpackToSrc, which is what legacy effectively
    // used for every narrower format.
    ComputeUnpackModes unpack_modes;
    if (src0_cb_data_format == DataFormat::Float32) {
        unpack_modes[GRAD_OUTPUT_DFB] = UnpackMode::UnpackToDest;
    }
    if (src1_cb_data_format == DataFormat::Float32) {
        unpack_modes[INPUT_DFB] = UnpackMode::UnpackToDest;
    }

    ComputeGen1Config compute_hw_config{
        .fpu_math_fidelity = MathFidelity::HiFi4,
        .enable_32_bit_dest = fp32_dest_acc_en,
        .unpack_modes = std::move(unpack_modes),
    };

    const char* compute_kernel_path = nullptr;
    if (args.approximate == "tanh") {
        compute_kernel_path =
            "ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/"
            "kernels/compute/eltwise_bw_gelu_approx_tanh.cpp";
    } else {
        compute_kernel_path =
            "ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/"
            "kernels/compute/eltwise_bw_gelu_poly.cpp";
    }

    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = compute_kernel_path,
        // Legacy ComputeConfig defaults opt_level to O3 and this kernel set none; Metal 2.0's
        // type-agnostic CompilerOptions defaults to O2, so state O3 explicitly to keep the compile
        // and link at the level the op had.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = GRAD_OUTPUT_DFB,
                    .accessor_name = "grad_out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT_DFB,
                    .accessor_name = "input",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = GRAD_IN_DFB,
                    .accessor_name = "grad_in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_tiles"},
            },
        .hw_config = ComputeHardwareConfig{std::move(compute_hw_config)},
    };

    ProgramSpec spec{
        .name = "gelu_backward",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = GRAD_OUTPUT, .spec = grad_output.tensor_spec()},
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = grad_in.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER, COMPUTE},
                    .target_nodes = all_cores,
                },
            },
    };

    // --- Per-core runtime args ---
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_tiles", num_tiles_per_core},
             {"start_id", num_tiles_written},
             {"block_height", 0u},
             {"block_width", 0u},
             {"num_cores_y", num_cores_y}});

        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"num_tiles", num_tiles_per_core}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_pages", num_tiles_per_core}, {"start_id", num_tiles_written}});

        num_tiles_written += num_tiles_per_core;
    }

    ProgramRunArgs run_args;
    // push_back rather than brace-init: an initializer_list would copy the per-node arg tables.
    run_args.kernel_run_args.reserve(3);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args = {
        {GRAD_OUTPUT, grad_output},
        {INPUT, input},
        {OUTPUT, grad_in},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
