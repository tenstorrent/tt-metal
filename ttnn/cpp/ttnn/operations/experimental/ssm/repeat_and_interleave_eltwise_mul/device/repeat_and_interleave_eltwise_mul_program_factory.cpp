// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_and_interleave_eltwise_mul_program_factory.hpp"

#include <map>
#include <string>
#include <utility>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {
constexpr uint32_t ONE_TILE = 1;
}  // namespace

ttnn::device_operation::ProgramArtifacts RepeatAndInterleaveEltwiseMulProgramFactory::create_program_artifacts(
    const RepeatMulParams& operation_attributes, const RepeatMulInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;
    auto& output = tensor_return_value;

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    // Metalium-native device tensors: these back the TensorParameter specs and the
    // TensorArguments, which the framework matches by MeshTensor identity.
    const auto& a_tensor = a.mesh_tensor();
    const auto& b_tensor = b.mesh_tensor();
    const auto& output_tensor = output.mesh_tensor();

    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat interm_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    // Parallelize on bshape[-1]
    auto num_output_blocks_total = bshape[-1] / TILE_WIDTH;
    const bool row_major = false;
    auto* device = a.device();
    auto device_compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(device_compute_with_storage_grid_size, num_output_blocks_total, row_major);

    uint32_t g1_numcores = core_group_1.num_cores();
    std::vector<CoreCoord> cores = grid_to_cores(
        num_cores, device_compute_with_storage_grid_size.x, device_compute_with_storage_grid_size.y, row_major);

    // Kernel identifiers
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // Dataflow buffer identifiers
    const DFBSpecName IN0{"in0"};
    const DFBSpecName IN1{"in1"};
    const DFBSpecName OUT{"out"};
    const DFBSpecName IN0_TRANSPOSED{"in0_transposed"};
    const DFBSpecName IN1_TRANSPOSED{"in1_transposed"};
    const DFBSpecName IN1_BCAST_ROW{"in1_bcast_row"};
    const DFBSpecName OUT_TRANSPOSED{"out_transposed"};

    // Tensor parameter identifiers
    const TensorParamName SRC0{"src0"};
    const TensorParamName SRC1{"src1"};
    const TensorParamName DST{"dst"};

    uint32_t dfb0_entries = ONE_TILE * 2;        // double buffer
    uint32_t dfb1_entries = ONE_TILE * 2;        // double buffer
    uint32_t output_dfb_entries = ONE_TILE * 2;  // double buffer
    uint32_t interm_num_entries = ONE_TILE * 2;  // double buffer

    // Kernel-source configuration. The compute path taken depends on which of the two inputs
    // arrives already widened, and the kernels select it with these defines.
    std::map<std::string, std::string> ssm_eltwise_defines;
    if (ashape[-1] == TILE_WIDTH) {
        ssm_eltwise_defines["REPEAT_IN0"] = "1";
    }
    const bool repeat_interleave_in1 = bshape[-1] == HIDDEN_SIZE;
    if (repeat_interleave_in1) {
        ssm_eltwise_defines["REPEAT_INTERLEAVE_IN1"] = "1";
    }
    const KernelSpec::CompilerOptions::Defines kernel_defines(ssm_eltwise_defines);

    ////////////////////////////////////////////////////////////////////////////
    //                      Build ProgramSpec
    ////////////////////////////////////////////////////////////////////////////

    // Dataflow buffers.
    //
    // in0_transposed and out_transposed are compute-private. Under REPEAT_INTERLEAVE_IN1 compute
    // both fills and drains them; without it that code is compiled out and compute merely names
    // them — it constructs the buffers, and reads in0_transposed's data format for a pack
    // reconfig — without touching either. Compute is the only kernel involved either way, so each
    // is bound as a self-loop: PRODUCER and CONSUMER on that one kernel. Every DFB needs both
    // endpoints, and the role labels only drive FIFO machinery that a non-touching kernel never
    // invokes.
    //
    // in1_transposed carries the transposed in1 tile from compute to the reader, which slices it
    // into single rows. Compute produces and the reader consumes, one endpoint apiece, in every
    // configuration — the same shape as out. Outside REPEAT_INTERLEAVE_IN1 neither kernel touches
    // the buffer, but the endpoints stay as they are because every DFB needs both.
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = IN0,
            .entry_size = in0_single_tile_size,
            .num_entries = dfb0_entries,
            .data_format_metadata = in0_data_format,
        },
        DataflowBufferSpec{
            .unique_id = IN1,
            .entry_size = in1_single_tile_size,
            .num_entries = dfb1_entries,
            .data_format_metadata = in1_data_format,
        },
        DataflowBufferSpec{
            .unique_id = OUT,
            .entry_size = output_single_tile_size,
            .num_entries = output_dfb_entries,
            .data_format_metadata = output_data_format,
        },
        DataflowBufferSpec{
            .unique_id = IN0_TRANSPOSED,
            .entry_size = interm_single_tile_size,
            .num_entries = interm_num_entries,
            .data_format_metadata = interm_data_format,
        },
        DataflowBufferSpec{
            .unique_id = IN1_TRANSPOSED,
            .entry_size = interm_single_tile_size,
            .num_entries = interm_num_entries,
            .data_format_metadata = interm_data_format,
        },
        DataflowBufferSpec{
            .unique_id = IN1_BCAST_ROW,
            .entry_size = interm_single_tile_size,
            .num_entries = interm_num_entries,
            .data_format_metadata = interm_data_format,
        },
        DataflowBufferSpec{
            .unique_id = OUT_TRANSPOSED,
            .entry_size = interm_single_tile_size,
            .num_entries = interm_num_entries,
            .data_format_metadata = interm_data_format,
        },
    };

    // Kernels
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/"
            "reader_ssm_eltwise_mul.cpp",
        .compiler_options = {.defines = kernel_defines},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN0,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = IN1,
                    .accessor_name = "in1",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = IN1_TRANSPOSED,
                    .accessor_name = "in1_transposed",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = IN1_BCAST_ROW,
                    .accessor_name = "in1_bcast_row",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = SRC0, .accessor_name = "src0"},
                TensorBinding{.tensor_parameter_name = SRC1, .accessor_name = "src1"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"in1_num_blocks", "in1_start_id", "in1_num_blocks_h", "in1_num_blocks_w", "in0_num_blocks_w"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/"
            "writer_ssm_eltwise_mul.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = DST, .accessor_name = "dst"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"out_num_blocks_w_per_core", "start_id", "out_num_blocks_h", "out_total_blocks_w"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    Group<DFBBinding> compute_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = IN0,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = IN1,
            .accessor_name = "in1",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = IN0_TRANSPOSED,
            .accessor_name = "in0_transposed",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = IN0_TRANSPOSED,
            .accessor_name = "in0_transposed",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = IN1_TRANSPOSED,
            .accessor_name = "in1_transposed",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = IN1_BCAST_ROW,
            .accessor_name = "in1_bcast_row",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = OUT_TRANSPOSED,
            .accessor_name = "out_transposed",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = OUT_TRANSPOSED,
            .accessor_name = "out_transposed",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
    };

    // Compute hw_config — Style B (the legacy factory set a Metal ComputeConfigDescriptor directly,
    // with no TTNN ComputeKernelConfig behind it). Build ComputeGen1Config field by field: the three
    // fields the descriptor set explicitly are carried over (math_approx_mode = false becomes
    // sfpu_precision_mode = Precise), and the three it left alone stay at ComputeGen1Config's
    // defaults, which coincide with the legacy descriptor's. No unpack_modes entry is required
    // because enable_32_bit_dest is false.
    const ComputeHardwareConfig compute_hw_config = ComputeGen1Config{
        .fpu_math_fidelity = operation_attributes.math_fidelity,
        .sfpu_precision_mode = Precision::Precise,
        .enable_32_bit_dest = false,
    };

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/"
            "ssm_eltwise_mul.cpp",
        // O3 is the level a compute kernel got from the legacy per-kernel-type default; Metal 2.0's
        // single CompilerOptions defaults to O2, so it has to be asked for.
        .compiler_options = {.defines = kernel_defines, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"in1_num_blocks", "in1_num_blocks_h"},
            },
        .hw_config = compute_hw_config,
    };

    ProgramSpec spec{
        .name = "repeat_and_interleave_eltwise_mul",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = SRC0, .spec = a_tensor.tensor_spec()},
                TensorParameter{.unique_id = SRC1, .spec = b_tensor.tensor_spec()},
                TensorParameter{.unique_id = DST, .spec = output_tensor.tensor_spec()},
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

    ////////////////////////////////////////////////////////////////////////////
    //                      Build ProgramRunArgs
    ////////////////////////////////////////////////////////////////////////////
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    // Set runtime args per core
    uint32_t num_blocks_per_core = 0;
    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        if (i < g1_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else {
            num_blocks_per_core = num_blocks_per_core_group_2;
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            cores[i],
            {{"in1_num_blocks", num_blocks_per_core},
             {"in1_start_id", num_blocks_written},
             {"in1_num_blocks_h", static_cast<uint32_t>(bshape[2] / TILE_HEIGHT)},
             {"in1_num_blocks_w", static_cast<uint32_t>(bshape[-1] / TILE_WIDTH)},
             {"in0_num_blocks_w", static_cast<uint32_t>(ashape[-1] / TILE_WIDTH)}});

        // update writer's block count based on input_b already repeat_interleaved or not
        uint32_t writer_num_tiles = num_blocks_per_core;
        uint32_t writer_start_id = num_blocks_written;
        if (bshape[-1] == HIDDEN_SIZE) {
            writer_num_tiles = num_blocks_per_core * TILE_WIDTH;
            writer_start_id = num_blocks_written * TILE_WIDTH;
        }
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            cores[i],
            {{"out_num_blocks_w_per_core", writer_num_tiles},
             {"start_id", writer_start_id},
             {"out_num_blocks_h", static_cast<uint32_t>(bshape[2] / TILE_HEIGHT)},
             {"out_total_blocks_w", HIDDEN_SIZE}});

        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            cores[i],
            {{"in1_num_blocks", num_blocks_per_core},
             {"in1_num_blocks_h", static_cast<uint32_t>(bshape[2] / TILE_HEIGHT)}});

        num_blocks_written += num_blocks_per_core;
    }

    ProgramRunArgs run_args{
        .kernel_run_args = {reader_run_args, writer_run_args, compute_run_args},
        .tensor_args =
            {
                {SRC0, a_tensor},
                {SRC1, b_tensor},
                {DST, output_tensor},
            },
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
