// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_matmul_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

// Spec names. Op-scoped namespace rather than an anonymous one: the experimental matmul factories
// share a CMake target, where same-named anonymous-namespace constants collide under a unity build.
namespace attn_matmul_metal2 {
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
const KernelSpecName COMPUTE{"compute"};

const DFBSpecName IN0{"in0"};
const DFBSpecName IN1{"in1"};
const DFBSpecName INTERMED0{"intermed0"};
const DFBSpecName INTERMED1{"intermed1"};
const DFBSpecName INTERMED2{"intermed2"};
const DFBSpecName OUT{"out"};

const TensorParamName INPUT_A{"input_a"};
const TensorParamName INPUT_B{"input_b"};
const TensorParamName OUTPUT{"output"};
}  // namespace attn_matmul_metal2

ttnn::device_operation::ProgramArtifacts AttnMatmulProgramFactory::create_program_artifacts(
    const AttnMatmulParams& operation_attributes, const AttnMatmulInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace attn_matmul_metal2;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    auto& output = tensor_return_value;
    const auto& a_mesh = a.mesh_tensor();
    const auto& b_mesh = b.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    tt::tt_metal::distributed::MeshDevice* device = a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat interm_data_format = fp32_dest_acc_en and in0_data_format == tt::DataFormat::Float32
                                            ? tt::DataFormat::Float32
                                            : tt::DataFormat::Float16_b;
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    if (in0_data_format == tt::DataFormat::Float32 or in1_data_format == tt::DataFormat::Float32 or
        output_data_format == tt::DataFormat::Float32) {
        TT_FATAL(fp32_dest_acc_en == true, "when inputs/output are in fp32 format, fp32_dest_acc_en must be set");
    }

    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);
    log_debug(tt::LogOp, "in0_data_format: {}", in0_data_format);
    log_debug(tt::LogOp, "in1_data_format: {}", in1_data_format);
    log_debug(tt::LogOp, "interm_data_format: {}", interm_data_format);
    log_debug(tt::LogOp, "output_data_format: {}", output_data_format);

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    // A block of work is one MtNt
    auto num_output_blocks_total = ashape[1];  // ashape[1] is Q num_heads; only parallelize on this
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_blocks_per_core_group_1,
         num_output_blocks_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(
                operation_attributes.compute_with_storage_grid_size, num_output_blocks_total);

    CoreRangeSet all_device_cores{CoreRange(
        {0, 0}, {device->compute_with_storage_grid_size().x - 1, device->compute_with_storage_grid_size().y - 1})};
    uint32_t total_num_cores = device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
    // The kernels run on every device core, so the runtime-arg loop walks the device grid, not the op's sub-grid.
    const uint32_t device_num_cores_y = device->compute_with_storage_grid_size().y;

    // C = torch.matmul(A.transpose(0, 2) * B).transpose(0, 2)
    const bool transpose_hw_bool = operation_attributes.transpose_hw.value_or(false);
    const uint32_t num_tokens_val =
        operation_attributes.num_tokens.value_or(0);  // should not be nullopt if transpose_hw=true
    constexpr uint32_t num_rows_in_one_tile = 32;

    uint32_t Mt = ashape[2] / TILE_HEIGHT;
    uint32_t Kt = ashape[3] / TILE_WIDTH;
    // For transpose_hw=true, in1_Kt is same as in0_Kt but on bshape[3]
    // For transpose_hw=false, in1_Kt is on bshape[2] but represents the max cache length to read from (ie. may not
    // equal in0_Kt)
    uint32_t in1_Kt = transpose_hw_bool ? Kt : bshape[2] / TILE_HEIGHT;
    uint32_t Nt = transpose_hw_bool ? num_tokens_val / TILE_HEIGHT : bshape[3] / TILE_WIDTH;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;
    uint32_t in1_KtNt_stride = transpose_hw_bool ? bshape[2] / TILE_HEIGHT * in1_Kt : in1_Kt * Nt;
    uint32_t in1_KtNt_skip = transpose_hw_bool ? (bshape[2] / TILE_HEIGHT - 1) * in1_Kt : (in1_Kt - Kt) * Nt;

    // ---- Dataflow buffers ----
    // in0's capacity (Kt entries) depends on the input shape; the default reflection hash keys the
    // program cache on the full tensor specs, so each unique Kt keeps its own cache entry.
    DataflowBufferSpec dfb_in0{
        .unique_id = IN0,
        .entry_size = in0_single_tile_size,
        .num_entries = Kt,
        .data_format_metadata = in0_data_format,
    };

    const uint32_t dfb_in1_num_input_tiles = 2;
    DataflowBufferSpec dfb_in1{
        .unique_id = IN1,
        .entry_size = in1_single_tile_size,
        .num_entries = dfb_in1_num_input_tiles,
        .data_format_metadata = in1_data_format,
    };

    DataflowBufferSpec dfb_intermed0{
        .unique_id = INTERMED0,
        .entry_size = interm_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = interm_data_format,
    };

    DataflowBufferSpec dfb_intermed1{
        .unique_id = INTERMED1,
        .entry_size = interm_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = interm_data_format,
    };

    DataflowBufferSpec dfb_intermed2{
        .unique_id = INTERMED2,
        .entry_size = interm_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = interm_data_format,
    };

    const uint32_t num_output_tiles = 2;
    DataflowBufferSpec dfb_out{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_data_format,
    };

    // ---- Compute hardware config ----
    ComputeHardwareConfig compute_hw = to_compute_hardware_config(operation_attributes.compute_kernel_config);
    compute_hw.sfpu_precision_mode = Precision::Precise;
    compute_hw.double_buffer_dest = true;
    if (compute_hw.enable_32_bit_dest) {
        if (in0_data_format == tt::DataFormat::Float32) {
            compute_hw.unpack_modes.emplace(IN0, UnpackMode::UnpackToSrc);
        }
        if (in1_data_format == tt::DataFormat::Float32) {
            compute_hw.unpack_modes.emplace(IN1, UnpackMode::UnpackToSrc);
        }
        if (interm_data_format == tt::DataFormat::Float32) {
            compute_hw.unpack_modes.emplace(INTERMED0, UnpackMode::UnpackToSrc);
            compute_hw.unpack_modes.emplace(INTERMED2, UnpackMode::UnpackToSrc);
        }
    }

    // ---- Kernels ----
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/kernels/dataflow/"
            "reader_transformer_attn_matmul.cpp",
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
                // The reader copies one row out of each untilized tile (intermed1, from compute) into
                // the tile it assembles for compute (intermed2).
                DFBBinding{
                    .dfb_spec_name = INTERMED1,
                    .accessor_name = "intermed1",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = INTERMED2,
                    .accessor_name = "intermed2",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT_A,
                    .accessor_name = "src0",
                },
                TensorBinding{
                    .tensor_parameter_name = INPUT_B,
                    .accessor_name = "src1",
                },
            },
        .compile_time_args =
            {
                {"transpose_hw", static_cast<uint32_t>(transpose_hw_bool)},
                {"fp32_acc_en", static_cast<uint32_t>(fp32_dest_acc_en and in0_data_format == tt::DataFormat::Float32)},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"Mt",
                     "Kt",
                     "Nt",
                     "MtKt",
                     "in1_KtNt_skip",
                     "in1_KtNt_mul_32",
                     "blocks",
                     "itileA_start",
                     "itileB_start"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    // The Metal 2.0 fork of eltwise/unary's shared interleaved writer. Its binding and argument
    // names (dfb::out, tensor::dst, num_pages, start_id) are the fork's interface, inherited here.
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
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
                TensorBinding{
                    .tensor_parameter_name = OUTPUT,
                    .accessor_name = "dst",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/kernels/compute/"
            "transformer_attn_matmul.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
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
                    .dfb_spec_name = INTERMED0,
                    .accessor_name = "intermed0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = INTERMED0,
                    .accessor_name = "intermed0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = INTERMED1,
                    .accessor_name = "intermed1",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = INTERMED2,
                    .accessor_name = "intermed2",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args =
            {
                {"transpose_hw", static_cast<uint32_t>(transpose_hw_bool)},  // transpose_hw for matmul_init
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"batch", "Mt", "Kt", "Nt"},
            },
        .hw_config = compute_hw,
    };

    // ---- Per-core runtime args (idle cores get zeroed args) ----
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    uint32_t num_output_blocks_per_core = 0;
    for (uint32_t i = 0, num_blocks_written = 0; i < total_num_cores; i++) {
        CoreCoord core = {i / device_num_cores_y, i % device_num_cores_y};

        if (core_group_1.contains(core)) {
            num_output_blocks_per_core = num_output_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_blocks_per_core = num_output_blocks_per_core_group_2;
        } else {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"Mt", 0u},
                 {"Kt", 0u},
                 {"Nt", 0u},
                 {"MtKt", 0u},
                 {"in1_KtNt_skip", 0u},
                 {"in1_KtNt_mul_32", 0u},
                 {"blocks", 0u},
                 {"itileA_start", 0u},
                 {"itileB_start", 0u}});
            AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values, core, {{"batch", 0u}, {"Mt", 0u}, {"Kt", 0u}, {"Nt", 0u}});
            AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"num_pages", 0u}, {"start_id", 0u}});
            continue;
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {
                {"Mt", Mt},
                {"Kt", Kt},
                {"Nt", Nt},
                {"MtKt", MtKt},
                {"in1_KtNt_skip", in1_KtNt_skip},
                {"in1_KtNt_mul_32", in1_KtNt_stride * num_rows_in_one_tile},
                {"blocks", num_output_blocks_per_core},
                {"itileA_start", num_blocks_written * MtKt},
                {"itileB_start", 0u},  // always read same in1 per core
            });
        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            core,
            {
                {"batch", 1u},
                {"Mt", 1u},
                {"Kt", Kt},
                {"Nt", num_output_blocks_per_core * MtNt},
            });
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"num_pages", num_output_blocks_per_core * MtNt},
                {"start_id", num_blocks_written * MtNt},
            });
        num_blocks_written += num_output_blocks_per_core;
    }

    ProgramSpec spec{
        .name = "attn_matmul",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers =
            {std::move(dfb_in0),
             std::move(dfb_in1),
             std::move(dfb_intermed0),
             std::move(dfb_intermed1),
             std::move(dfb_intermed2),
             std::move(dfb_out)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT_A, .spec = a.tensor_spec()},
                TensorParameter{.unique_id = INPUT_B, .spec = b.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER, COMPUTE},
                    .target_nodes = all_device_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args = {
        {INPUT_A, a_mesh},
        {INPUT_B, b_mesh},
        {OUTPUT, output_mesh},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
