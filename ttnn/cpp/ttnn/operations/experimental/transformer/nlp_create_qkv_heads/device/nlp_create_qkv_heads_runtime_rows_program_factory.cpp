// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Interleaved factory for runtime_rows: the program owns the full grid, the row count is one common
// runtime word and every per-core range is derived from it on device, so one program serves every
// sequence length of a given width and dtype. The default Interleaved factory is untouched.

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/math.hpp>

#include "nlp_create_qkv_heads_device_operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

namespace ttnn::operations::experimental::transformer {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Static part of the head-parallel decision; the dynamic part (fewer tile rows than cores) is taken
// on device from the common sequence length, so one program serves every length.
bool head_parallel_allowed(
    const NlpCreateHeadsDeviceOperation::operation_attributes_t& operation_attributes, bool input_tensor_kv_present) {
    const bool q_only = operation_attributes.num_kv_heads == 0;
    const uint32_t head_slots = operation_attributes.num_q_heads + 2 * operation_attributes.num_kv_heads;
    return head_slots > 1 && !operation_attributes.transpose_k_heads &&
           (q_only || (!input_tensor_kv_present && !operation_attributes.q_head_split.has_value()));
}

// The five tensor arguments both factories bind: the Q input, the optional separate KV input and the three
// outputs.  The names must match the TensorParameter unique_ids the factories declare.  Shared by
// create_program_artifacts and override_runtime_arguments of both factories so a renamed or added tensor
// parameter cannot drift between them.
using TensorRunArgs = decltype(tt::tt_metal::experimental::ProgramRunArgs::tensor_args);
TensorRunArgs build_qkv_tensor_run_args(
    const NlpCreateHeadsDeviceOperation::tensor_args_t& tensor_args,
    NlpCreateHeadsDeviceOperation::tensor_return_value_t& output) {
    const TensorParamName INPUT_Q{"input_q"};
    const TensorParamName INPUT_KV{"input_kv"};
    const TensorParamName Q{"q"};
    const TensorParamName K{"k"};
    const TensorParamName V{"v"};

    TensorRunArgs tensor_run_args = {
        {INPUT_Q, tensor_args.input_tensor_q.mesh_tensor()},
        {Q, std::get<0>(output).mesh_tensor()},
        {K, std::get<1>(output).mesh_tensor()},
        {V, std::get<2>(output).mesh_tensor()},
    };
    if (tensor_args.input_tensor_kv.has_value()) {
        tensor_run_args.emplace(INPUT_KV, tensor_args.input_tensor_kv->mesh_tensor());
    }
    return tensor_run_args;
}

// The sequence length in tiles is the one common runtime word of the Interleaved program; every
// kernel derives its per-core share from it on device.
constexpr const char* kLengthArg = "h_tiles";

uint32_t interleaved_h_tiles(const NlpCreateHeadsDeviceOperation::tensor_args_t& tensor_args) {
    return tensor_args.input_tensor_q.padded_shape()[2] / TILE_HEIGHT;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts
NlpCreateHeadsDeviceOperation::InterleavedRuntimeRows::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Spec resource names (function-local: the three factories share one unity translation unit).
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName QV{"qv"};        // Q and V head tiles, reader -> writer (K too unless transpose_k_heads)
    const DFBSpecName K_IN{"k_in"};    // K head tiles, reader -> compute (transpose_k_heads only)
    const DFBSpecName K_OUT{"k_out"};  // transposed K head tiles, compute -> writer (transpose_k_heads only)
    const TensorParamName INPUT_Q{"input_q"};
    const TensorParamName INPUT_KV{"input_kv"};
    const TensorParamName Q{"q"};
    const TensorParamName K{"k"};
    const TensorParamName V{"v"};

    const Tensor& input_tensor = tensor_args.input_tensor_q;
    // Read through the reference: the KV tensor argument below must name the tensor the framework sees.
    const std::optional<Tensor>& input_tensor_kv = tensor_args.input_tensor_kv;
    const uint32_t num_q_heads = operation_attributes.num_q_heads;
    const uint32_t num_kv_heads = operation_attributes.num_kv_heads;
    const uint32_t head_dim = operation_attributes.head_dim;
    const bool transpose_k_heads = operation_attributes.transpose_k_heads;
    auto& output = tensor_return_value;

    const auto& input_shape = input_tensor.padded_shape();

    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();

    uint32_t single_tile_size = tt::tile_size(data_format);
    TT_ASSERT(input_tensor.buffer()->size() % single_tile_size == 0);

    if (read_from_input_tensor_kv) {
        TT_ASSERT(input_tensor_kv->buffer()->size() % single_tile_size == 0);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_w_tiles = input_shape[3] / TILE_WIDTH;
    uint32_t in1_w_tiles = 0;
    if (read_from_input_tensor_kv) {
        in1_w_tiles = input_tensor_kv->padded_shape()[3] / TILE_WIDTH;
    }

    // Per output tensor args
    // Output shape for Q is: [B, num_q_heads, s, head_dim], shuffled from [B, 1, s, num_q_heads * head_dim]
    // Output shape for K/V is: [B, num_kv_heads, s, head_dim], shuffled from [B, 1, s, num_kv_heads * head_dim]
    // NOTE: Output h and w dims are identical for Q, K, V, so any arg that is related to these dims for q_* can be
    // shared for K, V. The row count (s / 32) is the common runtime word, not a compile-time arg.
    uint32_t q_out_w_tiles = head_dim / TILE_WIDTH;  // tiles along head_dim
    // Tiles the reader and writer move per NoC barrier; the transposed-K path feeds compute one tile at a time.
    uint32_t tile_batch = 8;
    while (q_out_w_tiles % tile_batch != 0) {
        tile_batch /= 2;
    }
    if (operation_attributes.transpose_k_heads) {
        tile_batch = 1;
    }
    uint32_t q_num_tiles = num_q_heads * q_out_w_tiles;
    uint32_t kv_num_tiles = num_kv_heads * q_out_w_tiles;
    const uint32_t batch = input_shape[0];
    const uint32_t h_tiles = CMAKE_UNIQUE_NAMESPACE::interleaved_h_tiles(tensor_args);
    const bool head_parallel_ok =
        CMAKE_UNIQUE_NAMESPACE::head_parallel_allowed(operation_attributes, read_from_input_tensor_kv);

    // The program owns the full grid; cores past the end of the row split at a short length run the
    // DFB protocol with zero blocks.
    const CoreCoord grid = input_tensor.device()->compute_with_storage_grid_size();
    const CoreRangeSet all_cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const std::vector<CoreCoord> cores = corerange_to_cores(all_cores, std::nullopt, true);
    const uint32_t num_cores = cores.size();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    ttnn::Tensor& q = std::get<0>(output);
    ttnn::Tensor& k = std::get<1>(output);
    ttnn::Tensor& v = std::get<2>(output);

    TT_ASSERT(q.buffer() != nullptr, "Output q buffer should be allocated on device!");
    TT_ASSERT(k.buffer() != nullptr, "Output k buffer should be allocated on device!");
    TT_ASSERT(v.buffer() != nullptr, "Output v buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    // Tensor parameters: the Q input, the optional separate KV input, and the three outputs.  The kernels
    // reach them through TensorAccessor(tensor::<name>); the base addresses ride the bindings.
    // Every one of them is as tall as the sequence, which the hash drops and the kernels take at runtime.
    const TensorSpecRelaxations rows_vary{.dynamic_tensor_shape = true};
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = INPUT_Q, .spec = input_tensor.tensor_spec(), .relaxations = rows_vary},
        TensorParameter{.unique_id = Q, .spec = q.tensor_spec(), .relaxations = rows_vary},
        TensorParameter{.unique_id = K, .spec = k.tensor_spec(), .relaxations = rows_vary},
        TensorParameter{.unique_id = V, .spec = v.tensor_spec(), .relaxations = rows_vary},
    };
    if (read_from_input_tensor_kv) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = INPUT_KV, .spec = input_tensor_kv->tensor_spec(), .relaxations = rows_vary});
    }

    KernelSpec::CompilerOptions::Defines reader_defines;
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (transpose_k_heads) {
        reader_defines.emplace("TRANSPOSE_K_HEADS", "1");
        writer_defines.emplace("TRANSPOSE_K_HEADS", "1");
    }
    if (read_from_input_tensor_kv) {
        reader_defines.emplace("READ_FROM_INPUT_TENSOR_KV", "1");
    }
    if (operation_attributes.kv_tied) {
        reader_defines.emplace("KV_TIED", "1");
    }

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_create_qkv_heads_runtime_rows.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = QV,
                    .accessor_name = "qv",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT_Q,
                    .accessor_name = "input_q",
                },
            },
        .compile_time_args =
            {
                {"q_num_tiles", q_num_tiles},
                {"kv_num_tiles", kv_num_tiles},
                {"head_parallel_ok", static_cast<uint32_t>(head_parallel_ok)},
                {"head_tiles", q_out_w_tiles},
                {"in0_w_tiles", in0_w_tiles},
                {"in1_w_tiles", in1_w_tiles},
                {"batch", batch},
                {"num_cores", num_cores},
                {"tile_batch", tile_batch},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"core_idx"}, .common_runtime_arg_names = {CMAKE_UNIQUE_NAMESPACE::kLengthArg}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    // TODO: Q, K, V doesn't necessarily need to be the same output mem config
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
            "writer_tm_tile_layout_nlp_create_qkv_heads_runtime_rows.cpp",
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = QV,
                    .accessor_name = "qv",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = Q,
                    .accessor_name = "q",
                },
                TensorBinding{
                    .tensor_parameter_name = K,
                    .accessor_name = "k",
                },
                TensorBinding{
                    .tensor_parameter_name = V,
                    .accessor_name = "v",
                },
            },
        .compile_time_args =
            {
                {"q_out_w_tiles", q_out_w_tiles},
                {"q_out_c", num_q_heads},
                {"kv_out_c", num_kv_heads},
                {"head_parallel_ok", static_cast<uint32_t>(head_parallel_ok)},
                // Non-zero only for the Q head split: output 0 takes the first split_width tiles of every
                // head row and output 1 (bound as K) the rest.
                {"split_width", static_cast<uint32_t>(operation_attributes.q_head_split.value_or(0) / TILE_WIDTH)},
                {"batch", batch},
                {"num_cores", num_cores},
                {"tile_batch", tile_batch},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"core_idx"}, .common_runtime_arg_names = {CMAKE_UNIQUE_NAMESPACE::kLengthArg}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    if (read_from_input_tensor_kv) {
        reader.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = INPUT_KV,
            .accessor_name = "input_kv",
        });
    }

    // Dataflow buffers
    // Two batches, so the reader fills one while the writer drains the other.
    uint32_t dfb_num_tiles = std::max<uint32_t>(4, 2 * tile_batch);

    uint32_t qv_num_tiles = dfb_num_tiles;
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = QV,
            .entry_size = single_tile_size,
            .num_entries = qv_num_tiles,
            .data_format_metadata = data_format,
        },
    };

    Group<KernelSpecName> work_unit_kernels = {READER, WRITER};
    Group<KernelSpec> kernels;

    // If we transpose_k_heads:
    // - reader will write K heads to k_in, instead of qv
    // - compute will wait on k_in and write to k_out
    // - writer will wait on k_out, instead of qv
    // Neither K buffer exists otherwise.
    if (transpose_k_heads) {
        // For FLOAT32 input, enable fp32 dest accumulation so the JIT data-format selection
        // resolves the unpack-dst buffer to Tf32 (10-bit mantissa) instead of Float16_b (7-bit
        // mantissa). Mirrors the per-dtype promotion in eltwise unary/binary primitives.
        const bool fp32_dest_acc_en = input_tensor.dtype() == tt_metal::DataType::FLOAT32;

        ComputeHardwareConfig compute_hw{.enable_32_bit_dest = fp32_dest_acc_en};
        if (fp32_dest_acc_en) {
            // The legacy descriptor left unpack_to_dest_mode empty (unpack to SrcA/B).  With a 32-bit dest
            // and a Float32 input buffer the mode has to be stated explicitly; this is the same mode.
            compute_hw.unpack_modes.emplace(K_IN, UnpackMode::UnpackToSrc);
        }

        kernels.push_back(KernelSpec{
            .unique_id = COMPUTE,
            .source = "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/compute/"
                      "transpose_k_heads.cpp",
            // The legacy compute descriptor resolved to O3; Metal 2.0 defaults every kernel to O2.
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = K_IN,
                        .accessor_name = "in",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = K_OUT,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                },
            .compile_time_args = {{"kv_num_tiles", kv_num_tiles}, {"batch", batch}, {"num_cores", num_cores}},
            .runtime_arg_schema =
                {.runtime_arg_names = {"core_idx"}, .common_runtime_arg_names = {CMAKE_UNIQUE_NAMESPACE::kLengthArg}},
            .hw_config = ComputeHardwareConfig{compute_hw},
        });
        work_unit_kernels.push_back(COMPUTE);

        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = K_IN,
            .accessor_name = "k",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = K_OUT,
            .accessor_name = "k",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });

        uint32_t k_in_num_tiles = dfb_num_tiles;
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = K_IN,
            .entry_size = single_tile_size,
            .num_entries = k_in_num_tiles,
            .data_format_metadata = data_format,
        });

        uint32_t k_out_num_tiles = dfb_num_tiles;
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = K_OUT,
            .entry_size = single_tile_size,
            .num_entries = k_out_num_tiles,
            .data_format_metadata = data_format,
        });
    }

    KernelRunArgs reader_run_args{
        .kernel = READER, .common_runtime_arg_values = {{CMAKE_UNIQUE_NAMESPACE::kLengthArg, h_tiles}}};
    KernelRunArgs writer_run_args{
        .kernel = WRITER, .common_runtime_arg_values = {{CMAKE_UNIQUE_NAMESPACE::kLengthArg, h_tiles}}};
    KernelRunArgs compute_run_args{
        .kernel = COMPUTE, .common_runtime_arg_values = {{CMAKE_UNIQUE_NAMESPACE::kLengthArg, h_tiles}}};
    for (uint32_t i = 0; i < num_cores; ++i) {
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, cores[i], {{"core_idx", i}});
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, cores[i], {{"core_idx", i}});
        if (transpose_k_heads) {
            AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, cores[i], {{"core_idx", i}});
        }
    }

    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));

    ProgramSpec spec{
        .name = "nlp_create_qkv_heads_interleaved_runtime_rows",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {WorkUnitSpec{
            .name = "all_cores",
            .kernels = std::move(work_unit_kernels),
            .target_nodes = all_cores,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    if (transpose_k_heads) {
        run_args.kernel_run_args.push_back(std::move(compute_run_args));
    }
    run_args.tensor_args = CMAKE_UNIQUE_NAMESPACE::build_qkv_tensor_run_args(tensor_args, output);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

// The hash drops the row count, so a hit rewrites the one common length word of every kernel (the
// per-core split is derived on device from it) and re-binds the five tensors.
tt::tt_metal::experimental::ProgramRunArgs
NlpCreateHeadsDeviceOperation::InterleavedRuntimeRows::override_runtime_arguments(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const uint32_t h_tiles = CMAKE_UNIQUE_NAMESPACE::interleaved_h_tiles(tensor_args);
    ProgramRunArgs params;
    params.kernel_run_args = {
        KernelRunArgs{
            .kernel = KernelSpecName{"reader"},
            .common_runtime_arg_values = {{CMAKE_UNIQUE_NAMESPACE::kLengthArg, h_tiles}}},
        KernelRunArgs{
            .kernel = KernelSpecName{"writer"},
            .common_runtime_arg_values = {{CMAKE_UNIQUE_NAMESPACE::kLengthArg, h_tiles}}},
    };
    if (operation_attributes.transpose_k_heads) {
        params.kernel_run_args.push_back(KernelRunArgs{
            .kernel = KernelSpecName{"compute"},
            .common_runtime_arg_values = {{CMAKE_UNIQUE_NAMESPACE::kLengthArg, h_tiles}}});
    }
    params.tensor_args = CMAKE_UNIQUE_NAMESPACE::build_qkv_tensor_run_args(tensor_args, tensor_return_value);
    return params;
}

}  // namespace ttnn::operations::experimental::transformer
