// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_multi_core_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <filesystem>

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts RotaryEmbeddingLlamaMultiCore::create_program_artifacts(
    const RotaryEmbeddingLlamaParams& operation_attributes,
    const RotaryEmbeddingLlamaInputs& tensor_args,
    tt::tt_metal::Tensor& output) {
    // Metal 2.0 named resource handles (declared as LOCALS for unity-build hygiene).
    const DFBSpecName IN_DFB{"in"};
    const DFBSpecName COS_DFB{"cos"};
    const DFBSpecName SIN_DFB{"sin"};
    const DFBSpecName TRANS_MAT_DFB{"trans_mat"};
    const DFBSpecName ROTATED_INTERM_DFB{"rotated_in_interm"};
    const DFBSpecName COS_INTERM_DFB{"cos_interm"};
    const DFBSpecName SIN_INTERM_DFB{"sin_interm"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName ZERO_DFB{"zero"};

    const TensorParamName IN_TENSOR{"in_tensor"};
    const TensorParamName COS_TENSOR{"cos_tensor"};
    const TensorParamName SIN_TENSOR{"sin_tensor"};
    const TensorParamName TRANS_MAT_TENSOR{"trans_mat_tensor"};
    const TensorParamName OUT_TENSOR{"out_tensor"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL{"compute"};

    const auto& input = tensor_args.input_tensor;
    const auto& cos = tensor_args.cos_cache;
    const auto& sin = tensor_args.sin_cache;
    const auto& trans_mat = tensor_args.trans_mat;

    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    const tt::DataFormat cos_cb_data_format = datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);

    const tt::DataFormat sin_cb_data_format = datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);

    const tt::DataFormat trans_mat_cb_data_format = datatype_to_dataformat_converter(trans_mat.dtype());
    const uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_cb_data_format);

    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const uint32_t batch = input.padded_shape()[0];
    const uint32_t n_heads = input.padded_shape()[1];
    const uint32_t seq_len_t = input.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t head_dim_t = input.padded_shape()[3] / TILE_WIDTH;
    const uint32_t cos_seq_len_t = cos.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t sin_seq_len_t = sin.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t rotary_seq_len_t = std::min({seq_len_t, cos_seq_len_t, sin_seq_len_t});

    if (seq_len_t != cos_seq_len_t || seq_len_t != sin_seq_len_t) {
        log_warning(
            tt::LogOp,
            "rotary_embedding_llama sequence tile coverage mismatch: input_Ht={}, cos_Ht={}, sin_Ht={}, "
            "rotary_Ht={}. Tiles beyond rotary_Ht will be zero-filled in the output.",
            seq_len_t,
            cos_seq_len_t,
            sin_seq_len_t,
            rotary_seq_len_t);
    }

    // Flag for whether or not sin/cos vary per head. If false, they will be broadcasted across heads.
    const bool freq_per_head = cos.padded_shape()[1] == n_heads;

    tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    CoreRange all_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    const uint32_t num_input_tiles = 2 * head_dim_t;
    const uint32_t num_output_tiles = num_input_tiles;

    bool row_major = true;

    // Parallelization
    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t batch_parallel_factor = std::min(batch, num_cores);
    const uint32_t seq_parallel_factor = std::min(num_cores / batch_parallel_factor, seq_len_t);
    const uint32_t batch_per_core = (batch + batch_parallel_factor - 1) / batch_parallel_factor;
    const uint32_t seq_per_core = (seq_len_t + seq_parallel_factor - 1) / seq_parallel_factor;

    const uint32_t num_sin_cos_rows_per_core = (seq_len_t + seq_parallel_factor - 1) / seq_parallel_factor;
    const uint32_t num_rows_per_core = num_sin_cos_rows_per_core * n_heads;

    uint32_t num_cos_sin_tiles = 2 * head_dim_t * num_sin_cos_rows_per_core;

    uint32_t input_cb_num_tiles = num_sin_cos_rows_per_core * num_input_tiles;

    // Reload implementation is used if sequence length is larger than some heuristic threshold where
    // the buffer size will be too large or if sin/cos are not broadcasted across heads.
    const bool use_reload_impl = num_rows_per_core > 8 || freq_per_head;
    if (use_reload_impl) {
        // Only size CBs to double buffer head_dim_t tiles for all inputs
        input_cb_num_tiles = num_input_tiles;
        num_cos_sin_tiles = num_input_tiles;
    }

    const uint32_t num_interm_tiles = head_dim_t;
    const std::string reload_define = use_reload_impl ? "1" : "0";

    // ----------------------------------------------------------------------------
    // Tensor parameters — all five tensors are Case 1 (the reader/writer build a
    // TensorAccessor from each binding). No borrowed-memory DFB in this interleaved factory.
    // ----------------------------------------------------------------------------
    TensorParameter in_param{.unique_id = IN_TENSOR, .spec = input.tensor_spec()};
    TensorParameter cos_param{.unique_id = COS_TENSOR, .spec = cos.tensor_spec()};
    TensorParameter sin_param{.unique_id = SIN_TENSOR, .spec = sin.tensor_spec()};
    TensorParameter trans_mat_param{.unique_id = TRANS_MAT_TENSOR, .spec = trans_mat.tensor_spec()};
    TensorParameter out_param{.unique_id = OUT_TENSOR, .spec = output.tensor_spec()};

    // ----------------------------------------------------------------------------
    // Dataflow buffers (all program-local FIFOs; none borrowed).
    //   reader -> compute: in (c_0), cos (c_1), sin (c_2), trans_mat (c_3)
    //   compute self-loop: rotated_in_interm (c_24), cos_interm (c_25), sin_interm (c_26)
    //   compute -> writer:  out (c_16)
    //   writer self-loop:   zero (c_27)
    // ----------------------------------------------------------------------------
    DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = input_cb_num_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec cos_dfb_spec{
        .unique_id = COS_DFB,
        .entry_size = cos_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = cos_cb_data_format,
    };
    DataflowBufferSpec sin_dfb_spec{
        .unique_id = SIN_DFB,
        .entry_size = sin_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = sin_cb_data_format,
    };
    DataflowBufferSpec trans_mat_dfb_spec{
        .unique_id = TRANS_MAT_DFB,
        .entry_size = trans_mat_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = trans_mat_cb_data_format,
    };
    DataflowBufferSpec rotated_in_interm_dfb_spec{
        .unique_id = ROTATED_INTERM_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec cos_interm_dfb_spec{
        .unique_id = COS_INTERM_DFB,
        .entry_size = cos_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = cos_cb_data_format,
    };
    DataflowBufferSpec sin_interm_dfb_spec{
        .unique_id = SIN_INTERM_DFB,
        .entry_size = sin_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = sin_cb_data_format,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_cb_data_format,
    };
    DataflowBufferSpec zero_dfb_spec{
        .unique_id = ZERO_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    // ----------------------------------------------------------------------------
    // Reader: reads input/cos/sin/trans_mat from DRAM (TensorAccessor) into c_0..c_3.
    // ----------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/dataflow/"
            "reader_rotary_embedding_llama_interleaved_start_id.cpp"},
        .compiler_options = {.defines = {{"RELOAD_IMPL", reload_define}}},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = COS_DFB, .accessor_name = "cos", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = SIN_DFB, .accessor_name = "sin", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = TRANS_MAT_DFB,
                 .accessor_name = "trans_mat",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = ZERO_DFB, .accessor_name = "zero", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = IN_TENSOR, .accessor_name = "in"},
             TensorBinding{.tensor_parameter_name = COS_TENSOR, .accessor_name = "cos"},
             TensorBinding{.tensor_parameter_name = SIN_TENSOR, .accessor_name = "sin"},
             TensorBinding{.tensor_parameter_name = TRANS_MAT_TENSOR, .accessor_name = "trans_mat"}},
        .compile_time_args =
            {{"n_heads", n_heads},
             {"Ht", seq_len_t},
             {"Wt", head_dim_t},
             {"freq_per_head", static_cast<uint32_t>(freq_per_head)},
             {"cos_Ht", cos_seq_len_t},
             {"sin_Ht", sin_seq_len_t},
             {"rotary_Ht", rotary_seq_len_t}},
        .runtime_arg_schema = {.runtime_arg_names = {"batch_start", "batch_end", "seq_t_start", "seq_t_end"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // ----------------------------------------------------------------------------
    // Compute: consumes c_0..c_3, produces c_16; c_24/25/26 are intra self-loops.
    // ----------------------------------------------------------------------------
    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source = std::filesystem::path{
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/"
            "rotary_embedding_llama_metal2.cpp"},
        .compiler_options = {.defines = {{"RELOAD_IMPL", reload_define}}},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = COS_DFB, .accessor_name = "cos", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = SIN_DFB, .accessor_name = "sin", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = TRANS_MAT_DFB,
                 .accessor_name = "trans_mat",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = ROTATED_INTERM_DFB,
                 .accessor_name = "rotated_in_interm",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = ROTATED_INTERM_DFB,
                 .accessor_name = "rotated_in_interm",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = COS_INTERM_DFB,
                 .accessor_name = "cos_interm",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = COS_INTERM_DFB,
                 .accessor_name = "cos_interm",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = SIN_INTERM_DFB,
                 .accessor_name = "sin_interm",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = SIN_INTERM_DFB,
                 .accessor_name = "sin_interm",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args = {{"Wt", head_dim_t}, {"n_heads", n_heads}, {"rotary_Ht", rotary_seq_len_t}},
        .runtime_arg_schema = {.runtime_arg_names = {"batch_start", "batch_end", "seq_t_start", "seq_t_end"}},
        .hw_config =
            ComputeHardwareConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
            },
    };
    compute_spec.advanced_options.dfb_self_loop_connectivities = {
        {ROTATED_INTERM_DFB, DFBSelfLoopConnectivity::INTRA},
        {COS_INTERM_DFB, DFBSelfLoopConnectivity::INTRA},
        {SIN_INTERM_DFB, DFBSelfLoopConnectivity::INTRA},
    };

    // ----------------------------------------------------------------------------
    // Writer: consumes c_16, writes to DRAM (TensorAccessor); c_27 is an intra self-loop.
    // ----------------------------------------------------------------------------
    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/dataflow/"
            "writer_rotary_embedding_llama_interleaved_start_id_metal2.cpp"},
        .compiler_options = {.defines = {{"RELOAD_IMPL", reload_define}}},
        // The 'zero' DFB holds Wt zero tiles, filled once by the reader (PRODUCER) and consumed
        // here. A data-movement kernel cannot self-loop a DFB on Gen1, so the fill lives in the
        // reader rather than the writer; the writer wait_fronts the Wt zeros once and reuses them
        // for every zero-fill tail tile (pop_front at the end), exactly as the legacy writer did.
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = ZERO_DFB, .accessor_name = "zero", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "output"}},
        .compile_time_args =
            {{"n_heads", n_heads}, {"Wt", head_dim_t}, {"Ht", seq_len_t}, {"rotary_Ht", rotary_seq_len_t}},
        .runtime_arg_schema = {.runtime_arg_names = {"batch_start", "batch_end", "seq_t_start", "seq_t_end"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    // ----------------------------------------------------------------------------
    // Per-core runtime args: {batch_start, batch_end, seq_t_start, seq_t_end}. Cores with no
    // work keep the {0,0,0,0} default (their kernel loops are empty) — mirrors the legacy factory,
    // which placed all three kernels on the full grid and skipped idle cores via these args.
    // ----------------------------------------------------------------------------
    struct CoreArgs {
        uint32_t start_batch = 0;
        uint32_t end_batch = 0;
        uint32_t start_seq = 0;
        uint32_t end_seq = 0;
    };
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);
    std::vector<CoreArgs> per_core_args(cores.size());

    for (uint32_t batch_parallel = 0; batch_parallel < batch_parallel_factor; batch_parallel++) {
        for (uint32_t seq_parallel = 0; seq_parallel < seq_parallel_factor; seq_parallel++) {
            uint32_t core_idx = (batch_parallel * seq_parallel_factor) + seq_parallel;
            uint32_t start_batch = batch_parallel * batch_per_core;
            uint32_t end_batch = std::min(start_batch + batch_per_core, batch);
            uint32_t start_seq = seq_parallel * seq_per_core;
            uint32_t end_seq = std::min(start_seq + seq_per_core, seq_len_t);

            if (start_seq >= seq_len_t || start_batch >= batch) {
                continue;
            }
            per_core_args[core_idx] = CoreArgs{start_batch, end_batch, start_seq, end_seq};
        }
    }

    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    KernelRunArgs compute_run{.kernel = COMPUTE_KERNEL};
    reader_run.runtime_arg_values.reserve(cores.size());
    writer_run.runtime_arg_values.reserve(cores.size());
    compute_run.runtime_arg_values.reserve(cores.size());
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& a = per_core_args[i];
        KernelRunArgs::RuntimeArgValues vals{
            {"batch_start", a.start_batch},
            {"batch_end", a.end_batch},
            {"seq_t_start", a.start_seq},
            {"seq_t_end", a.end_seq}};
        reader_run.runtime_arg_values.push_back({cores[i], vals});
        writer_run.runtime_arg_values.push_back({cores[i], vals});
        compute_run.runtime_arg_values.push_back({cores[i], vals});
    }

    WorkUnitSpec wu{
        .name = "rotary_embedding_llama_interleaved",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = CoreRangeSet(all_cores),
    };

    ProgramSpec spec{
        .name = "rotary_embedding_llama_interleaved",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers =
            {in_dfb_spec,
             cos_dfb_spec,
             sin_dfb_spec,
             trans_mat_dfb_spec,
             rotated_in_interm_dfb_spec,
             cos_interm_dfb_spec,
             sin_interm_dfb_spec,
             out_dfb_spec,
             zero_dfb_spec},
        .tensor_parameters = {in_param, cos_param, sin_param, trans_mat_param, out_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run, compute_run};
    run_args.tensor_args = {
        {IN_TENSOR, TensorArgument{std::cref(input.mesh_tensor())}},
        {COS_TENSOR, TensorArgument{std::cref(cos.mesh_tensor())}},
        {SIN_TENSOR, TensorArgument{std::cref(sin.mesh_tensor())}},
        {TRANS_MAT_TENSOR, TensorArgument{std::cref(trans_mat.mesh_tensor())}},
        {OUT_TENSOR, TensorArgument{std::cref(output.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
