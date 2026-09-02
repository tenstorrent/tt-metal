// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_multi_core_program_factory.hpp"
#include "rotary_embedding_llama_metal2_common.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using namespace ttnn::experimental::prim::rope_metal2;

ttnn::device_operation::ProgramArtifacts RotaryEmbeddingLlamaMultiCore::create_program_artifacts(
    const RotaryEmbeddingLlamaParams& operation_attributes,
    const RotaryEmbeddingLlamaInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    const auto& input = tensor_args.input_tensor.mesh_tensor();
    const auto& cos = tensor_args.cos_cache.mesh_tensor();
    const auto& sin = tensor_args.sin_cache.mesh_tensor();
    const auto& trans_mat = tensor_args.trans_mat.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();

    const tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    const tt::DataFormat cos_cb_data_format = tt_metal::datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);

    const tt::DataFormat sin_cb_data_format = tt_metal::datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);

    const tt::DataFormat trans_mat_cb_data_format = tt_metal::datatype_to_dataformat_converter(trans_mat.dtype());
    const uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_cb_data_format);

    const tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
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

    tt_metal::IDevice* device = tensor_args.input_tensor.device();

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

    // ------------------------------------------------------------------
    // Dataflow buffers (one per legacy CB; all plain — no borrowed memory in the interleaved case).
    // entry_size == legacy page_size; num_entries == legacy tile count.
    // ------------------------------------------------------------------
    DataflowBufferSpec input_dfb{
        .unique_id = INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = input_cb_num_tiles,
        .data_format_metadata = input_cb_data_format};
    DataflowBufferSpec cos_dfb{
        .unique_id = COS_DFB,
        .entry_size = cos_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = cos_cb_data_format};
    DataflowBufferSpec sin_dfb{
        .unique_id = SIN_DFB,
        .entry_size = sin_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = sin_cb_data_format};
    DataflowBufferSpec trans_mat_dfb{
        .unique_id = TRANS_MAT_DFB,
        .entry_size = trans_mat_single_tile_size,
        .num_entries = 1,  // We only take one tile of trans_mat
        .data_format_metadata = trans_mat_cb_data_format};
    DataflowBufferSpec rotated_interm_dfb{
        .unique_id = ROTATED_INTERM_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = input_cb_data_format};
    DataflowBufferSpec cos_interm_dfb{
        .unique_id = COS_INTERM_DFB,
        .entry_size = cos_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = cos_cb_data_format};
    DataflowBufferSpec sin_interm_dfb{
        .unique_id = SIN_INTERM_DFB,
        .entry_size = sin_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = sin_cb_data_format};
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_cb_data_format};
    // Zero-fill staging region for the writer (legacy c_27). Ported first as a self-looped DFB
    // (writer = PRODUCER + CONSUMER), a shape Gen2 rejects on DM kernels; now a Scratchpad
    // (dm_self_loop_dfbs.md). The writer only ever used it as raw local memory (fill once, read
    // repeatedly from the base), so no FIFO semantics are lost. data_format_metadata had no
    // consumer: the writer takes raw addresses / NOC-sources it, and no LLK touches it.
    ScratchpadSpec zero_scratchpad{
        .unique_id = ZERO_SCRATCH, .size_per_node = output_single_tile_size * num_interm_tiles};

    // ------------------------------------------------------------------
    // Tensor parameters (all Case 1 — accessor-read; legacy Buffer* RTAs + TensorAccessorArgs collapse).
    // ------------------------------------------------------------------
    TensorParameter input_param{.unique_id = INPUT_PARAM, .spec = input.tensor_spec()};
    TensorParameter cos_param{.unique_id = COS_PARAM, .spec = cos.tensor_spec()};
    TensorParameter sin_param{.unique_id = SIN_PARAM, .spec = sin.tensor_spec()};
    TensorParameter trans_mat_param{.unique_id = TRANS_MAT_PARAM, .spec = trans_mat.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_PARAM, .spec = output.tensor_spec()};

    // ------------------------------------------------------------------
    // hw_config. Style B (build ComputeGen1Config directly): the legacy ComputeConfigDescriptor set
    // only math_fidelity + fp32_dest_acc_en, leaving the rest at descriptor defaults. Routing through
    // to_compute_hardware_config would instead translate the *resolved* math_approx_mode (default true)
    // into sfpu_precision_mode=Approximate, which the legacy descriptor discarded (Precise). All DFBs
    // are bfloat16, so no unpack_modes entry is required even when enable_32_bit_dest is true.
    // ------------------------------------------------------------------
    ComputeHardwareConfig compute_hw_config =
        ComputeGen1Config{.fpu_math_fidelity = math_fidelity, .enable_32_bit_dest = fp32_dest_acc_en};
    if (device->arch() == tt::ARCH::QUASAR) {
        // Gen2 copies the fields the Gen1 config sets (gen2_hardware_configs.md shape 4).
        // TODO(#52269): Quasar unpack_modes are copied from Gen1 and not yet optimized for Quasar.
        compute_hw_config = ComputeGen2Config{
            .fpu_math_fidelity = math_fidelity,
            .enable_32_bit_dest = fp32_dest_acc_en,
        };
    }

    const KernelSpec::CompilerOptions::Defines reload_define{{"RELOAD_IMPL", use_reload_impl ? "1" : "0"}};

    // ------------------------------------------------------------------
    // Kernels
    // ------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER,
        .source = kReaderInterleavedSource,
        .compiler_options = {.defines = reload_define},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = COS_DFB, .accessor_name = "cos", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = SIN_DFB, .accessor_name = "sin", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = TRANS_MAT_DFB,
                 .accessor_name = "trans_mat",
                 .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = INPUT_PARAM, .accessor_name = "input"},
             TensorBinding{.tensor_parameter_name = COS_PARAM, .accessor_name = "cos"},
             TensorBinding{.tensor_parameter_name = SIN_PARAM, .accessor_name = "sin"},
             TensorBinding{.tensor_parameter_name = TRANS_MAT_PARAM, .accessor_name = "trans_mat"}},
        .compile_time_args =
            {{"n_heads", n_heads},
             {"Ht", seq_len_t},
             {"Wt", head_dim_t},
             {"freq_per_head", static_cast<uint32_t>(freq_per_head)},
             {"cos_Ht", cos_seq_len_t},
             {"sin_Ht", sin_seq_len_t},
             {"rotary_Ht", rotary_seq_len_t}},
        .runtime_arg_schema = {.runtime_arg_names = {"batch_start", "batch_end", "seq_t_start", "seq_t_end"}},
        .hw_config = create_reader_datamovement_config(device->arch())};

    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = kWriterSource,
        .compiler_options = {.defines = reload_define},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        // ZERO is a single-toucher (writer fills + reads it): a Scratchpad, not a DFB (see above).
        .scratchpad_bindings = {ScratchpadBinding{.scratchpad_spec_name = ZERO_SCRATCH, .accessor_name = "zero"}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_PARAM, .accessor_name = "output"}},
        .compile_time_args =
            {{"n_heads", n_heads}, {"Wt", head_dim_t}, {"Ht", seq_len_t}, {"rotary_Ht", rotary_seq_len_t}},
        .runtime_arg_schema = {.runtime_arg_names = {"batch_start", "batch_end", "seq_t_start", "seq_t_end"}},
        .hw_config = create_writer_datamovement_config(device->arch())};

    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = kComputeSource,
        .compiler_options = {.defines = reload_define, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = COS_DFB, .accessor_name = "cos", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = SIN_DFB, .accessor_name = "sin", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = TRANS_MAT_DFB,
                 .accessor_name = "trans_mat",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
             // Intermediate CBs: compute is the sole toucher → self-loop (PRODUCER + CONSUMER).
             DFBBinding{
                 .dfb_spec_name = ROTATED_INTERM_DFB,
                 .accessor_name = "rotated_interm",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = ROTATED_INTERM_DFB,
                 .accessor_name = "rotated_interm",
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
                 .endpoint_type = DFBEndpointType::CONSUMER}},
        .compile_time_args = {{"Wt", head_dim_t}, {"n_heads", n_heads}, {"rotary_Ht", rotary_seq_len_t}},
        .runtime_arg_schema = {.runtime_arg_names = {"batch_start", "batch_end", "seq_t_start", "seq_t_end"}},
        .hw_config = compute_hw_config};

    // ------------------------------------------------------------------
    // Per-node runtime args (batch×seq parallelization; idle cores zero-filled exactly as legacy).
    // ------------------------------------------------------------------
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    struct CoreArgs {
        uint32_t start_batch = 0;
        uint32_t end_batch = 0;
        uint32_t start_seq = 0;
        uint32_t end_seq = 0;
    };
    std::vector<CoreArgs> per_core_args(cores.size());

    for (uint32_t batch_parallel = 0; batch_parallel < batch_parallel_factor; batch_parallel++) {
        for (uint32_t seq_parallel = 0; seq_parallel < seq_parallel_factor; seq_parallel++) {
            uint32_t core_idx = (batch_parallel * seq_parallel_factor) + seq_parallel;
            uint32_t start_batch = batch_parallel * batch_per_core;
            uint32_t end_batch = std::min(start_batch + batch_per_core, batch);
            uint32_t start_seq = seq_parallel * seq_per_core;
            uint32_t end_seq = std::min(start_seq + seq_per_core, seq_len_t);

            if (start_seq >= seq_len_t || start_batch >= batch) {
                // Important to skip cores which have no work to do, otherwise they will wait
                // on cos/sin data which will never arrive.
                continue;
            }
            per_core_args[core_idx] = CoreArgs{start_batch, end_batch, start_seq, end_seq};
        }
    }

    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};
    KernelRunArgs compute_run{.kernel = COMPUTE};
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& a = per_core_args[i];
        const NodeCoord node = cores[i];
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            node,
            {{"batch_start", a.start_batch},
             {"batch_end", a.end_batch},
             {"seq_t_start", a.start_seq},
             {"seq_t_end", a.end_seq}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            node,
            {{"batch_start", a.start_batch},
             {"batch_end", a.end_batch},
             {"seq_t_start", a.start_seq},
             {"seq_t_end", a.end_seq}});
        AddRuntimeArgsForNode(
            compute_run.runtime_arg_values,
            node,
            {{"batch_start", a.start_batch},
             {"batch_end", a.end_batch},
             {"seq_t_start", a.start_seq},
             {"seq_t_end", a.end_seq}});
    }

    // ------------------------------------------------------------------
    // Assemble spec + run-args.
    // ------------------------------------------------------------------
    ProgramSpec spec{
        .name = "rotary_embedding_llama_multi_core",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers =
            {input_dfb, cos_dfb, sin_dfb, trans_mat_dfb, rotated_interm_dfb, cos_interm_dfb, sin_interm_dfb, out_dfb},
        .scratchpads = {zero_scratchpad},
        .tensor_parameters = {input_param, cos_param, sin_param, trans_mat_param, output_param},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}}};

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run, compute_run};
    run_args.tensor_args = {
        {INPUT_PARAM, TensorArgument{input}},
        {COS_PARAM, TensorArgument{cos}},
        {SIN_PARAM, TensorArgument{sin}},
        {TRANS_MAT_PARAM, TensorArgument{trans_mat}},
        {OUTPUT_PARAM, TensorArgument{output}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
