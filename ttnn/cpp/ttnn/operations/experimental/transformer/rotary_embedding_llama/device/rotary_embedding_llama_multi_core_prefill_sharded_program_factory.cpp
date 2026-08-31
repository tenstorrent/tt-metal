// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_multi_core_prefill_sharded_program_factory.hpp"
#include "rotary_embedding_llama_metal2_common.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <vector>

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using namespace ttnn::experimental::prim::rope_metal2;

ttnn::device_operation::ProgramArtifacts RotaryEmbeddingLlamaMultiCorePrefillSharded::create_program_artifacts(
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
            "rotary_embedding_llama prefill sequence tile coverage mismatch: input_Ht={}, cos_Ht={}, sin_Ht={}, "
            "rotary_Ht={}. Tiles beyond rotary_Ht will be zero-filled in the output.",
            seq_len_t,
            cos_seq_len_t,
            sin_seq_len_t,
            rotary_seq_len_t);
    }

    // Flag for whether or not sin/cos vary per head. If false, they will be broadcasted across heads.
    const bool freq_per_head = cos.padded_shape()[1] == n_heads;

    // Whether cos/sin and trans_mat are pre-loaded into per-core L1 shards.
    const bool cos_sin_sharded = cos.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    const bool trans_mat_sharded = trans_mat.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;

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

    // Reload implementation is always used when cos/sin are HEIGHT_SHARDED
    // (since the CB is globally-allocated from the shard).
    const bool use_reload_impl = num_rows_per_core > 8 || freq_per_head || cos_sin_sharded;
    if (use_reload_impl) {
        // Only size CBs to double buffer head_dim_t tiles for all inputs
        input_cb_num_tiles = num_input_tiles;
        num_cos_sin_tiles = num_input_tiles;
    }
    // Borrowed cos/sin (a globally-allocated L1 view of the resident shard) is only expressible in
    // Metal 2.0 when the shard grid covers ALL cores. A DFB has a single `borrowed_from` and its
    // placement is derived from the union of its bound kernels' nodes, so — unlike the legacy CB —
    // it cannot present a per-node borrowed/plain split (borrowed on the shard grid, plain on the
    // remaining cores). When the shard is partial we therefore fall back to the reload path (read
    // each row via TensorAccessor), which is layout-agnostic and runs on all_cores exactly as the
    // interleaved path does. This keeps the port a faithful all_cores placement; the only observable
    // difference from legacy is that a partial-shard config legacy would have served from the fast L1
    // view now takes the (output-identical) reload path.
    const bool cos_sin_shard_full = cos_sin_sharded && cos.shard_spec()->grid.num_cores() == num_cores;
    const bool cos_sin_sharded_reload =
        cos_sin_sharded && (seq_per_core > 1 || !cos_sin_shard_full || seq_len_t > cos_seq_len_t);
    if (cos_sin_sharded) {
        num_cos_sin_tiles = cos_sin_sharded_reload ? num_input_tiles : head_dim_t;
    }
    // Globally-allocated (borrowed) CB for trans_mat likewise requires the shard grid to cover all
    // cores; otherwise fall back to TensorAccessor reads (the non-global-cb kernel path).
    const bool trans_mat_use_global_cb = trans_mat_sharded && trans_mat.shard_spec()->grid.num_cores() == num_cores;

    const uint32_t num_interm_tiles = head_dim_t;

    // Borrowed-memory selection. Borrow only in the full-shard fast path (above), where the borrowed
    // DFB is placed on all_cores and every node has its own shard to back it.
    const bool cos_sin_borrowed = cos_sin_sharded && !cos_sin_sharded_reload;
    const bool cos_sin_accessor = !cos_sin_borrowed;  // reader reads cos/sin via TensorAccessor
    const bool trans_mat_borrowed = trans_mat_use_global_cb;
    const bool trans_mat_accessor = !trans_mat_borrowed;

    // ------------------------------------------------------------------
    // Dataflow buffers.
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
        .data_format_metadata = cos_cb_data_format,
        .borrowed_from = cos_sin_borrowed ? std::optional<TensorParamName>{COS_PARAM} : std::nullopt};
    DataflowBufferSpec sin_dfb{
        .unique_id = SIN_DFB,
        .entry_size = sin_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = sin_cb_data_format,
        .borrowed_from = cos_sin_borrowed ? std::optional<TensorParamName>{SIN_PARAM} : std::nullopt};
    DataflowBufferSpec trans_mat_dfb{
        .unique_id = TRANS_MAT_DFB,
        .entry_size = trans_mat_single_tile_size,
        .num_entries = 1,  // We only take one tile of trans_mat
        .data_format_metadata = trans_mat_cb_data_format,
        .borrowed_from = trans_mat_borrowed ? std::optional<TensorParamName>{TRANS_MAT_PARAM} : std::nullopt};
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
    DataflowBufferSpec zero_dfb{
        .unique_id = ZERO_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = output_cb_data_format};

    // ------------------------------------------------------------------
    // Tensor parameters. INPUT/OUTPUT always accessor-read. COS/SIN and TRANS_MAT are borrowed_from
    // in the L1-resident configs and accessor-read otherwise — either way each is referenced.
    // ------------------------------------------------------------------
    TensorParameter input_param{.unique_id = INPUT_PARAM, .spec = input.tensor_spec()};
    TensorParameter cos_param{.unique_id = COS_PARAM, .spec = cos.tensor_spec()};
    TensorParameter sin_param{.unique_id = SIN_PARAM, .spec = sin.tensor_spec()};
    TensorParameter trans_mat_param{.unique_id = TRANS_MAT_PARAM, .spec = trans_mat.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_PARAM, .spec = output.tensor_spec()};

    // hw_config — Style B (see the interleaved factory for the rationale).
    const ComputeHardwareConfig compute_hw_config{
        .fpu_math_fidelity = math_fidelity, .enable_32_bit_dest = fp32_dest_acc_en};

    // ------------------------------------------------------------------
    // Reader kernel. cos_sin_sharded / trans_mat_use_global_cb move from CTAs to preprocessor defines
    // so the conditional cos/sin/trans_mat TensorAccessor references parse away when unbound.
    // ------------------------------------------------------------------
    KernelSpec::CompilerOptions::Defines reader_defines;
    reader_defines.insert({"RELOAD_IMPL", use_reload_impl ? "1" : "0"});
    reader_defines.insert({"COS_SIN_SHARDED_RELOAD", cos_sin_sharded_reload ? "1" : "0"});
    if (cos_sin_sharded) {
        reader_defines.insert({"COS_SIN_SHARDED", "1"});
    }
    if (trans_mat_use_global_cb) {
        reader_defines.insert({"TRANS_MAT_USE_GLOBAL_CB", "1"});
    }

    Group<TensorBinding> reader_tensor_bindings{
        TensorBinding{.tensor_parameter_name = INPUT_PARAM, .accessor_name = "input"}};
    if (cos_sin_accessor) {
        reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = COS_PARAM, .accessor_name = "cos"});
        reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = SIN_PARAM, .accessor_name = "sin"});
    }
    if (trans_mat_accessor) {
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = TRANS_MAT_PARAM, .accessor_name = "trans_mat"});
    }

    KernelSpec reader_spec{
        .unique_id = READER,
        .source = kReaderPrefillShardedSource,
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = COS_DFB, .accessor_name = "cos", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = SIN_DFB, .accessor_name = "sin", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = TRANS_MAT_DFB,
                 .accessor_name = "trans_mat",
                 .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = reader_tensor_bindings,
        .compile_time_args =
            {{"n_heads", n_heads},
             {"Ht", seq_len_t},
             {"Wt", head_dim_t},
             {"freq_per_head", static_cast<uint32_t>(freq_per_head)},
             {"cos_Ht", cos_seq_len_t},
             {"sin_Ht", sin_seq_len_t},
             {"rotary_Ht", rotary_seq_len_t}},
        .runtime_arg_schema = {.runtime_arg_names = {"batch_start", "batch_end", "seq_t_start", "seq_t_end"}},
        .hw_config = create_reader_datamovement_config()};

    // Writer / compute — identical to the interleaved factory (shared kernel sources).
    const KernelSpec::CompilerOptions::Defines reload_define{{"RELOAD_IMPL", use_reload_impl ? "1" : "0"}};

    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = kWriterSource,
        .compiler_options = {.defines = reload_define},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = ZERO_DFB, .accessor_name = "zero", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = ZERO_DFB, .accessor_name = "zero", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_PARAM, .accessor_name = "output"}},
        .compile_time_args =
            {{"n_heads", n_heads}, {"Wt", head_dim_t}, {"Ht", seq_len_t}, {"rotary_Ht", rotary_seq_len_t}},
        .runtime_arg_schema = {.runtime_arg_names = {"batch_start", "batch_end", "seq_t_start", "seq_t_end"}},
        .hw_config = create_writer_datamovement_config()};

    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = kComputeSource,
        .compiler_options = {.defines = reload_define},
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
    // Placement is all_cores (faithful to legacy): borrowed DFBs are used only on the full-shard fast
    // path, where every core has its own shard to back the borrow, so no placement narrowing is needed.
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

    ProgramSpec spec{
        .name = "rotary_embedding_llama_multi_core_prefill_sharded",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers =
            {input_dfb,
             cos_dfb,
             sin_dfb,
             trans_mat_dfb,
             rotated_interm_dfb,
             cos_interm_dfb,
             sin_interm_dfb,
             out_dfb,
             zero_dfb},
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
