// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_fused_qk_device_operation_types.hpp"
#include "rotary_embedding_llama_fused_qk_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts RotaryEmbeddingLlamaFusedQKProgramFactory::create_program_artifacts(
    const RotaryEmbeddingLlamaFusedQkParams& operation_attributes,
    const RotaryEmbeddingLlamaFusedQkInputs& tensor_args,
    RotaryEmbeddingLlamaFusedQkResult& tensor_return_value) {
    // Named-resource vocabulary. Function-local on purpose: ttnn unity builds can concatenate
    // factory .cpp files into one TU, where same-named anonymous-namespace constants collide.
    // The names extend the sibling rotary_embedding_llama op's landed Metal 2.0 vocabulary
    // (input/cos/sin/trans_mat/rotated_interm/cos_interm/sin_interm/out) for the fused q/k split.
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName Q_INPUT_DFB{"q_input"};
    const DFBSpecName K_INPUT_DFB{"k_input"};
    const DFBSpecName COS_DFB{"cos"};
    const DFBSpecName SIN_DFB{"sin"};
    const DFBSpecName TRANS_MAT_DFB{"trans_mat"};
    const DFBSpecName ROTATED_INTERM_DFB{"rotated_interm"};
    const DFBSpecName COS_INTERM_DFB{"cos_interm"};
    const DFBSpecName SIN_INTERM_DFB{"sin_interm"};
    const DFBSpecName Q_OUT_DFB{"q_out"};
    const DFBSpecName K_OUT_DFB{"k_out"};
    const TensorParamName Q_INPUT_PARAM{"q_input"};
    const TensorParamName K_INPUT_PARAM{"k_input"};
    const TensorParamName COS_PARAM{"cos"};
    const TensorParamName SIN_PARAM{"sin"};
    const TensorParamName TRANS_MAT_PARAM{"trans_mat"};
    const TensorParamName Q_OUTPUT_PARAM{"q_output"};
    const TensorParamName K_OUTPUT_PARAM{"k_output"};

    const auto& q_input = tensor_args.q_input.mesh_tensor();
    const auto& k_input = tensor_args.k_input.mesh_tensor();
    const auto& cos = tensor_args.cos.mesh_tensor();
    const auto& sin = tensor_args.sin.mesh_tensor();
    const auto& trans_mat = tensor_args.trans_mat.mesh_tensor();
    const auto& q_output = std::get<0>(tensor_return_value).mesh_tensor();
    const auto& k_output = std::get<1>(tensor_return_value).mesh_tensor();

    const tt::DataFormat input_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(q_input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_dfb_data_format);

    const tt::DataFormat cos_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_single_tile_size = tt::tile_size(cos_dfb_data_format);

    const tt::DataFormat sin_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_single_tile_size = tt::tile_size(sin_dfb_data_format);

    const tt::DataFormat trans_mat_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(trans_mat.dtype());
    const uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_dfb_data_format);

    const tt::DataFormat output_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(q_output.dtype());
    const uint32_t output_single_tile_size = tt::tile_size(output_dfb_data_format);

    const std::optional<tt::tt_metal::ShardSpec>& q_shard_spec = q_input.shard_spec();
    const std::optional<tt::tt_metal::ShardSpec>& k_shard_spec = k_input.shard_spec();

    const uint32_t q_n_heads_t =
        operation_attributes.row_major_QK ? 1 : q_shard_spec->shape[0] / tt::constants::TILE_HEIGHT;
    const uint32_t k_n_heads_t =
        operation_attributes.row_major_QK ? 1 : k_shard_spec->shape[0] / tt::constants::TILE_HEIGHT;

    const uint32_t head_dim_t =
        operation_attributes.row_major_QK ? 1 : q_shard_spec->shape[1] / tt::constants::TILE_WIDTH;

    tt::tt_metal::IDevice* device = tensor_args.q_input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    CoreRangeSet q_cores = q_shard_spec->grid;

    CoreRangeSet k_cores = k_shard_spec->grid;

    // The compute kernel must only run on cores that actually receive per-core (unique) runtime args
    // (the is_q/is_k flag below). Using the bounding box of the cos/sin grid would place the
    // kernel on "hole" cores that are inside the bounding box but belong to neither q nor k. Those
    // cores have zero runtime args set, so reading the is_q runtime arg goes out of bounds and trips
    // the watcher assert (SIGABRT). q and k grids are guaranteed non-overlapping by validate().
    CoreRangeSet work_cores = q_cores.merge(k_cores);

    const uint32_t num_q_input_tiles = q_n_heads_t * head_dim_t;
    const uint32_t num_q_output_tiles = num_q_input_tiles;

    const uint32_t num_k_input_tiles = k_n_heads_t * head_dim_t;
    const uint32_t num_k_output_tiles = num_k_input_tiles;

    // Parallelization

    const uint32_t batch_per_core = 1;  // TODO: To make general, add support for batch_per_core > 1

    const uint32_t num_sin_cos_rows_per_core = batch_per_core;
    uint32_t num_cos_sin_tiles = head_dim_t * num_sin_cos_rows_per_core;

    // ------------------------------------------------------------------
    // Dataflow buffers. The q/k inputs, cos/sin/trans_mat, and q/k outputs borrow their resident
    // L1 shards (the legacy dynamic-address `.buffer` backing); the three intermediates are plain. The lone
    // compute kernel is the sole toucher of every DFB -> self-loop each (PRODUCER + CONSUMER).
    // Placement is derived from the kernel bindings, so the DFBs live on work_cores.
    // ------------------------------------------------------------------
    DataflowBufferSpec q_input_dfb{
        .unique_id = Q_INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_q_input_tiles,
        .data_format_metadata = input_dfb_data_format,
        .borrowed_from = Q_INPUT_PARAM,
    };
    DataflowBufferSpec k_input_dfb{
        .unique_id = K_INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_k_input_tiles,
        .data_format_metadata = input_dfb_data_format,
        .borrowed_from = K_INPUT_PARAM,
    };
    DataflowBufferSpec cos_dfb{
        .unique_id = COS_DFB,
        .entry_size = cos_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = cos_dfb_data_format,
        .borrowed_from = COS_PARAM,
    };
    DataflowBufferSpec sin_dfb{
        .unique_id = SIN_DFB,
        .entry_size = sin_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = sin_dfb_data_format,
        .borrowed_from = SIN_PARAM,
    };
    // We only take one tile of trans_mat
    uint32_t num_trans_mat_tiles = 1;
    DataflowBufferSpec trans_mat_dfb{
        .unique_id = TRANS_MAT_DFB,
        .entry_size = trans_mat_single_tile_size,
        .num_entries = num_trans_mat_tiles,
        .data_format_metadata = trans_mat_dfb_data_format,
        .borrowed_from = TRANS_MAT_PARAM,
    };
    uint32_t num_interm_tiles = head_dim_t;
    DataflowBufferSpec rotated_interm_dfb{
        .unique_id = ROTATED_INTERM_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = input_dfb_data_format,
    };
    DataflowBufferSpec cos_interm_dfb{
        .unique_id = COS_INTERM_DFB,
        .entry_size = cos_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = cos_dfb_data_format,
    };
    DataflowBufferSpec sin_interm_dfb{
        .unique_id = SIN_INTERM_DFB,
        .entry_size = sin_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = sin_dfb_data_format,
    };
    DataflowBufferSpec q_out_dfb{
        .unique_id = Q_OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_q_output_tiles,
        .data_format_metadata = output_dfb_data_format,
        .borrowed_from = Q_OUTPUT_PARAM,
    };
    DataflowBufferSpec k_out_dfb{
        .unique_id = K_OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_k_output_tiles,
        .data_format_metadata = output_dfb_data_format,
        .borrowed_from = K_OUTPUT_PARAM,
    };

    // ------------------------------------------------------------------
    // Tensor parameters. Referenced via the DFB borrowed_from links (no kernel TensorBinding —
    // a compute kernel cannot bind a TensorAccessor). The spec validator accepts borrowed_from as
    // the required reference.
    // ------------------------------------------------------------------
    TensorParameter q_input_param{.unique_id = Q_INPUT_PARAM, .spec = q_input.tensor_spec()};
    TensorParameter k_input_param{.unique_id = K_INPUT_PARAM, .spec = k_input.tensor_spec()};
    TensorParameter cos_param{.unique_id = COS_PARAM, .spec = cos.tensor_spec()};
    TensorParameter sin_param{.unique_id = SIN_PARAM, .spec = sin.tensor_spec()};
    TensorParameter trans_mat_param{.unique_id = TRANS_MAT_PARAM, .spec = trans_mat.tensor_spec()};
    TensorParameter q_output_param{.unique_id = Q_OUTPUT_PARAM, .spec = q_output.tensor_spec()};
    TensorParameter k_output_param{.unique_id = K_OUTPUT_PARAM, .spec = k_output.tensor_spec()};

    // Set up the kernel
    const std::filesystem::path compute_kernel_path =
        operation_attributes.row_major_QK
            ? "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/"
              "compute/rotary_embedding_llama_sharded_row_major.cpp"
            : "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/"
              "compute/rotary_embedding_llama_sharded.cpp";

    // hw_config mirrors the legacy ComputeConfigDescriptor subset: the legacy factory resolved the
    // full TTNN compute config but copied only math_fidelity and fp32_dest_acc_en onto the
    // descriptor, leaving math_approx_mode / dst_full_sync_en at descriptor defaults. Those
    // defaults coincide with ComputeGen1Config's (sfpu_precision_mode = Precise,
    // double_buffer_dest = true), so only the two copied fields are set here. No unpack_modes
    // entries: every DFB is bfloat16 (validate() forces BFLOAT16 tensors), so the Float32
    // required-entry rule never triggers even when enable_32_bit_dest is on.
    const ComputeHardwareConfig compute_hw_config =
        ComputeGen1Config{.fpu_math_fidelity = math_fidelity, .enable_32_bit_dest = fp32_dest_acc_en};

    auto self_loop = [](const DFBSpecName& dfb, const std::string& name) {
        return Group<DFBBinding>{
            DFBBinding{
                .dfb_spec_name = dfb,
                .accessor_name = name,
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = dfb,
                .accessor_name = name,
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
        };
    };
    Group<DFBBinding> compute_bindings;
    for (const auto& [dfb, name] : {
             std::pair{Q_INPUT_DFB, std::string{"q_input"}},
             std::pair{K_INPUT_DFB, std::string{"k_input"}},
             std::pair{COS_DFB, std::string{"cos"}},
             std::pair{SIN_DFB, std::string{"sin"}},
             std::pair{TRANS_MAT_DFB, std::string{"trans_mat"}},
             std::pair{ROTATED_INTERM_DFB, std::string{"rotated_interm"}},
             std::pair{COS_INTERM_DFB, std::string{"cos_interm"}},
             std::pair{SIN_INTERM_DFB, std::string{"sin_interm"}},
             std::pair{Q_OUT_DFB, std::string{"q_out"}},
             std::pair{K_OUT_DFB, std::string{"k_out"}},
         }) {
        auto pair = self_loop(dfb, name);
        compute_bindings.push_back(pair[0]);
        compute_bindings.push_back(pair[1]);
    }

    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = compute_kernel_path,
        // Explicit O3: the legacy compute kernel resolved to O3 (ComputeConfigDescriptor default),
        // while Metal 2.0's CompilerOptions defaults to O2. Both kernel variants sit within ~4 bytes
        // of the TRISC2 code-size limit with the profiler on, so the build flags genuinely matter.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = compute_bindings,
        .compile_time_args =
            {
                {"q_Ht", q_n_heads_t},
                {"k_Ht", k_n_heads_t},
                {"Wt", head_dim_t},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"is_q"}},
        .hw_config = compute_hw_config,
    };

    ProgramSpec spec{
        .name = "rotary_embedding_llama_fused_qk",
        .kernels = {compute_spec},
        .dataflow_buffers =
            {q_input_dfb,
             k_input_dfb,
             cos_dfb,
             sin_dfb,
             trans_mat_dfb,
             rotated_interm_dfb,
             cos_interm_dfb,
             sin_interm_dfb,
             q_out_dfb,
             k_out_dfb},
        .tensor_parameters =
            {q_input_param, k_input_param, cos_param, sin_param, trans_mat_param, q_output_param, k_output_param},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {COMPUTE}, .target_nodes = work_cores}},
    };

    // Runtime args to differentiate between q, k or no work groups
    // TODO: Turn off unused compute cores? (technically, it doesn't matter since only compute kernel)
    // Running into code size issues on TRISC2 with profiler turned on; need to reduce stack size by 4B
    // constexpr bool has_work = true;
    constexpr uint32_t is_q_arg = 1;  // If not q, must be k
    constexpr uint32_t is_k_arg = 0;
    const auto q_cores_vec = corerange_to_cores(q_cores, std::nullopt, /*row_wise=*/true);
    const auto k_cores_vec = corerange_to_cores(k_cores, std::nullopt, /*row_wise=*/true);
    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (const auto& core : q_cores_vec) {
        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"is_q", is_q_arg}});
    }
    for (const auto& core : k_cores_vec) {
        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"is_q", is_k_arg}});
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(compute_run_args)};
    // Borrowed DFBs draw their backing L1 addresses from these tensor args.
    run_args.tensor_args = {
        {Q_INPUT_PARAM, TensorArgument{q_input}},
        {K_INPUT_PARAM, TensorArgument{k_input}},
        {COS_PARAM, TensorArgument{cos}},
        {SIN_PARAM, TensorArgument{sin}},
        {TRANS_MAT_PARAM, TensorArgument{trans_mat}},
        {Q_OUTPUT_PARAM, TensorArgument{q_output}},
        {K_OUTPUT_PARAM, TensorArgument{k_output}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
