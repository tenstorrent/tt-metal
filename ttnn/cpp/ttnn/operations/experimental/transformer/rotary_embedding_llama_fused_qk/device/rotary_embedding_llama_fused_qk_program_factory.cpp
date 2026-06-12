// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_fused_qk_device_operation_types.hpp"
#include "rotary_embedding_llama_fused_qk_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <filesystem>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts RotaryEmbeddingLlamaFusedQKProgramFactory::create_program_spec(
    const RotaryEmbeddingLlamaFusedQkParams& operation_attributes,
    const RotaryEmbeddingLlamaFusedQkInputs& tensor_args,
    RotaryEmbeddingLlamaFusedQkResult& tensor_return_value) {
    // Metal 2.0 named resource handles (declared as LOCALS for unity-build hygiene).
    const DFBSpecName Q_IN_DFB{"q_in"};
    const DFBSpecName K_IN_DFB{"k_in"};
    const DFBSpecName COS_DFB{"cos"};
    const DFBSpecName SIN_DFB{"sin"};
    const DFBSpecName TRANS_MAT_DFB{"trans_mat"};
    const DFBSpecName ROTATED_INTERM_DFB{"rotated_in_interm"};
    const DFBSpecName COS_INTERM_DFB{"cos_interm"};
    const DFBSpecName SIN_INTERM_DFB{"sin_interm"};
    const DFBSpecName Q_OUT_DFB{"q_out"};
    const DFBSpecName K_OUT_DFB{"k_out"};

    const TensorParamName Q_IN_TENSOR{"q_in_tensor"};
    const TensorParamName K_IN_TENSOR{"k_in_tensor"};
    const TensorParamName COS_TENSOR{"cos_tensor"};
    const TensorParamName SIN_TENSOR{"sin_tensor"};
    const TensorParamName TRANS_MAT_TENSOR{"trans_mat_tensor"};
    const TensorParamName Q_OUT_TENSOR{"q_out_tensor"};
    const TensorParamName K_OUT_TENSOR{"k_out_tensor"};

    const KernelSpecName COMPUTE_KERNEL{"compute"};

    const auto& q_input = tensor_args.q_input;
    const auto& k_input = tensor_args.k_input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    const auto& trans_mat = tensor_args.trans_mat;
    auto& q_output = std::get<0>(tensor_return_value);
    auto& k_output = std::get<1>(tensor_return_value);

    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(q_input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    const tt::DataFormat cos_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);

    const tt::DataFormat sin_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);

    const tt::DataFormat trans_mat_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(trans_mat.dtype());
    const uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_cb_data_format);

    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(q_output.dtype());
    const uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const std::optional<tt::tt_metal::ShardSpec>& q_shard_spec = q_input.shard_spec();
    const std::optional<tt::tt_metal::ShardSpec>& k_shard_spec = k_input.shard_spec();

    const uint32_t q_n_heads_t =
        operation_attributes.row_major_QK ? 1 : q_shard_spec->shape[0] / tt::constants::TILE_HEIGHT;
    const uint32_t k_n_heads_t =
        operation_attributes.row_major_QK ? 1 : k_shard_spec->shape[0] / tt::constants::TILE_HEIGHT;

    const uint32_t head_dim_t =
        operation_attributes.row_major_QK ? 1 : q_shard_spec->shape[1] / tt::constants::TILE_WIDTH;

    tt::tt_metal::IDevice* device = q_input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    CoreRangeSet q_cores = q_shard_spec->grid;

    CoreRangeSet k_cores = k_shard_spec->grid;

    // The compute kernel runs on exactly the cores that hold Q or K work. is_q (per core)
    // selects the branch, so every placed node must carry the RTA -> target this union
    // (q and k shard grids are disjoint: a core processes either Q or K, never both).
    CoreRangeSet all_cores = q_cores.merge(k_cores);

    const uint32_t num_q_input_tiles = q_n_heads_t * head_dim_t;
    const uint32_t num_q_output_tiles = num_q_input_tiles;

    const uint32_t num_k_input_tiles = k_n_heads_t * head_dim_t;
    const uint32_t num_k_output_tiles = num_k_input_tiles;

    // Parallelization

    const uint32_t batch_per_core = 1;  // TODO: To make general, add support for batch_per_core > 1

    const uint32_t num_sin_cos_rows_per_core = batch_per_core;
    uint32_t num_cos_sin_tiles = head_dim_t * num_sin_cos_rows_per_core;

    // We only take one tile of trans_mat
    uint32_t num_trans_mat_tiles = 1;

    uint32_t num_interm_tiles = head_dim_t;

    // ----------------------------------------------------------------------------
    // Tensor parameters. Each of the 7 backing tensors backs exactly one borrowed
    // DFB (resident, sharded L1). NONE is bound as a TensorBinding: the compute
    // kernel touches every CB by id only (FIFO ops / LLK operands), never builds a
    // TensorAccessor and never reads a base address. The borrowed DFB resolves its
    // address from the matching TensorArgument at runtime.
    // ----------------------------------------------------------------------------
    TensorParameter q_in_param{.unique_id = Q_IN_TENSOR, .spec = q_input.tensor_spec()};
    TensorParameter k_in_param{.unique_id = K_IN_TENSOR, .spec = k_input.tensor_spec()};
    TensorParameter cos_param{.unique_id = COS_TENSOR, .spec = cos.tensor_spec()};
    TensorParameter sin_param{.unique_id = SIN_TENSOR, .spec = sin.tensor_spec()};
    TensorParameter trans_mat_param{.unique_id = TRANS_MAT_TENSOR, .spec = trans_mat.tensor_spec()};
    TensorParameter q_out_param{.unique_id = Q_OUT_TENSOR, .spec = q_output.tensor_spec()};
    TensorParameter k_out_param{.unique_id = K_OUT_TENSOR, .spec = k_output.tensor_spec()};

    // ----------------------------------------------------------------------------
    // Dataflow buffers. One DFB per legacy CBDescriptor.
    //   - 7 borrowed-memory DFBs (legacy CBs with .buffer set): q_in (c_0), k_in (c_1),
    //     cos (c_2), sin (c_3), trans_mat (c_4), q_out (c_16), k_out (c_17). These are
    //     fake CBs (one-ended address sources/sinks) — see the self-loop bindings below.
    //   - 3 program-local scratch DFBs (legacy CBs with no .buffer): rotated_in_interm
    //     (c_24), cos_interm (c_25), sin_interm (c_26). Real intra-kernel FIFOs.
    // entry_size / num_entries are fixed at spec construction (no per-execution override).
    // ----------------------------------------------------------------------------
    DataflowBufferSpec q_in_dfb_spec{
        .unique_id = Q_IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_q_input_tiles,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = Q_IN_TENSOR,
    };
    DataflowBufferSpec k_in_dfb_spec{
        .unique_id = K_IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_k_input_tiles,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = K_IN_TENSOR,
    };
    DataflowBufferSpec cos_dfb_spec{
        .unique_id = COS_DFB,
        .entry_size = cos_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = cos_cb_data_format,
        .borrowed_from = COS_TENSOR,
    };
    DataflowBufferSpec sin_dfb_spec{
        .unique_id = SIN_DFB,
        .entry_size = sin_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = sin_cb_data_format,
        .borrowed_from = SIN_TENSOR,
    };
    DataflowBufferSpec trans_mat_dfb_spec{
        .unique_id = TRANS_MAT_DFB,
        .entry_size = trans_mat_single_tile_size,
        .num_entries = num_trans_mat_tiles,
        .data_format_metadata = trans_mat_cb_data_format,
        .borrowed_from = TRANS_MAT_TENSOR,
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
    DataflowBufferSpec q_out_dfb_spec{
        .unique_id = Q_OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_q_output_tiles,
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = Q_OUT_TENSOR,
    };
    DataflowBufferSpec k_out_dfb_spec{
        .unique_id = K_OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_k_output_tiles,
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = K_OUT_TENSOR,
    };

    // Host-selected compute source (identical bindings/CTAs for both).
    const std::string compute_kernel_path =
        operation_attributes.row_major_QK
            ? "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/"
              "compute/rotary_embedding_llama_sharded_row_major.cpp"
            : "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/"
              "compute/rotary_embedding_llama_sharded.cpp";

    // ----------------------------------------------------------------------------
    // Single compute KernelSpec. Every DFB is bound on this kernel as a self-loop
    // (PRODUCER + CONSUMER), because the op is compute-only and the data is resident,
    // so each DFB is one-ended from the validator's view. Each self-loop DFB is
    // declared INTRA (intra-thread) via dfb_self_loop_connectivities.
    //   - q_in/k_in/cos/sin/trans_mat/q_out/k_out: borrowed-memory FAKE-CB self-loops.
    //   - rotated_in_interm/cos_interm/sin_interm: real intra-kernel FIFO self-loops.
    // ----------------------------------------------------------------------------
    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source = std::filesystem::path{compute_kernel_path},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = Q_IN_DFB, .accessor_name = "q_in", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = Q_IN_DFB, .accessor_name = "q_in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = K_IN_DFB, .accessor_name = "k_in", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = K_IN_DFB, .accessor_name = "k_in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = COS_DFB, .accessor_name = "cos", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = COS_DFB, .accessor_name = "cos", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = SIN_DFB, .accessor_name = "sin", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = SIN_DFB, .accessor_name = "sin", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = TRANS_MAT_DFB,
                 .accessor_name = "trans_mat",
                 .endpoint_type = DFBEndpointType::PRODUCER},
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
             DFBBinding{
                 .dfb_spec_name = Q_OUT_DFB, .accessor_name = "q_out", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = Q_OUT_DFB, .accessor_name = "q_out", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = K_OUT_DFB, .accessor_name = "k_out", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = K_OUT_DFB, .accessor_name = "k_out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .compile_time_args = {{"q_Ht", q_n_heads_t}, {"k_Ht", k_n_heads_t}, {"Wt", head_dim_t}},
        .runtime_arg_schema = {.runtime_arg_names = {"is_q"}},
        .hw_config =
            ComputeHardwareConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
            },
    };
    // Compute-kernel self-loops must declare their thread connectivity. All are INTRA
    // (each TRISC thread self-loops; no cross-thread production).
    compute_spec.advanced_options.dfb_self_loop_connectivities = {
        {Q_IN_DFB, DFBSelfLoopConnectivity::INTRA},
        {K_IN_DFB, DFBSelfLoopConnectivity::INTRA},
        {COS_DFB, DFBSelfLoopConnectivity::INTRA},
        {SIN_DFB, DFBSelfLoopConnectivity::INTRA},
        {TRANS_MAT_DFB, DFBSelfLoopConnectivity::INTRA},
        {ROTATED_INTERM_DFB, DFBSelfLoopConnectivity::INTRA},
        {COS_INTERM_DFB, DFBSelfLoopConnectivity::INTRA},
        {SIN_INTERM_DFB, DFBSelfLoopConnectivity::INTRA},
        {Q_OUT_DFB, DFBSelfLoopConnectivity::INTRA},
        {K_OUT_DFB, DFBSelfLoopConnectivity::INTRA},
    };

    // ----------------------------------------------------------------------------
    // Per-core runtime arg: is_q (1 for cores in q_cores, 0 for cores in k_cores).
    // The compute kernel is placed on exactly the working cores (all_cores = q ∪ k =
    // the cos/sin shard grid); Metal 2.0 requires the named RTA on every placed node,
    // so the WorkUnit targets all_cores (NOT its bounding box, which would include
    // unused gap cores that carry no runtime args).
    // ----------------------------------------------------------------------------
    constexpr uint32_t is_q_arg = 1;  // If not q, must be k
    constexpr uint32_t is_k_arg = 0;
    const auto q_cores_vec = corerange_to_cores(q_cores, std::nullopt, /*row_wise=*/true);
    const auto k_cores_vec = corerange_to_cores(k_cores, std::nullopt, /*row_wise=*/true);

    KernelRunArgs compute_run{.kernel = COMPUTE_KERNEL};
    compute_run.runtime_arg_values.reserve(q_cores_vec.size() + k_cores_vec.size());
    for (const auto& core : q_cores_vec) {
        compute_run.runtime_arg_values.push_back({core, KernelRunArgs::RuntimeArgValues{{"is_q", is_q_arg}}});
    }
    for (const auto& core : k_cores_vec) {
        compute_run.runtime_arg_values.push_back({core, KernelRunArgs::RuntimeArgValues{{"is_q", is_k_arg}}});
    }

    WorkUnitSpec wu{
        .name = "rotary_embedding_llama_fused_qk",
        .kernels = {COMPUTE_KERNEL},
        .target_nodes = all_cores,
    };

    ProgramSpec spec{
        .name = "rotary_embedding_llama_fused_qk",
        .kernels = {compute_spec},
        .dataflow_buffers =
            {q_in_dfb_spec,
             k_in_dfb_spec,
             cos_dfb_spec,
             sin_dfb_spec,
             trans_mat_dfb_spec,
             rotated_in_interm_dfb_spec,
             cos_interm_dfb_spec,
             sin_interm_dfb_spec,
             q_out_dfb_spec,
             k_out_dfb_spec},
        .tensor_parameters = {q_in_param, k_in_param, cos_param, sin_param, trans_mat_param, q_out_param, k_out_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {compute_run};
    run_args.tensor_args = {
        {Q_IN_TENSOR, TensorArgument{std::cref(q_input.mesh_tensor())}},
        {K_IN_TENSOR, TensorArgument{std::cref(k_input.mesh_tensor())}},
        {COS_TENSOR, TensorArgument{std::cref(cos.mesh_tensor())}},
        {SIN_TENSOR, TensorArgument{std::cref(sin.mesh_tensor())}},
        {TRANS_MAT_TENSOR, TensorArgument{std::cref(trans_mat.mesh_tensor())}},
        {Q_OUT_TENSOR, TensorArgument{std::cref(q_output.mesh_tensor())}},
        {K_OUT_TENSOR, TensorArgument{std::cref(k_output.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
