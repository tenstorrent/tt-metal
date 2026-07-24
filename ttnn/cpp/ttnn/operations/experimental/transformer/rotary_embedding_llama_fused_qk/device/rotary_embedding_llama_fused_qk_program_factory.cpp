// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_fused_qk_device_operation_types.hpp"
#include "rotary_embedding_llama_fused_qk_program_factory.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts RotaryEmbeddingLlamaFusedQKProgramFactory::create_program_artifacts(
    const RotaryEmbeddingLlamaFusedQkParams& operation_attributes,
    const RotaryEmbeddingLlamaFusedQkInputs& tensor_args,
    RotaryEmbeddingLlamaFusedQkResult& tensor_return_value) {
    const auto& q_input = tensor_args.q_input;
    const auto& k_input = tensor_args.k_input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    const auto& trans_mat = tensor_args.trans_mat;
    auto& q_output = std::get<0>(tensor_return_value);
    auto& k_output = std::get<1>(tensor_return_value);

    // MeshTensors backing the borrowed DFBs / tensor parameters.
    const auto& q_input_mt = q_input.mesh_tensor();
    const auto& k_input_mt = k_input.mesh_tensor();
    const auto& cos_mt = cos.mesh_tensor();
    const auto& sin_mt = sin.mesh_tensor();
    const auto& trans_mat_mt = trans_mat.mesh_tensor();
    const auto& q_output_mt = q_output.mesh_tensor();
    const auto& k_output_mt = k_output.mesh_tensor();

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
    const std::optional<tt::tt_metal::ShardSpec>& cos_sin_shard_spec = cos.shard_spec();

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

    CoreRangeSet all_cores = cos_sin_shard_spec->grid;
    CoreRangeSet all_cores_bb = all_cores.bounding_box();
    CoreRangeSet unused_cores = all_cores_bb.subtract(all_cores);

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

    // Resource names (DFBs, tensor parameters).  DFBSpecName and TensorParamName are distinct strong
    // types, so the seven borrowed buffers share a spelling across the two namespaces without collision.
    const m2::DFBSpecName Q_INPUT_DFB{"q_input"};
    const m2::DFBSpecName K_INPUT_DFB{"k_input"};
    const m2::DFBSpecName COS_DFB{"cos"};
    const m2::DFBSpecName SIN_DFB{"sin"};
    const m2::DFBSpecName TRANS_MAT_DFB{"trans_mat"};
    const m2::DFBSpecName ROTATED_INTERM_DFB{"rotated_input_interm"};
    const m2::DFBSpecName COS_INTERM_DFB{"cos_interm"};
    const m2::DFBSpecName SIN_INTERM_DFB{"sin_interm"};
    const m2::DFBSpecName Q_OUTPUT_DFB{"q_output"};
    const m2::DFBSpecName K_OUTPUT_DFB{"k_output"};

    const m2::TensorParamName Q_INPUT_T{"q_input"};
    const m2::TensorParamName K_INPUT_T{"k_input"};
    const m2::TensorParamName COS_T{"cos"};
    const m2::TensorParamName SIN_T{"sin"};
    const m2::TensorParamName TRANS_MAT_T{"trans_mat"};
    const m2::TensorParamName Q_OUTPUT_T{"q_output"};
    const m2::TensorParamName K_OUTPUT_T{"k_output"};

    const m2::KernelSpecName COMPUTE{"compute"};

    // Set up the DFBs.  Every DFB is bound to the (single) compute kernel, so each carries
    // data_format_metadata.  entry_size/num_entries reproduce the legacy CB page_size / tile count
    // (entry_size * num_entries == the legacy total_size).  The seven tensor DFBs are borrowed-memory
    // (borrowed_from a TensorParameter); the three interm DFBs are local.
    m2::Group<m2::DataflowBufferSpec> dataflow_buffers = {
        // q_input (c_0)
        m2::DataflowBufferSpec{
            .unique_id = Q_INPUT_DFB,
            .entry_size = input_single_tile_size,
            .num_entries = num_q_input_tiles,
            .data_format_metadata = input_cb_data_format,
            .borrowed_from = Q_INPUT_T,
        },
        // k_input (c_1)
        m2::DataflowBufferSpec{
            .unique_id = K_INPUT_DFB,
            .entry_size = input_single_tile_size,
            .num_entries = num_k_input_tiles,
            .data_format_metadata = input_cb_data_format,
            .borrowed_from = K_INPUT_T,
        },
        // cos (c_2)
        m2::DataflowBufferSpec{
            .unique_id = COS_DFB,
            .entry_size = cos_single_tile_size,
            .num_entries = num_cos_sin_tiles,
            .data_format_metadata = cos_cb_data_format,
            .borrowed_from = COS_T,
        },
        // sin (c_3)
        m2::DataflowBufferSpec{
            .unique_id = SIN_DFB,
            .entry_size = sin_single_tile_size,
            .num_entries = num_cos_sin_tiles,
            .data_format_metadata = sin_cb_data_format,
            .borrowed_from = SIN_T,
        },
        // trans_mat (c_4)
        m2::DataflowBufferSpec{
            .unique_id = TRANS_MAT_DFB,
            .entry_size = trans_mat_single_tile_size,
            .num_entries = num_trans_mat_tiles,
            .data_format_metadata = trans_mat_cb_data_format,
            .borrowed_from = TRANS_MAT_T,
        },
        // rotated_input_interm (c_24) — local
        m2::DataflowBufferSpec{
            .unique_id = ROTATED_INTERM_DFB,
            .entry_size = input_single_tile_size,
            .num_entries = num_interm_tiles,
            .data_format_metadata = input_cb_data_format,
        },
        // cos_interm (c_25) — local
        m2::DataflowBufferSpec{
            .unique_id = COS_INTERM_DFB,
            .entry_size = cos_single_tile_size,
            .num_entries = num_interm_tiles,
            .data_format_metadata = cos_cb_data_format,
        },
        // sin_interm (c_26) — local
        m2::DataflowBufferSpec{
            .unique_id = SIN_INTERM_DFB,
            .entry_size = sin_single_tile_size,
            .num_entries = num_interm_tiles,
            .data_format_metadata = sin_cb_data_format,
        },
        // q_output (c_16)
        m2::DataflowBufferSpec{
            .unique_id = Q_OUTPUT_DFB,
            .entry_size = output_single_tile_size,
            .num_entries = num_q_output_tiles,
            .data_format_metadata = output_cb_data_format,
            .borrowed_from = Q_OUTPUT_T,
        },
        // k_output (c_17)
        m2::DataflowBufferSpec{
            .unique_id = K_OUTPUT_DFB,
            .entry_size = output_single_tile_size,
            .num_entries = num_k_output_tiles,
            .data_format_metadata = output_cb_data_format,
            .borrowed_from = K_OUTPUT_T,
        },
    };

    // Every DFB is touched by exactly one kernel (the sole compute kernel), so each is a single-toucher
    // self-loop: bind COMPUTE as both PRODUCER and CONSUMER (shared accessor name → one dfb:: handle).
    m2::Group<m2::DFBBinding> compute_dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = Q_INPUT_DFB, .accessor_name = "q_in_cb", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = Q_INPUT_DFB, .accessor_name = "q_in_cb", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = K_INPUT_DFB, .accessor_name = "k_in_cb", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = K_INPUT_DFB, .accessor_name = "k_in_cb", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = COS_DFB, .accessor_name = "cos_cb", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = COS_DFB, .accessor_name = "cos_cb", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = SIN_DFB, .accessor_name = "sin_cb", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = SIN_DFB, .accessor_name = "sin_cb", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = TRANS_MAT_DFB,
            .accessor_name = "trans_mat_cb",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = TRANS_MAT_DFB,
            .accessor_name = "trans_mat_cb",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = ROTATED_INTERM_DFB,
            .accessor_name = "rotated_in_interm_cb",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = ROTATED_INTERM_DFB,
            .accessor_name = "rotated_in_interm_cb",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = COS_INTERM_DFB,
            .accessor_name = "cos_interm_cb",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = COS_INTERM_DFB,
            .accessor_name = "cos_interm_cb",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = SIN_INTERM_DFB,
            .accessor_name = "sin_interm_cb",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = SIN_INTERM_DFB,
            .accessor_name = "sin_interm_cb",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = Q_OUTPUT_DFB, .accessor_name = "q_out_cb", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = Q_OUTPUT_DFB, .accessor_name = "q_out_cb", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = K_OUTPUT_DFB, .accessor_name = "k_out_cb", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = K_OUTPUT_DFB, .accessor_name = "k_out_cb", .endpoint_type = m2::DFBEndpointType::CONSUMER},
    };

    const std::string compute_kernel_path =
        operation_attributes.row_major_QK
            ? "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/"
              "compute/rotary_embedding_llama_sharded_row_major.cpp"
            : "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/"
              "compute/rotary_embedding_llama_sharded.cpp";

    // Compute hardware config.  The op resolves a TTNN ComputeKernelConfig but the legacy factory only
    // threaded math_fidelity + fp32_dest_acc_en into its Metal ComputeConfigDescriptor, leaving every
    // other field at the descriptor default (math_approx_mode=false, dst_full_sync_en=false,
    // bfp8_pack_precise=false).  Reproduce those *resolved* values exactly by building a ComputeGen1Config
    // directly and setting only the two threaded knobs; the remaining ComputeGen1Config defaults
    // (sfpu_precision_mode=Precise, double_buffer_dest=true, bfp_pack_precision_mode=Approximate) match
    // the legacy descriptor defaults.  (Routing through to_compute_hardware_config would instead import
    // the resolved math_approx_mode=true and silently flip sfpu_precision_mode to Approximate.)
    m2::ComputeHardwareConfig compute_hw = m2::ComputeGen1Config{
        .fpu_math_fidelity = math_fidelity,
        .enable_32_bit_dest = fp32_dest_acc_en,
    };

    m2::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(compute_kernel_path),
        .dfb_bindings = compute_dfb_bindings,
        .compile_time_args =
            {
                {"q_Ht", q_n_heads_t},
                {"k_Ht", k_n_heads_t},
                {"Wt", head_dim_t},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"is_q"}},
        .hw_config = compute_hw,
    };

    m2::WorkUnitSpec work_unit{
        .name = "rotary_embedding_llama_fused_qk",
        .kernels = {COMPUTE},
        .target_nodes = all_cores_bb,
    };

    m2::ProgramSpec spec{
        .name = "rotary_embedding_llama_fused_qk",
        .kernels = {compute_spec},
        .dataflow_buffers = dataflow_buffers,
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = Q_INPUT_T, .spec = q_input_mt.tensor_spec()},
                m2::TensorParameter{.unique_id = K_INPUT_T, .spec = k_input_mt.tensor_spec()},
                m2::TensorParameter{.unique_id = COS_T, .spec = cos_mt.tensor_spec()},
                m2::TensorParameter{.unique_id = SIN_T, .spec = sin_mt.tensor_spec()},
                m2::TensorParameter{.unique_id = TRANS_MAT_T, .spec = trans_mat_mt.tensor_spec()},
                m2::TensorParameter{.unique_id = Q_OUTPUT_T, .spec = q_output_mt.tensor_spec()},
                m2::TensorParameter{.unique_id = K_OUTPUT_T, .spec = k_output_mt.tensor_spec()},
            },
        .work_units = {work_unit},
    };

    // Runtime args: is_q differentiates q, k and unused work groups.  1 on q-cores, 0 on k-cores.
    // Legacy left bounding-box cores outside q/k at the zero default (=> k branch); Metal 2.0 requires
    // an RTA for every target node, so set those unused cores to 0 explicitly.
    constexpr uint32_t is_q_arg = 1;  // If not q, must be k
    constexpr uint32_t is_k_arg = 0;
    const auto q_cores_vec = corerange_to_cores(q_cores, std::nullopt, /*row_wise=*/true);
    const auto k_cores_vec = corerange_to_cores(k_cores, std::nullopt, /*row_wise=*/true);
    const auto unused_cores_vec = corerange_to_cores(unused_cores, std::nullopt, /*row_wise=*/true);

    m2::KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (const auto& core : q_cores_vec) {
        m2::AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"is_q", is_q_arg}});
    }
    for (const auto& core : k_cores_vec) {
        m2::AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"is_q", is_k_arg}});
    }
    for (const auto& core : unused_cores_vec) {
        m2::AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"is_q", is_k_arg}});
    }

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args = {std::move(compute_run_args)};
    run_params.tensor_args.emplace(Q_INPUT_T, m2::ProgramRunArgs::TensorArgument{q_input_mt});
    run_params.tensor_args.emplace(K_INPUT_T, m2::ProgramRunArgs::TensorArgument{k_input_mt});
    run_params.tensor_args.emplace(COS_T, m2::ProgramRunArgs::TensorArgument{cos_mt});
    run_params.tensor_args.emplace(SIN_T, m2::ProgramRunArgs::TensorArgument{sin_mt});
    run_params.tensor_args.emplace(TRANS_MAT_T, m2::ProgramRunArgs::TensorArgument{trans_mat_mt});
    run_params.tensor_args.emplace(Q_OUTPUT_T, m2::ProgramRunArgs::TensorArgument{q_output_mt});
    run_params.tensor_args.emplace(K_OUTPUT_T, m2::ProgramRunArgs::TensorArgument{k_output_mt});

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::experimental::prim
