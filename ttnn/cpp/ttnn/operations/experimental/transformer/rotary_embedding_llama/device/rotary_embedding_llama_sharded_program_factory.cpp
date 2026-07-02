// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_sharded_program_factory.hpp"

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

ttnn::device_operation::ProgramArtifacts RotaryEmbeddingLlamaMultiCoreSharded::create_program_artifacts(
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

    const TensorParamName IN_TENSOR{"in_tensor"};
    const TensorParamName COS_TENSOR{"cos_tensor"};
    const TensorParamName SIN_TENSOR{"sin_tensor"};
    const TensorParamName TRANS_MAT_TENSOR{"trans_mat_tensor"};
    const TensorParamName OUT_TENSOR{"out_tensor"};

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

    const bool in_sharded = input.shard_spec().has_value();
    const std::optional<ShardSpec> shard_spec = in_sharded ? input.shard_spec() : output.shard_spec();

    const uint32_t batch = input.padded_shape()[1];
    const uint32_t n_heads_t = shard_spec->shape[0] / constants::TILE_HEIGHT;
    const uint32_t head_dim_t = shard_spec->shape[1] / constants::TILE_WIDTH;

    tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    const CoreRange all_cores = shard_spec->grid.bounding_box();
    const uint32_t num_cores_x = all_cores.grid_size().x;
    const uint32_t num_cores_y = all_cores.grid_size().y;

    const uint32_t num_input_tiles = n_heads_t * head_dim_t;
    const uint32_t num_output_tiles = num_input_tiles;

    // Parallelization
    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t batch_parallel_factor = std::min(batch, num_cores);
    const uint32_t batch_per_core = (batch + batch_parallel_factor - 1) /
                                    batch_parallel_factor;  // TODO: To make general, add support for batch_per_core > 1

    const uint32_t num_sin_cos_rows_per_core = batch_per_core;
    const uint32_t num_cos_sin_tiles = head_dim_t * num_sin_cos_rows_per_core;

    // We only take one tile of trans_mat
    const uint32_t num_trans_mat_tiles = 1;

    const uint32_t num_interm_tiles = head_dim_t;

    // ----------------------------------------------------------------------------
    // Tensor parameters. Each of the 5 backing tensors backs exactly one borrowed
    // DFB (resident, sharded L1). NONE is bound as a TensorBinding: the compute kernel
    // touches every CB by id only (FIFO ops / LLK operands), never builds a
    // TensorAccessor and never reads a base address. The borrowed DFB resolves its
    // address from the matching TensorArgument at runtime.
    // ----------------------------------------------------------------------------
    TensorParameter in_param{.unique_id = IN_TENSOR, .spec = input.tensor_spec()};
    TensorParameter cos_param{.unique_id = COS_TENSOR, .spec = cos.tensor_spec()};
    TensorParameter sin_param{.unique_id = SIN_TENSOR, .spec = sin.tensor_spec()};
    TensorParameter trans_mat_param{.unique_id = TRANS_MAT_TENSOR, .spec = trans_mat.tensor_spec()};
    TensorParameter out_param{.unique_id = OUT_TENSOR, .spec = output.tensor_spec()};

    // ----------------------------------------------------------------------------
    // Dataflow buffers. One DFB per legacy CBDescriptor.
    //   - 5 borrowed-memory DFBs (legacy CBs with .buffer set): in (c_0), cos (c_1),
    //     sin (c_2), trans_mat (c_3), out (c_16).
    //   - 3 program-local scratch DFBs (legacy CBs with no .buffer): rotated_in_interm
    //     (c_24), cos_interm (c_25), sin_interm (c_26). Real intra-kernel FIFOs.
    // ----------------------------------------------------------------------------
    DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = IN_TENSOR,
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
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = OUT_TENSOR,
    };

    // ----------------------------------------------------------------------------
    // Single compute KernelSpec. Every DFB is bound on this kernel as a self-loop
    // (PRODUCER + CONSUMER): the op is compute-only and the data is resident, so each
    // DFB is one-ended from the validator's view. Each self-loop DFB is declared INTRA.
    //   - in/cos/sin/trans_mat/out: borrowed-memory FAKE-CB self-loops.
    //   - rotated_in_interm/cos_interm/sin_interm: real intra-kernel FIFO self-loops.
    // ----------------------------------------------------------------------------
    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/"
        "rotary_embedding_llama_sharded.cpp";

    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source = std::filesystem::path{compute_kernel_path},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
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
             DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .compile_time_args = {{"Wt", head_dim_t}, {"Ht", n_heads_t}},
        .hw_config =
            ComputeHardwareConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
            },
    };
    // Compute-kernel self-loops must declare their thread connectivity. All are INTRA
    // (each TRISC thread self-loops; no cross-thread production).
    compute_spec.advanced_options.dfb_self_loop_connectivities = {
        {IN_DFB, DFBSelfLoopConnectivity::INTRA},
        {COS_DFB, DFBSelfLoopConnectivity::INTRA},
        {SIN_DFB, DFBSelfLoopConnectivity::INTRA},
        {TRANS_MAT_DFB, DFBSelfLoopConnectivity::INTRA},
        {ROTATED_INTERM_DFB, DFBSelfLoopConnectivity::INTRA},
        {COS_INTERM_DFB, DFBSelfLoopConnectivity::INTRA},
        {SIN_INTERM_DFB, DFBSelfLoopConnectivity::INTRA},
        {OUT_DFB, DFBSelfLoopConnectivity::INTRA},
    };

    WorkUnitSpec wu{
        .name = "rotary_embedding_llama_sharded",
        .kernels = {COMPUTE_KERNEL},
        .target_nodes = CoreRangeSet(all_cores),
    };

    ProgramSpec spec{
        .name = "rotary_embedding_llama_sharded",
        .kernels = {compute_spec},
        .dataflow_buffers =
            {in_dfb_spec,
             cos_dfb_spec,
             sin_dfb_spec,
             trans_mat_dfb_spec,
             rotated_in_interm_dfb_spec,
             cos_interm_dfb_spec,
             sin_interm_dfb_spec,
             out_dfb_spec},
        .tensor_parameters = {in_param, cos_param, sin_param, trans_mat_param, out_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {IN_TENSOR, TensorArgument{std::cref(input.mesh_tensor())}},
        {COS_TENSOR, TensorArgument{std::cref(cos.mesh_tensor())}},
        {SIN_TENSOR, TensorArgument{std::cref(sin.mesh_tensor())}},
        {TRANS_MAT_TENSOR, TensorArgument{std::cref(trans_mat.mesh_tensor())}},
        {OUT_TENSOR, TensorArgument{std::cref(output.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
