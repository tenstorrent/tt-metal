// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_sharded_program_factory.hpp"
#include "rotary_embedding_llama_metal2_common.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using namespace ttnn::experimental::prim::rope_metal2;

ttnn::device_operation::ProgramArtifacts RotaryEmbeddingLlamaMultiCoreSharded::create_program_artifacts(
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

    bool in_sharded = input.shard_spec().has_value();
    std::optional<ShardSpec> shard_spec = in_sharded ? input.shard_spec() : output.shard_spec();

    const uint32_t batch = input.padded_shape()[1];
    const uint32_t n_heads_t = shard_spec->shape[0] / constants::TILE_HEIGHT;
    const uint32_t head_dim_t = shard_spec->shape[1] / constants::TILE_WIDTH;

    tt_metal::IDevice* device = tensor_args.input_tensor.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    CoreRange all_cores = shard_spec->grid.bounding_box();
    uint32_t num_cores_x = all_cores.grid_size().x;
    uint32_t num_cores_y = all_cores.grid_size().y;

    const uint32_t num_input_tiles = n_heads_t * head_dim_t;
    const uint32_t num_output_tiles = num_input_tiles;

    // Parallelization
    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t batch_parallel_factor = std::min(batch, num_cores);
    const uint32_t batch_per_core = (batch + batch_parallel_factor - 1) /
                                    batch_parallel_factor;  // TODO: To make general, add support for batch_per_core > 1

    const uint32_t num_sin_cos_rows_per_core = batch_per_core;
    const uint32_t num_cos_sin_tiles = head_dim_t * num_sin_cos_rows_per_core;
    const uint32_t num_interm_tiles = head_dim_t;

    // ------------------------------------------------------------------
    // Dataflow buffers. Decode borrows io + cos/sin/trans_mat from their resident L1 shards
    // (legacy dynamic-CB `.buffer` backing); the intermediates are plain. The lone compute kernel is the
    // sole toucher of every DFB → self-loop each (PRODUCER + CONSUMER).
    // ------------------------------------------------------------------
    DataflowBufferSpec input_dfb{
        .unique_id = INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = INPUT_PARAM};
    DataflowBufferSpec cos_dfb{
        .unique_id = COS_DFB,
        .entry_size = cos_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = cos_cb_data_format,
        .borrowed_from = COS_PARAM};
    DataflowBufferSpec sin_dfb{
        .unique_id = SIN_DFB,
        .entry_size = sin_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = sin_cb_data_format,
        .borrowed_from = SIN_PARAM};
    DataflowBufferSpec trans_mat_dfb{
        .unique_id = TRANS_MAT_DFB,
        .entry_size = trans_mat_single_tile_size,
        .num_entries = 1,  // We only take one tile of trans_mat
        .data_format_metadata = trans_mat_cb_data_format,
        .borrowed_from = TRANS_MAT_PARAM};
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
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = OUTPUT_PARAM};

    // ------------------------------------------------------------------
    // Tensor parameters. Referenced via the DFB borrowed_from links (no kernel TensorBinding —
    // a compute kernel cannot bind a TensorAccessor). The spec validator accepts borrowed_from as
    // the required reference.
    // ------------------------------------------------------------------
    TensorParameter input_param{.unique_id = INPUT_PARAM, .spec = input.tensor_spec()};
    TensorParameter cos_param{.unique_id = COS_PARAM, .spec = cos.tensor_spec()};
    TensorParameter sin_param{.unique_id = SIN_PARAM, .spec = sin.tensor_spec()};
    TensorParameter trans_mat_param{.unique_id = TRANS_MAT_PARAM, .spec = trans_mat.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_PARAM, .spec = output.tensor_spec()};

    // hw_config — Style B (see the interleaved factory for the rationale).
    const ComputeHardwareConfig compute_hw_config =
        ComputeGen1Config{.fpu_math_fidelity = math_fidelity, .enable_32_bit_dest = fp32_dest_acc_en};

    auto self_loop = [](const DFBSpecName& dfb, const std::string& name) {
        return Group<DFBBinding>{
            DFBBinding{.dfb_spec_name = dfb, .accessor_name = name, .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = dfb, .accessor_name = name, .endpoint_type = DFBEndpointType::CONSUMER}};
    };
    Group<DFBBinding> compute_bindings;
    for (const auto& [dfb, name] : {
             std::pair{INPUT_DFB, std::string{"input"}},
             std::pair{COS_DFB, std::string{"cos"}},
             std::pair{SIN_DFB, std::string{"sin"}},
             std::pair{TRANS_MAT_DFB, std::string{"trans_mat"}},
             std::pair{ROTATED_INTERM_DFB, std::string{"rotated_interm"}},
             std::pair{COS_INTERM_DFB, std::string{"cos_interm"}},
             std::pair{SIN_INTERM_DFB, std::string{"sin_interm"}},
             std::pair{OUT_DFB, std::string{"out"}},
         }) {
        auto pair = self_loop(dfb, name);
        compute_bindings.push_back(pair[0]);
        compute_bindings.push_back(pair[1]);
    }

    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = kComputeShardedSource,
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = compute_bindings,
        .compile_time_args = {{"Wt", head_dim_t}, {"Ht", n_heads_t}},
        .hw_config = compute_hw_config};

    ProgramSpec spec{
        .name = "rotary_embedding_llama_sharded",
        .kernels = {compute_spec},
        .dataflow_buffers =
            {input_dfb, cos_dfb, sin_dfb, trans_mat_dfb, rotated_interm_dfb, cos_interm_dfb, sin_interm_dfb, out_dfb},
        .tensor_parameters = {input_param, cos_param, sin_param, trans_mat_param, output_param},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {COMPUTE}, .target_nodes = all_cores}}};

    // Compute kernel has no runtime args → no KernelRunArgs entry. Borrowed DFBs draw their backing
    // L1 address from the tensor_args below.
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {INPUT_PARAM, TensorArgument{input}},
        {COS_PARAM, TensorArgument{cos}},
        {SIN_PARAM, TensorArgument{sin}},
        {TRANS_MAT_PARAM, TensorArgument{trans_mat}},
        {OUTPUT_PARAM, TensorArgument{output}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
