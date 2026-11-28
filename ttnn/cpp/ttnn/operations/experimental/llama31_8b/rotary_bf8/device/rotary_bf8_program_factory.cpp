// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "rotary_bf8_device_operation.hpp"
#include "rotary_bf8_table.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_metal2_common.hpp"

namespace ttnn::experimental::llama31_8b {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using namespace ttnn::experimental::prim::rope_metal2;

ttnn::device_operation::ProgramArtifacts RotaryBf8ProgramFactory::create_program_artifacts(
    const RotaryBf8Operation::operation_attributes_t& operation_attributes,
    const RotaryBf8Operation::tensor_args_t& tensor_args,
    RotaryBf8Operation::tensor_return_value_t& output_tensor) {
    const auto& e = rotary_bf8_table::ENTRIES[operation_attributes.table_row];
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& cos = tensor_args.cos_cache.mesh_tensor();
    const auto& sin = tensor_args.sin_cache.mesh_tensor();
    const auto& trans_mat = tensor_args.trans_mat.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();

    const tt::DataFormat input_format = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat cos_format = datatype_to_dataformat_converter(cos.dtype());
    const tt::DataFormat sin_format = datatype_to_dataformat_converter(sin.dtype());
    const tt::DataFormat trans_mat_format = datatype_to_dataformat_converter(trans_mat.dtype());
    const tt::DataFormat output_format = datatype_to_dataformat_converter(output.dtype());

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(tensor_args.input.device()->arch(), operation_attributes.compute_kernel_config);

    DataflowBufferSpec input_dfb{
        .unique_id = INPUT_DFB,
        .entry_size = tt::tile_size(input_format),
        .num_entries = e.in_pages,
        .data_format_metadata = input_format};
    DataflowBufferSpec cos_dfb{
        .unique_id = COS_DFB,
        .entry_size = tt::tile_size(cos_format),
        .num_entries = e.cos_sin_pages,
        .data_format_metadata = cos_format};
    DataflowBufferSpec sin_dfb{
        .unique_id = SIN_DFB,
        .entry_size = tt::tile_size(sin_format),
        .num_entries = e.cos_sin_pages,
        .data_format_metadata = sin_format};
    DataflowBufferSpec trans_mat_dfb{
        .unique_id = TRANS_MAT_DFB,
        .entry_size = tt::tile_size(trans_mat_format),
        .num_entries = 1,
        .data_format_metadata = trans_mat_format};
    DataflowBufferSpec rotated_interm_dfb{
        .unique_id = ROTATED_INTERM_DFB,
        .entry_size = tt::tile_size(input_format),
        .num_entries = e.interm_pages,
        .data_format_metadata = input_format};
    DataflowBufferSpec cos_interm_dfb{
        .unique_id = COS_INTERM_DFB,
        .entry_size = tt::tile_size(cos_format),
        .num_entries = e.interm_pages,
        .data_format_metadata = cos_format};
    DataflowBufferSpec sin_interm_dfb{
        .unique_id = SIN_INTERM_DFB,
        .entry_size = tt::tile_size(sin_format),
        .num_entries = e.interm_pages,
        .data_format_metadata = sin_format};
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = tt::tile_size(output_format),
        .num_entries = e.out_pages,
        .data_format_metadata = output_format};
    ScratchpadSpec zero_scratchpad{
        .unique_id = ZERO_SCRATCH, .size_per_node = tt::tile_size(output_format) * e.interm_pages};

    TensorParameter input_param{.unique_id = INPUT_PARAM, .spec = input.tensor_spec()};
    TensorParameter cos_param{.unique_id = COS_PARAM, .spec = cos.tensor_spec()};
    TensorParameter sin_param{.unique_id = SIN_PARAM, .spec = sin.tensor_spec()};
    TensorParameter trans_mat_param{.unique_id = TRANS_MAT_PARAM, .spec = trans_mat.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_PARAM, .spec = output.tensor_spec()};

    const KernelSpec::CompilerOptions::Defines reload_define{{"RELOAD_IMPL", e.reload ? "1" : "0"}};

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
            {{"n_heads", e.n_heads},
             {"Ht", e.seq_tiles},
             {"Wt", e.head_dim_tiles},
             {"freq_per_head", 0u},
             {"cos_Ht", e.cos_seq_tiles},
             {"sin_Ht", e.cos_seq_tiles},
             {"rotary_Ht", e.rotary_seq_tiles}},
        .runtime_arg_schema = {.runtime_arg_names = kPrefillRangeArgNames},
        .hw_config = create_reader_datamovement_config(true)};

    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = kWriterSource,
        .compiler_options = {.defines = reload_define},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .scratchpad_bindings = {ScratchpadBinding{.scratchpad_spec_name = ZERO_SCRATCH, .accessor_name = "zero"}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_PARAM, .accessor_name = "output"}},
        .compile_time_args =
            {{"n_heads", e.n_heads}, {"Wt", e.head_dim_tiles}, {"Ht", e.seq_tiles}, {"rotary_Ht", e.rotary_seq_tiles}},
        .runtime_arg_schema = {.runtime_arg_names = kPrefillRangeArgNames},
        .hw_config = create_writer_datamovement_config(true)};

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
        .compile_time_args = {{"Wt", e.head_dim_tiles}, {"rotary_Ht", e.rotary_seq_tiles}},
        .runtime_arg_schema = {.runtime_arg_names = kPrefillComputeArgNames},
        .hw_config = ComputeHardwareConfig{
            .fpu_math_fidelity = math_fidelity,
            .enable_32_bit_dest = fp32_dest_acc_en,
            .config_1xx = ComputeHardwareConfig::Compute1XXConfig{.bfp_pack_precision_mode = Precision::Precise}}};

    std::vector<rotary_bf8_table::CoreWork> work(e.grid_x * e.grid_y, rotary_bf8_table::CoreWork{});
    for (const auto& c : e.cores) {
        work[(c.y * e.grid_x) + c.x] = c;
    }

    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};
    KernelRunArgs compute_run{.kernel = COMPUTE};
    for (std::uint32_t y = 0; y < e.grid_y; ++y) {
        for (std::uint32_t x = 0; x < e.grid_x; ++x) {
            const auto& w = work[(y * e.grid_x) + x];
            const std::uint32_t batch_end = w.seq_end > w.seq_start ? 1u : 0u;
            const NodeCoord node = CoreCoord{x, y};
            const std::initializer_list<std::pair<std::string, ArgValue>> range_args = {
                {"batch_start", 0u},
                {"batch_end", batch_end},
                {"seq_t_start", w.seq_start},
                {"seq_t_end", w.seq_end},
                {"head_start", w.head_start},
                {"head_end", w.head_end}};
            for (auto* run : {&reader_run, &writer_run}) {
                AddRuntimeArgsForNode(run->runtime_arg_values, node, range_args);
            }
            AddRuntimeArgsForNode(
                compute_run.runtime_arg_values,
                node,
                {{"batch_start", 0u},
                 {"batch_end", batch_end},
                 {"seq_t_start", w.seq_start},
                 {"seq_t_end", w.seq_end},
                 {"n_heads", w.head_end - w.head_start}});
        }
    }

    ProgramSpec spec{
        .name = "llama31_8b_rotary_bf8",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers =
            {input_dfb, cos_dfb, sin_dfb, trans_mat_dfb, rotated_interm_dfb, cos_interm_dfb, sin_interm_dfb, out_dfb},
        .scratchpads = {zero_scratchpad},
        .tensor_parameters = {input_param, cos_param, sin_param, trans_mat_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = CoreRange({0, 0}, {e.grid_x - 1, e.grid_y - 1})}}};

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

}  // namespace ttnn::experimental::llama31_8b
