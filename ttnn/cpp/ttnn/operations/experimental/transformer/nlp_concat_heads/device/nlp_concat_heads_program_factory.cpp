// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "nlp_concat_heads_program_factory.hpp"
#include "nlp_concat_heads_device_operation.hpp"
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <string>
#include <vector>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts NLPConcatHeadsProgramFactory::create_program_spec(
    const NlpConcatHeadsParams& /*operation_attributes*/, const Tensor& input, Tensor& output) {
    // Metal 2.0 named resource handles (locals, for unity-build hygiene).
    const DFBSpecName IN_DFB{"in"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};

    constexpr const char* SHARDED_KERNEL =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_concat_heads_sharded.cpp";
    constexpr const char* INTERLEAVED_READER =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_concat_heads.cpp";
    constexpr const char* INTERLEAVED_WRITER =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_metal2.cpp";

    const auto& a = input;
    const auto& ashape = a.padded_shape();

    const tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);
    const bool in_sharded = a.is_sharded();

    const CoreCoord compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();

    uint32_t per_tensor_tiles = ashape[1] * ashape[3] / TILE_WIDTH;
    const uint32_t in0_h_tiles = ashape[2] / TILE_HEIGHT;
    const uint32_t in0_w_tiles = ashape[3] / TILE_WIDTH;    // head_dim
    const uint32_t in0_c = per_tensor_tiles / in0_w_tiles;  // num_heads
    const uint32_t in0_HtWt = in0_h_tiles * in0_w_tiles;
    const uint32_t in0_CHtWt = in0_c * in0_HtWt;

    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_blocks = ashape[0] * ashape[2] / TILE_HEIGHT;
    uint32_t num_cores = 0, num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0;
    CoreRangeSet all_cores = CoreRangeSet(), core_group_1 = CoreRangeSet(), core_group_2 = CoreRangeSet();
    bool row_major = false;
    if (in_sharded) {
        all_cores = a.shard_spec().value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        num_blocks_per_core_group_1 = a.shard_spec().value().shape[0] / a.padded_shape()[-2];
        per_tensor_tiles = a.shard_spec().value().shape[0] * a.shard_spec().value().shape[1] / TILE_HW;
        row_major = a.shard_spec().value().orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR;
    } else {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);
    }
    const uint32_t g1_numcores = core_group_1.num_cores();

    TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    ProgramSpec spec;
    spec.name = "nlp_concat_heads";
    spec.tensor_parameters = {input_param, output_param};

    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};

    if (in_sharded) {
        // Sharded: input (c_0) and output (c_16) are borrowed-memory CBs used purely as
        // local address source/sink (the kernel does a self-NoC head-rearrange). Both are
        // fake CBs -> bound reader=PRODUCER / writer=CONSUMER to satisfy the validator (the
        // two kernels, splitting nheads across the two RISCs, run on the same nodes).
        DataflowBufferSpec in_dfb{
            .unique_id = IN_DFB,
            .entry_size = single_tile_size,
            .num_entries = per_tensor_tiles,
            .data_format_metadata = cb_data_format,
            .borrowed_from = INPUT,
        };
        DataflowBufferSpec out_dfb{
            .unique_id = OUT_DFB,
            .entry_size = single_tile_size,
            .num_entries = per_tensor_tiles,
            .data_format_metadata = cb_data_format,
            .borrowed_from = OUTPUT,
        };

        const KernelSpec::CompileTimeArgs cta{
            {"in0_h_tiles", in0_h_tiles},
            {"head_dim_size_bytes", in0_w_tiles * single_tile_size},
            {"out_row_size_bytes", num_blocks_per_core_group_1 * in0_w_tiles * single_tile_size},
            {"block_size", num_blocks_per_core_group_1 * in0_HtWt}};
        const std::vector<std::string> rta_names{"nheads", "start_read_offset_bytes", "start_write_offset_bytes"};

        KernelSpec reader_spec{
            .unique_id = READER_KERNEL,
            .source = std::filesystem::path{SHARDED_KERNEL},
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER},
                 DFBBinding{
                     .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args = cta,
            .runtime_arg_schema = {.runtime_arg_names = rta_names},
            .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
        };
        KernelSpec writer_spec{
            .unique_id = WRITER_KERNEL,
            .source = std::filesystem::path{SHARDED_KERNEL},
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
            .compile_time_args = cta,
            .runtime_arg_schema = {.runtime_arg_names = rta_names},
            .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
        };

        const uint32_t nheads_first_risc = div_up(num_blocks_per_core_group_1, 2);
        const uint32_t nheads_second_risc = num_blocks_per_core_group_1 - nheads_first_risc;
        for (const NodeCoord& core : corerange_to_cores(all_cores, num_cores, /*row_wise=*/true)) {
            reader_run.runtime_arg_values.push_back(
                {core,
                 {{"nheads", nheads_first_risc}, {"start_read_offset_bytes", 0u}, {"start_write_offset_bytes", 0u}}});
            writer_run.runtime_arg_values.push_back(
                {core,
                 {{"nheads", nheads_second_risc},
                  {"start_read_offset_bytes", nheads_first_risc * in0_HtWt * single_tile_size},
                  {"start_write_offset_bytes", nheads_first_risc * in0_w_tiles * single_tile_size}}});
        }

        spec.kernels = {reader_spec, writer_spec};
        spec.dataflow_buffers = {in_dfb, out_dfb};
        spec.work_units = {WorkUnitSpec{
            .name = "nlp_concat_heads_sharded", .kernels = {READER_KERNEL, WRITER_KERNEL}, .target_nodes = all_cores}};
    } else {
        // Interleaved: input read page-by-page via TensorAccessor into the src CB (c_0); the
        // forked writer consumes the src CB and writes pages to the output tensor. No borrowed CB.
        DataflowBufferSpec in_dfb{
            .unique_id = IN_DFB,
            .entry_size = single_tile_size,
            .num_entries = per_tensor_tiles * 2,  // double buffer
            .data_format_metadata = cb_data_format,
        };

        KernelSpec reader_spec{
            .unique_id = READER_KERNEL,
            .source = std::filesystem::path{INTERLEAVED_READER},
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
            .compile_time_args =
                {{"in0_h_tiles", in0_h_tiles}, {"in0_w_tiles", in0_w_tiles}, {"in0_c", in0_c}, {"in0_HtWt", in0_HtWt}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_blocks", "in0_h_dim", "in0_tensor_tile_id"}},
            .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
        };
        KernelSpec writer_spec{
            .unique_id = WRITER_KERNEL,
            .source = std::filesystem::path{INTERLEAVED_WRITER},
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER}},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
            .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
        };

        const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);
        for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); ++i) {
            const NodeCoord& core = cores[i];
            const uint32_t num_blocks_per_core =
                i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

            const uint32_t in0_h_dim = num_blocks_written % in0_h_tiles;
            const uint32_t in0_tensor_tile_id =
                (num_blocks_written / in0_h_tiles * in0_CHtWt) + (in0_h_dim * in0_w_tiles);

            reader_run.runtime_arg_values.push_back(
                {core,
                 {{"num_blocks", num_blocks_per_core},
                  {"in0_h_dim", in0_h_dim},
                  {"in0_tensor_tile_id", in0_tensor_tile_id}}});
            writer_run.runtime_arg_values.push_back(
                {core,
                 {{"num_pages", num_blocks_per_core * per_tensor_tiles},
                  {"start_id", num_blocks_written * per_tensor_tiles}}});
            num_blocks_written += num_blocks_per_core;
        }

        spec.kernels = {reader_spec, writer_spec};
        spec.dataflow_buffers = {in_dfb};
        spec.work_units = {WorkUnitSpec{
            .name = "nlp_concat_heads_interleaved",
            .kernels = {READER_KERNEL, WRITER_KERNEL},
            .target_nodes = all_cores}};
    }

    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT, TensorArgument{std::cref(input.mesh_tensor())}},
        {OUTPUT, TensorArgument{std::cref(output.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
