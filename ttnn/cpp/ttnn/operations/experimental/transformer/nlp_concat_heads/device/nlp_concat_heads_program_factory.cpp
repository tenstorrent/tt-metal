// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include "nlp_concat_heads_program_factory.hpp"
#include "nlp_concat_heads_device_operation.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

using ttnn::device_operation::ProgramArtifacts;

ProgramArtifacts NLPConcatHeadsProgramFactory::create_program_artifacts(
    const NlpConcatHeadsParams& /*operation_attributes*/, const Tensor& input, Tensor& output) {
    // Program-scope resource names (typed constants).
    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::DFBSpecName SRC0{"src0"};
    const m2::DFBSpecName OUT0{"out0"};
    const m2::TensorParamName INPUT{"input"};
    const m2::TensorParamName OUTPUT{"output"};

    const auto& a = input;
    const auto& ashape = a.padded_shape();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    bool in_sharded = a.is_sharded();

    CoreCoord compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_tensor_tiles = ashape[1] * ashape[3] / TILE_WIDTH;  // 142

    // Per output tensor args
    // Output shape is: [B, 1, s, 4544]
    uint32_t in0_h_tiles = ashape[2] / TILE_HEIGHT;
    uint32_t in0_w_tiles = ashape[3] / TILE_WIDTH;    // head_dim
    uint32_t in0_c = per_tensor_tiles / in0_w_tiles;  // num_heads
    uint32_t in0_HtWt = in0_h_tiles * in0_w_tiles;
    uint32_t in0_CHtWt = in0_c * in0_HtWt;

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of per_tensor_tiles per core
    uint32_t num_blocks = ashape[0] * ashape[2] / TILE_HEIGHT;
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
    uint32_t g1_numcores = core_group_1.num_cores();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Buffer* out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer should be allocated on device!");

    if (in_sharded) {
        ////////////////////////////////////////////////////////////////////////
        //                      Sharded path
        ////////////////////////////////////////////////////////////////////////
        const std::filesystem::path sharded_src(
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_concat_heads_sharded.cpp");

        // CTAs shared by both instances of the dual-instance work-split; the magic CB indices
        // (src0_cb_index / out_cb_index) drop out — they become the SRC0 / OUT0 DFB bindings.
        m2::KernelSpec::CompileTimeArgs shared_ctas = {
            {"in0_h_tiles", in0_h_tiles},
            {"head_dim_size_bytes", in0_w_tiles * single_tile_size},
            {"out_row_size_bytes", num_blocks_per_core_group_1 * in0_w_tiles * single_tile_size},
            {"block_size", num_blocks_per_core_group_1 * in0_HtWt},
        };

        // Dual-instance work-split (two-toucher → 1P+1C): same source, reader-config + writer-config
        // instances over all_cores. Bind the reader instance PRODUCER and the writer instance CONSUMER
        // for both borrowed DFBs (both touches are sync-free raw addressing, so the role is cosmetic
        // on Gen1). Accessor names are identical across instances (same source → same dfb:: symbols).
        m2::KernelSpec reader_spec{
            .unique_id = READER,
            .source = sharded_src,
            .dfb_bindings =
                {m2::DFBBinding{
                     .dfb_spec_name = SRC0, .accessor_name = "in0", .endpoint_type = m2::DFBEndpointType::PRODUCER},
                 m2::DFBBinding{
                     .dfb_spec_name = OUT0, .accessor_name = "out0", .endpoint_type = m2::DFBEndpointType::PRODUCER}},
            .compile_time_args = shared_ctas,
            .runtime_arg_schema =
                {.runtime_arg_names = {"nheads", "start_read_offset_bytes", "start_write_offset_bytes"}},
            .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
        };

        m2::KernelSpec writer_spec{
            .unique_id = WRITER,
            .source = sharded_src,
            .dfb_bindings =
                {m2::DFBBinding{
                     .dfb_spec_name = SRC0, .accessor_name = "in0", .endpoint_type = m2::DFBEndpointType::CONSUMER},
                 m2::DFBBinding{
                     .dfb_spec_name = OUT0, .accessor_name = "out0", .endpoint_type = m2::DFBEndpointType::CONSUMER}},
            .compile_time_args = shared_ctas,
            .runtime_arg_schema =
                {.runtime_arg_names = {"nheads", "start_read_offset_bytes", "start_write_offset_bytes"}},
            .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
        };

        // Borrowed-memory DFBs: SRC0 backed by the input shard, OUT0 by the output shard. No
        // TensorBinding on the kernels — the borrowed DFB is the "use"; the backing L1 address
        // resolves from the corresponding tensor_arg at runtime.
        m2::DataflowBufferSpec src0_dfb{
            .unique_id = SRC0,
            .entry_size = single_tile_size,
            .num_entries = per_tensor_tiles,
            .data_format_metadata = cb_data_format,
            .borrowed_from = INPUT,
        };
        // NOTE: legacy allocated CB 16 under `if (out_sharded)`, but the kernel binds it
        // unconditionally; sharded-in ⇒ sharded-out holds in practice (a borrowed output CB requires
        // L1-sharded output). Allocated unconditionally in this branch to match the kernel's use.
        m2::DataflowBufferSpec out0_dfb{
            .unique_id = OUT0,
            .entry_size = single_tile_size,
            .num_entries = per_tensor_tiles,
            .data_format_metadata = cb_data_format,
            .borrowed_from = OUTPUT,
        };

        m2::ProgramSpec spec{
            .name = "nlp_concat_heads_sharded",
            .kernels = {reader_spec, writer_spec},
            .dataflow_buffers = {src0_dfb, out0_dfb},
            .tensor_parameters =
                {{.unique_id = INPUT, .spec = input.tensor_spec()},
                 {.unique_id = OUTPUT, .spec = output.tensor_spec()}},
            .work_units = {m2::WorkUnitSpec{
                .name = "main",
                .kernels = {READER, WRITER},
                .target_nodes = all_cores,
            }},
        };

        uint32_t nheads_first_risc = div_up(num_blocks_per_core_group_1, 2);
        uint32_t nheads_second_risc = num_blocks_per_core_group_1 - nheads_first_risc;
        // Mirror SetRuntimeArgs(program, kernel, all_cores, args) by emplacing the same
        // per-core args on every logical core in the sharded range set.
        m2::KernelRunArgs reader_ra{.kernel = READER};
        m2::KernelRunArgs writer_ra{.kernel = WRITER};
        for (const auto& core : corerange_to_cores(all_cores, num_cores, /*row_wise=*/true)) {
            m2::AddRuntimeArgsForNode(
                reader_ra.runtime_arg_values,
                core,
                {{"nheads", nheads_first_risc}, {"start_read_offset_bytes", 0u}, {"start_write_offset_bytes", 0u}});
            m2::AddRuntimeArgsForNode(
                writer_ra.runtime_arg_values,
                core,
                {{"nheads", nheads_second_risc},
                 {"start_read_offset_bytes", nheads_first_risc * in0_HtWt * single_tile_size},
                 {"start_write_offset_bytes", nheads_first_risc * in0_w_tiles * single_tile_size}});
        }

        m2::ProgramRunArgs run_params;
        run_params.kernel_run_args = {std::move(reader_ra), std::move(writer_ra)};
        run_params.tensor_args.emplace(INPUT, m2::TensorArgument{input.mesh_tensor()});
        run_params.tensor_args.emplace(OUTPUT, m2::TensorArgument{output.mesh_tensor()});

        return ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_params)};
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Interleaved path
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_concat_heads.cpp"),
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = SRC0, .accessor_name = "src0", .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "in0"}},
        .compile_time_args =
            {{"in0_h_tiles", in0_h_tiles}, {"in0_w_tiles", in0_w_tiles}, {"in0_c", in0_c}, {"in0_HtWt", in0_HtWt}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_blocks", "in0_h_dim", "in0_tensor_tile_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
    };

    // Interleaved writer is the forked _metal2 copy of the eltwise/unary donor (the legacy copy
    // stays for its unmigrated co-borrowers). Consumes SRC0 (the reader's ring) and writes the
    // output tensor via a TensorBinding.
    m2::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_metal2.cpp"),
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = SRC0, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
    };

    // Double-buffered ring (index 0 in legacy); reader PRODUCER, writer CONSUMER (plain 1:1).
    m2::DataflowBufferSpec src0_dfb{
        .unique_id = SRC0,
        .entry_size = single_tile_size,
        .num_entries = per_tensor_tiles * 2,  // double buffer
        .data_format_metadata = cb_data_format,
    };

    m2::ProgramSpec spec{
        .name = "nlp_concat_heads_interleaved",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {src0_dfb},
        .tensor_parameters =
            {{.unique_id = INPUT, .spec = input.tensor_spec()}, {.unique_id = OUTPUT, .spec = output.tensor_spec()}},
        .work_units = {m2::WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER},
            .target_nodes = all_cores,
        }},
    };

    m2::KernelRunArgs reader_ra{.kernel = READER};
    m2::KernelRunArgs writer_ra{.kernel = WRITER};
    const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);
    for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        uint32_t num_blocks_per_core = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        uint32_t in0_h_dim = num_blocks_written % in0_h_tiles;
        uint32_t in0_tensor_tile_id = (num_blocks_written / in0_h_tiles * in0_CHtWt) + (in0_h_dim * in0_w_tiles);

        m2::AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values,
            core,
            {{"num_blocks", num_blocks_per_core},
             {"in0_h_dim", in0_h_dim},
             {"in0_tensor_tile_id", in0_tensor_tile_id}});

        m2::AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values,
            core,
            {{"num_pages", num_blocks_per_core * per_tensor_tiles},
             {"start_id", num_blocks_written * per_tensor_tiles}});
        num_blocks_written += num_blocks_per_core;
    }

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args = {std::move(reader_ra), std::move(writer_ra)};
    run_params.tensor_args.emplace(INPUT, m2::TensorArgument{input.mesh_tensor()});
    run_params.tensor_args.emplace(OUTPUT, m2::TensorArgument{output.mesh_tensor()});

    return ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_params)};
}

}  // namespace ttnn::experimental::prim
