// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "nlp_concat_heads_program_factory.hpp"
#include "nlp_concat_heads_device_operation.hpp"
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// Op-private kernels, converted in place.
constexpr const char* kNlpConcatHeadsReaderSource =
    "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
    "reader_tm_tile_layout_nlp_concat_heads.cpp";
constexpr const char* kNlpConcatHeadsShardedSource =
    "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
    "reader_tm_tile_layout_nlp_concat_heads_sharded.cpp";
// Metal 2.0 fork of the eltwise/unary donor, living beside the original. The legacy copy still
// serves the many factories that have not migrated; bind the fork, never convert the original.
constexpr const char* kNlpConcatHeadsWriterSource =
    "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
    "writer_unary_interleaved_start_id_metal2.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts NLPConcatHeadsProgramFactory::create_program_artifacts(
    const NlpConcatHeadsParams& /*operation_attributes*/, const Tensor& input, Tensor& output) {
    const auto& a = input;
    const auto& ashape = a.padded_shape();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    bool in_sharded = a.is_sharded();

    const auto* device = a.device();
    CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();

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

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    // Resource names. SRC0_DFB / OUT_DFB replace the legacy magic CB indices 0 and 16.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName SRC0_DFB{"src0"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    const TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    const TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    if (in_sharded) {
        const CoreRangeSet all_cores = a.shard_spec().value().grid;
        const uint32_t num_cores = all_cores.num_cores();
        const uint32_t num_blocks_per_core_group_1 = a.shard_spec().value().shape[0] / a.padded_shape()[-2];
        per_tensor_tiles = a.shard_spec().value().shape[0] * a.shard_spec().value().shape[1] / TILE_HW;

        // Both DFBs are borrowed views onto the resident input / output shards, so their L1
        // addresses come from the bound tensors rather than from a Program allocation.
        //
        // OUT_DFB is declared unconditionally here, unlike the legacy CB which was created only for
        // a sharded output. The sharded kernel constructs and writes through this buffer on every
        // path, and a kernel cannot name a binding the spec does not declare, so the binding has to
        // exist wherever the kernel does. (A sharded input with an interleaved output is reachable
        // through validation but has no test coverage and no defined behaviour today; it is an open
        // question for the op's owners, not something to paper over here.)
        const DataflowBufferSpec src0_dfb{
            .unique_id = SRC0_DFB,
            .entry_size = single_tile_size,
            .num_entries = per_tensor_tiles,
            .data_format_metadata = cb_data_format,
            .borrowed_from = INPUT,
        };
        const DataflowBufferSpec out_dfb{
            .unique_id = OUT_DFB,
            .entry_size = single_tile_size,
            .num_entries = per_tensor_tiles,
            .data_format_metadata = cb_data_format,
            .borrowed_from = OUTPUT,
        };

        // One kernel source, two instances over the same grid: they split each core's head range
        // between the two DM RISCs. Both instances raw-touch both DFBs, so the endpoint roles below
        // are assigned to satisfy the one-producer-one-consumer invariant; on Gen1 the DFB lowers to
        // a plain circular buffer whose interface both RISCs share, so the label drives no
        // device-side behaviour and the CONSUMER-bound instance still writes its own head range.
        const KernelSpec::CompileTimeArgs sharded_compile_time_args{
            {"in0_h_tiles", in0_h_tiles},
            {"head_dim_size_bytes", in0_w_tiles * single_tile_size},
            {"out_row_size_bytes", num_blocks_per_core_group_1 * in0_w_tiles * single_tile_size},
            {"block_size", num_blocks_per_core_group_1 * in0_HtWt},
        };
        const KernelSpec::RuntimeArgSchema sharded_runtime_arg_schema{
            .runtime_arg_names = {"nheads", "start_read_offset_bytes", "start_write_offset_bytes"},
        };

        const KernelSpec reader{
            .unique_id = READER,
            .source = kNlpConcatHeadsShardedSource,
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = SRC0_DFB,
                     .accessor_name = "in0",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 },
                 DFBBinding{
                     .dfb_spec_name = OUT_DFB,
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 }},
            .compile_time_args = sharded_compile_time_args,
            .runtime_arg_schema = sharded_runtime_arg_schema,
            .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        };
        const KernelSpec writer{
            .unique_id = WRITER,
            .source = kNlpConcatHeadsShardedSource,
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = SRC0_DFB,
                     .accessor_name = "in0",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = OUT_DFB,
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 }},
            .compile_time_args = sharded_compile_time_args,
            .runtime_arg_schema = sharded_runtime_arg_schema,
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        };

        uint32_t nheads_first_risc = div_up(num_blocks_per_core_group_1, 2);
        uint32_t nheads_second_risc = num_blocks_per_core_group_1 - nheads_first_risc;
        KernelRunArgs reader_run_args{.kernel = READER};
        KernelRunArgs writer_run_args{.kernel = WRITER};
        // Mirror SetRuntimeArgs(program, kernel, all_cores, args) by emplacing the same
        // per-core args on every logical core in the sharded range set.
        for (const auto& core : corerange_to_cores(all_cores, num_cores, /*row_wise=*/true)) {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {
                    {"nheads", nheads_first_risc},
                    {"start_read_offset_bytes", 0u},
                    {"start_write_offset_bytes", 0u},
                });
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {
                    {"nheads", nheads_second_risc},
                    {"start_read_offset_bytes", nheads_first_risc * in0_HtWt * single_tile_size},
                    {"start_write_offset_bytes", nheads_first_risc * in0_w_tiles * single_tile_size},
                });
        }

        ProgramSpec spec{
            .name = "nlp_concat_heads_sharded",
            .kernels = {reader, writer},
            .dataflow_buffers = {src0_dfb, out_dfb},
            .tensor_parameters = {input_param, output_param},
            .work_units = {WorkUnitSpec{
                .name = "nlp_concat_heads_sharded",
                .kernels = {READER, WRITER},
                .target_nodes = all_cores,
            }},
        };

        ProgramRunArgs run_args;
        run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
        run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

        return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
    }

    uint32_t num_cores = 0, num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0;
    CoreRangeSet all_cores = CoreRangeSet(), core_group_1 = CoreRangeSet(), core_group_2 = CoreRangeSet();
    std::tie(
        num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2) =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);
    uint32_t g1_numcores = core_group_1.num_cores();

    // Double-buffered: the reader fills one tensor's worth of tiles while the writer drains the other.
    uint32_t cb_src0_num_tiles = per_tensor_tiles * 2;
    const DataflowBufferSpec src0_dfb{
        .unique_id = SRC0_DFB,
        .entry_size = single_tile_size,
        .num_entries = cb_src0_num_tiles,
        .data_format_metadata = cb_data_format,
    };

    const KernelSpec reader{
        .unique_id = READER,
        .source = kNlpConcatHeadsReaderSource,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC0_DFB,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"in0_h_tiles", in0_h_tiles}, {"in0_w_tiles", in0_w_tiles}, {"in0_c", in0_c}, {"in0_HtWt", in0_HtWt}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_blocks", "in0_h_dim", "in0_tensor_tile_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };
    const KernelSpec writer{
        .unique_id = WRITER,
        .source = kNlpConcatHeadsWriterSource,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC0_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // Column-major core order, matching split_work_to_cores' own default assignment order.
    const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, /*row_wise=*/false);
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        uint32_t num_blocks_per_core = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        uint32_t in0_h_dim = num_blocks_written % in0_h_tiles;
        uint32_t in0_tensor_tile_id = (num_blocks_written / in0_h_tiles * in0_CHtWt) + (in0_h_dim * in0_w_tiles);

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {
                {"num_blocks", num_blocks_per_core},
                {"in0_h_dim", in0_h_dim},
                {"in0_tensor_tile_id", in0_tensor_tile_id},
            });
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"num_pages", num_blocks_per_core * per_tensor_tiles},
                {"start_id", num_blocks_written * per_tensor_tiles},
            });
        num_blocks_written += num_blocks_per_core;
    }

    ProgramSpec spec{
        .name = "nlp_concat_heads",
        .kernels = {reader, writer},
        .dataflow_buffers = {src0_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "nlp_concat_heads",
            .kernels = {READER, WRITER},
            .target_nodes = all_cores,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
