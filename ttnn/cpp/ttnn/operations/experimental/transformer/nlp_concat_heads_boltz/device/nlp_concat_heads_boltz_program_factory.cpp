// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "nlp_concat_heads_boltz_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts
NLPConcatHeadsBoltzDeviceOperation::NLPConcatHeadsBoltzProgramFactory::create_program_artifacts(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;

    // The Metal 2.0 binding layer works with the Metalium tensor type; extract once at entry.
    const auto& input_mesh_tensor = tensor_args.input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();
    CoreCoord compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();

    const auto& ashape = a.padded_shape();

    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(data_format);
    bool in_sharded = a.is_sharded();
    bool out_sharded = output.is_sharded();
    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_tensor_tiles = ashape[0] * ashape[3] / TILE_WIDTH;  // 142

    // Per output tensor args
    // Output shape is: [B, 1, s, 4544]
    uint32_t in0_h_tiles = ashape[1] * ashape[2] / TILE_HEIGHT;
    uint32_t in0_w_tiles = ashape[3] / TILE_WIDTH;    // head_dim
    uint32_t in0_c = per_tensor_tiles / in0_w_tiles;  // num_heads
    uint32_t in0_HtWt = in0_h_tiles * in0_w_tiles;
    uint32_t in0_CHtWt = in0_c * in0_HtWt;

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of per_tensor_tiles per core
    uint32_t num_blocks = ashape[1] * ashape[2] / TILE_HEIGHT;
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
    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName IN0_DFB{"in0"};
    const DFBSpecName OUT0_DFB{"out0"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    KernelSpec reader_spec;
    KernelSpec writer_spec;
    if (in_sharded) {
        KernelSpec::CompileTimeArgs compile_time_args = {
            {"in0_h_tiles", in0_h_tiles},
            {"head_dim_size_bytes", in0_w_tiles * single_tile_size},
            {"out_row_size_bytes", num_blocks_per_core_group_1 * in0_w_tiles * single_tile_size},
            {"block_size", num_blocks_per_core_group_1 * in0_HtWt},
        };
        // Both instances touch both borrowed DFBs: each reserves the whole buffer once (nothing is ever
        // pushed or consumed) and then reads / writes through raw pointer peeks at disjoint offsets. The
        // PRODUCER / CONSUMER labels split the two instances 1P+1C to satisfy the one-producer +
        // one-consumer spec invariant; on WH/BH they drive no FIFO machinery the kernel relies on.
        Group<DFBBinding> reader_dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0_DFB,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }};
        Group<DFBBinding> writer_dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0_DFB,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }};
        if (out_sharded) {
            reader_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = OUT0_DFB,
                .accessor_name = "out0",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            writer_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = OUT0_DFB,
                .accessor_name = "out0",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }

        // Both RISCVs run the same sharded kernel, split by the per-core runtime args below.
        reader_spec = KernelSpec{
            .unique_id = READER,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_boltz/device/kernels/dataflow/"
                "reader_tm_tile_layout_nlp_concat_heads_boltz_sharded.cpp",
            .dfb_bindings = std::move(reader_dfb_bindings),
            .compile_time_args = compile_time_args,
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"nheads", "start_read_offset_bytes", "start_write_offset_bytes"},
                },
            .hw_config = create_reader_datamovement_config(),
        };
        writer_spec = KernelSpec{
            .unique_id = WRITER,
            .source = reader_spec.source,
            .dfb_bindings = std::move(writer_dfb_bindings),
            .compile_time_args = std::move(compile_time_args),
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"nheads", "start_read_offset_bytes", "start_write_offset_bytes"},
                },
            .hw_config = create_writer_datamovement_config(),
        };
    } else {
        reader_spec = KernelSpec{
            .unique_id = READER,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_boltz/device/kernels/dataflow/"
                "reader_tm_tile_layout_nlp_concat_heads_boltz.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = IN0_DFB,
                .accessor_name = "in0",
                .endpoint_type = DFBEndpointType::PRODUCER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = INPUT,
                .accessor_name = "in0_tensor",
            }},
            .compile_time_args =
                {
                    {"in0_h_tiles", in0_h_tiles},
                    {"in0_w_tiles", in0_w_tiles},
                    {"in0_c", in0_c},
                    {"in0_HtWt", in0_HtWt},
                },
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"num_blocks", "in0_h_dim", "in0_tensor_tile_id"},
                },
            .hw_config = create_reader_datamovement_config(),
        };
        // The interleaved writer is the shared Metal 2.0 fork of writer_unary_interleaved_start_id.cpp; its
        // binding vocabulary (dfb::out, tensor::dst, args num_pages / start_id) is the fork's interface and
        // this factory conforms to it.
        writer_spec = KernelSpec{
            .unique_id = WRITER,
            .source =
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                "writer_unary_interleaved_start_id_metal2.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = IN0_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = OUTPUT,
                .accessor_name = "dst",
            }},
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"num_pages", "start_id"},
                },
            .hw_config = create_writer_datamovement_config(),
        };
    }

    // Create dataflow buffers.  A sharded input/output is reached through a DFB borrowed from its
    // tensor: naming the tensor (never an address) is what lets the framework re-peg it on a cache hit.
    uint32_t in0_dfb_num_entries = per_tensor_tiles;
    if (!in_sharded) {
        in0_dfb_num_entries *= 2;  // double buffer
    }
    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0_DFB,
        .entry_size = single_tile_size,
        .num_entries = in0_dfb_num_entries,
        .data_format_metadata = data_format,
        .borrowed_from = in_sharded ? std::optional<TensorParamName>(INPUT) : std::nullopt,
    });

    // The output DFB is only touched by the sharded kernel; with an interleaved input a sharded output is
    // written through the writer's tensor binding, so no DFB is allocated for it.
    if (in_sharded && out_sharded) {
        uint32_t out0_dfb_num_entries = per_tensor_tiles;
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT0_DFB,
            .entry_size = single_tile_size,
            .num_entries = out0_dfb_num_entries,
            .data_format_metadata = data_format,
            .borrowed_from = OUTPUT,
        });
    }

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    if (in_sharded) {
        // Every core gets the same args (the legacy code set them over the whole CoreRangeSet);
        // the two halves of the head range are split across the two RISCVs.
        uint32_t nheads_first_risc = div_up(num_blocks_per_core_group_1, 2);
        uint32_t nheads_second_risc = num_blocks_per_core_group_1 - nheads_first_risc;
        for (const CoreCoord& core : corerange_to_cores(all_cores, std::nullopt, row_major)) {
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
    } else {
        const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);
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
    }

    ProgramSpec spec{
        .name = "nlp_concat_heads_boltz",
        .kernels = {std::move(reader_spec), std::move(writer_spec)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units = {WorkUnitSpec{
            .name = "nlp_concat_heads_boltz",
            .kernels = {READER, WRITER},
            .target_nodes = all_cores,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, input_mesh_tensor},
        {OUTPUT, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
