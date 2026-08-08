// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_sharded_program_factory.hpp"

#include <cmath>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TilizeWithValPaddingMultiCoreShardedFactory::create_program_artifacts(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor, Tensor& tensor_return_value) {
    const Tensor& a = input_tensor;
    const Tensor& output = tensor_return_value;
    auto pad_value = operation_attributes.pad_value;
    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    auto input_shard_spec = a.shard_spec().value();
    auto output_shard_spec = output.shard_spec().value();

    auto all_cores = output_shard_spec.grid;

    uint32_t num_batches = output.physical_volume() / (output.padded_shape()[-2] * output.padded_shape()[-1]);

    uint32_t num_input_rows = input_shard_spec.shape[0];
    uint32_t input_shard_width_bytes = input_shard_spec.shape[1] * a.element_size();
    uint32_t ntiles_per_core = output_shard_spec.shape[0] * output_shard_spec.shape[1] / TILE_HW;
    uint32_t ntiles_per_batch = ntiles_per_core / num_batches;
    uint32_t ntiles_per_block = output_shard_spec.shape[1] / TILE_WIDTH;
    uint32_t nblocks_per_core = output_shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t num_padded_rows = output.padded_shape()[-2] - a.padded_shape()[-2];

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    // ---------------------------------------------------------------------
    // Program-scope resource names (typed handles → generated dfb:: / tensor:: tokens)
    // ---------------------------------------------------------------------
    const DFBSpecName SRC_SHARD{"src_shard"};  // legacy src0 CB (c_1): the input shard, borrowed
    const DFBSpecName STAGE{"stage"};          // legacy src1 CB (c_0): row-major staging for tilize
    const DFBSpecName PAD{"pad"};              // legacy src2 CB (c_2): one row of pad value
    const DFBSpecName OUT_SHARD{"out_shard"};  // legacy output CB (c_16): the output shard, borrowed
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    ProgramSpec spec;
    spec.name = "tilize_with_val_padding_multi_core_sharded";

    // ---------------------------------------------------------------------
    // Tensor parameters. These carry no kernel TensorBinding: the sharded kernels never build a
    // TensorAccessor. They exist to back the two borrowed-memory DFBs below, whose L1 addresses are
    // resolved per enqueue from the corresponding TensorArgument (which is also what patches them on
    // a program-cache hit — the job the legacy CBDescriptor::buffer assignment did). Each is declared
    // under the same condition that gated the legacy `cb.buffer = …` assignment, so a DFB that would
    // not have borrowed keeps its own L1 allocation and pulls in no tensor plumbing.
    // ---------------------------------------------------------------------
    if (src_sharded) {
        spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = a.tensor_spec()});
    }
    if (out_sharded) {
        spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});
    }

    // ---------------------------------------------------------------------
    // DataflowBufferSpecs (replaces the legacy c_1 / c_0 / c_2 / c_16 CBDescriptors)
    // ---------------------------------------------------------------------
    // Sharded input DFB — built on the input buffer's borrowed memory.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SRC_SHARD,
        .entry_size = input_shard_width_bytes,
        .num_entries = num_input_rows,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = src_sharded ? std::optional<TensorParamName>{INPUT} : std::nullopt,
    });

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = STAGE,
        .entry_size = input_single_tile_size,
        .num_entries = ntiles_per_batch * 2,
        .data_format_metadata = input_cb_data_format,
    });

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = PAD,
        .entry_size = input_shard_width_bytes,
        .num_entries = 1,
        .data_format_metadata = input_cb_data_format,
    });

    // Sharded output DFB — built on the output buffer's borrowed memory.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_SHARD,
        .entry_size = output_single_tile_size,
        .num_entries = ntiles_per_core,
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = out_sharded ? std::optional<TensorParamName>{OUTPUT} : std::nullopt,
    });

    /** reader
     */
    // SRC_SHARD and PAD are each touched by the reader alone — it reserves them and peeks a pointer,
    // with no second kernel on the other end of the FIFO — so each is self-looped: the reader binds
    // both the producer and the consumer endpoint. STAGE is a normal 1P+1C FIFO into the compute
    // kernel.
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
                  "reader_unary_pad_height_width_sharded.cpp",
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = SRC_SHARD,
                 .accessor_name = "in0",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = SRC_SHARD,
                 .accessor_name = "in0",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = STAGE,
                 .accessor_name = "in1",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = PAD,
                 .accessor_name = "pad",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = PAD,
                 .accessor_name = "pad",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             }},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_input_rows",
                  "input_width_bytes",
                  "input_block_size",
                  "num_padded_tiles_per_batch",
                  "num_padded_rows",
                  "num_batches",
                  "packed_pad_value"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
    });

    /** writer
     */
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                  "writer_unary_sharded_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_SHARD,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
    });

    /** compute
     */
    // Legacy ComputeConfigDescriptor set only fp32_dest_acc_en and unpack_to_dest_mode; every other
    // field stayed at its default, which ComputeGen1Config reproduces exactly. The legacy
    // unpack_to_dest_mode vector was Default everywhere except v[c_0] = UnpackToDestFp32 when
    // fp32_llk_acc — c_0 is this factory's staging CB, i.e. the tilize input DFB (Default ==
    // UnpackToSrc is expressed by omitting the entry).
    ComputeGen1Config compute_gen1{.enable_32_bit_dest = fp32_llk_acc};
    if (fp32_llk_acc) {
        compute_gen1.unpack_modes = ComputeUnpackModes{{STAGE, UnpackMode::UnpackToDest}};
    }
    ComputeHardwareConfig compute_hw{std::move(compute_gen1)};

    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp",
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = STAGE,
                 .accessor_name = "in",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = OUT_SHARD,
                 .accessor_name = "out",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .compile_time_args =
            {{"per_core_block_cnt", static_cast<uint32_t>(nblocks_per_core)},
             {"per_core_block_tile_cnt", static_cast<uint32_t>(ntiles_per_block)}},
        .hw_config = compute_hw,
    });

    spec.work_units.push_back(
        WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores});

    uint32_t packed_pad_value = detail::get_packed_value(a, pad_value);

    // Sharded readers/writers: the DFBs themselves carry the buffer bindings, so no buffer address is
    // needed in runtime args.
    ProgramRunArgs run_args;
    KernelRunArgs reader_ra{.kernel = READER};
    KernelRunArgs writer_ra{.kernel = WRITER};
    for (const auto& core : corerange_to_cores(all_cores)) {
        AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values,
            core,
            {{"num_input_rows", num_input_rows},
             {"input_width_bytes", input_shard_width_bytes},
             {"input_block_size", (num_input_rows / num_batches) * input_shard_width_bytes},
             {"num_padded_tiles_per_batch", ntiles_per_batch},
             {"num_padded_rows", num_padded_rows},
             {"num_batches", num_batches},
             {"packed_pad_value", packed_pad_value}});
        AddRuntimeArgsForNode(writer_ra.runtime_arg_values, core, {{"num_units", ntiles_per_core}});
    }

    run_args.kernel_run_args.push_back(std::move(reader_ra));
    run_args.kernel_run_args.push_back(std::move(writer_ra));

    if (src_sharded) {
        run_args.tensor_args.emplace(INPUT, TensorArgument{a.mesh_tensor()});
    }
    if (out_sharded) {
        run_args.tensor_args.emplace(OUTPUT, TensorArgument{output.mesh_tensor()});
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
