// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_sharded_program_factory.hpp"

#include <cmath>
#include <filesystem>
#include <vector>

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts UntilizeWithUnpaddingMultiCoreShardedProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    // ---- Resource names ----
    const DFBSpecName IN_DFB{"in"};                    // legacy c_0 (borrowed: resident input shard)
    const DFBSpecName OUT_DFB{"out"};                  // legacy c_16 (untilized output, regular FIFO)
    const DFBSpecName OUT_SHARDED_DFB{"out_sharded"};  // legacy c_17 (borrowed: resident output shard)
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();
    // Special handling for tensors of W=16 and H%32==0
    // In this case skip untilizing on compute and in writer kernel just copy face0 and face2,
    // and skip face1 and face3.
    bool unpad_tensor_w_16 = output.padded_shape()[-1] == 16 && output.padded_shape()[-2] % TILE_HEIGHT == 0;
    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t num_rows_block = 0, block_row_size = 0, last_block_row_size_unpadded = 0, num_output_rows_unpadded = 0;
    CoreCoord end_core;
    uint32_t last_idx = 0;
    auto shard_spec = a.shard_spec().value();

    // I am not sure it is correct to ever use the shard_spec here.
    auto out_shard_spec = output.shard_spec().has_value() ? output.shard_spec().value() : shard_spec;

    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto all_cores = shard_spec.grid;
    uint32_t ntiles_per_block = shard_spec.shape[1] / TILE_WIDTH;
    uint32_t nblocks_per_core = shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t global_batch = a.physical_volume() / (a.padded_shape()[-2] * a.padded_shape()[-1]);
    uint32_t batch =
        a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED
            ? std::max(1u, (shard_spec.shape[0] * shard_spec.shape[1]) / (a.padded_shape()[-2] * a.padded_shape()[-1]))
            : global_batch;
    uint32_t ntiles_per_batch = ntiles_per_block * nblocks_per_core / batch;

    num_rows_block = out_shard_spec.shape[0];
    block_row_size = out_shard_spec.shape[1] * output.element_size();  // in0_block_w * TILE_WIDTH * dtype_nbytes
    last_block_row_size_unpadded = block_row_size - (tt::round_up(output.padded_shape()[-1], out_shard_spec.shape[1]) -
                                                     output.padded_shape()[-1]) *
                                                        output.element_size();
    uint32_t num_output_rows = output.physical_volume() / output.padded_shape()[-1];
    num_output_rows_unpadded =
        num_rows_block - (tt::round_up(num_output_rows, out_shard_spec.shape[0]) - num_output_rows);
    if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        last_idx = tt::div_up(output.padded_shape()[-1], out_shard_spec.shape[1]) - 1;
    } else if (a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        last_idx = tt::div_up(num_output_rows, out_shard_spec.shape[0]) - 1;
    } else {
        end_core = {
            tt::div_up(output.padded_shape()[-1], out_shard_spec.shape[1]) - 1,
            tt::div_up(num_output_rows, out_shard_spec.shape[0]) - 1};
    }
    if (!row_major) {
        std::swap(end_core.x, end_core.y);
    }

    uint32_t num_input_tiles = ntiles_per_block * nblocks_per_core;
    uint32_t num_output_tiles = out_sharded ? (unpad_tensor_w_16 ? 16 : ntiles_per_batch * 2) : ntiles_per_block * 2;
    uint32_t aligned_page_size = static_cast<uint32_t>(output.buffer()->aligned_page_size());

    bool float32_dtype = input_cb_data_format == tt::DataFormat::Float32 ||
                         input_cb_data_format == tt::DataFormat::UInt32 ||
                         input_cb_data_format == tt::DataFormat::Int32;

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    // ------------------------------------------------------------------------
    // Dataflow buffers.
    //  - IN (c_0): borrowed from the resident input shard (reader does a fake-push; compute
    //    consumes). Legacy globally allocated this CB onto a.buffer().
    //  - OUT (c_16): regular FIFO holding the untilized output (compute produces; writer consumes).
    //  - OUT_SHARDED (c_17): out-sharded only — borrowed from the resident output shard. The writer
    //    is its only endpoint (producer), so it is bound as a producer/consumer self-loop to satisfy
    //    the DFB topology invariant (one producer + one consumer per node).
    // ------------------------------------------------------------------------
    DataflowBufferSpec in_dfb{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_cb_data_format,
        // This factory is only selected for sharded input, so src is always resident in L1.
        .borrowed_from = src_sharded ? std::optional<TensorParamName>{INPUT} : std::nullopt,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_cb_data_format,
    };
    DataflowBufferSpec out_sharded_dfb{
        .unique_id = OUT_SHARDED_DFB,
        .entry_size = aligned_page_size,
        .num_entries = num_output_rows_unpadded,
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = OUTPUT,
    };

    // ---- Tensor parameters ----
    TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ------------------------------------------------------------------------
    // Reader: sharded fake-push of the resident input into IN.
    // ------------------------------------------------------------------------
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/dataflow/"
            "reader_unary_sharded.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}},
        .hw_config =
            ttnn::create_reader_datamovement_config(a.device()->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // ------------------------------------------------------------------------
    // Writer: three runtime-selected variants.
    //  - out-sharded, W==16 -> writer_unary_unpad_width_16_sharded
    //  - out-sharded        -> writer_unary_unpad_batch_rows_sharded
    //  - interleaved out    -> writer_unary_stick_layout_interleaved_blocks (OUTPUT TensorAccessor)
    // ------------------------------------------------------------------------
    KernelSpec writer{
        .unique_id = WRITER,
        .hw_config =
            ttnn::create_writer_datamovement_config(a.device()->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };
    if (out_sharded) {
        // Both out-sharded writers consume OUT (untilized tiles) and produce the resident OUT_SHARDED
        // shard. OUT_SHARDED has no real consumer, so we add a self-loop consumer binding.
        writer.dfb_bindings = {
            DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = OUT_SHARDED_DFB,
                .accessor_name = "out_sharded",
                .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = OUT_SHARDED_DFB,
                .accessor_name = "out_sharded",
                .endpoint_type = DFBEndpointType::CONSUMER},
        };
        if (unpad_tensor_w_16) {
            writer.source = std::filesystem::path(
                "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/dataflow/"
                "writer_unary_unpad_width_16_sharded.cpp");
            writer.compile_time_args = {{"tile_size_in_bytes", output_single_tile_size}};
            writer.runtime_arg_schema = {
                .runtime_arg_names = {"num_unpadded_output_rows", "num_padded_tiles_per_core"}};
        } else {
            writer.source = std::filesystem::path(
                "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/dataflow/"
                "writer_unary_unpad_batch_rows_sharded.cpp");
            writer.compile_time_args = {{"aligned_page_size", aligned_page_size}};
            writer.runtime_arg_schema = {
                .runtime_arg_names = {
                    "num_unpadded_output_rows",
                    "num_padded_tiles_per_batch",
                    "num_unpadded_rows_per_batch",
                    "padded_block_row_size_bytes",
                    "unpadded_block_row_size_bytes",
                    "batch"}};
        }
    } else {
        writer.source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_stick_layout_interleaved_blocks.cpp");
        writer.dfb_bindings = {
            DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}};
        writer.tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}};
        writer.compile_time_args = {{"float32_dtype", float32_dtype ? 1u : 0u}};
        writer.runtime_arg_schema = {
            .runtime_arg_names = {
                "num_rows_block",
                "block_row_size",
                "batch",
                "num_blocks_h",
                "num_blocks_w",
                "last_block_row_size_unpadded",
                "num_output_rows_unpadded",
                "block_start_row_id",
                "block_start_row_offset"}};
    }

    // ------------------------------------------------------------------------
    // Compute: untilize (general) or eltwise_copy (W==16, dtype-convert only).
    // ------------------------------------------------------------------------
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }

    ttnn::ComputeKernelConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_dest_acc_en};
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(a.device()->arch(), compute_config);
    if (fp32_dest_acc_en) {
        std::visit(
            [&](auto& c) { c.unpack_modes.emplace(IN_DFB, tt::tt_metal::UnpackMode::UnpackToDest); }, compute_hw);
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .compiler_options = {.defines = std::move(compute_defines)},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .hw_config = compute_hw,
    };
    if (unpad_tensor_w_16) {
        compute.source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/compute/"
            "eltwise_copy.cpp");
        compute.compile_time_args = {{"per_core_tile_cnt", num_input_tiles}};
    } else {
        compute.source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/compute/"
            "untilize.cpp");
        compute.compile_time_args = {
            {"per_core_block_cnt", nblocks_per_core}, {"per_core_block_tile_cnt", ntiles_per_block}};
    }

    // ------------------------------------------------------------------------
    // Per-core runtime args. Legacy broadcast the same reader/writer args to every core via
    // SetRuntimeArgs(all_cores, ...) except the interleaved-out writer, which differs per core.
    // ------------------------------------------------------------------------
    const std::vector<CoreCoord> all_core_coords = corerange_to_cores(all_cores, std::nullopt, row_major);

    KernelRunArgs::RuntimeArgValues reader_node_args;
    KernelRunArgs::RuntimeArgValues writer_node_args;

    for (const auto& core : all_core_coords) {
        const NodeCoord node = core;
        reader_node_args["num_tiles_per_core"][node] = num_input_tiles;
    }

    if (out_sharded) {
        for (const auto& core : all_core_coords) {
            const NodeCoord node = core;
            if (unpad_tensor_w_16) {
                AddRuntimeArgsForNode(
                    writer_node_args,
                    node,
                    {
                        {"num_unpadded_output_rows", num_output_rows_unpadded},
                        {"num_padded_tiles_per_core", num_input_tiles},
                    });
            } else {
                AddRuntimeArgsForNode(
                    writer_node_args,
                    node,
                    {
                        {"num_unpadded_output_rows", num_output_rows_unpadded},
                        {"num_padded_tiles_per_batch", ntiles_per_batch},
                        {"num_unpadded_rows_per_batch", out_shard_spec.shape[0] / batch},
                        {"padded_block_row_size_bytes", shard_spec.shape[1] * output.element_size()},
                        {"unpadded_block_row_size_bytes", block_row_size},
                        {"batch", batch},
                    });
            }
        }
    } else {
        for (uint32_t i = 0; i < all_core_coords.size(); ++i) {
            const NodeCoord node = all_core_coords[i];

            uint32_t block_start_row_offset;
            uint32_t block_start_row_id_offset;
            uint32_t row_size_unpadded = block_row_size;
            uint32_t num_rows_unpadded = num_rows_block;
            if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
                block_start_row_offset = i * block_row_size;
                block_start_row_id_offset = 0;
                if (i > last_idx) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                } else {
                    num_rows_unpadded = num_output_rows_unpadded;
                    if (i == last_idx) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                }
            } else if (a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
                block_start_row_offset = 0;
                block_start_row_id_offset = i * num_rows_block;
                if (i > last_idx) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                } else {
                    if (i == last_idx) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                    row_size_unpadded = last_block_row_size_unpadded;
                }
            } else {
                const CoreCoord& core = all_core_coords[i];
                if (row_major) {
                    block_start_row_offset = core.x * block_row_size;
                    block_start_row_id_offset = core.y * num_rows_block;
                    if (core.x == end_core.x) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.y == end_core.y) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                } else {
                    block_start_row_offset = core.y * block_row_size;
                    block_start_row_id_offset = core.x * num_rows_block;
                    if (core.y == end_core.y) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.x == end_core.x) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                }
                if (core.x > end_core.x || core.y > end_core.y) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                }
            }

            // Legacy positional RTAs (dst_addr dropped -> carried by the OUTPUT TensorAccessor):
            //   {dst_addr, num_rows_block, block_row_size, 1, 1, 1, row_size_unpadded,
            //    num_rows_unpadded, block_start_row_id_offset, block_start_row_offset}
            AddRuntimeArgsForNode(
                writer_node_args,
                node,
                {
                    {"num_rows_block", num_rows_block},
                    {"block_row_size", block_row_size},
                    {"batch", 1u},
                    {"num_blocks_h", 1u},
                    {"num_blocks_w", 1u},
                    {"last_block_row_size_unpadded", row_size_unpadded},
                    {"num_output_rows_unpadded", num_rows_unpadded},
                    {"block_start_row_id", block_start_row_id_offset},
                    {"block_start_row_offset", block_start_row_offset},
                });
        }
    }

    Group<KernelSpec> kernels = {reader, writer, compute};

    Group<DataflowBufferSpec> dataflow_buffers = {in_dfb, out_dfb};
    if (out_sharded) {
        dataflow_buffers.push_back(out_sharded_dfb);
    }

    WorkUnitSpec wu{
        .name = "untilize_with_unpadding_multi_core_sharded",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = all_cores,
    };

    // ---- ProgramSpec ----
    ProgramSpec spec{
        .name = "untilize_with_unpadding_multi_core_sharded",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    // ---- ProgramRunArgs ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_args{.kernel = READER, .runtime_arg_values = std::move(reader_node_args)};
    KernelRunArgs writer_args{.kernel = WRITER, .runtime_arg_values = std::move(writer_node_args)};
    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args)};
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
