// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_sharded_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts UntilizeWithUnpaddingMultiCoreShardedProgramFactory::create_program_spec(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();
    // Special handling for tensors of W=16 and H%32==0
    // In this case skip untilizing on compute and in writer kernel just copy face0 and face2,
    // and skip face1 and face3.
    bool unpad_tensor_w_16 = output.padded_shape()[-1] == 16 && output.padded_shape()[-2] % TILE_HEIGHT == 0;
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t num_rows_block = 0, block_row_size = 0, last_block_row_size_unpadded = 0, num_output_rows_unpadded = 0;
    // output_row_size is only consumed by the (now-dropped) legacy config-C writer CTA; the Metal 2.0
    // writer never reads it, so split it out of the multi-var decl and mark it unused to keep -Werror happy.
    [[maybe_unused]] uint32_t output_row_size = 0;
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
    block_row_size = out_shard_spec.shape[1] * output.element_size();     // in0_block_w * TILE_WIDTH * dtype_nbytes
    output_row_size = output.padded_shape()[-1] * output.element_size();  // output row size bytes
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

    // ---- Resource names ----
    const DFBSpecName IN{"in"};
    const DFBSpecName OUT{"out"};
    const DFBSpecName SHARDED_OUT{"sharded_out"};
    const TensorParamName SRC{"src"};
    const TensorParamName DST{"dst"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // ---- Dataflow buffers ----
    // IN (legacy CB c_0): sharded input lives in L1, so borrow the input buffer iff src_sharded
    // (the reader merely publishes the resident tiles via push_back — there is no NOC read). The
    // borrowed backing L1 address resolves at runtime from the SRC TensorArgument.
    DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = src_sharded ? std::optional<TensorParamName>{SRC} : std::nullopt,
    };
    // OUT (legacy CB c_16): plain compute-output DFB (produced by compute, consumed by writer).
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_cb_data_format,
    };
    // SHARDED_OUT (legacy CB c_17): present only when out_sharded. Borrowed onto the output buffer's
    // L1 — the writer advances the write pointer by aligned_page_size (which may exceed block_row_size
    // due to buffer alignment padding), so the DFB page size must match to avoid overflow. The backing
    // L1 address resolves at runtime from the DST TensorArgument.
    DataflowBufferSpec sharded_out_dfb{
        .unique_id = SHARDED_OUT,
        .entry_size = aligned_page_size,
        .num_entries = num_output_rows_unpadded,
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = DST,
    };

    // ---- Tensor parameters ----
    TensorParameter src_param{.unique_id = SRC, .spec = input.tensor_spec()};
    TensorParameter dst_param{.unique_id = DST, .spec = output.tensor_spec()};

    // ---- Reader kernel (sharded input; forked m2 copy of the shared eltwise/unary reader) ----
    // The sharded input lives in L1, so the reader just advances the IN write pointer (push_back) to
    // publish the resident tiles to the compute consumer — IN PRODUCER / compute IN CONSUMER is a real pair.
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
            "reader_unary_sharded_m2.cpp",
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // ---- Writer kernel (one of three sources, selected at host time) ----
    // Config A: out_sharded && !unpad_tensor_w_16 -> batch-rows sharded writer (untilize compute output)
    // Config B: out_sharded &&  unpad_tensor_w_16 -> width-16 sharded writer (copy compute output)
    // Config C: !out_sharded                      -> interleaved stick-layout writer (writes to interleaved DST)
    KernelSpec writer{
        .unique_id = WRITER,
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };
    if (out_sharded) {
        // OUT consumed here; SHARDED_OUT is a one-ended fake CB (the writer is the only kernel touching
        // it — reserve_back / get_write_ptr / push_back). It IS the sharded output buffer in L1; nothing
        // produces or consumes it as a FIFO. To satisfy the DFB ">=1 PRODUCER and >=1 CONSUMER" validator
        // rule, bind it as a SELF-LOOP on the writer (PRODUCER + CONSUMER, shared accessor_name). This is
        // an INTERIM validator-satisfying device, NOT a real FIFO; to be replaced by the forthcoming
        // "local" TensorAccessor variant.
        writer.dfb_bindings = {
            DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = SHARDED_OUT,
                .accessor_name = "sharded_out",
                .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = SHARDED_OUT,
                .accessor_name = "sharded_out",
                .endpoint_type = DFBEndpointType::CONSUMER},
        };
        if (unpad_tensor_w_16) {
            // Config B: width-16 writer. No CTA (it reads get_tile_size(dfb::sharded_out)).
            writer.source =
                "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                "writer_unary_unpad_width_16_sharded.cpp";
            writer.runtime_arg_schema = {
                .runtime_arg_names = {"num_unpadded_output_rows", "num_padded_tiles_per_core"}};
        } else {
            // Config A: batch-rows writer.
            writer.source =
                "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                "writer_unary_unpad_batch_rows_sharded.cpp";
            writer.compile_time_args = {
                {"aligned_page_size", aligned_page_size},
            };
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
        // Config C: interleaved stick-layout writer. dst addr -> ta::dst (the dst_addr RTA disappears).
        writer.source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_stick_layout_interleaved_blocks_m2.cpp";
        writer.dfb_bindings = {
            DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
        };
        writer.tensor_bindings = {
            TensorBinding{.tensor_parameter_name = DST, .accessor_name = "dst"},
        };
        writer.compile_time_args = {
            {"float32_dtype",
             (std::uint32_t)(input_cb_data_format == tt::DataFormat::Float32 or
                             input_cb_data_format == tt::DataFormat::UInt32 or
                             input_cb_data_format == tt::DataFormat::Int32)},
        };
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

    // ---- Compute kernel (untilize, or eltwise_copy for the unpad-width-16 path) ----
    KernelSpec::CompilerOptions::Defines compute_kernel_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines.insert({"DST_ACCUM_MODE", "1"});
    }
    ComputeHardwareConfig compute_hw_config{.fp32_dest_acc_en = fp32_dest_acc_en};
    if (fp32_dest_acc_en) {
        compute_hw_config.unpack_to_dest_mode.insert({IN, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32});
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .compiler_options = {.defines = std::move(compute_kernel_defines)},
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .hw_config = std::move(compute_hw_config),
    };
    if (unpad_tensor_w_16) {
        // Use copy compute kernel just for a potential data type conversion.
        compute.source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/compute/eltwise_copy_m2.cpp";
        compute.compile_time_args = {
            {"per_core_tile_cnt", num_input_tiles},
        };
    } else {
        compute.source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/compute/untilize_m2.cpp";
        compute.compile_time_args = {
            {"per_core_block_cnt", nblocks_per_core},
            {"per_core_block_tile_cnt", ntiles_per_block},
        };
    }

    // Runtime args: legacy code uses SetRuntimeArgs(program, kernel, all_cores, args) which broadcasts
    // the same args to every core. Enumerate cores and emit one per-node entry per core (the !out_sharded
    // writer args genuinely vary per core).
    const std::vector<CoreCoord> all_core_coords = corerange_to_cores(all_cores, std::nullopt, row_major);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    reader_run_args.runtime_arg_values.reserve(all_core_coords.size());
    writer_run_args.runtime_arg_values.reserve(all_core_coords.size());

    const uint32_t reader_num_tiles_per_core = ntiles_per_block * nblocks_per_core;
    for (const auto& core : all_core_coords) {
        reader_run_args.runtime_arg_values.push_back(
            KernelRunArgs::NodeRuntimeArgs{.node = core, .args = {{"num_tiles_per_core", reader_num_tiles_per_core}}});
    }

    if (out_sharded) {
        writer_run_args.runtime_arg_values.reserve(all_core_coords.size());
        for (const auto& core : all_core_coords) {
            if (unpad_tensor_w_16) {
                writer_run_args.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
                    .node = core,
                    .args = {
                        {"num_unpadded_output_rows", num_output_rows_unpadded},
                        {"num_padded_tiles_per_core", num_input_tiles}}});
            } else {
                writer_run_args.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
                    .node = core,
                    .args = {
                        {"num_unpadded_output_rows", num_output_rows_unpadded},
                        {"num_padded_tiles_per_batch", ntiles_per_batch},
                        {"num_unpadded_rows_per_batch", out_shard_spec.shape[0] / batch},
                        {"padded_block_row_size_bytes", shard_spec.shape[1] * output.element_size()},
                        {"unpadded_block_row_size_bytes", block_row_size},
                        {"batch", batch}}});
            }
        }
    } else {
        writer_run_args.runtime_arg_values.reserve(all_core_coords.size());
        for (uint32_t i = 0; i < all_core_coords.size(); ++i) {
            CoreCoord core = all_core_coords[i];

            // writer runtime args
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

            // The three constant-1 RTAs (legacy slots 3/4/5) map to batch / num_blocks_h / num_blocks_w.
            writer_run_args.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
                .node = core,
                .args = {
                    {"num_rows_block", num_rows_block},
                    {"block_row_size", block_row_size},
                    {"batch", std::uint32_t{1}},
                    {"num_blocks_h", std::uint32_t{1}},
                    {"num_blocks_w", std::uint32_t{1}},
                    {"last_block_row_size_unpadded", row_size_unpadded},
                    {"num_output_rows_unpadded", num_rows_unpadded},
                    {"block_start_row_id", block_start_row_id_offset},
                    {"block_start_row_offset", block_start_row_offset}}});
        }
    }

    // ---- Work unit ----
    // All kernels run on all_cores, so every local DFB's producer and consumer share the SAME
    // WorkUnitSpec — the local-DFB invariant holds trivially.
    WorkUnitSpec work_unit{.name = "uwu_sharded", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores};

    // Dataflow buffers: IN, OUT always; SHARDED_OUT only when out_sharded.
    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(std::move(in_dfb));
    dataflow_buffers.push_back(std::move(out_dfb));
    if (out_sharded) {
        dataflow_buffers.push_back(std::move(sharded_out_dfb));
    }

    ProgramSpec spec{
        .name = "untilize_with_unpadding_multi_core_sharded",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = {std::move(src_param), std::move(dst_param)},
        .work_units = {std::move(work_unit)},
    };

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)},
        .tensor_args =
            {
                {SRC, TensorArgument{input.mesh_tensor()}},
                {DST, TensorArgument{output.mesh_tensor()}},
            },
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
