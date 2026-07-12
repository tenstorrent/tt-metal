// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write_tiled_sharded_input_program_factory.hpp"

#include <cstdint>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "slice_write_device_operation_types.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/experimental/padded_slice/device/padded_slice_utils.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace ttnn::operations::experimental::detail;

namespace ttnn::prim::qsr {

using namespace tt::tt_metal::experimental;

namespace {
SliceWriteRuntimeArgs get_slice_write_runtime_args_tiled_sharded_input(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const std::vector<CoreCoord>& cores) {
    auto* output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();
    auto actual_input_shape = input_tensor.logical_shape();
    for (uint32_t i = 0; i < actual_input_shape.rank(); i++) {
        actual_input_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }
    auto output_shape = output_tensor.padded_shape();
    log_debug(tt::LogOp, "Slice Write Output Shape: {}", output_shape);

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    auto shard_spec = input_tensor.shard_spec().value();
    auto input_cores = shard_spec.grid;

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    bool is_width_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    uint32_t num_cores_channels = get_num_cores_channels_from_sharded_tensor(input_tensor);

    std::uint32_t num_dims = static_cast<std::uint32_t>(actual_input_shape.rank());
    std::vector<uint32_t> num_output_tiles_per_dim(num_dims);
    std::vector<uint32_t> num_input_tiles_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_tiles_per_dim(num_dims);
    std::vector<uint32_t> accumulated_input_total_tiles_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);
    std::vector<uint32_t> size_till_end(num_dims);

    num_input_tiles_per_dim[0] = tt::div_up(actual_input_shape[-1], (TILE_WIDTH * num_cores_channels));
    num_input_tiles_per_dim[1] = tt::div_up(actual_input_shape[-2], TILE_HEIGHT);

    num_output_tiles_per_dim[0] = tt::div_up(output_shape[-1], TILE_WIDTH) - num_input_tiles_per_dim[0];
    num_output_tiles_per_dim[1] = tt::div_up(output_shape[-2], TILE_HEIGHT) - num_input_tiles_per_dim[1];
    num_output_tiles_per_dim[1] *= tt::div_up(output_shape[-1], TILE_WIDTH);

    uint32_t num_tiles_per_channel = num_input_tiles_per_dim[0];

    log_debug(
        tt::LogOp,
        "Output Start : {}, Output End : {}, Actual Input Shape : {}, \n Input Shape : {}, Output Shape : {}",
        output_tensor_start,
        output_tensor_end,
        actual_input_shape,
        input_shape,
        output_shape);

    accumulated_total_tiles_per_dim[0] = tt::div_up(output_shape[-1], TILE_WIDTH);
    accumulated_total_tiles_per_dim[1] = tt::div_up(output_shape[-2], TILE_HEIGHT) * accumulated_total_tiles_per_dim[0];

    uint32_t output_channel_tiles = accumulated_total_tiles_per_dim[0];
    accumulated_input_total_tiles_per_dim[0] = num_input_tiles_per_dim[0];
    accumulated_input_total_tiles_per_dim[1] = num_input_tiles_per_dim[1] * accumulated_input_total_tiles_per_dim[0];
    for (int32_t i = 2; i < num_dims; i++) {
        uint32_t num_unpadded_dim = actual_input_shape[-(i + 1)];
        uint32_t num_total_dim = output_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_tiles_per_dim[i - 1];
        num_input_tiles_per_dim[i] = num_unpadded_dim;
        num_output_tiles_per_dim[i] = num_padded_dim;
        accumulated_total_tiles_per_dim[i] = num_total_dim * accumulated_total_tiles_per_dim[i - 1];
        accumulated_input_total_tiles_per_dim[i] = num_unpadded_dim * accumulated_input_total_tiles_per_dim[i - 1];
    }

    log_debug(
        tt::LogOp,
        "Slice Write Input Tiles {}, Output Tiles {}, Acc Output Tiles {}, Acc Input Tiles {}",
        num_input_tiles_per_dim,
        num_output_tiles_per_dim,
        accumulated_total_tiles_per_dim,
        accumulated_input_total_tiles_per_dim);

    using namespace tt::tt_metal::experimental;
    TT_FATAL(
        output_tensor_start[-1] == 0,
        "slice_write expects output start for the last dimension to be 0. Got {}",
        output_tensor_start[-1]);

    log_debug(tt::LogOp, "Output Buffer address: {}", output_buffer->address());
    std::vector<uint32_t> common_writer_kernel_args = {
        output_buffer->address(),
        input_single_tile_size,
        input_single_tile_size,
        input_single_tile_size,
        num_dims,
        0,
        0,
        0,
        0};

    common_writer_kernel_args.insert(
        common_writer_kernel_args.end(), num_input_tiles_per_dim.begin(), num_input_tiles_per_dim.end());
    common_writer_kernel_args.insert(
        common_writer_kernel_args.end(), num_output_tiles_per_dim.begin(), num_output_tiles_per_dim.end());

    auto num_cores_total = cores.size();

    TT_FATAL(
        shard_spec.shape[0] % TILE_HEIGHT == 0,
        "Shard Height {} should be a multiple of tile height",
        shard_spec.shape[0]);
    const auto num_tiles_nhw_per_core = shard_spec.shape[0] / TILE_HEIGHT;

    SliceWriteRuntimeArgs ret_val(num_cores_total);

    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(output_tensor, output_tensor_start);
    uint32_t core_index = 0;
    for (const auto& core : cores) {
        uint32_t core_w_index = 0;
        uint32_t core_h_index = core_index;
        if (is_block_sharded) {
            core_w_index = rm_orientation ? core.x : core.y;
            core_h_index = rm_orientation ? core.y : core.x;
        } else if (is_width_sharded) {
            core_h_index = 0;
            core_w_index = core_index;
        }

        const uint32_t num_sticks_read = core_h_index * num_tiles_nhw_per_core;
        const uint32_t width_offset = core_w_index * num_tiles_per_channel;

        const uint32_t channels_tiles_this_core = std::min(output_channel_tiles - width_offset, num_tiles_per_channel);
        id_per_dim[0] = 0;
        uint32_t unpadded_written = num_sticks_read;
        uint32_t start_id = id_per_dim[0] + start_offset + width_offset;
        int max_num_tiles_this_core = 0;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_input_tiles_per_dim[j];
            if (j == num_dims - 1 && unpadded_written == num_input_tiles_per_dim[j]) {
                // Handle edge case where last dimension is completely written
                id_per_dim[j] = num_input_tiles_per_dim[j];
            }
            unpadded_written = unpadded_written / num_input_tiles_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_tiles_per_dim[j - 1];
            size_till_end[j] = num_input_tiles_per_dim[j] - id_per_dim[j] - ((j == 1) ? 0 : 1);
            max_num_tiles_this_core += size_till_end[j] * accumulated_input_total_tiles_per_dim[j - 1];
        }
        WriterKernelArgs writer_kernel_args = common_writer_kernel_args;

        uint32_t num_tiles_this_core = std::min<uint32_t>(
            num_tiles_nhw_per_core * num_tiles_per_channel, std::max<int>(max_num_tiles_this_core, 0));

        log_trace(
            tt::LogOp,
            "Start ID: {}, Start ID per dim : {} , Size till end : {}, Channel Tiles : {}, Max Tiles: {}, Num Tiles: "
            "{} for Core: {}",
            start_id,
            id_per_dim,
            size_till_end,
            channels_tiles_this_core,
            max_num_tiles_this_core,
            num_tiles_this_core,
            core);
        uint32_t addr_offset = 5;  // output buffer addr, output_row_size_bytes, input_row_size_bytes, num_dims
        writer_kernel_args[addr_offset++] = start_id;
        writer_kernel_args[addr_offset++] = num_tiles_this_core;
        writer_kernel_args[addr_offset++] = num_tiles_this_core;
        writer_kernel_args[addr_offset] = 1;
        writer_kernel_args.insert(writer_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());
        writer_kernel_args.push_back(num_tiles_per_channel - channels_tiles_this_core);

        ReaderKernelArgs reader_kernel_args = {num_tiles_this_core};
        ret_val[core_index] = {reader_kernel_args, writer_kernel_args};
        core_index++;
    }

    return ret_val;
}
}  // namespace

namespace {
const TensorParamName SWT_IN{"slice_write_tiled_in"};
const TensorParamName SWT_OUT{"slice_write_tiled_out"};
const DFBSpecName SWT_IN_DFB{"slice_write_tiled_in_dfb"};
const KernelSpecName SWT_READER{"slice_write_tiled_reader"};
const KernelSpecName SWT_WRITER{"slice_write_tiled_writer"};
}  // namespace

ttnn::device_operation::ProgramArtifacts SliceWriteTiledShardedInputProgramFactory::create_program_artifacts(
    const SliceWriteParams& operation_attributes, const SliceWriteInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& output_tensor_start = operation_attributes.slice_start;
    const auto& output_tensor_end = operation_attributes.slice_end;

    const auto& input_padded_shape = input.padded_shape();
    const auto output_shape = output.logical_shape();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    TT_FATAL(input.dtype() == output.dtype(), "slice_write input & output must have the same dtype");
    TT_FATAL(input_cb_data_format == output_cb_data_format, "slice_write input/output data formats must match");
    TT_FATAL(output_tensor_start[-1] == 0, "slice_write expects output start for the last dimension to be 0");
    TT_FATAL(
        output_tensor_start[-2] % TILE_HEIGHT == 0,
        "slice_write expects output start for the second-last dim to be a multiple of tile height");
    TT_FATAL(
        input_padded_shape[-2] % TILE_HEIGHT == 0,
        "slice_write expects input second-last dim to be a multiple of tile height");
    TT_FATAL(input.layout() == Layout::TILE, "slice_write (tiled) expects TILE input, got {}", input.layout());
    TT_FATAL(output.layout() == Layout::TILE, "slice_write (tiled) expects TILE output, got {}", output.layout());
    TT_FATAL(input.shard_spec().has_value(), "slice_write input must be sharded");

    auto shard_spec = input.shard_spec().value();
    CoreRangeSet input_cores = shard_spec.grid;
    const bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    TT_FATAL(
        shard_spec.shape[0] % TILE_HEIGHT == 0,
        "slice_write (tiled): shard height {} must be a multiple of tile height {}",
        shard_spec.shape[0],
        TILE_HEIGHT);

    const uint32_t num_tiles_height_per_core = shard_spec.shape[0] / TILE_HEIGHT;
    const uint32_t num_tiles_channel_per_core = shard_spec.shape[1] / TILE_WIDTH;
    const uint32_t num_cores_channels = get_num_cores_channels_from_sharded_tensor(input);
    const uint32_t num_dims = static_cast<uint32_t>(input.logical_shape().rank());

    // The width-unpadding path (skip padding-channel tiles) is not yet ported to the Metal-2 quasar writer.
    // The resnet stem does not hit it (channel tiles fill the output width exactly). Guard it explicitly.
    TT_FATAL(
        !(num_tiles_channel_per_core * TILE_WIDTH * num_cores_channels > static_cast<uint32_t>(output_shape[-1])),
        "Quasar tiled slice_write: width-unpadding (UNPAD_INPUT_WIDTH) path not yet ported.");

    std::vector<CoreCoord> iter_cores = corerange_to_cores(input_cores, std::nullopt, rm_orientation);

    // ---- ProgramSpec ----
    ProgramSpec spec;
    spec.name = "slice_write_tiled";
    spec.tensor_parameters = {
        TensorParameter{.unique_id = SWT_IN, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = SWT_OUT, .spec = output.tensor_spec()},
    };

    // INPUT DFB borrowed onto the resident tiled sharded shard (reader produces, writer drains to dst).
    DataflowBufferSpec in_dfb{
        .unique_id = SWT_IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_height_per_core * num_tiles_channel_per_core,
        .data_format_metadata = input_cb_data_format,
    };
    in_dfb.borrowed_from = SWT_IN;
    spec.dataflow_buffers.push_back(in_dfb);

    // Reader: mark the resident tiled input shard available (no data fetch).
    KernelSpec reader{
        .unique_id = SWT_READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice_write/device/kernels/dataflow/"
            "slice_write_reader_sharded.cpp"};
    reader.dfb_bindings = {ProducerOf(SWT_IN_DFB, "in0")};
    reader.runtime_arg_schema = {.runtime_arg_names = {"num_sticks"}};
    reader.hw_config = DataMovementHardwareConfig{
        .role = DataMovementRoleHint::READER,
        .gen2_config = DataMovementHardwareConfig::Gen2Config{.disable_dfb_implicit_sync_for_all = true}};

    // Writer: drain the input DFB -> interleaved output, writing each TILE at start_id + the padded-tile-dim walk.
    KernelSpec writer{
        .unique_id = SWT_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice_write/device/kernels/dataflow/"
            "slice_write_writer_interleaved.cpp"};
    writer.dfb_bindings = {ConsumerOf(SWT_IN_DFB, "in0")};
    writer.tensor_bindings = {TensorBinding{.tensor_parameter_name = SWT_OUT, .accessor_name = "dst"}};
    writer.runtime_arg_schema = {
        .runtime_arg_names = {
            "dst_byte_offset",
            "output_stick_size",
            "input_stick_size",
            "stick_size_offset",
            "num_dims",
            "start_id",
            "num_sticks_per_core",
            "num_sticks_per_core_read",
            "num_read_per_barrier"}};
    writer.advanced_options.num_runtime_varargs = 3 * num_dims;
    writer.hw_config = DataMovementHardwareConfig{
        .role = DataMovementRoleHint::WRITER,
        .gen2_config = DataMovementHardwareConfig::Gen2Config{.disable_dfb_implicit_sync_for_all = true}};

    spec.kernels = {reader, writer};
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {SWT_READER, SWT_WRITER}, .target_nodes = input_cores}};

    // ---- Per-core runtime args ----
    auto per_core = get_slice_write_runtime_args_tiled_sharded_input(
        input, output, output_tensor_start, output_tensor_end, iter_cores);
    const uint32_t out_addr = output.buffer()->address();

    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = SWT_READER};
    KernelRunArgs writer_run{.kernel = SWT_WRITER};
    uint32_t idx = 0;
    for (const auto& core : iter_cores) {
        const auto& r = per_core[idx].first;   // {num_tiles_this_core}
        const auto& w = per_core[idx].second;  // 9 scalars + 3*num_dims tile-dim vals + 1 padding tail
        reader_run.runtime_arg_values.push_back(
            KernelRunArgs::NodeRuntimeArgs{.node = core, .args = {{"num_sticks", r[0]}}});
        // tiled: w[0] is the raw output base addr (no width offset); the per-core offset is carried by start_id.
        const uint32_t dst_byte_offset = w[0] - out_addr;
        writer_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = core,
            .args = {
                {"dst_byte_offset", dst_byte_offset},
                {"output_stick_size", w[1]},
                {"input_stick_size", w[2]},
                {"stick_size_offset", w[3]},
                {"num_dims", w[4]},
                {"start_id", w[5]},
                {"num_sticks_per_core", w[6]},
                {"num_sticks_per_core_read", w[7]},
                {"num_read_per_barrier", w[8]}}});
        // Vararg tail: [num_input_tiles_per_dim, num_output_tiles_per_dim, id_per_dim] (3*num_dims). The tiled
        // helper appends one more value (padding tiles) used only by the UNPAD path — excluded here (guarded off).
        writer_run.advanced_options.runtime_varargs[core] =
            std::vector<uint32_t>(w.begin() + 9, w.begin() + 9 + 3 * num_dims);
        idx++;
    }
    run_args.kernel_run_args.push_back(reader_run);
    run_args.kernel_run_args.push_back(writer_run);
    run_args.tensor_args.emplace(SWT_IN, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(SWT_OUT, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
