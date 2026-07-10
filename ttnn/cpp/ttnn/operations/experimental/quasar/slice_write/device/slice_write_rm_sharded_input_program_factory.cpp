// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write_rm_sharded_input_program_factory.hpp"

#include <cstdint>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "slice_write_device_operation_types.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim::qsr {

using namespace tt::tt_metal::experimental;

namespace {

SliceWriteRuntimeArgs get_slice_write_runtime_args_rm_sharded_input(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const std::vector<CoreCoord>& cores,
    uint32_t max_read_size) {
    auto* output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.logical_shape();
    for (uint32_t i = 0; i < input_shape.rank(); i++) {
        input_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }
    log_debug(tt::LogOp, "Slice Write Input Shape: {}", input_shape);
    auto output_shape = output_tensor.logical_shape();
    log_debug(tt::LogOp, "Slice Write Output Shape: {}", output_shape);

    TT_FATAL(
        input_tensor.element_size() == output_tensor.element_size(),
        "Input & output should have the same element size");
    TT_FATAL(input_tensor.dtype() == output_tensor.dtype(), "Input & output should have the same dtype");

    TT_FATAL(input_tensor.shard_spec().has_value(), "Input tensor should be sharded");

    auto shard_spec = input_tensor.shard_spec().value();
    auto input_cores = shard_spec.grid;
    auto input_shard_shape = shard_spec.shape;

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    uint32_t output_row_size_bytes = output_shape[-1] * input_tensor.element_size();
    uint32_t input_row_size_bytes = input_shard_shape[1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);
    std::vector<int> size_till_end(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    std::vector<uint32_t> accumulated_input_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_input_sticks_per_dim[0] = 1;
    num_output_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;
    accumulated_input_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = input_shape[-(i + 1)];
        uint32_t num_total_dim = output_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_input_sticks_per_dim[i] = num_unpadded_dim;
        num_output_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
        accumulated_input_total_per_dim[i] = num_unpadded_dim * accumulated_input_total_per_dim[i - 1];
    }

    std::string unpadded_sticks_str;
    for (auto& i : num_input_sticks_per_dim) {
        unpadded_sticks_str += std::to_string(i) + ", ";
    }
    std::string padded_sticks_str;
    for (auto& i : num_output_sticks_per_dim) {
        padded_sticks_str += std::to_string(i) + ", ";
    }
    std::string accumulated_str;
    for (auto& i : accumulated_total_per_dim) {
        accumulated_str += std::to_string(i) + ", ";
    }
    log_debug(tt::LogOp, "Slice Write Accumulated Sticks: {}", accumulated_str);
    log_debug(tt::LogOp, "Slice Write Unpadded Sticks: {}", unpadded_sticks_str);
    log_debug(tt::LogOp, "Slice Write Padded Sticks: {}", padded_sticks_str);
    log_debug(tt::LogOp, "Accumulated Input : {}", accumulated_input_total_per_dim);

    using namespace tt::tt_metal::experimental;
    auto src_buffer_alignment = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();
    uint32_t input_row_size_bytes_offset = tt::round_up(input_row_size_bytes, src_buffer_alignment);
    TT_FATAL(
        output_tensor_start[-1] == 0,
        "slice_write expects output start for the last dimension to be 0. Got {}",
        output_tensor_start[-1]);

    log_debug(tt::LogOp, "Output Buffer address: {}", output_buffer->address());
    std::vector<uint32_t> common_writer_kernel_args = {
        output_buffer->address() + (output_tensor_start[-1] * output_tensor.element_size()),
        output_row_size_bytes,
        input_row_size_bytes,
        input_row_size_bytes_offset,
        num_dims,
        0,
        0,
        0,
        0};

    common_writer_kernel_args.insert(
        common_writer_kernel_args.end(), num_input_sticks_per_dim.begin(), num_input_sticks_per_dim.end());
    common_writer_kernel_args.insert(
        common_writer_kernel_args.end(), num_output_sticks_per_dim.begin(), num_output_sticks_per_dim.end());

    auto num_cores_total = cores.size();

    const auto num_sticks_per_core = shard_spec.shape[0];
    // issue more reads before calling barrier
    const uint32_t num_sticks_per_core_read =
        tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core, input_row_size_bytes_offset, max_read_size);
    const uint32_t num_read_per_barrier = num_sticks_per_core / num_sticks_per_core_read;

    log_debug(
        tt::LogOp,
        "num_sticks_per_core = {}, num_sticks_per_core_read = {}, num_read_per_barrier = {}",
        num_sticks_per_core,
        num_sticks_per_core_read,
        num_read_per_barrier);
    SliceWriteRuntimeArgs ret_val(num_cores_total);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(output_tensor, output_tensor_start);
    uint32_t core_index = 0;
    for (const auto& core : cores) {
        uint32_t core_w_index = 0;
        uint32_t core_h_index = core_index;
        if (is_block_sharded) {
            core_w_index = rm_orientation ? core.x : core.y;
            core_h_index = rm_orientation ? core.y : core.x;
        }
        const uint32_t num_sticks_read = core_h_index * num_sticks_per_core;
        const uint32_t width_offset = core_w_index * input_row_size_bytes;

        id_per_dim[0] = num_sticks_read % num_input_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_read / num_input_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;
        int max_num_sticks_this_core = 0;
        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_input_sticks_per_dim[j];
            if (j == num_dims - 1 && unpadded_written == num_input_sticks_per_dim[j]) {
                // Handle edge case where last dimension is completely written
                id_per_dim[j] = num_input_sticks_per_dim[j];
            }
            unpadded_written = unpadded_written / num_input_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
            size_till_end[j] = output_tensor_end[-1 - j] - output_tensor_start[-1 - j] - id_per_dim[j] - 1;
            max_num_sticks_this_core += size_till_end[j] * accumulated_input_total_per_dim[j - 1];
        }

        uint32_t this_input_row_size_bytes = std::min(input_row_size_bytes, output_row_size_bytes - width_offset);
        WriterKernelArgs writer_kernel_args = common_writer_kernel_args;
        writer_kernel_args[0] += width_offset;
        writer_kernel_args[2] = this_input_row_size_bytes;

        uint32_t num_sticks_this_core =
            std::min<uint32_t>(num_sticks_per_core, std::max<int>(max_num_sticks_this_core + 1, 0));

        log_trace(
            tt::LogOp,
            "Start ID: {}, Start ID per dim : {} , Size till end : {} Num Sticks: {}, this_input_row_size_bytes: {} "
            "for Core: {}",
            start_id,
            id_per_dim,
            size_till_end,
            num_sticks_this_core,
            this_input_row_size_bytes,
            core);
        uint32_t addr_offset = 5;  // output buffer addr, output_row_size_bytes, input_row_size_bytes, num_dims
        writer_kernel_args[addr_offset++] = start_id;
        writer_kernel_args[addr_offset++] = num_sticks_this_core;
        writer_kernel_args[addr_offset++] = num_sticks_this_core;
        writer_kernel_args[addr_offset] = num_read_per_barrier;
        writer_kernel_args.insert(writer_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        ReaderKernelArgs reader_kernel_args = {num_sticks_per_core};
        ret_val[core_index] = {reader_kernel_args, writer_kernel_args};
        core_index++;
    }

    return ret_val;
}
}  // namespace

namespace {
const TensorParamName SW_IN{"slice_write_in"};
const TensorParamName SW_OUT{"slice_write_out"};
const DFBSpecName SW_IN_DFB{"slice_write_in_dfb"};
const KernelSpecName SW_READER{"slice_write_reader"};
const KernelSpecName SW_WRITER{"slice_write_writer"};
}  // namespace

ttnn::device_operation::ProgramArtifacts SliceWriteRMShardedInputProgramFactory::create_program_artifacts(
    const SliceWriteParams& operation_attributes, const SliceWriteInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& output_tensor_start = operation_attributes.slice_start;
    const auto& output_tensor_end = operation_attributes.slice_end;

    const auto input_shape = input.logical_shape();

    TT_FATAL(input.shard_spec().has_value(), "slice_write input must be sharded");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
            input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
        "slice_write input must be height or block sharded");
    auto shard_spec = input.shard_spec().value();
    CoreRangeSet input_cores = shard_spec.grid;
    const bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    const uint32_t num_input_sticks_per_core = shard_spec.shape[0];
    const uint32_t input_row_size_bytes = shard_spec.shape[1] * input.element_size();
    const auto src_buffer_alignment = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                          ? ::hal::get_dram_alignment()
                                          : ::hal::get_l1_alignment();
    const uint32_t input_row_size_bytes_offset = tt::round_up(input_row_size_bytes, src_buffer_alignment);
    const uint32_t max_read_size = 4096;
    const uint32_t num_dims = static_cast<uint32_t>(input_shape.rank());

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(
        cb_data_format == tt::tt_metal::datatype_to_dataformat_converter(output.dtype()),
        "slice_write input/output data formats must match");

    std::vector<CoreCoord> iter_cores = corerange_to_cores(input_cores, std::nullopt, rm_orientation);

    // ---- ProgramSpec ----
    ProgramSpec spec;
    spec.name = "slice_write_rm";
    spec.tensor_parameters = {
        TensorParameter{.unique_id = SW_IN, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = SW_OUT, .spec = output.tensor_spec()},
    };

    // INPUT DFB borrowed onto the resident sharded input shard (reader produces, writer drains to dst).
    DataflowBufferSpec in_dfb{
        .unique_id = SW_IN_DFB,
        .entry_size = input_row_size_bytes_offset,
        .num_entries = num_input_sticks_per_core,
        .data_format_metadata = cb_data_format,
    };
    in_dfb.borrowed_from = SW_IN;
    spec.dataflow_buffers.push_back(in_dfb);

    // Reader: mark the resident input shard available (no data fetch).
    KernelSpec reader{
        .unique_id = SW_READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice_write/device/kernels/dataflow/"
            "slice_write_reader_sharded.cpp"};
    reader.dfb_bindings = {ProducerOf(SW_IN_DFB, "in0")};
    reader.runtime_arg_schema = {.runtime_arg_names = {"num_sticks"}};
    reader.hw_config = DataMovementHardwareConfig{
        .role = DataMovementRoleHint::READER,
        .gen2_config = DataMovementHardwareConfig::Gen2Config{.disable_dfb_implicit_sync_for_all = true}};

    // Writer: drain the input DFB -> interleaved output at start_id + the padded-dim walk.
    KernelSpec writer{
        .unique_id = SW_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice_write/device/kernels/dataflow/"
            "slice_write_writer_interleaved.cpp"};
    writer.dfb_bindings = {ConsumerOf(SW_IN_DFB, "in0")};
    writer.tensor_bindings = {TensorBinding{.tensor_parameter_name = SW_OUT, .accessor_name = "dst"}};
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
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {SW_READER, SW_WRITER}, .target_nodes = input_cores}};

    // ---- Per-core runtime args ----
    auto per_core = get_slice_write_runtime_args_rm_sharded_input(
        input, output, output_tensor_start, output_tensor_end, iter_cores, max_read_size);
    const uint32_t out_addr = output.buffer()->address();

    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = SW_READER};
    KernelRunArgs writer_run{.kernel = SW_WRITER};
    uint32_t i = 0;
    for (const auto& core : iter_cores) {
        const auto& r = per_core[i].first;   // {num_sticks_per_core}
        const auto& w = per_core[i].second;  // writer args
        reader_run.runtime_arg_values.push_back(
            KernelRunArgs::NodeRuntimeArgs{.node = core, .args = {{"num_sticks", r[0]}}});
        // w[0] is the (aligned) dst base addr + width_offset; recover the per-core byte offset from the base.
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
        writer_run.advanced_options.runtime_varargs[core] = std::vector<uint32_t>(w.begin() + 9, w.end());
        i++;
    }
    run_args.kernel_run_args.push_back(reader_run);
    run_args.kernel_run_args.push_back(writer_run);
    run_args.tensor_args.emplace(SW_IN, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(SW_OUT, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
