// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write_rm_sharded_input_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

#include "slice_write_device_operation_types.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

namespace ttnn::experimental::prim {

using tt::tt_metal::BufferType;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::corerange_to_cores;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::datatype_to_dataformat_converter;
using tt::tt_metal::ShardOrientation;
using tt::tt_metal::TensorMemoryLayout;
namespace hal = tt::tt_metal::hal;

using tt::tt_metal::experimental::AdvancedKernelRunArgs;
using tt::tt_metal::experimental::ConsumerOf;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProducerOf;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::TensorArgument;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace {

const TensorParamName RM_SHARDED_INPUT{"input"};
const TensorParamName RM_SHARDED_OUTPUT{"output"};
const DFBSpecName RM_SHARDED_INPUT_DFB{"input"};
const KernelSpecName RM_SHARDED_READER{"reader"};
const KernelSpecName RM_SHARDED_WRITER{"writer"};

SliceWriteRuntimeArgs get_slice_write_runtime_args_rm_sharded_input(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const std::vector<CoreCoord>& cores,
    uint32_t max_read_size) {
    auto input_shape = input_tensor.logical_shape();
    for (uint32_t i = 0; i < input_shape.rank(); ++i) {
        input_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }
    const auto output_shape = output_tensor.logical_shape();

    TT_FATAL(
        input_tensor.element_size() == output_tensor.element_size(),
        "Input & output should have the same element size");
    TT_FATAL(input_tensor.dtype() == output_tensor.dtype(), "Input & output should have the same dtype");
    TT_FATAL(input_tensor.shard_spec().has_value(), "Input tensor should be sharded");

    const auto shard_spec = input_tensor.shard_spec().value();
    const bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const uint32_t output_row_size_bytes = output_shape[-1] * input_tensor.element_size();
    const uint32_t input_row_size_bytes = shard_spec.shape[1] * input_tensor.element_size();

    const uint32_t num_dims = static_cast<uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);
    std::vector<int> size_till_end(num_dims);
    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    std::vector<uint32_t> accumulated_input_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly.
    num_input_sticks_per_dim[0] = 1;
    num_output_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;
    accumulated_input_total_per_dim[0] = 1;
    for (uint32_t i = 1; i < num_dims; ++i) {
        const uint32_t num_unpadded_dim = input_shape[-(i + 1)];
        const uint32_t num_total_dim = output_shape[-(i + 1)];
        const uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_input_sticks_per_dim[i] = num_unpadded_dim;
        num_output_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
        accumulated_input_total_per_dim[i] = num_unpadded_dim * accumulated_input_total_per_dim[i - 1];
    }

    const uint32_t src_alignment =
        input_tensor.buffer()->buffer_type() == BufferType::DRAM ? hal::get_dram_alignment() : hal::get_l1_alignment();
    const uint32_t input_row_size_bytes_offset = tt::round_up(input_row_size_bytes, src_alignment);
    TT_FATAL(
        output_tensor_start[-1] == 0,
        "slice_write expects output start for the last dimension to be 0. Got {}",
        output_tensor_start[-1]);

    const uint32_t num_sticks_per_core = shard_spec.shape[0];
    const uint32_t num_sticks_per_core_read =
        tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core, input_row_size_bytes_offset, max_read_size);
    const uint32_t num_read_per_barrier = num_sticks_per_core / num_sticks_per_core_read;
    const uint32_t start_offset =
        ttnn::operations::data_movement::get_rm_start_offset(output_tensor, output_tensor_start);

    SliceWriteRuntimeArgs result(cores.size());
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
        for (uint32_t j = 1; j < num_dims; ++j) {
            id_per_dim[j] = unpadded_written % num_input_sticks_per_dim[j];
            if (j == num_dims - 1 && unpadded_written == num_input_sticks_per_dim[j]) {
                // Handle edge case where last dimension is completely written.
                id_per_dim[j] = num_input_sticks_per_dim[j];
            }
            unpadded_written /= num_input_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
            size_till_end[j] = output_tensor_end[-1 - j] - output_tensor_start[-1 - j] - id_per_dim[j] - 1;
            max_num_sticks_this_core += size_till_end[j] * accumulated_input_total_per_dim[j - 1];
        }

        const uint32_t this_input_row_size_bytes = std::min(input_row_size_bytes, output_row_size_bytes - width_offset);
        const uint32_t num_sticks_this_core =
            std::min<uint32_t>(num_sticks_per_core, std::max<int>(max_num_sticks_this_core + 1, 0));
        log_trace(
            tt::LogOp,
            "Start ID: {}, Start ID per dim: {}, Size till end: {}, Num Sticks: {}, input row bytes: {} for Core: {}",
            start_id,
            id_per_dim,
            size_till_end,
            num_sticks_this_core,
            this_input_row_size_bytes,
            core);
        ReaderKernelArgs reader_args = {num_sticks_per_core};
        WriterKernelArgs writer_args = {
            width_offset,
            output_row_size_bytes,
            this_input_row_size_bytes,
            input_row_size_bytes_offset,
            num_dims,
            start_id,
            num_sticks_this_core,
            num_sticks_this_core,
            num_read_per_barrier};
        writer_args.insert(writer_args.end(), num_input_sticks_per_dim.begin(), num_input_sticks_per_dim.end());
        writer_args.insert(writer_args.end(), num_output_sticks_per_dim.begin(), num_output_sticks_per_dim.end());
        writer_args.insert(writer_args.end(), id_per_dim.begin(), id_per_dim.end());
        result[core_index] = {std::move(reader_args), std::move(writer_args)};
        ++core_index;
    }
    return result;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts SliceWriteRMShardedInputProgramFactory::create_program_artifacts(
    const SliceWriteParams& operation_attributes, const SliceWriteInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    const auto input_shape = input.logical_shape();
    TT_FATAL(input.shard_spec().has_value(), "Input tensor should be sharded");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
            input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
        "Input tensor should be height or block sharded");

    const auto shard_spec = input.shard_spec().value();
    const CoreRangeSet input_cores = shard_spec.grid;
    const bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const auto iter_cores = corerange_to_cores(input_cores, std::nullopt, rm_orientation);
    const uint32_t input_row_size_bytes = shard_spec.shape[1] * input.element_size();
    const uint32_t src_alignment =
        input.buffer()->buffer_type() == BufferType::DRAM ? hal::get_dram_alignment() : hal::get_l1_alignment();
    const uint32_t input_page_size = tt::round_up(input_row_size_bytes, src_alignment);
    constexpr uint32_t max_read_size = 4096;

    ProgramSpec spec;
    spec.name = "slice_write_rm_sharded_input";
    spec.tensor_parameters = {
        TensorParameter{.unique_id = RM_SHARDED_INPUT, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = RM_SHARDED_OUTPUT, .spec = output.tensor_spec()},
    };
    spec.dataflow_buffers = {DataflowBufferSpec{
        .unique_id = RM_SHARDED_INPUT_DFB,
        .entry_size = input_page_size,
        .num_entries = shard_spec.shape[0],
        .data_format_metadata = datatype_to_dataformat_converter(input.dtype()),
        .borrowed_from = RM_SHARDED_INPUT,
    }};
    KernelSpec reader{
        .unique_id = RM_SHARDED_READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
            "slice_write_reader_sharded.cpp",
        .dfb_bindings = {ProducerOf(RM_SHARDED_INPUT_DFB, "input")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch(), true),
    };
    KernelSpec writer{
        .unique_id = RM_SHARDED_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
            "slice_write_writer_interleaved.cpp",
        .dfb_bindings = {ConsumerOf(RM_SHARDED_INPUT_DFB, "input")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = RM_SHARDED_OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"output_byte_offset",
                  "output_stick_size",
                  "input_stick_size",
                  "stick_size_offset",
                  "num_dims",
                  "start_id",
                  "num_sticks_per_core",
                  "num_sticks_per_core_read",
                  "num_read_per_barrier",
                  "padding_width_units"}},
        .hw_config = ttnn::create_writer_datamovement_config(input.device()->arch(), true),
        .advanced_options = {.num_runtime_varargs = 3 * static_cast<uint32_t>(input_shape.rank())},
    };
    writer.compile_time_args = {{"unpad_input_width", 0}};
    spec.kernels = {std::move(reader), std::move(writer)};
    spec.work_units = {
        WorkUnitSpec{.name = "main", .kernels = {RM_SHARDED_READER, RM_SHARDED_WRITER}, .target_nodes = input_cores}};

    const auto all_runtime_args = get_slice_write_runtime_args_rm_sharded_input(
        input, output, operation_attributes.slice_start, operation_attributes.slice_end, iter_cores, max_read_size);
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = RM_SHARDED_READER};
    KernelRunArgs writer_run{.kernel = RM_SHARDED_WRITER};
    for (uint32_t i = 0; i < iter_cores.size(); ++i) {
        const auto& core = iter_cores[i];
        const auto& r = all_runtime_args[i].first;
        const auto& w = all_runtime_args[i].second;
        AddRuntimeArgsForNode(reader_run.runtime_arg_values, core, {{"num_units", r[0]}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"output_byte_offset", w[0]},
             {"output_stick_size", w[1]},
             {"input_stick_size", w[2]},
             {"stick_size_offset", w[3]},
             {"num_dims", w[4]},
             {"start_id", w[5]},
             {"num_sticks_per_core", w[6]},
             {"num_sticks_per_core_read", w[7]},
             {"num_read_per_barrier", w[8]},
             {"padding_width_units", 0}});
        writer_run.advanced_options.runtime_varargs.emplace(
            core, AdvancedKernelRunArgs::Varargs(w.begin() + 9, w.end()));
    }
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args.emplace(RM_SHARDED_INPUT, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(RM_SHARDED_OUTPUT, TensorArgument{output.mesh_tensor()});
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
