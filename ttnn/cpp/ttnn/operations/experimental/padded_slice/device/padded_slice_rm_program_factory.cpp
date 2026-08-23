// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "padded_slice_rm_program_factory.hpp"
#include "padded_slice_utils.hpp"

#include <optional>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/math.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include <algorithm>
#include <cstdint>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <vector>
#include <tt-metalium/tt_align.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

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
using tt::tt_metal::experimental::ScratchpadBinding;
using tt::tt_metal::experimental::ScratchpadSpec;
using tt::tt_metal::experimental::ScratchpadSpecName;
using tt::tt_metal::experimental::TensorArgument;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

static std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>
get_padded_slice_runtime_args_rm_sharded_output(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& actual_output_shape,
    const std::vector<CoreCoord>& cores) {
    auto input_shape = input_tensor.logical_shape();
    auto output_shard_spec = output_tensor.shard_spec().value();
    auto output_shard_shape = output_shard_spec.shape;

    auto num_cores_total = cores.size();

    bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    bool is_block_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    bool is_width_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    [[maybe_unused]] uint32_t num_cores_channels =
        ttnn::operations::experimental::detail::get_num_cores_channels_from_sharded_tensor(output_tensor);
    int input_page_size = input_shape[-1] * input_tensor.element_size();
    [[maybe_unused]] uint32_t input_row_size_bytes =
        tt::div_up(input_shape[-1], num_cores_channels) * input_tensor.element_size();

    uint32_t output_row_size_bytes = output_shard_shape[1] * input_tensor.element_size();
    uint32_t output_row_size_elems = output_shard_shape[1];

    log_debug(
        tt::LogOp,
        "input_row_size_bytes: {}, input_page_size: {}, output_row_size_bytes: {}",
        input_row_size_bytes,
        input_page_size,
        output_row_size_bytes);
    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_output_sticks_per_dim[0] = 1;
    num_input_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    log_debug(tt::LogOp, "Output Shape : {}, Input Shape : {}", actual_output_shape, input_shape);
    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_output_dim = actual_output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_input_dim = (num_total_dim - num_output_dim) * accumulated_total_per_dim[i - 1];
        num_output_sticks_per_dim[i] = num_output_dim;
        num_input_sticks_per_dim[i] = num_input_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    for (int i = 0; i < num_dims; i++) {
        log_debug(
            tt::LogOp,
            "i = {}, num_output_sticks_per_dim: {}, num_input_sticks_per_dim: {}, accumulated_total_per_dim: {}",
            i,
            num_output_sticks_per_dim[i],
            num_input_sticks_per_dim[i],
            accumulated_total_per_dim[i]);
    }
    auto dst_buffer_alignment = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();

    uint32_t begins_bytes = output_tensor_start[-1] * input_tensor.element_size();
    uint32_t output_row_size_bytes_offset = tt::round_up(output_row_size_bytes, dst_buffer_alignment);
    std::vector<uint32_t> common_reader_kernel_args = {
        begins_bytes, input_page_size, output_row_size_bytes, output_row_size_bytes_offset, num_dims, 0, 0, 0, 0};

    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_output_sticks_per_dim.begin(), num_output_sticks_per_dim.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_input_sticks_per_dim.begin(), num_input_sticks_per_dim.end());

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    const uint32_t num_sticks_per_core = output_shard_spec.shape[0];
    // Fill every physical shard row, including the alignment-rounded tail beyond the logical slice.
    // This preserves the legacy deterministic behavior: tail rows follow the same wrapped source
    // geometry as real rows instead of retaining allocator-dependent contents.

    log_debug(tt::LogOp, "num_stick_per_core: {}", num_sticks_per_core);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input_tensor, output_tensor_start);

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

        const uint32_t num_sticks_written = core_h_index * num_sticks_per_core;
        const int width_offset = core_w_index * output_row_size_bytes_offset;

        id_per_dim[0] = num_sticks_written % num_output_sticks_per_dim[0];
        uint32_t output_written = num_sticks_written / num_output_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;
        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = output_written % num_output_sticks_per_dim[j];
            output_written = output_written / num_output_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }

        int this_input_row_size_bytes =
            std::max(std::min<int>(output_row_size_bytes, input_page_size - width_offset), 0);
        uint32_t this_core_num_sticks = num_sticks_per_core;
        if (this_input_row_size_bytes == 0) {
            this_core_num_sticks = 0;
        }
        std::vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        reader_kernel_args[0] += width_offset;
        reader_kernel_args[2] = this_input_row_size_bytes;
        uint32_t addr_offset = 5;
        reader_kernel_args[addr_offset++] = start_id;
        reader_kernel_args[addr_offset++] = this_core_num_sticks;
        reader_kernel_args[addr_offset++] = this_core_num_sticks;
        reader_kernel_args[addr_offset] = this_core_num_sticks;
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        log_trace(
            tt::LogOp,
            "For Core {}, start_id : {}, start_addr : {}, width_offset : {}, this_core_num_sticks : {}, "
            "this_input_row_size_bytes : {}",
            core,
            start_id,
            reader_kernel_args[0],
            width_offset,
            this_core_num_sticks,
            this_input_row_size_bytes);

        std::vector<uint32_t> writer_kernel_args = {
            this_core_num_sticks, output_row_size_elems, this_input_row_size_bytes, output_row_size_bytes};
        ret_val[core_index] = {reader_kernel_args, writer_kernel_args};
        core_index++;
    }

    return ret_val;
}

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

const TensorParamName INPUT{"input"};
const TensorParamName OUTPUT{"output"};
const DFBSpecName OUTPUT_DFB{"output"};
const ScratchpadSpecName ALIGNMENT_SCRATCH{"alignment_scratch"};
const ScratchpadSpecName PADDING_SCRATCH{"padding_scratch"};
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};

}  // namespace CMAKE_UNIQUE_NAMESPACE
using namespace CMAKE_UNIQUE_NAMESPACE;
}  // namespace

ttnn::device_operation::ProgramArtifacts PaddedSliceRMProgramFactory::create_program_artifacts(
    const PaddedSliceParams& operation_attributes, const PaddedSliceInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& output_tensor_start = operation_attributes.padded_slice_start;
    const auto& output_tensor_end = operation_attributes.padded_slice_end;

    const ttnn::Shape output_shape = output.logical_shape();
    ttnn::Shape actual_output_shape = output_tensor_end;
    for (int i = 0; i < output_shape.rank(); i++) {
        actual_output_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    TT_FATAL(output.is_sharded(), "Output Tensor must be sharded.");
    const auto output_shard_spec = output.shard_spec().value();
    const uint32_t output_row_size_bytes = output_shard_spec.shape[1] * output.element_size();

    const CoreRangeSet total_cores = output_shard_spec.grid;
    const bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    const std::vector<CoreCoord> iter_cores = corerange_to_cores(total_cores, std::nullopt, rm_orientation);

    const uint32_t num_cores_channels =
        ttnn::operations::experimental::detail::get_num_cores_channels_from_sharded_tensor(output);

    log_debug(tt::LogOp, "Input Shape {}, Padded Shape : {}", a.logical_shape(), a.padded_shape());

    const uint32_t input_row_size_bytes = a.logical_shape()[-1] * a.element_size() / num_cores_channels;
    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    TT_FATAL(
        output.buffer()->buffer_type() == tt::tt_metal::BufferType::L1,
        "Output buffer should be L1 for padded_slice operation with tiled inputs");

    const uint32_t src_buffer_alignment = a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                              ? ::hal::get_dram_alignment()
                                              : ::hal::get_l1_alignment();
    const uint32_t dst_buffer_alignment = ::hal::get_l1_alignment();

    TT_FATAL(
        output_row_size_bytes % dst_buffer_alignment == 0,
        "Output row size {} must be aligned to the destination buffer {} alignment {}",
        output_row_size_bytes,
        output.buffer()->buffer_type(),
        dst_buffer_alignment);
    const uint32_t alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
    const uint32_t begins_bytes = output_tensor_start[-1] * a.element_size();
    // The direct reader requires both its per-core row stride and its first source byte to be
    // aligned. A last-dimension slice start can require staging even when the output row is aligned.
    const bool is_non_aligned = output_row_size_bytes % alignment != 0 || begins_bytes % src_buffer_alignment != 0;
    const bool pad_output_row = output_row_size_bytes > input_row_size_bytes;

    // The kernel advances the write pointer by the aligned row size (stick_size_offset),
    // so the CB page size must match to avoid overflow.
    const uint32_t output_cb_page_size =
        is_non_aligned ? tt::round_up(output_row_size_bytes, dst_buffer_alignment) : output_row_size_bytes;
    const uint32_t num_output_sticks_per_core = output_shard_spec.shape[0];
    constexpr uint32_t num_trids = 2;

    ProgramSpec spec;
    spec.name = "padded_slice_rm";
    spec.tensor_parameters = {
        TensorParameter{.unique_id = INPUT, .spec = a.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
    };
    spec.dataflow_buffers = {DataflowBufferSpec{
        .unique_id = OUTPUT_DFB,
        .entry_size = output_cb_page_size,
        .num_entries = num_output_sticks_per_core,
        .data_format_metadata = cb_data_format,
        .borrowed_from = OUTPUT,
    }};
    if (is_non_aligned) {
        // Scratch page must accommodate padded_stick_size + worst-case misalignment.
        const uint32_t scratch_page = tt::align(output_row_size_bytes + src_buffer_alignment, src_buffer_alignment);
        spec.scratchpads.push_back(
            ScratchpadSpec{.unique_id = ALIGNMENT_SCRATCH, .size_per_node = scratch_page * num_trids});
    }
    if (pad_output_row) {
        spec.scratchpads.push_back(
            ScratchpadSpec{.unique_id = PADDING_SCRATCH, .size_per_node = output_row_size_bytes});
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = is_non_aligned ? "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
                                   "padded_slice_reader_rm_interleaved_start_id_non_aligned.cpp"
                                 : "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
                                   "padded_slice_reader_rm_interleaved_start_id.cpp",
        .dfb_bindings = {ProducerOf(OUTPUT_DFB, "output")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"src_byte_offset",
                  "padded_stick_size",
                  "unpadded_stick_size",
                  "stick_size_offset",
                  "num_dims",
                  "start_id",
                  "num_sticks_per_core",
                  "num_sticks_per_core_read",
                  "num_read_per_barrier"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
        .advanced_options = {.num_runtime_varargs = 3 * static_cast<uint32_t>(a.logical_shape().rank())},
    };
    if (is_non_aligned) {
        reader.compile_time_args = {{"src_buffer_alignment", src_buffer_alignment}, {"num_trids", num_trids}};
        reader.scratchpad_bindings = {
            ScratchpadBinding{.scratchpad_spec_name = ALIGNMENT_SCRATCH, .accessor_name = "alignment"}};
    }

    KernelSpec writer{
        .unique_id = WRITER,
        .source = pad_output_row
                      ? "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
                        "writer_unary_sharded_padded_rm.cpp"
                      : "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
                        "writer_unary_sharded.cpp",
        .dfb_bindings = {ConsumerOf(OUTPUT_DFB, "output")},
        .runtime_arg_schema =
            {.runtime_arg_names = pad_output_row
                                      ? std::vector<std::string>{
                                            "num_units",
                                            "num_elements_per_row",
                                            "unpadded_row_size_bytes",
                                            "padded_row_size_bytes"}
                                      : std::vector<std::string>{"num_units"}},
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
    };
    if (pad_output_row) {
        writer.compile_time_args = {{"output_elem_size", output.element_size()}};
        writer.scratchpad_bindings = {
            ScratchpadBinding{.scratchpad_spec_name = PADDING_SCRATCH, .accessor_name = "padding"}};
    }

    spec.kernels = {reader, writer};
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = total_cores}};

    const auto all_runtime_args = get_padded_slice_runtime_args_rm_sharded_output(
        a, output, output_tensor_start, actual_output_shape, iter_cores);

    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};
    for (uint32_t i = 0; i < iter_cores.size(); ++i) {
        const auto& core = iter_cores[i];
        const auto& r = all_runtime_args[i].first;
        const auto& w = all_runtime_args[i].second;
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"src_byte_offset", r[0]},
             {"padded_stick_size", r[1]},
             {"unpadded_stick_size", r[2]},
             {"stick_size_offset", r[3]},
             {"num_dims", r[4]},
             {"start_id", r[5]},
             {"num_sticks_per_core", r[6]},
             {"num_sticks_per_core_read", r[7]},
             {"num_read_per_barrier", r[8]}});
        reader_run.advanced_options.runtime_varargs.emplace(
            core, AdvancedKernelRunArgs::Varargs(r.begin() + 9, r.end()));

        AddRuntimeArgsForNode(writer_run.runtime_arg_values, core, {{"num_units", w[0]}});
        if (pad_output_row) {
            AddRuntimeArgsForNode(
                writer_run.runtime_arg_values,
                core,
                {{"num_elements_per_row", w[1]}, {"unpadded_row_size_bytes", w[2]}, {"padded_row_size_bytes", w[3]}});
        }
    }
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args.emplace(INPUT, TensorArgument{a.mesh_tensor()});
    run_args.tensor_args.emplace(OUTPUT, TensorArgument{output.mesh_tensor()});

    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
