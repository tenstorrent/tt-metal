// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write_rm_interleaved_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "slice_write_device_operation_types.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

namespace ttnn::experimental::prim {

using tt::tt_metal::BufferType;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::datatype_to_dataformat_converter;
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
using tt::tt_metal::experimental::ScratchpadBinding;
using tt::tt_metal::experimental::ScratchpadSpec;
using tt::tt_metal::experimental::ScratchpadSpecName;
using tt::tt_metal::experimental::TensorArgument;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace {

const TensorParamName RM_INTERLEAVED_INPUT{"input"};
const TensorParamName RM_INTERLEAVED_OUTPUT{"output"};
const DFBSpecName RM_INTERLEAVED_INPUT_DFB{"input"};
const ScratchpadSpecName RM_INTERLEAVED_OUTPUT_ROWS_SCRATCH{"output_rows"};
const KernelSpecName RM_INTERLEAVED_READER{"reader"};
const KernelSpecName RM_INTERLEAVED_WRITER{"writer"};

SliceWriteRuntimeArgs get_slice_write_runtime_args_rm(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& stride,
    uint32_t num_cores_total,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_sticks_per_core_group_1,
    uint32_t num_sticks_per_core_group_2,
    uint32_t input_slot_size,
    uint32_t uniform_num_read_per_barrier) {
    const auto input_shape = input_tensor.padded_shape();
    const auto output_shape = output_tensor.padded_shape();

    TT_FATAL(
        input_tensor.element_size() == output_tensor.element_size(),
        "Input & output should have the same element size");
    TT_FATAL(input_tensor.dtype() == output_tensor.dtype(), "Input & output should have the same dtype");

    const uint32_t output_row_size_bytes = output_shape[-1] * input_tensor.element_size();
    const uint32_t input_row_size_bytes = input_shape[-1] * input_tensor.element_size();
    const bool strided = std::any_of(stride.cbegin(), stride.cend(), [](int val) { return val != 1; });

    const uint32_t num_dims = static_cast<uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);
    std::vector<uint32_t> reverse_stride(num_dims);
    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly.
    // Dimension zero represents the row itself so the kernel can use the same coordinate-carry loop for rank 1
    // through rank 8.
    num_input_sticks_per_dim[0] = 1;
    num_output_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;
    reverse_stride[0] = stride[num_dims - 1];

    for (uint32_t i = 1; i < num_dims; ++i) {
        const uint32_t num_unpadded_dim = input_shape[-(i + 1)];
        const uint32_t num_total_dim = output_shape[-(i + 1)];
        reverse_stride[i] = stride[num_dims - (i + 1)];
        uint32_t num_padded_dim = 0;
        if (strided) {
            const uint32_t dims_traversed = reverse_stride[i] * (num_unpadded_dim - 1);
            const uint32_t num_dims_to_skip = num_total_dim - dims_traversed;
            num_padded_dim = num_dims_to_skip * accumulated_total_per_dim[i - 1];
        } else {
            num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        }
        num_input_sticks_per_dim[i] = num_unpadded_dim;
        num_output_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    SliceWriteRuntimeArgs result(num_cores_total);
    const uint32_t start_offset =
        ttnn::operations::data_movement::get_rm_start_offset(output_tensor, output_tensor_start);
    for (uint32_t i = 0, num_sticks_read = 0; i < num_cores_total; ++i) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_sticks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        }

        uint32_t num_sticks_per_core_read = 0;
        if (num_sticks_per_core != 0) {
            num_sticks_per_core_read = tt::div_up(num_sticks_per_core, uniform_num_read_per_barrier);
        }

        id_per_dim[0] = num_sticks_read % num_input_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_read / num_input_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;
        for (uint32_t j = 1; j < num_dims; ++j) {
            id_per_dim[j] = unpadded_written % num_input_sticks_per_dim[j];
            unpadded_written /= num_input_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1] * reverse_stride[j];
        }

        ReaderKernelArgs reader_args = {
            input_row_size_bytes,
            input_slot_size,
            num_sticks_per_core,
            num_sticks_per_core_read,
            uniform_num_read_per_barrier,
            num_sticks_read};
        WriterKernelArgs writer_args = {
            output_row_size_bytes,
            input_row_size_bytes,
            input_slot_size,
            num_dims,
            start_id,
            num_sticks_per_core,
            num_sticks_per_core_read,
            uniform_num_read_per_barrier};
        writer_args.insert(writer_args.end(), num_input_sticks_per_dim.begin(), num_input_sticks_per_dim.end());
        writer_args.insert(writer_args.end(), num_output_sticks_per_dim.begin(), num_output_sticks_per_dim.end());
        writer_args.insert(writer_args.end(), id_per_dim.begin(), id_per_dim.end());
        writer_args.insert(writer_args.end(), reverse_stride.begin(), reverse_stride.end());
        result[i] = {std::move(reader_args), std::move(writer_args)};
        num_sticks_read += num_sticks_per_core;
    }
    return result;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts SliceWriteRMInterleavedProgramFactory::create_program_artifacts(
    const SliceWriteParams& operation_attributes, const SliceWriteInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    const auto& output_tensor_start = operation_attributes.slice_start;
    const auto& stride = operation_attributes.step;
    const auto input_shape = input.padded_shape();
    const auto output_shape = output.padded_shape();

    const auto grid = input.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_x = grid.x;
    const uint32_t num_cores_y = grid.y;
    const uint32_t num_cores_total = num_cores_x * num_cores_y;
    const CoreRangeSet total_cores(CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1}));
    const uint32_t num_unpadded_sticks = input.physical_volume() / input_shape[-1];
    const auto [num_cores, all_cores, core_group_1, core_group_2, sticks_group_1, sticks_group_2] =
        tt::tt_metal::split_work_to_cores(grid, num_unpadded_sticks);

    constexpr uint32_t max_read_size = 4096;
    const uint32_t input_row_size_bytes = input_shape[-1] * input.element_size();
    const uint32_t output_row_size_bytes = output_shape[-1] * output.element_size();
    const uint32_t src_alignment =
        input.buffer()->buffer_type() == BufferType::DRAM ? hal::get_dram_alignment() : hal::get_l1_alignment();
    const uint32_t dst_alignment =
        output.buffer()->buffer_type() == BufferType::DRAM ? hal::get_dram_alignment() : hal::get_l1_alignment();
    const uint32_t alignment = std::max(src_alignment, dst_alignment);
    const uint32_t begins_bytes = output_tensor_start[-1] * input.element_size();
    const uint32_t page_alignment_offset = begins_bytes % src_alignment;
    const bool last_dim_strided = stride[-1] != 1;

    const uint32_t input_slot_size = tt::round_up(input_row_size_bytes + page_alignment_offset, src_alignment);
    const uint32_t output_slot_size = tt::round_up(output_row_size_bytes, alignment);
    const uint32_t transfer_slot_size =
        last_dim_strided ? std::max(input_slot_size, output_slot_size) : input_slot_size;
    const uint32_t max_num_sticks_per_core = std::max(sticks_group_1, sticks_group_2);
    const uint32_t max_num_sticks_per_core_pad32 = tt::round_up(max_num_sticks_per_core, 32);
    const uint32_t uniform_num_reads =
        tt::tt_metal::merge_num_sticks_to_read(max_num_sticks_per_core_pad32, transfer_slot_size, max_read_size);
    const uint32_t uniform_num_read_per_barrier = max_num_sticks_per_core_pad32 / uniform_num_reads;
    const auto all_runtime_args = get_slice_write_runtime_args_rm(
        input,
        output,
        output_tensor_start,
        stride,
        num_cores_total,
        num_cores_y,
        core_group_1,
        core_group_2,
        sticks_group_1,
        sticks_group_2,
        input_slot_size,
        uniform_num_read_per_barrier);
    ProgramSpec spec;
    spec.name = "slice_write_rm_interleaved";
    spec.tensor_parameters = {
        TensorParameter{.unique_id = RM_INTERLEAVED_INPUT, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = RM_INTERLEAVED_OUTPUT, .spec = output.tensor_spec()},
    };
    spec.dataflow_buffers = {DataflowBufferSpec{
        .unique_id = RM_INTERLEAVED_INPUT_DFB,
        .entry_size = input_slot_size,
        .num_entries = 2 * uniform_num_read_per_barrier,
        .data_format_metadata = datatype_to_dataformat_converter(input.dtype()),
    }};
    if (last_dim_strided) {
        // The writer reads, edits, and writes these rows itself; model this as
        // private raw storage rather than a producer/consumer DFB self-loop.
        spec.scratchpads = {ScratchpadSpec{
            .unique_id = RM_INTERLEAVED_OUTPUT_ROWS_SCRATCH,
            .size_per_node = uniform_num_read_per_barrier * output_slot_size}};
    }

    KernelSpec reader{
        .unique_id = RM_INTERLEAVED_READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
            "slice_write_reader_interleaved.cpp",
        .dfb_bindings = {ProducerOf(RM_INTERLEAVED_INPUT_DFB, "input")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = RM_INTERLEAVED_INPUT, .accessor_name = "input"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"stick_size",
                  "stick_size_offset",
                  "num_sticks_per_core",
                  "num_sticks_per_core_read",
                  "num_read_per_barrier",
                  "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch(), true),
    };
    KernelSpec writer{
        .unique_id = RM_INTERLEAVED_WRITER,
        .source = last_dim_strided ? "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
                                     "slice_write_writer_interleaved_strided.cpp"
                                   : "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
                                     "slice_write_writer_interleaved_contiguous.cpp",
        .dfb_bindings = {ConsumerOf(RM_INTERLEAVED_INPUT_DFB, "input")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = RM_INTERLEAVED_OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"output_stick_size",
                  "input_stick_size",
                  "stick_size_offset",
                  "num_dims",
                  "start_id",
                  "num_sticks_per_core",
                  "num_sticks_per_core_read",
                  "num_read_per_barrier"}},
        .hw_config = ttnn::create_writer_datamovement_config(input.device()->arch(), true),
        .advanced_options = {.num_runtime_varargs = 4 * static_cast<uint32_t>(input_shape.rank())},
    };
    reader.compile_time_args = {{"page_offset", page_alignment_offset}};
    if (last_dim_strided) {
        writer.compile_time_args = {
            {"alignment_offset", page_alignment_offset},
            {"page_begins_offset", begins_bytes},
            {"element_size", output.element_size()},
            {"output_row_stride", output_slot_size}};
        writer.scratchpad_bindings = {ScratchpadBinding{
            .scratchpad_spec_name = RM_INTERLEAVED_OUTPUT_ROWS_SCRATCH, .accessor_name = "output_rows"}};
    } else {
        writer.compile_time_args = {{"alignment_offset", page_alignment_offset}, {"page_begins_offset", begins_bytes}};
    }
    spec.kernels = {std::move(reader), std::move(writer)};
    spec.work_units = {WorkUnitSpec{
        .name = "main", .kernels = {RM_INTERLEAVED_READER, RM_INTERLEAVED_WRITER}, .target_nodes = total_cores}};

    ProgramRunArgs run_args;
    KernelRunArgs reader_run{
        .kernel = RM_INTERLEAVED_READER,
        .runtime_arg_values = {
            {"stick_size", {}},
            {"stick_size_offset", {}},
            {"num_sticks_per_core", {}},
            {"num_sticks_per_core_read", {}},
            {"num_read_per_barrier", {}},
            {"start_id", {}}}};
    KernelRunArgs writer_run{
        .kernel = RM_INTERLEAVED_WRITER,
        .runtime_arg_values = {
            {"output_stick_size", {}},
            {"input_stick_size", {}},
            {"stick_size_offset", {}},
            {"num_dims", {}},
            {"start_id", {}},
            {"num_sticks_per_core", {}},
            {"num_sticks_per_core_read", {}},
            {"num_read_per_barrier", {}}}};
    // Resolve the predeclared tables once: name-first population avoids repeating the Table lookup for every core.
    // get() is deliberately non-inserting so a miss cannot reallocate the table and invalidate an earlier reference.
    auto& reader_stick_size = *reader_run.runtime_arg_values.get("stick_size");
    auto& reader_stick_size_offset = *reader_run.runtime_arg_values.get("stick_size_offset");
    auto& reader_num_sticks = *reader_run.runtime_arg_values.get("num_sticks_per_core");
    auto& reader_num_reads = *reader_run.runtime_arg_values.get("num_sticks_per_core_read");
    auto& reader_barrier_size = *reader_run.runtime_arg_values.get("num_read_per_barrier");
    auto& reader_start_id = *reader_run.runtime_arg_values.get("start_id");
    auto& writer_output_stick_size = *writer_run.runtime_arg_values.get("output_stick_size");
    auto& writer_input_stick_size = *writer_run.runtime_arg_values.get("input_stick_size");
    auto& writer_stick_size_offset = *writer_run.runtime_arg_values.get("stick_size_offset");
    auto& writer_num_dims = *writer_run.runtime_arg_values.get("num_dims");
    auto& writer_start_id = *writer_run.runtime_arg_values.get("start_id");
    auto& writer_num_sticks = *writer_run.runtime_arg_values.get("num_sticks_per_core");
    auto& writer_num_reads = *writer_run.runtime_arg_values.get("num_sticks_per_core_read");
    auto& writer_barrier_size = *writer_run.runtime_arg_values.get("num_read_per_barrier");
    for (uint32_t i = 0; i < num_cores_total; ++i) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const auto& r = all_runtime_args[i].first;
        const auto& w = all_runtime_args[i].second;
        reader_stick_size[core] = r[0];
        reader_stick_size_offset[core] = r[1];
        reader_num_sticks[core] = r[2];
        reader_num_reads[core] = r[3];
        reader_barrier_size[core] = r[4];
        reader_start_id[core] = r[5];
        writer_output_stick_size[core] = w[0];
        writer_input_stick_size[core] = w[1];
        writer_stick_size_offset[core] = w[2];
        writer_num_dims[core] = w[3];
        writer_start_id[core] = w[4];
        writer_num_sticks[core] = w[5];
        writer_num_reads[core] = w[6];
        writer_barrier_size[core] = w[7];
        writer_run.advanced_options.runtime_varargs.emplace(
            core, AdvancedKernelRunArgs::Varargs(w.begin() + 8, w.end()));
    }
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args.emplace(RM_INTERLEAVED_INPUT, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(RM_INTERLEAVED_OUTPUT, TensorArgument{output.mesh_tensor()});
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
