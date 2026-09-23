// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "padded_slice_rm_program_factory.hpp"
#include "padded_slice_utils.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include "hostdevcommon/kernel_structs.h"
#include "optional"
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include <algorithm>
#include <cstdint>
#include <ranges>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <vector>
#include <tt-metalium/tt_align.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t kPaddedSliceRmReaderKernelIdx = 0;
constexpr uint32_t kPaddedSliceRmReaderAddressArgIdx = 0;
constexpr uint32_t kPaddedSliceRmOutputCbIndex = 0;

CBDescriptor make_padded_slice_rm_cb(
    uint32_t cb_index,
    const CoreRangeSet& core_ranges,
    uint32_t page_size,
    uint32_t num_pages,
    tt::DataFormat data_format,
    Buffer* buffer = nullptr) {
    return CBDescriptor{
        .total_size = num_pages * page_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = data_format,
            .page_size = page_size,
        }}},
        .buffer = buffer,
    };
}

uint32_t padded_slice_rm_reader_address(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const CoreCoord& core,
    uint32_t core_index) {
    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "Input buffer should be allocated on device");
    TT_FATAL(output_buffer != nullptr, "Output buffer should be allocated on device!");

    const auto output_shard_spec = output_tensor.shard_spec().value();
    const bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const bool is_block_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool is_width_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    const uint32_t output_row_size_bytes = output_shard_spec.shape[1] * input_tensor.element_size();

    const auto src_buffer_alignment =
        input_buffer->buffer_type() == BufferType::DRAM ? hal::get_dram_alignment() : hal::get_l1_alignment();
    const auto dst_buffer_alignment =
        output_buffer->buffer_type() == BufferType::DRAM ? hal::get_dram_alignment() : hal::get_l1_alignment();
    const uint32_t begins_bytes = output_tensor_start[-1] * input_tensor.element_size();
    const uint32_t misalignment = begins_bytes % src_buffer_alignment;
    const uint32_t output_row_size_bytes_offset = tt::round_up(output_row_size_bytes, dst_buffer_alignment);

    uint32_t core_w_index = 0;
    if (is_block_sharded) {
        core_w_index = rm_orientation ? core.x : core.y;
    } else if (is_width_sharded) {
        core_w_index = core_index;
    }
    const int width_offset = static_cast<int>(core_w_index * output_row_size_bytes_offset);
    // read from nearest aligned address
    uint32_t read_address = input_buffer->address() + begins_bytes - misalignment;
    read_address += static_cast<uint32_t>(width_offset);
    return read_address;
}

void update_padded_slice_rm_output_cb(Program& program, Buffer& output_buffer) {
    for (const auto& cb : program.circular_buffers()) {
        if (cb->buffer_indices().contains(static_cast<uint8_t>(kPaddedSliceRmOutputCbIndex))) {
            UpdateDynamicCircularBufferAddress(program, cb->id(), output_buffer);
        }
    }
}

}  // namespace

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
    using namespace tt::tt_metal::experimental;
    auto dst_buffer_alignment = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();

    uint32_t output_row_size_bytes_offset = tt::round_up(output_row_size_bytes, dst_buffer_alignment);
    std::vector<uint32_t> common_reader_kernel_args = {
        0u,  // read from nearest aligned address
        input_page_size,
        output_row_size_bytes,
        output_row_size_bytes_offset,
        num_dims,
        0,
        0,
        0,
        0};

    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_output_sticks_per_dim.begin(), num_output_sticks_per_dim.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_input_sticks_per_dim.begin(), num_input_sticks_per_dim.end());

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    const uint32_t num_sticks_per_core = output_shard_spec.shape[0];

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
        reader_kernel_args[0] =
            padded_slice_rm_reader_address(input_tensor, output_tensor, output_tensor_start, core, core_index);
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

ProgramDescriptor PaddedSliceRMProgramFactory::create_descriptor(
    const PaddedSliceParams& operation_attributes, const PaddedSliceInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& output_tensor_start = operation_attributes.padded_slice_start;
    const auto& output_tensor_end = operation_attributes.padded_slice_end;

    const ttnn::Shape output_shape = output.logical_shape();
    ttnn::Shape actual_output_shape = output_tensor_end;
    for (int i = 0; i < output_shape.rank(); i++) {
        actual_output_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }

    ProgramDescriptor desc;

    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    TT_FATAL(src0_buffer != nullptr, "Input buffer should be allocated on device");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    TT_FATAL(output.is_sharded(), "Output Tensor must be sharded.");
    auto output_shard_spec = output.shard_spec().value();

    uint32_t output_row_size_bytes = output_shard_spec.shape[1] * output.element_size();

    CoreRangeSet total_cores = output.shard_spec().value().grid;
    bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    std::vector<CoreCoord> iter_cores = corerange_to_cores(total_cores, std::nullopt, rm_orientation);

    uint32_t num_cores_channels =
        ttnn::operations::experimental::detail::get_num_cores_channels_from_sharded_tensor(output);

    bool pad_output_row = false;
    log_debug(tt::LogOp, "Input Shape {}, Padded Shape : {}", a.logical_shape(), a.padded_shape());

    uint32_t input_row_size_bytes = a.logical_shape()[-1] * a.element_size();
    input_row_size_bytes = input_row_size_bytes / num_cores_channels;

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    TT_FATAL(
        dst_buffer->buffer_type() == tt::tt_metal::BufferType::L1,
        "Output buffer should be L1 for padded_slice operation with tiled inputs");

    uint32_t output_cb_index = kPaddedSliceRmOutputCbIndex;
    uint32_t temp_pad_cb_index = 1;
    uint32_t non_aligned_temp_cb_index = 2;

    auto src_buffer_alignment = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();

    TT_FATAL(
        output_row_size_bytes % dst_buffer_alignment == 0,
        "Output row size {} must be aligned to the destination buffer {} alignment {}",
        output_row_size_bytes,
        dst_buffer->buffer_type(),
        dst_buffer_alignment);
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    auto is_non_aligned = false;
    if (output_row_size_bytes % alignment) {
        is_non_aligned = true;
    }

    // The kernel advances the write pointer by the aligned row size (stick_size_offset),
    // so the CB page size must match to avoid overflow.
    uint32_t output_cb_page_size =
        is_non_aligned ? tt::round_up(output_row_size_bytes, dst_buffer_alignment) : output_row_size_bytes;

    uint32_t num_output_sticks_per_core = output_shard_spec.shape[0];

    desc.cbs.push_back(make_padded_slice_rm_cb(
        output_cb_index, total_cores, output_cb_page_size, num_output_sticks_per_core, cb_data_format, dst_buffer));
    if (output_row_size_bytes > input_row_size_bytes) {
        pad_output_row = true;
        desc.cbs.push_back(
            make_padded_slice_rm_cb(temp_pad_cb_index, total_cores, output_row_size_bytes, 1, cb_data_format));
    } else {
        non_aligned_temp_cb_index = temp_pad_cb_index;  // Use the unused temp pad index so that CBs are continuous.
    }
    uint32_t num_trids = 2;
    if (is_non_aligned) {
        // Scratch page must accommodate padded_stick_size + worst-case misalignment.
        const uint32_t scratch_page_size =
            tt::align((a.logical_shape()[-1] * a.element_size()) + src_buffer_alignment, src_buffer_alignment);
        desc.cbs.push_back(make_padded_slice_rm_cb(
            non_aligned_temp_cb_index, total_cores, scratch_page_size, num_trids, cb_data_format));
    }

    std::vector<uint32_t> writer_compile_time_args_vec = {(std::uint32_t)output_cb_index};

    std::vector<uint32_t> reader_compile_time_args_vec = {
        (uint32_t)is_non_aligned, non_aligned_temp_cb_index, src_buffer_alignment, num_trids};
    tt::tt_metal::TensorAccessorArgs(src0_buffer).append_to(reader_compile_time_args_vec);

    KernelDescriptor reader_kernel;
    reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
        "padded_slice_reader_rm_interleaved_start_id.cpp";
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = total_cores;
    reader_kernel.compile_time_args = reader_compile_time_args_vec;
    reader_kernel.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_kernel;
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = total_cores;
    writer_kernel.config = WriterConfigDescriptor{};
    if (pad_output_row) {
        writer_compile_time_args_vec.push_back((std::uint32_t)temp_pad_cb_index);
        writer_compile_time_args_vec.push_back(output.element_size());
        writer_kernel.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
            "writer_unary_sharded_padded_rm.cpp";
    } else {
        writer_kernel.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp";
    }
    writer_kernel.compile_time_args = std::move(writer_compile_time_args_vec);

    auto all_runtime_args = get_padded_slice_runtime_args_rm_sharded_output(
        a, output, output_tensor_start, actual_output_shape, iter_cores);

    uint32_t i = 0;
    for (const auto& core : iter_cores) {
        reader_kernel.runtime_args.emplace_back(core, all_runtime_args[i].first);
        writer_kernel.runtime_args.emplace_back(core, all_runtime_args[i].second);
        i++;
    }

    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    return desc;
}

void PaddedSliceRMProgramFactory::override_runtime_arguments(
    Program& program,
    const PaddedSliceParams& operation_attributes,
    const PaddedSliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*coord*/) {
    const auto& src_tensor = tensor_args.input;
    auto& dst_tensor = output;
    TT_FATAL(dst_tensor.is_sharded(), "Output tensor must be sharded");
    auto* dst_buffer = dst_tensor.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    update_padded_slice_rm_output_cb(program, *dst_buffer);

    const auto output_shard_spec = dst_tensor.shard_spec().value();
    const bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const auto iter_cores = corerange_to_cores(output_shard_spec.grid, std::nullopt, rm_orientation);

    uint32_t core_index = 0;
    for (const auto& core : iter_cores) {
        auto& reader_args = GetRuntimeArgs(program, kPaddedSliceRmReaderKernelIdx, core);
        reader_args[kPaddedSliceRmReaderAddressArgIdx] = padded_slice_rm_reader_address(
            src_tensor, dst_tensor, operation_attributes.padded_slice_start, core, core_index);
        ++core_index;
    }
}

}  // namespace ttnn::experimental::prim
