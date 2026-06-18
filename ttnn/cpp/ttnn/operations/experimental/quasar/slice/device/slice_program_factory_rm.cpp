// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_program_factory_rm.hpp"

#include <optional>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::quasar {

namespace {

inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_slice_runtime_args_rm(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    uint32_t num_cores,
    const std::vector<CoreCoord>& all_cores_vec,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_sticks_per_core_group_1,
    uint32_t num_sticks_per_core_group_2,
    uint32_t max_read_size) {
    auto* output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();

    uint32_t padded_row_size_bytes = input_shape[-1] * input_tensor.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    num_unpadded_sticks_per_dim[0] = 1;
    num_padded_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_sticks_per_dim[i] = num_unpadded_dim;
        num_padded_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    auto src_buffer_alignment = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
    uint32_t begins_bytes = output_tensor_start[-1] * input_tensor.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;
    uint32_t unpadded_row_size_bytes_offset = tt::round_up(unpadded_row_size_bytes, alignment);
    uint32_t start_addr = input_tensor.buffer()->address();

    // shard_W * elem_size for B/W-sharded (splits row across shards); full row otherwise.
    // Fallback is padded for the reader tensor, unpadded for the writer tensor.
    const auto per_shard_page_size_bytes = [&](const Tensor& t, uint32_t row_bytes) -> uint32_t {
        const auto& mc = t.memory_config();
        if (mc.is_sharded() && (mc.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                                mc.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED)) {
            const auto& spec = mc.shard_spec().value();
            return spec.shape[1] * t.element_size();
        }
        return row_bytes;
    };
    const uint32_t reader_page_size = per_shard_page_size_bytes(input_tensor, padded_row_size_bytes);

    std::vector<uint32_t> common_reader_kernel_args = {
        start_addr + begins_bytes - misalignment,  // read from nearest aligned address,
        reader_page_size,
        unpadded_row_size_bytes,
        unpadded_row_size_bytes_offset,
        num_dims,
        misalignment,
        0,
        0,
        0,
        0};
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_unpadded_sticks_per_dim.begin(), num_unpadded_sticks_per_dim.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_padded_sticks_per_dim.begin(), num_padded_sticks_per_dim.end());

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val;
    ret_val.reserve(num_cores);

    uint32_t start_offset =
        ttnn::operations::experimental::quasar::get_rm_start_offset(input_tensor, output_tensor_start);
    uint32_t num_sticks_written = 0;
    for (const auto& core : all_cores_vec) {
        uint32_t num_sticks_per_core;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            num_sticks_per_core = 0;
        }

        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            auto num_sticks_per_core_pad32 = num_sticks_per_core + ((32 - num_sticks_per_core % 32) % 32);
            num_sticks_per_core_read = tt::tt_metal::merge_num_sticks_to_read(
                num_sticks_per_core_pad32, unpadded_row_size_bytes_offset, max_read_size);
            num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
        }

        id_per_dim[0] = num_sticks_written % num_unpadded_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / num_unpadded_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }
        std::vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        uint32_t addr_offset = 6;
        reader_kernel_args[addr_offset++] = start_id;
        reader_kernel_args[addr_offset++] = num_sticks_per_core;
        reader_kernel_args[addr_offset++] = num_sticks_per_core_read;
        reader_kernel_args[addr_offset] = num_read_per_barrier;
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        const uint32_t writer_page_size = per_shard_page_size_bytes(output_tensor, unpadded_row_size_bytes);
        std::vector<uint32_t> writer_kernel_args = {
            output_buffer->address(),
            unpadded_row_size_bytes,
            unpadded_row_size_bytes_offset,
            num_sticks_per_core,
            num_sticks_per_core_read,
            num_read_per_barrier,
            num_sticks_written,
            writer_page_size,
        };
        num_sticks_written += num_sticks_per_core;
        ret_val.emplace_back(reader_kernel_args, writer_kernel_args);
    }

    return ret_val;
}

constexpr uint32_t MAX_READ_SIZE = 4096;

std::tuple<uint32_t, uint32_t, uint32_t> compute_cb_size(
    const Tensor& input,
    const Tensor& output,
    const Shape& output_tensor_start,
    const uint32_t num_sticks_per_core_group_1,
    const uint32_t num_sticks_per_core_group_2) {
    auto src_buffer_alignment = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    const auto single_alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
    auto alignment = single_alignment;

    uint32_t begins_bytes = output_tensor_start[-1] * input.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;

    if (misalignment != 0) {
        alignment *= 2;
    }
    const uint32_t unpadded_row_size_bytes = output.padded_shape()[-1] * input.element_size();
    const uint32_t cb_page_size = tt::round_up(unpadded_row_size_bytes, alignment);
    const uint32_t stick_stride_for_merge = tt::round_up(unpadded_row_size_bytes, single_alignment);
    const uint32_t num_input_pages = num_sticks_per_core_group_1 > num_sticks_per_core_group_2
                                         ? num_sticks_per_core_group_1
                                         : num_sticks_per_core_group_2;
    uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
    if (num_input_pages != 0) {
        auto num_sticks_per_core_pad32 = num_input_pages + ((32 - num_input_pages % 32) % 32);
        num_sticks_per_core_read =
            tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core_pad32, stick_stride_for_merge, MAX_READ_SIZE);
        num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
    }

    return std::make_tuple(cb_page_size, num_read_per_barrier, misalignment);
}

}  // namespace

}  // namespace ttnn::operations::experimental::quasar

namespace ttnn::prim::qsr {

tt::tt_metal::ProgramDescriptor SliceRmProgramFactory::create_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    tt::tt_metal::IDevice* device = input.device();
    ProgramDescriptor desc;

    uint32_t num_unpadded_sticks = output.physical_volume() / output.padded_shape()[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_sticks)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

    tt::tt_metal::Buffer* src0_buffer = input.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    constexpr uint8_t src0_cb_index = 0;

    // CB sizing varies with slice_start; padded_shape folds into compute_program_hash() so each
    // unique CB layout gets its own cache entry (total_size/page_size are not patched on cache hit).
    const auto [cb_page_size, num_read_per_barrier, misalignment] =
        ttnn::operations::experimental::quasar::compute_cb_size(
            input, output, args.slice_start, num_sticks_per_core_group_1, num_sticks_per_core_group_2);

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_read_per_barrier * 2 * cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = cb_page_size,
        }}},
    });

    std::vector<uint32_t> writer_compile_time_args_vec = {static_cast<uint32_t>(src0_cb_index)};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args_vec);

    std::vector<uint32_t> reader_compile_time_args_vec;
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args_vec);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/"
        "slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args_vec);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/"
        "slice_writer_unary_stick_layout_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args_vec);
    writer_desc.config = WriterConfigDescriptor{};

    auto all_cores_vec = corerange_to_cores(all_cores);
    auto all_runtime_args = ttnn::operations::experimental::quasar::get_slice_runtime_args_rm(
        input,
        output,
        args.slice_start,
        num_cores,
        all_cores_vec,
        core_group_1,
        core_group_2,
        num_sticks_per_core_group_1,
        num_sticks_per_core_group_2,
        ttnn::operations::experimental::quasar::MAX_READ_SIZE);

    reader_desc.runtime_args.reserve(all_cores_vec.size());
    writer_desc.runtime_args.reserve(all_cores_vec.size());
    for (size_t i = 0; i < all_cores_vec.size(); ++i) {
        reader_desc.runtime_args.emplace_back(all_cores_vec[i], std::move(all_runtime_args[i].first));
        writer_desc.runtime_args.emplace_back(all_cores_vec[i], std::move(all_runtime_args[i].second));
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim::qsr
