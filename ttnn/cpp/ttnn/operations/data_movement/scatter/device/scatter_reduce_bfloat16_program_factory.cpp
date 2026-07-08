// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_reduce_bfloat16_program_factory.hpp"

#include "scatter_common.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor ScatterReduceBfloat16ProgramFactory::create_descriptor(
    const ScatterParams& args, const ScatterInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor{tensor_args.input_tensor};
    const auto& input_shape{input_tensor.logical_shape()};
    const auto& index_tensor{tensor_args.index_tensor};
    const auto& index_shape{index_tensor.logical_shape()};
    const auto& src_tensor{tensor_args.src_tensor};
    const auto& src_shape{src_tensor.logical_shape()};
    const auto& output_shape{output_tensor.logical_shape()};

    auto* input_buffer = input_tensor.buffer();
    auto* index_buffer = index_tensor.buffer();
    auto* src_buffer = src_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();

    const uint32_t input_stick_size = input_shape[-1];
    const uint32_t index_stick_size = index_shape[-1];
    const uint32_t source_stick_size = src_shape[-1];
    const uint32_t output_stick_size = output_shape[-1];

    // input dtype byte sizes
    const uint32_t input_datum_size = input_tensor.element_size();
    const uint32_t index_datum_size = index_tensor.element_size();
    const uint32_t source_datum_size = src_tensor.element_size();
    const uint32_t output_datum_size = output_tensor.element_size();
    const uint32_t fp32_temp_datum_size = sizeof(float);

    // input row byte sizes
    const uint32_t input_stick_size_bytes = input_stick_size * input_datum_size;
    const uint32_t index_stick_size_bytes = index_stick_size * index_datum_size;
    const uint32_t source_stick_size_bytes = source_stick_size * source_datum_size;
    const uint32_t output_stick_size_bytes = output_stick_size * output_datum_size;

    // maximal input/index/source/output chunk size, divisible by 32, calculated as follows:
    // BH available L1 mem size of nearly 1.5 MB...
    // ... divided by 5 to be able to allocate five equally long row chunks (coming from input/index/source/output
    // tensors)
    // ... divided by 4 to account for 4-byte datum sizes of each tensor (fp32, int32)
    // ... minimized by 120% to account for reserved memory
    const uint32_t input_and_output_max_chunk_size = calculate_optimal_chunk_size(input_tensor);
    const uint32_t index_and_source_max_chunk_size = input_and_output_max_chunk_size;
    const uint32_t input_and_output_chunk_size = std::min(input_stick_size, input_and_output_max_chunk_size);
    const uint32_t index_chunk_size = std::min(index_stick_size, index_and_source_max_chunk_size);
    const uint32_t source_chunk_size = std::min(source_stick_size, index_and_source_max_chunk_size);
    const uint32_t input_and_output_chunk_size_bytes = input_and_output_chunk_size * input_datum_size;
    const uint32_t index_chunk_size_bytes = index_chunk_size * index_datum_size;
    const uint32_t source_chunk_size_bytes = source_chunk_size * source_datum_size;
    const uint32_t fp32_temp_chunk_size_bytes = input_and_output_chunk_size * fp32_temp_datum_size;

    // pad pages to 32
    const uint32_t input_page_size_bytes = ceil32(input_and_output_chunk_size_bytes);
    const uint32_t index_page_size_bytes = ceil32(index_chunk_size_bytes);
    const uint32_t source_page_size_bytes = ceil32(source_chunk_size_bytes);
    const uint32_t output_page_size_bytes = ceil32(input_and_output_chunk_size_bytes);
    const uint32_t fp32_temp_page_size_bytes = ceil32(fp32_temp_chunk_size_bytes);

    constexpr const char* reader_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/scatter/device/kernels/dataflow/reader_bf16_reduction_scatter.cpp";
    constexpr const char* writer_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/scatter/device/kernels/dataflow/writer_bf16_reduction_scatter.cpp";

    std::vector<uint32_t> compile_time_args{
        input_buffer->address(),
        index_buffer->address(),
        src_buffer->address(),
        output_buffer->address(),
        static_cast<uint32_t>(ScatterCB::INPUT),
        static_cast<uint32_t>(ScatterCB::INDEX),
        static_cast<uint32_t>(ScatterCB::SRC),
        static_cast<uint32_t>(ScatterCB::DST),
        static_cast<uint32_t>(ScatterCB::FP32_TEMP),
        input_stick_size,
        index_stick_size,
        source_stick_size,
        output_stick_size,
        input_stick_size_bytes,
        index_stick_size_bytes,
        source_stick_size_bytes,
        output_stick_size_bytes,
        input_shape.rank()};
    TensorAccessorArgs(*input_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*index_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*output_buffer).append_to(compile_time_args);

    auto* device = input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t work_units = input_tensor.logical_volume() / input_stick_size;
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
            args.sub_core_grid.has_value()
                ? tt::tt_metal::split_work_to_cores(*args.sub_core_grid, work_units)
                : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, work_units);

    const auto fp32_temp_dtype = DataType::FLOAT32;
    const auto farthest_x_y =
        args.sub_core_grid.has_value() ? args.sub_core_grid->bounding_box().end_coord : compute_with_storage_grid_size;
    const uint32_t all_cores_in_bounding_box = (farthest_x_y.x + 1) * (farthest_x_y.y + 1);

    ProgramDescriptor desc;

    auto add_cb = [&](ScatterCB scatter_cb, DataType dtype, uint32_t page_size_bytes) {
        const uint32_t cb_id = static_cast<uint32_t>(scatter_cb);
        desc.cbs.push_back(CBDescriptor{
            .total_size = page_size_bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_id),
                .data_format = datatype_to_dataformat_converter(dtype),
                .page_size = page_size_bytes,
            }}},
        });
    };

    add_cb(ScatterCB::INPUT, input_tensor.dtype(), input_page_size_bytes);
    add_cb(ScatterCB::INDEX, index_tensor.dtype(), index_page_size_bytes);
    add_cb(ScatterCB::SRC, src_tensor.dtype(), source_page_size_bytes);
    add_cb(ScatterCB::DST, output_tensor.dtype(), output_page_size_bytes);
    add_cb(ScatterCB::FP32_TEMP, fp32_temp_dtype, fp32_temp_page_size_bytes);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_path;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_path;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t stick_offset = 0;
    for (uint32_t i = 0; i < all_cores_in_bounding_box; ++i) {
        const CoreCoord core{i / (farthest_x_y.y + 1), i % (farthest_x_y.y + 1)};
        uint32_t sticks_per_core;
        if (core_group_1.contains(core)) {
            sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            sticks_per_core = num_sticks_per_core_group_2;
        } else {
            continue;
        }

        // Buffer* entries become BufferBinding slots; addresses are patched on cache hits
        // without rebuilding the descriptor.
        KernelDescriptor::RTArgList reader_runtime_args;
        reader_runtime_args.reserve(9 + (input_shape.rank() - 1) + (index_shape.rank() - 1));
        reader_runtime_args.push_back(input_buffer);
        reader_runtime_args.push_back(index_buffer);
        reader_runtime_args.push_back(src_buffer);
        reader_runtime_args.push_back(stick_offset);
        reader_runtime_args.push_back(sticks_per_core);
        reader_runtime_args.push_back(input_and_output_chunk_size);
        reader_runtime_args.push_back(index_chunk_size);
        reader_runtime_args.push_back(source_chunk_size);
        reader_runtime_args.push_back(static_cast<uint32_t>(args.opt_reduction));
        for (const auto* it = input_shape.cbegin(); it != input_shape.cend() - 1; ++it) {
            reader_runtime_args.push_back(static_cast<uint32_t>(*it));
        }
        for (const auto* it = index_shape.cbegin(); it != index_shape.cend() - 1; ++it) {
            reader_runtime_args.push_back(static_cast<uint32_t>(*it));
        }
        reader_desc.emplace_runtime_args(core, reader_runtime_args);

        writer_desc.emplace_runtime_args(
            core, {output_buffer, stick_offset, sticks_per_core, input_and_output_chunk_size});

        stick_offset += sticks_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
