// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "isin_program_factory.hpp"

#include "../isin_common.hpp"

#include <algorithm>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

enum class IsInCB : std::underlying_type_t<tt::CBIndex> {
    ELEMENTS = tt::CBIndex::c_0,
    TEST_ELEMENTS = tt::CBIndex::c_1,
    OUTPUT = tt::CBIndex::c_2
};

static CBDescriptor make_cb(
    const DataType& dtype,
    const IsInCB& is_in_cb,
    const CoreRangeSet& core_range_set,
    const uint32_t& page_size_bytes) {
    const uint32_t cb_id{static_cast<uint32_t>(is_in_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    return CBDescriptor{
        .total_size = page_size_bytes,
        .core_ranges = core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = cb_data_format,
            .page_size = page_size_bytes,
        }}},
    };
}

ProgramDescriptor IsInProgramFactory::create_descriptor(
    const IsinParams& args, const IsinInputs& tensor_args, Tensor& output_tensor) {
    ProgramDescriptor desc;

    const auto& elements_tensor = tensor_args.elements_tensor;
    const auto& test_elements_tensor = tensor_args.test_elements_tensor;

    const auto& elements_dtype = elements_tensor.dtype();
    const auto& test_elements_dtype = test_elements_tensor.dtype();

    const bool& invert = args.invert;
    const uint32_t& single_fetch_subchunk_size = args.single_fetch_subchunk_size;

    auto* elements_buffer = elements_tensor.buffer();
    auto* test_elements_buffer = test_elements_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();
    TT_FATAL(elements_buffer != nullptr, "Elements tensor's buffer is null");
    TT_FATAL(test_elements_buffer != nullptr, "Test elements tensor's buffer is null");
    TT_FATAL(output_buffer != nullptr, "Output tensor's buffer is null");

    // input dtype byte sizes
    const uint32_t& elements_datum_size = elements_tensor.element_size();
    const uint32_t& test_elements_datum_size = test_elements_tensor.element_size();
    const uint32_t& output_datum_size = output_tensor.element_size();

    // input row byte sizes
    const uint32_t& elements_subchunk_size_bytes = single_fetch_subchunk_size * elements_datum_size;
    const uint32_t& test_elements_subchunk_size_bytes = single_fetch_subchunk_size * test_elements_datum_size;
    const uint32_t& output_subchunk_size_bytes = single_fetch_subchunk_size * output_datum_size;

    auto* device = elements_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // Slots 0-2 are unused; TensorAccessorArgs starts at 11 and kernels read addresses from runtime args.
    std::vector<uint32_t> compile_time_args{
        0u,
        0u,
        0u,
        static_cast<uint32_t>(IsInCB::ELEMENTS),
        static_cast<uint32_t>(IsInCB::TEST_ELEMENTS),
        static_cast<uint32_t>(IsInCB::OUTPUT),
        elements_tensor.logical_volume(),
        test_elements_tensor.logical_volume(),
        single_fetch_subchunk_size,
        static_cast<uint32_t>(invert),
        elements_tensor.element_size()};
    tt::tt_metal::TensorAccessorArgs(*elements_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*test_elements_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*output_buffer).append_to(compile_time_args);

    // The final tensor to be dealt with by the isin device operation is flattened to 1D and the number of subchunks
    // (each of elements, test_elements and output) that they have to be split into depends on the L1 available size per
    // core (to hold elements, test_elements and output subchunks at the same time). The number of cores utilized is at
    // least the number of subchunks work has been split into.
    const uint32_t subchunks_num =
        (elements_tensor.logical_volume() + single_fetch_subchunk_size - 1) / single_fetch_subchunk_size;
    const auto
        [num_cores,                       // number of cores utilized
         all_cores,                       // set of all cores used
         core_group_1,                    // Primary core group
         core_group_2,                    // Secondary core group
         num_subchunks_per_core_group_1,  // Number of subchunks each core in the primary group processes
         num_subchunks_per_core_group_2   // Number of subchunks each core in the secondary group processes
    ] = split_work_to_cores(compute_with_storage_grid_size, subchunks_num);
    desc.cbs.push_back(make_cb(elements_dtype, IsInCB::ELEMENTS, all_cores, elements_subchunk_size_bytes));
    desc.cbs.push_back(
        make_cb(test_elements_dtype, IsInCB::TEST_ELEMENTS, all_cores, test_elements_subchunk_size_bytes));
    desc.cbs.push_back(make_cb(OUTPUT_TENSOR_DATA_TYPE, IsInCB::OUTPUT, all_cores, output_subchunk_size_bytes));

    constexpr const char* READER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/experimental/isin/device/kernels/dataflow/isin_reader.cpp";
    constexpr const char* WRITER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/experimental/isin/device/kernels/dataflow/isin_writer.cpp";

    KernelDescriptor reader_kernel;
    reader_kernel.kernel_source = READER_KERNEL_PATH;
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = all_cores;
    reader_kernel.compile_time_args = compile_time_args;
    reader_kernel.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_kernel;
    writer_kernel.kernel_source = WRITER_KERNEL_PATH;
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = all_cores;
    writer_kernel.compile_time_args = std::move(compile_time_args);
    writer_kernel.config = WriterConfigDescriptor{};

    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const auto cores = grid_to_cores(num_cores, compute_with_storage_grid_size.x, num_cores_y);
    uint32_t subchunks_offset = 0;
    std::for_each(cores.cbegin(), cores.cend(), [&](const CoreCoord& core) {
        uint32_t subchunks_per_core = 0;
        if (core_group_1.contains(core)) {
            subchunks_per_core = num_subchunks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            subchunks_per_core = num_subchunks_per_core_group_2;
        } else {
            TT_THROW("Core not in any predefined core range.");
        }

        reader_kernel.emplace_runtime_args(
            core, {elements_buffer, test_elements_buffer, subchunks_per_core, subchunks_offset});
        writer_kernel.emplace_runtime_args(core, {output_buffer, subchunks_per_core, subchunks_offset});

        subchunks_offset += subchunks_per_core;
    });

    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    return desc;
}
}  // namespace ttnn::experimental::prim
