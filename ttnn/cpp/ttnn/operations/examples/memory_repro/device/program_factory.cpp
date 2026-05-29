// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "memory_repro_device_operation.hpp"

namespace ttnn::operations::examples {

using namespace tt;
using namespace tt::tt_metal;

ProgramDescriptor MemoryReproDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
    // Two single-core ranges -- one CB per core, deliberately of different sizes.
    CoreRange core_0_range(CoreCoord{0, 0}, CoreCoord{0, 0});
    CoreRange core_1_range(CoreCoord{1, 0}, CoreCoord{1, 0});
    CoreRangeSet core_0_set(core_0_range);
    CoreRangeSet core_1_set(core_1_range);
    CoreRangeSet all_cores_set(std::vector<CoreRange>{core_0_range, core_1_range});

    constexpr uint32_t cb0_total_size = 1048576;  // 1.0 MiB (1024 * 1024)
    constexpr uint32_t cb1_total_size = 768000;   // 750 KiB (750 * 1024)
    constexpr uint32_t cb_page_size = 1024;
    constexpr uint32_t cb_index = CBIndex::c_0;
    constexpr tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb0_total_size,
        .core_ranges = core_0_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_index,
            .data_format = cb_data_format,
            .page_size = cb_page_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb1_total_size,
        .core_ranges = core_1_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_index,
            .data_format = cb_data_format,
            .page_size = cb_page_size,
        }}},
    });

    KernelDescriptor blank_kernel;
    blank_kernel.kernel_source = "tt_metal/kernels/dataflow/blank.cpp";
    blank_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    blank_kernel.core_ranges = all_cores_set;
    blank_kernel.config = ReaderConfigDescriptor{};

    desc.kernels.push_back(std::move(blank_kernel));

    return desc;
}

}  // namespace ttnn::operations::examples
