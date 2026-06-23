// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bh_dram_read_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::examples {

using namespace tt;
using namespace tt::tt_metal;

ProgramDescriptor BhDramReadDeviceOperation::DramBankCore::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto* src_buffer = input_tensor.buffer();
    auto* device = input_tensor.device();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t total_pages = input_tensor.physical_volume() / constants::TILE_HW;

    // One worker core per DRAM bank.
    const uint32_t num_banks = static_cast<uint32_t>(device->num_dram_channels());
    CoreCoord grid = device->compute_with_storage_grid_size();
    TT_FATAL(
        num_banks <= grid.x * grid.y,
        "bh_dram_read: {} DRAM banks exceed {} available worker cores",
        num_banks,
        grid.x * grid.y);

    CoreRangeSet all_cores = num_cores_to_corerangeset(num_banks, grid, /*row_wise=*/true);
    std::vector<CoreCoord> cores = corerange_to_cores(all_cores, num_banks, /*row_wise=*/true);

    // ---- Build the ProgramDescriptor ----
    ProgramDescriptor desc;

    // Circular buffer: double-buffered staging for the discarded reads.
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    constexpr uint32_t num_input_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // Reader-only kernel. The TensorAccessor maps page_id -> bank, so reading
    // page_ids {bank_id, bank_id + num_banks, ...} keeps every read on the
    // core's assigned bank.
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/examples/bh_dram_read/device/kernels/dataflow/reader_bh_dram_read.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    // Per-core runtime args: each core reads the pages that live in its bank.
    for (uint32_t bank_id = 0; bank_id < num_banks; ++bank_id) {
        // Pages in bank b (interleaved round-robin): b, b+num_banks, b+2*num_banks, ...
        uint32_t pages_in_bank = (total_pages > bank_id) ? ((total_pages - bank_id - 1) / num_banks + 1) : 0;
        reader_desc.runtime_args.emplace_back(
            cores[bank_id],
            KernelDescriptor::CoreRuntimeArgs{src_buffer->address(), pages_in_bank, bank_id, num_banks});
    }

    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

}  // namespace ttnn::operations::examples
