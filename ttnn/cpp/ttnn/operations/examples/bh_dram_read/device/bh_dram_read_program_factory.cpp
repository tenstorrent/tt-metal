// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bh_dram_read_device_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::examples {

using namespace tt;
using namespace tt::tt_metal;

// Must match the kernel: NUM_TRIDS transactions of PACKET bytes are kept in
// flight, so the L1 scratch (CB) holds NUM_TRIDS packet-sized slots.
static constexpr uint32_t PACKET_BYTES = 16384;  // NOC_MAX_BURST_SIZE on Blackhole
static constexpr uint32_t NUM_TRIDS = 8;

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
    // For an interleaved DRAM buffer each bank stores its pages contiguously at
    // the DRAM-aligned page size, so a bank's region is pages_in_bank * stride.
    const uint32_t aligned_page_size = static_cast<uint32_t>(src_buffer->aligned_page_size());

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

    // Circular buffer: plain tile-paged L1 scratch, sized to hold NUM_TRIDS
    // max-size packets worth of over-read data. Page size is the tile (not the
    // packet) -- a packet just spans several tile pages, and the kernel indexes
    // the buffer directly by packet slot.
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = NUM_TRIDS * PACKET_BYTES,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // Reader-only kernel. Each core streams its bank's contiguous region in
    // max-size NOC packets, addressing the bank directly via bank_id.
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/examples/bh_dram_read/device/kernels/dataflow/reader_bh_dram_read.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.config = ReaderConfigDescriptor{};

    // Per-core runtime args: each core reads the bytes that live in its bank.
    for (uint32_t bank_id = 0; bank_id < num_banks; ++bank_id) {
        // Pages in bank b (interleaved round-robin): b, b+num_banks, b+2*num_banks, ...
        uint32_t pages_in_bank = (total_pages > bank_id) ? ((total_pages - bank_id - 1) / num_banks + 1) : 0;
        uint32_t region_bytes = pages_in_bank * aligned_page_size;
        reader_desc.runtime_args.emplace_back(
            cores[bank_id], KernelDescriptor::CoreRuntimeArgs{src_buffer->address(), region_bytes, bank_id});
    }

    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

}  // namespace ttnn::operations::examples
