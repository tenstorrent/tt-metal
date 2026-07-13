// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bh_dram_read_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <cstdlib>

namespace ttnn::operations::examples {

using namespace tt;
using namespace tt::tt_metal;

// NUM_TRIDS transactions of PACKET bytes are kept in flight, so the L1 scratch
// (CB) holds NUM_TRIDS packet-sized slots. NUM_TRIDS is the buffering depth; it
// can be overridden via the BH_DRAM_READ_NUM_TRIDS env var for tuning sweeps
// (valid trids are 1..15 on Blackhole).
static constexpr uint32_t PACKET_BYTES = 16384;  // NOC_MAX_BURST_SIZE on Blackhole
// A trid sweep showed bandwidth saturates at depth 2 (double-buffering fully
// hides DRAM latency); 2..15 are byte-identical, only depth 1 is slower. So the
// default is 2 -- same bandwidth as 8 at 4x less L1. See bh-dram-read-progress.md.
static constexpr uint32_t DEFAULT_NUM_TRIDS = 2;

static uint32_t get_num_trids() {
    if (const char* e = std::getenv("BH_DRAM_READ_NUM_TRIDS")) {
        int v = std::atoi(e);
        if (v >= 1 && v <= 15) {
            return static_cast<uint32_t>(v);
        }
    }
    return DEFAULT_NUM_TRIDS;
}

ProgramDescriptor BhDramReadDeviceOperation::DramBankCore::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto* src_buffer = input_tensor.buffer();
    auto* device = input_tensor.device();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    const uint32_t num_trids = get_num_trids();  // buffering depth (outstanding reads per core)

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

    // Place each bank's reader on the worker core that is optimal for NOC0 reads
    // from that bank (bank-adjacent placement). Indexed by bank id.
    std::vector<CoreCoord> cores = device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0);
    TT_FATAL(
        cores.size() >= num_banks,
        "bh_dram_read: optimal assignment returned {} cores < {} banks",
        cores.size(),
        num_banks);
    std::vector<CoreRange> core_ranges;
    core_ranges.reserve(num_banks);
    for (uint32_t b = 0; b < num_banks; ++b) {
        core_ranges.emplace_back(cores[b]);
    }
    CoreRangeSet all_cores(core_ranges);

    // ---- Build the ProgramDescriptor ----
    ProgramDescriptor desc;

    // Circular buffer: plain tile-paged L1 scratch, sized to hold NUM_TRIDS
    // max-size packets worth of over-read data. Page size is the tile (not the
    // packet) -- a packet just spans several tile pages, and the kernel indexes
    // the buffer directly by packet slot.
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_trids * PACKET_BYTES,
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
    reader_desc.compile_time_args = {num_trids};
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
