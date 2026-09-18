// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for pixel_unshuffle on NCHW ROW_MAJOR interleaved input.
//
// Work unit: one INPUT ROW, which feeds exactly r output sticks (one per rw).
// Distributing rows rather than output sticks removes the r-fold input over-fetch the
// stick-per-read scheme had, and lets each row be read with a sequential page index.
//
// The cost of this op is the deinterleave - a strided RISC-V L1 copy; there is no
// hardware strided/gather NOC read to offload it to. So the rows of each core are split
// across BOTH dataflow RISCs, which run the same kernel over disjoint row ranges and
// each own private input/scratch L1 buffers. A reader/writer pair left one RISC idle
// after its DRAM reads while the other carried every element.
//
// Each RISC runs a shallow software pipeline over its rows, tagging each slot with its
// own NOC transaction id so a read, a copy and a set of writes are all in flight at once.

#include "pixel_unshuffle_device_op.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/common/constants.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

using namespace tt::constants;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor PixelUnshuffle::MultiCore::create_descriptor(
    const operation_attributes_t& op_attr, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    const Tensor& input = tensor_args.input_tensor;
    const Tensor& output = output_tensor;
    auto* device = input.device();

    const auto& in_shape = input.logical_shape();
    const uint32_t N = in_shape[0];
    const uint32_t C = in_shape[1];
    const uint32_t H = in_shape[2];
    const uint32_t W = in_shape[3];
    const uint32_t r = op_attr.downscale_factor;
    const uint32_t Ho = H / r;
    const uint32_t Wo = W / r;

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(src_buffer != nullptr && dst_buffer != nullptr, "PixelUnshuffle: buffers must be allocated.");

    tt::DataFormat in_data_fmt = datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat out_data_fmt = datatype_to_dataformat_converter(output.dtype());

    const uint32_t datum_sz = tt::datum_size(in_data_fmt);
    const uint32_t out_datum_sz = tt::datum_size(out_data_fmt);
    const uint32_t stick_nbytes_in = W * datum_sz;
    const uint32_t stick_nbytes_out = Wo * out_datum_sz;

    // Align CB page to DRAM read alignment (typically 32 bytes)
    const uint32_t aligned_stick_nbytes_in = tt::align(stick_nbytes_in, hal::get_dram_alignment());

    // Pipeline depth in rows: each RISC holds `depth` input rows and `depth` groups of r
    // output sticks, and tags each slot with its own NOC transaction id, so `depth` reads
    // and `depth` groups of writes are in flight at once. Reads are the critical path,
    // so this depth is what keeps DRAM busy.
    constexpr uint32_t depth = 2;

    // Work unit is one input row; each row produces r output sticks.
    const uint32_t total_rows = N * C * H;

    auto compute_grid = device->compute_with_storage_grid_size();
    const uint32_t ncores_x = compute_grid.x;
    const uint32_t ncores_y = compute_grid.y;
    const uint32_t ncores = ncores_x * ncores_y;

    CoreRangeSet all_cores{CoreRange({0, 0}, {ncores_x - 1, ncores_y - 1})};
    auto cores = grid_to_cores(ncores, ncores_x, ncores_y, true);

    ProgramDescriptor desc;

    const uint32_t aligned_stick_nbytes_out = tt::align(stick_nbytes_out, hal::get_dram_alignment());

    // Four private L1 buffers: an input buffer and a scratch buffer per dataflow RISC.
    // They are never pushed/popped - the kernels use them as plain allocations and index
    // slots by offset - so the two RISCs never synchronise with each other.
    const uint32_t cb_in_idx[2] = {tt::CBIndex::c_0, tt::CBIndex::c_1};
    const uint32_t cb_scratch_idx[2] = {tt::CBIndex::c_2, tt::CBIndex::c_3};

    for (uint32_t k = 0; k < 2; k++) {
        CBDescriptor cb_in;
        cb_in.total_size = depth * aligned_stick_nbytes_in;
        cb_in.core_ranges = all_cores;
        cb_in.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_in_idx[k]),
            .data_format = in_data_fmt,
            .page_size = aligned_stick_nbytes_in,
        });
        desc.cbs.push_back(std::move(cb_in));

        CBDescriptor cb_scratch;
        cb_scratch.total_size = depth * r * aligned_stick_nbytes_out;
        cb_scratch.core_ranges = all_cores;
        cb_scratch.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_scratch_idx[k]),
            .data_format = out_data_fmt,
            .page_size = aligned_stick_nbytes_out,
        });
        desc.cbs.push_back(std::move(cb_scratch));
    }

    // Both RISCs run the same kernel source; only the CB indices (compile-time) and the
    // row range (runtime) differ.
    constexpr const char* kernel_src =
        "ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/kernels/dataflow/"
        "pixel_unshuffle_nchw.cpp";

    KernelDescriptor kernels[2];
    for (uint32_t k = 0; k < 2; k++) {
        std::vector<uint32_t> cta = {
            stick_nbytes_in,                               // 0
            aligned_stick_nbytes_in,                       // 1
            stick_nbytes_out,                              // 2
            aligned_stick_nbytes_out,                      // 3
            cb_in_idx[k],                                  // 4
            cb_scratch_idx[k],                             // 5
            r,                                             // 6
            W,                                             // 7
            C,                                             // 8
            H,                                             // 9
            Ho,                                            // 10
            static_cast<uint32_t>(op_attr.channel_order),  // 11 (0=CHANNEL_MAJOR, 1=SPATIAL_MAJOR)
            depth,                                         // 12
        };
        TensorAccessorArgs(*src_buffer).append_to(cta);
        TensorAccessorArgs(*dst_buffer).append_to(cta);

        kernels[k].kernel_source = kernel_src;
        kernels[k].source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernels[k].core_ranges = all_cores;
        kernels[k].compile_time_args = std::move(cta);
    }
    kernels[0].config = ReaderConfigDescriptor{};  // NCRISC
    kernels[1].config = WriterConfigDescriptor{};  // BRISC

    // Rows are handed out INTERLEAVED across all dataflow RISCs, not as contiguous
    // blocks. Input page index == row index, and an interleaved DRAM buffer maps page p
    // to bank p % num_banks, so a contiguous block per core made every RISC target the
    // same bank at the same step (block base 12*g is 0 mod 12 for every g) - one bank of
    // twelve doing all the work. Interleaving makes RISC g read rows g, g+nriscs, ... so
    // concurrent reads spread evenly over every bank.
    const uint32_t nriscs = static_cast<uint32_t>(cores.size()) * 2;
    for (uint32_t i = 0; i < cores.size(); i++) {
        const CoreCoord core = cores[i];
        for (uint32_t k = 0; k < 2; k++) {
            const uint32_t g = i * 2 + k;  // global dataflow-RISC index
            const uint32_t count = (g < total_rows) ? tt::div_up(total_rows - g, nriscs) : 0u;
            kernels[k].emplace_runtime_args(core, {src_buffer, dst_buffer, g, count, nriscs});
        }
    }

    desc.kernels.push_back(std::move(kernels[0]));
    desc.kernels.push_back(std::move(kernels[1]));

    return desc;
}

}  // namespace ttnn::operations::data_movement
