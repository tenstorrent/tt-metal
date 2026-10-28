// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for pixel_shuffle on NCHW ROW_MAJOR input (the mirror of the pixel_unshuffle one).
//
// Work unit: one OUTPUT ROW (n, c, h_out), which is the interleave of exactly r input rows (one per rw).
// The r input rows are read with one NoC read each (TensorAccessor page = input row index), the RISC
// interleaves them into one output stick in a private scratch slot and writes it with one NoC write.
// As in pixel_unshuffle the interleave is the cost (a RISC-V L1 copy), so the rows of every core are
// split across BOTH dataflow RISCs, which run the same kernel over disjoint, interleaved row ranges with
// private L1 buffers, each with a shallow two-slot software pipeline (the next row's r reads are issued
// before the current row is copied).

#include "pixel_shuffle_device_op.hpp"

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

tt::tt_metal::ProgramDescriptor PixelShuffle::MultiCore::create_descriptor(
    const operation_attributes_t& op_attr, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    const Tensor& input = tensor_args.input_tensor;
    const Tensor& output = output_tensor;
    auto* device = input.device();

    const auto& in_shape = input.logical_shape();
    const uint32_t N = in_shape[0];
    const uint32_t C_in = in_shape[1];
    const uint32_t H = in_shape[2];
    const uint32_t W = in_shape[3];
    const uint32_t r = op_attr.upscale_factor;
    const uint32_t C = C_in / (r * r);
    const uint32_t Ho = H * r;
    const uint32_t Wo = W * r;

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(src_buffer != nullptr && dst_buffer != nullptr, "PixelShuffle: buffers must be allocated.");

    const tt::DataFormat in_df = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    const uint32_t datum_sz = tt::datum_size(in_df);
    const uint32_t stick_nbytes_in = W * datum_sz;
    const uint32_t stick_nbytes_out = Wo * tt::datum_size(out_df);
    const uint32_t aligned_stick_nbytes_in = tt::align(stick_nbytes_in, hal::get_dram_alignment());
    const uint32_t aligned_stick_nbytes_out = tt::align(stick_nbytes_out, hal::get_dram_alignment());

    constexpr uint32_t depth = 2;  // pipeline slots: r input sticks + 1 output stick each
    const uint32_t total_rows = N * C * Ho;  // output rows

    auto compute_grid = device->compute_with_storage_grid_size();
    const uint32_t ncores_x = compute_grid.x;
    const uint32_t ncores_y = compute_grid.y;
    const uint32_t ncores = ncores_x * ncores_y;
    CoreRangeSet all_cores{CoreRange({0, 0}, {ncores_x - 1, ncores_y - 1})};
    auto cores = grid_to_cores(ncores, ncores_x, ncores_y, true);

    ProgramDescriptor desc;
    const uint32_t cb_in_idx[2] = {tt::CBIndex::c_0, tt::CBIndex::c_1};
    const uint32_t cb_scratch_idx[2] = {tt::CBIndex::c_2, tt::CBIndex::c_3};
    for (uint32_t k = 0; k < 2; k++) {
        CBDescriptor cb_in;  // depth slots of r input sticks
        cb_in.total_size = depth * r * aligned_stick_nbytes_in;
        cb_in.core_ranges = all_cores;
        cb_in.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_in_idx[k]), .data_format = in_df, .page_size = aligned_stick_nbytes_in});
        desc.cbs.push_back(std::move(cb_in));
        CBDescriptor cb_scratch;  // depth output sticks
        cb_scratch.total_size = depth * aligned_stick_nbytes_out;
        cb_scratch.core_ranges = all_cores;
        cb_scratch.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_scratch_idx[k]), .data_format = out_df, .page_size = aligned_stick_nbytes_out});
        desc.cbs.push_back(std::move(cb_scratch));
    }

    constexpr const char* kernel_src =
        "ttnn/cpp/ttnn/operations/data_movement/pixel_shuffle/device/kernels/dataflow/pixel_shuffle_nchw.cpp";
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
            C,                                             // 8 output channels
            H,                                             // 9 input height
            static_cast<uint32_t>(op_attr.channel_order),  // 10 (0=CHANNEL_MAJOR, 1=SPATIAL_MAJOR)
            depth,                                         // 11
        };
        TensorAccessorArgs(*src_buffer).append_to(cta);
        TensorAccessorArgs(*dst_buffer).append_to(cta);
        kernels[k].kernel_source = kernel_src;
        kernels[k].source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernels[k].core_ranges = all_cores;
        kernels[k].compile_time_args = std::move(cta);
    }
    kernels[0].config = ReaderConfigDescriptor{};
    kernels[1].config = WriterConfigDescriptor{};

    // Output rows handed out interleaved across all dataflow RISCs (see pixel_unshuffle: a contiguous block
    // per core would make every RISC hit the same DRAM bank at the same step).
    const uint32_t nriscs = static_cast<uint32_t>(cores.size()) * 2;
    for (uint32_t i = 0; i < cores.size(); i++) {
        for (uint32_t k = 0; k < 2; k++) {
            const uint32_t g = i * 2 + k;
            const uint32_t count = (g < total_rows) ? tt::div_up(total_rows - g, nriscs) : 0u;
            kernels[k].emplace_runtime_args(cores[i], {src_buffer, dst_buffer, g, count, nriscs});
        }
    }
    desc.kernels.push_back(std::move(kernels[0]));
    desc.kernels.push_back(std::move(kernels[1]));
    return desc;
}

}  // namespace ttnn::operations::data_movement
