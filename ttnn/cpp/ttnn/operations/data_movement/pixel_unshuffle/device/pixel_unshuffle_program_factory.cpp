// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for pixel_unshuffle on NCHW ROW_MAJOR interleaved input.
//
// Work unit: one output stick = (n, c_out, h_out) triple, Wo elements wide.
// For each output stick (n, c_out, h_out):
//   c_in        = c_out / r²
//   rh          = (c_out % r²) / r
//   rw          = c_out % r
//   input_page  = n*C*H + c_in*H + h_out*r + rh   (full W-element input row)
//   output_page = n*C_out*Ho + c_out*Ho + h_out    (Wo-element output row)
//
// Reader reads one full input stick (W elements) into CB per output stick.
// Writer picks every r-th element (stride r, offset rw) and writes Wo elements.

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
    const uint32_t r2 = r * r;
    const uint32_t C_out = C * r2;
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

    // Total output sticks distributed across all cores
    const uint32_t total_sticks = N * C_out * Ho;

    auto compute_grid = device->compute_with_storage_grid_size();
    const uint32_t ncores_x = compute_grid.x;
    const uint32_t ncores_y = compute_grid.y;
    const uint32_t ncores = ncores_x * ncores_y;

    const uint32_t sticks_per_core = tt::div_up(total_sticks, ncores);

    CoreRangeSet all_cores{CoreRange({0, 0}, {ncores_x - 1, ncores_y - 1})};
    auto cores = grid_to_cores(ncores, ncores_x, ncores_y, true);

    ProgramDescriptor desc;

    // CB0: holds one full input stick (W elements)
    const uint32_t cb_in0_idx = tt::CBIndex::c_0;
    {
        CBDescriptor cb;
        cb.total_size = 2 * aligned_stick_nbytes_in;  // double-buffer
        cb.core_ranges = all_cores;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_in0_idx),
            .data_format = in_data_fmt,
            .page_size = aligned_stick_nbytes_in,
        });
        desc.cbs.push_back(std::move(cb));
    }

    // CB1: scratch buffer for packing Wo output elements before DRAM write
    const uint32_t cb_scratch_idx = tt::CBIndex::c_1;
    {
        const uint32_t aligned_stick_nbytes_out = tt::align(stick_nbytes_out, hal::get_dram_alignment());
        CBDescriptor cb;
        cb.total_size = aligned_stick_nbytes_out;
        cb.core_ranges = all_cores;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_scratch_idx),
            .data_format = out_data_fmt,
            .page_size = aligned_stick_nbytes_out,
        });
        desc.cbs.push_back(std::move(cb));
    }

    // Reader compile-time args (9 args before TensorAccessorArgs)
    std::vector<uint32_t> reader_cta = {
        stick_nbytes_in,          // 0
        cb_in0_idx,               // 1
        aligned_stick_nbytes_in,  // 2
        r,                        // 3
        Ho,                       // 4
        W,                        // 5
        C,                        // 6
        H,                        // 7
        N,                        // 8
    };
    TensorAccessorArgs(*src_buffer).append_to(reader_cta);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/kernels/dataflow/"
        "reader_pixel_unshuffle_nchw.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_cta);
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer compile-time args (11 args before TensorAccessorArgs)
    // datum_nbytes is derived inside the kernel as stick_nbytes_in / W — no extra CTA needed.
    std::vector<uint32_t> writer_cta = {
        stick_nbytes_in,          // 0
        cb_in0_idx,               // 1
        aligned_stick_nbytes_in,  // 2
        r,                        // 3
        Ho,                       // 4
        W,                        // 5
        C,                        // 6
        H,                        // 7
        N,                        // 8
        stick_nbytes_out,         // 9
        cb_scratch_idx,           // 10
    };
    TensorAccessorArgs(*dst_buffer).append_to(writer_cta);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/kernels/dataflow/"
        "writer_pixel_unshuffle_nchw.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_cta);
    writer_desc.config = WriterConfigDescriptor{};

    // Per-core runtime args
    for (uint32_t i = 0; i < cores.size(); i++) {
        const CoreCoord core = cores[i];
        const uint32_t start = i * sticks_per_core;
        const uint32_t count = (start < total_sticks) ? std::min(sticks_per_core, total_sticks - start) : 0u;

        reader_desc.emplace_runtime_args(core, {src_buffer, start, count});
        writer_desc.emplace_runtime_args(core, {dst_buffer, start, count});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::operations::data_movement
