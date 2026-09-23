// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for pixel_unshuffle with channels_last output: NHWC [N, Ho, Wo, Cp],
// ROW_MAJOR, HEIGHT_SHARDED in L1, written core-locally.
//
// Why this exists: a height-sharded conv2d consumes exactly this tensor - one row-major
// stick of Cp channels per pixel, pixels split over cores in flattened (n, ho, wo) order.
// Producing it directly removes the tilize + permute + re-shard (and the halo's untilize)
// that the NCHW output otherwise needs in front of the conv.
//
// Work split: core i owns the pixels of shard i, [i*shard_h, (i+1)*shard_h). Its two
// dataflow RISCs split every image row of that range at a DRAM-aligned column, each
// reading only the r*C input half-rows it needs and gathering them into the local shard
// with plain L1 stores - no NOC writes at all, and no read amplification between RISCs.
// The output buffer is exposed to the kernel as a CB bound to the sharded buffer.

#include "pixel_unshuffle_device_op.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>

#include "ttnn/common/constants.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

using namespace tt::constants;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor PixelUnshuffle::MultiCoreChannelsLast::create_descriptor(
    const operation_attributes_t& op_attr, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    const Tensor& input = tensor_args.input_tensor;
    const Tensor& output = output_tensor;

    const auto& in_shape = input.logical_shape();
    const uint32_t N = in_shape[0];
    const uint32_t C = in_shape[1];
    const uint32_t H = in_shape[2];
    const uint32_t W = in_shape[3];
    const uint32_t r = op_attr.downscale_factor;
    const uint32_t Ho = H / r;
    const uint32_t Wo = W / r;
    const uint32_t Cp = op_attr.padded_channels;

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(src_buffer != nullptr && dst_buffer != nullptr, "PixelUnshuffle: buffers must be allocated.");

    const tt::DataFormat in_data_fmt = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat out_data_fmt = datatype_to_dataformat_converter(output.dtype());
    const uint32_t datum_sz = tt::datum_size(in_data_fmt);

    // One input row (all W elements) is the staging unit; a RISC reads a DRAM-aligned span
    // of it. Slots are sized for a full row so any span fits.
    const uint32_t row_nbytes_in = W * datum_sz;
    const uint32_t aligned_row_nbytes_in = tt::align(row_nbytes_in, hal::get_dram_alignment());
    const uint32_t rows_per_item = C * r;  // input rows feeding one output image row
    constexpr uint32_t depth = 2;          // image rows in flight per RISC

    // Output: one stick per pixel, Cp elements, page-aligned by the tensor layout.
    const uint32_t out_row_nbytes = static_cast<uint32_t>(dst_buffer->aligned_page_size());
    TT_FATAL(
        out_row_nbytes == Cp * tt::datum_size(out_data_fmt),
        "PixelUnshuffle(channels_last): output stick {} B != padded_channels {} x {} B.",
        out_row_nbytes,
        Cp,
        tt::datum_size(out_data_fmt));

    const auto& shard_spec = output.shard_spec().value();
    const uint32_t shard_h = shard_spec.shape[0];
    const bool row_wise = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const std::vector<CoreCoord> cores = corerange_to_cores(shard_spec.grid, std::nullopt, row_wise);
    const CoreRangeSet& all_cores = shard_spec.grid;
    const uint32_t total_pixels = N * Ho * Wo;

    ProgramDescriptor desc;

    // Output shard as a CB bound to the sharded buffer: the kernel writes at get_write_ptr(cb).
    {
        CBDescriptor cb_out;
        cb_out.total_size = static_cast<uint32_t>(dst_buffer->aligned_size_per_bank());
        cb_out.core_ranges = all_cores;
        cb_out.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_16),
            .data_format = out_data_fmt,
            .page_size = out_row_nbytes,
        });
        cb_out.buffer = dst_buffer;
        desc.cbs.push_back(std::move(cb_out));
    }

    // Private input staging per RISC: depth x (r*C rows). Never pushed/popped.
    const uint32_t cb_in_idx[2] = {tt::CBIndex::c_0, tt::CBIndex::c_1};
    for (uint32_t k = 0; k < 2; k++) {
        CBDescriptor cb_in;
        cb_in.total_size = depth * rows_per_item * aligned_row_nbytes_in;
        cb_in.core_ranges = all_cores;
        cb_in.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_in_idx[k]),
            .data_format = in_data_fmt,
            .page_size = aligned_row_nbytes_in,
        });
        desc.cbs.push_back(std::move(cb_in));
    }

    constexpr const char* kernel_src =
        "ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/kernels/dataflow/"
        "pixel_unshuffle_nhwc_sharded.cpp";

    KernelDescriptor kernels[2];
    for (uint32_t k = 0; k < 2; k++) {
        std::vector<uint32_t> cta = {
            W,                                             // 0
            C,                                             // 1
            H,                                             // 2
            r,                                             // 3
            Ho,                                            // 4
            Wo,                                            // 5
            datum_sz,                                      // 6
            Cp,                                            // 7
            out_row_nbytes,                                // 8
            aligned_row_nbytes_in,                         // 9
            cb_in_idx[k],                                  // 10
            tt::CBIndex::c_16,                             // 11
            static_cast<uint32_t>(op_attr.channel_order),  // 12 (0=CHANNEL_MAJOR, 1=SPATIAL_MAJOR)
            depth,                                         // 13
            k,                                             // 14 which half of each image row
        };
        TensorAccessorArgs(*src_buffer).append_to(cta);

        kernels[k].kernel_source = kernel_src;
        kernels[k].source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernels[k].core_ranges = all_cores;
        kernels[k].compile_time_args = std::move(cta);
    }
    kernels[0].config = ReaderConfigDescriptor{};  // NCRISC
    kernels[1].config = WriterConfigDescriptor{};  // BRISC

    for (uint32_t i = 0; i < cores.size(); i++) {
        const uint32_t pix0 = std::min(i * shard_h, total_pixels);
        const uint32_t npix = std::min(shard_h, total_pixels - pix0);
        for (uint32_t k = 0; k < 2; k++) {
            kernels[k].emplace_runtime_args(cores[i], {src_buffer, pix0, npix});
        }
    }

    desc.kernels.push_back(std::move(kernels[0]));
    desc.kernels.push_back(std::move(kernels[1]));
    return desc;
}

}  // namespace ttnn::operations::data_movement
